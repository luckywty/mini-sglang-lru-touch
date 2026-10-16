"""Reproduce the real host/device progress skew without executing a GPU model.

Run with PYTHONPATH=runtime-H/python or runtime-S/python after the common repair.
The actual Req and Scheduler result handler are used; only I/O/CUDA completion
and resource-release effects are substituted.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.decode import DecodeManager


def scheduler():
    s=object.__new__(Scheduler)
    s.finished_reqs=set()
    s.eos_token_id=99
    s.decode_manager=DecodeManager(1)
    s.cache_manager=SimpleNamespace(lazy_free_region=nullcontext,cache_req=lambda *a,**k:None)
    s.replies=[]
    s.released=[]
    s.send_result=lambda replies:s.replies.extend(replies)
    s._free_req_resources=lambda req:s.released.append(req.uid)
    return s


def req(max_tokens,ignore=True):
    return Req(input_ids=torch.tensor([1,2],dtype=torch.int32),table_idx=0,cached_len=0,
               output_len=max_tokens,uid=7,cache_handle=None,
               sampling_params=SamplingParams(max_tokens=max_tokens,ignore_eos=ignore))


def consume(s,r,token):
    b=Batch(reqs=[r],phase='decode')
    inp=SimpleNamespace(batch=b)
    out=(None,torch.tensor([token],dtype=torch.int32),SimpleNamespace(synchronize=lambda:None))
    s._process_last_data((inp,out))


def test_next_device_submission_cannot_end_previous_host_token():
    s=scheduler();r=req(3)
    # Prefill output is ready; one next decode has already been submitted.
    r.complete_one();r.complete_one()
    consume(s,r,10)
    assert not s.replies[-1].finished
    # Last decode submitted, but host is consuming the penultimate output.
    r.complete_one()
    consume(s,r,11)
    assert not s.replies[-1].finished
    consume(s,r,12)
    assert [m.finished for m in s.replies]==[False,False,True]
    assert s.released==[7]


def test_eos_discards_already_submitted_speculative_result_once():
    s=scheduler();r=req(20,ignore=False)
    r.complete_one();r.complete_one()
    consume(s,r,99)
    assert len(s.replies)==1 and s.replies[0].finished
    consume(s,r,33)
    assert len(s.replies)==1 and s.released==[7]
    assert r.input_ids.tolist()==[1,2,99]


def test_abort_of_inflight_prefill_cannot_recached_owned_pages():
    import minisgl.core as core
    from diagnose_abort_overlap import reproduce
    old=core._GLOBAL_CTX
    try:
        report=reproduce()
        assert report['error'] is None
        assert report['replies_after_abort']==0
        assert report['after_late_result']==report['before_late_result']
        state=report['after_late_result']
        assert state['free_pages']+state['cache_pages']==report['num_pages']
    finally:
        core._GLOBAL_CTX=old
