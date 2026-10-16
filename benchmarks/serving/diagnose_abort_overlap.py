"""CPU reproduction of Abort between prefill submission and host consumption.

Real scheduler, managers, Req and radix cache; only the already-completed GPU
event and its next-token result are substituted. This is not performance data.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import minisgl.core as core
from minisgl.core import Batch, Req, SamplingParams
from minisgl.message import AbortBackendMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.table import TableManager


def reproduce():
    from minisgl.distributed import set_tp_info,try_get_tp_info
    if try_get_tp_info() is None:set_tp_info(0,1)
    core._GLOBAL_CTX=None;core.set_global_ctx(core.Context(page_size=1))
    page_table=torch.full((1,32),-1,dtype=torch.int32)
    cm=CacheManager(32,1,page_table,'radix');tm=TableManager(1,page_table)
    handle=cm.prefix_cache.match_prefix(torch.arange(7,dtype=torch.int32)).cuda_handle
    cm.lock(handle)
    r=Req(input_ids=torch.arange(8,dtype=torch.int32),table_idx=tm.allocate(),cached_len=0,
          output_len=4,uid=7,sampling_params=SamplingParams(max_tokens=4,ignore_eos=True),cache_handle=handle)
    cm.allocate_paged([r]);r.complete_one()
    s=object.__new__(Scheduler);s.finished_reqs=set();s.eos_token_id=99
    s.cache_manager=cm;s.table_manager=tm;s.decode_manager=DecodeManager(1,{r})
    s.prefill_manager=SimpleNamespace(abort_req=lambda uid:None,runnable=False)
    replies=[];s.send_result=lambda xs:replies.extend(xs)
    s._process_one_msg(AbortBackendMsg(uid=7))
    before={'free_pages':len(cm.free_slots),'cache_pages':cm.prefix_cache.size_info.total_size,
            'free_tables':tm.available_size}
    data=(SimpleNamespace(batch=Batch([r],'prefill')),
          (None,torch.tensor([42],dtype=torch.int32),SimpleNamespace(synchronize=lambda:None)))
    error=None
    try:
        s._process_last_data(data);cm.check_integrity()
    except Exception as e:error=repr(e)
    return {'source':str(Path(__import__('minisgl.scheduler.scheduler',fromlist=['']).__file__)),
            'scenario':'Abort after submitted final prefill, before its host result',
            'before_late_result':before,'after_late_result':{'free_pages':len(cm.free_slots),
                'cache_pages':cm.prefix_cache.size_info.total_size,'free_tables':tm.available_size},
            'num_pages':cm.num_pages,'replies_after_abort':len(replies),'error':error}


if __name__=='__main__':print(json.dumps(reproduce(),indent=2))
