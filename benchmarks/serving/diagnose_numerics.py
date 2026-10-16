"""Fixed-input, single-next-token logit audit. Never a performance benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path('/home/wangwei/ai-infra/Promotion')
OUT=ROOT/'benchmark-results/preflight/numerics'
MODEL='/home/wangwei/ai-infra/models/Qwen3-0.6B'


def prepare():
    OUT.mkdir(exist_ok=True)
    checks=json.loads((ROOT/'benchmark-results/preflight/greedy-comparison.json').read_text())
    contexts={}
    for check in checks:
        if not check['differences']:continue
        directory=Path(check['reference'])
        receipt=json.loads((directory/'run.json').read_text())
        rows={r['request_id']:r for r in map(json.loads,Path(receipt['trace']).read_text().splitlines())}
        for rid,diff in check['differences'].items():
            a,b=diff['reference'],diff['actual']
            k=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
            ids=rows[rid]['input_ids']+a[:k]
            key=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            contexts[key]=dict(input_ids=ids,reference_token=a[k],candidate_token=b[k],
                               first_differing_output_index=k,request_id=rid,
                               reference=check['reference'],candidate=check['candidate'])
    (OUT/'contexts.json').write_text(json.dumps(list(contexts.values()),indent=2))
    # Registered before inspecting logits. This is a post-discrepancy numerical
    # diagnostic, explicitly not a retroactive assertion of bitwise equivalence.
    thresholds=dict(weight_dtype='bfloat16',torch_reference='HF sdpa BF16',
                    per_context_max_absolute_logit_error=0.5,
                    per_context_p99_absolute_logit_error=0.125,
                    changed_argmax_candidate_within_reference_best_logit=0.125,
                    selection='every first differing prefix from the frozen W0 audit',
                    caveat='Post-discrepancy registration; fixed inputs and thresholds before logit evaluation.')
    (OUT/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
    print('frozen diagnostic contexts',len(contexts),flush=True)


def mini(budget,output):
    import torch
    from minisgl.llm import LLM
    from minisgl.core import SamplingParams
    from minisgl.scheduler.prefill import ChunkedReq
    from minisgl.core import get_global_ctx

    contexts=json.loads((OUT/'contexts.json').read_text())
    llm=LLM(MODEL,dtype=torch.bfloat16,max_running_req=16,attention_backend='fi',cache_type='radix',
            page_size=1,num_page_override=16384,max_seq_len_override=2304,
            max_extend_tokens=budget,cuda_graph_max_bs=16,memory_ratio=.7)
    logits_by_uid={}
    active=[]
    original_forward=llm.engine.forward_batch
    def forward(batch,args):
        active[:]=[batch]
        return original_forward(batch,args)
    llm.engine.forward_batch=forward
    original_sample=llm.engine.sampler.sample
    def sample(logits,args):
        b=active[0]
        for i,r in enumerate(b.reqs):
            if not isinstance(r,ChunkedReq):
                # Deliberate diagnostic-only copy; absent from serving sources.
                logits_by_uid[r.uid]=logits[i].detach().float().cpu().numpy()
        return original_sample(logits,args)
    llm.engine.sampler.sample=sample
    with torch.inference_mode():
        generated=llm.generate([c['input_ids'] for c in contexts],SamplingParams(max_tokens=1,ignore_eos=True))
    assert len(logits_by_uid)==len(contexts)
    np.savez(output,logits=np.stack([logits_by_uid[i] for i in range(len(contexts))]))
    Path(str(output)+'.json').write_text(json.dumps(dict(budget=budget,predictions=[int(np.argmax(logits_by_uid[i])) for i in range(len(contexts))],
                       memory_peak_allocated=torch.cuda.max_memory_allocated(),
                       memory_peak_reserved=torch.cuda.max_memory_reserved()),indent=2))
    llm.shutdown()


def hf(output):
    import torch
    from transformers import AutoModelForCausalLM
    contexts=json.loads((OUT/'contexts.json').read_text())
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,
                           dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    rows=[]
    with torch.inference_mode():
        for c in contexts:
            ids=torch.tensor([c['input_ids']],dtype=torch.long,device='cuda')
            # Only the final-position logits are needed; keep the same BF16 model.
            logits=model(ids,use_cache=False,logits_to_keep=1).logits[0,-1]
            rows.append(logits.float().cpu().numpy())
    np.savez(output,logits=np.stack(rows))
    Path(str(output)+'.json').write_text(json.dumps(dict(dtype='bfloat16',backend='sdpa',
                           peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()),indent=2))


def compare():
    contexts=json.loads((OUT/'contexts.json').read_text())
    threshold=json.loads((OUT/'thresholds.json').read_text())
    arrays={p.stem:np.load(p)['logits'] for p in OUT.glob('*.npz')}
    rows=[]
    reference=arrays['hf']
    for name,logits in arrays.items():
        for i,(a,b) in enumerate(zip(reference,logits)):
            err=np.abs(a-b)
            best=int(np.argmax(a));candidate=int(np.argmax(b))
            gap=float(a[best]-a[candidate])
            rows.append(dict(source=name,context_index=i,input_tokens=len(contexts[i]['input_ids']),
               request_id=contexts[i]['request_id'],original_first_difference=contexts[i]['first_differing_output_index'],
               hf_argmax=best,mini_argmax=candidate,max_abs=float(np.max(err)),p99_abs=float(np.quantile(err,.99)),
               rms=float(np.sqrt(np.mean(err**2))),candidate_gap_in_hf=gap,
               original_ref_token=contexts[i]['reference_token'],original_candidate_token=contexts[i]['candidate_token'],
               original_pair_hf_logit_gap=float(abs(a[contexts[i]['reference_token']]-a[contexts[i]['candidate_token']])),
               passed=bool(np.max(err)<=threshold['per_context_max_absolute_logit_error'] and
                           np.quantile(err,.99)<=threshold['per_context_p99_absolute_logit_error'] and
                           gap<=threshold['changed_argmax_candidate_within_reference_best_logit'])))
    result=dict(passed=all(r['passed'] for r in rows),contexts=len(contexts),rows=rows,
                worst_max_abs=max(r['max_abs'] for r in rows),worst_p99_abs=max(r['p99_abs'] for r in rows),
                worst_changed_argmax_gap=max(r['candidate_gap_in_hf'] for r in rows))
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','mini','hf','compare'])
    p.add_argument('--budget',type=int,default=256);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='mini':mini(a.budget,a.output)
    elif a.phase=='hf':hf(a.output)
    else:compare()
