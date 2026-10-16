"""HF BF16 reference for the exact frozen natural-EOS prompts, not a benchmark.

Run with serving processes stopped; diagnostic logits copies stay in this
standalone process and never enter the serving hot path.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_experiments import OUT, MODEL, atomic


def main():
    target=OUT/'preflight/natural-hf-reference.json'
    if target.exists():raise RuntimeError('Reference exists; preserve it and inspect before rerunning')
    rows=[json.loads(x) for x in (OUT/'traces/W0-natural.jsonl').read_text().splitlines()]
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
                                            attn_implementation='sdpa').to('cuda').eval()
    results=[]
    with torch.inference_mode():
        for r in rows:
            ids=torch.tensor([r['input_ids']],dtype=torch.long,device='cuda')
            logits=model(ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            values,indices=torch.topk(logits,10)
            out=model.generate(ids,max_new_tokens=r['max_output_tokens'],do_sample=False,
                               pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
            generated=out[0,ids.shape[1]:].cpu().tolist()
            results.append({'request_id':r['request_id'],'input_ids':r['input_ids'],
                'generated_ids':generated,'output_text':tok.decode(generated,skip_special_tokens=True),
                'first_token_top10':[{'id':int(i),'text':tok.decode([int(i)]),'logit':float(v)} for i,v in zip(indices.cpu(),values.cpu())]})
            print(r['request_id'],results[-1]['output_text'],flush=True)
    atomic(target,{'model':MODEL,'dtype':'bfloat16','backend':'HF SDPA','role':'semantic diagnostic, not performance measurement',
                   'rows':results,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})


if __name__=='__main__':main()
