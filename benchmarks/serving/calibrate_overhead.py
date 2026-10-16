"""Paired collector on/off and client-affinity checks, outside formal results."""
import json
import os
from pathlib import Path

import numpy as np

from generate_workloads import Factory,save
from run_experiments import OUT,run_cell,atomic
from analyze_results import audit_run,paired_ci


def calibration_metrics(directory):
    directory=Path(directory)
    client=json.loads((directory/'client.json').read_text())
    rows=client['requests']
    assert len(rows)==512 and all(r['status']=='success' and len(r['ack_ns'])==128 for r in rows)
    t=(max(r['terminal_ns'] for r in rows)-client['start_ns'])/1e9
    gaps=[(b-a)/1e6 for r in rows for a,b in zip(r['ack_ns'],r['ack_ns'][1:])]
    return dict(directory=str(directory),output_tok_s=512*128/t,
                itl_client_ack_ms_p99=float(np.quantile(gaps,.99)),
                ttft_text_ms_p95=float(np.quantile([(r['first_text_ns']-r['send_ns'])/1e6 for r in rows],.95)),
                loop_lag_p99_ms=client['loop_lag_p99_ms'])


def main():
    root=OUT/'preflight/collector-calibration-light';root.mkdir(exist_ok=True)
    trace=root/'W1-calibration.jsonl'
    if not trace.exists():
        rows,_=Factory(seed=81932).make('W1',512,1,'closed')
        save(trace,rows,dict(role='collector overhead calibration',concurrency=16))
    observations=[]
    for i in range(5):
        order=[('off','8,9'),('light','8,9'),('light','8')]
        order=order[i%3:]+order[:i%3]
        if i%2:order=list(reversed(order))
        for mode,affinity in order:
            os.sched_setaffinity(0,{8,9} if affinity=='8,9' else {8})
            directory=root/f'block-{i}-{mode}-cpu{affinity.replace(",", "_")}'
            info=run_cell('S1000',trace,directory,mode='closed-loop',concurrency=16,
                           settle_seconds=0,collector_mode=mode)
            if info['status']!='complete':raise RuntimeError(info)
            if mode=='light':assert audit_run(directory)['status']=='valid'
            observation=calibration_metrics(directory);observation.update(block=i,mode=mode,affinity=affinity)
            observations.append(observation)
    os.sched_setaffinity(0,{8,9})
    off={r['block']:r for r in observations if r['mode']=='off'}
    full={r['block']:r for r in observations if r['mode']=='light' and r['affinity']=='8,9'}
    single={r['block']:r for r in observations if r['affinity']=='8'}
    collector_pairs=[(off[i],full[i]) for i in range(5)]
    affinity_pairs=[(full[i],single[i]) for i in range(5)]
    collector={m:paired_ci(collector_pairs,m,latency=m!='output_tok_s') for m in ('output_tok_s','itl_client_ack_ms_p99')}
    affinity={m:paired_ci(affinity_pairs,m,latency=m!='output_tok_s') for m in ('output_tok_s','itl_client_ack_ms_p99')}
    passed=(abs(collector['output_tok_s']['median_improvement_pct'])<=1 and
            abs(collector['itl_client_ack_ms_p99']['median_improvement_pct'])<=2 and
            abs(affinity['output_tok_s']['median_improvement_pct'])<=3)
    receipt=dict(passed=bool(passed),observations=observations,collector=collector,client_affinity=affinity,
                 limitations='Short calibration runs; use measured paired CIs and preserve all runs. Off mode still uses common begin/dump boundary control.',
                 frozen_client_transport='prewarmed raw HTTP1.1',server_cpus='0-7',client_cpus='8,9')
    atomic(root/'acceptance.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='observations'}),flush=True)


if __name__=='__main__':main()
