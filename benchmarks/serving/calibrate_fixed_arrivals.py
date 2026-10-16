"""Fixed-arrival tail calibration; capacity cost remains an off/on comparison.

Closed-loop ACK feedback changes arrivals, so its ITL differences cannot identify
observer cost. Token mode is the minimal exact server-time reference; a fully
disabled observer has no server-time records and cannot supply server ITL.
"""
import json
import os
from pathlib import Path

from analyze_results import audit_run,paired_ci
from generate_workloads import Factory,save
from run_experiments import OUT,run_cell,atomic


def main():
    root=OUT/'preflight/fixed-arrival-calibration';root.mkdir(exist_ok=True)
    trace=root/'W1-poisson-8.jsonl'
    if not trace.exists():
        rows,_=Factory(seed=81933).make('W1',512,8.,'poisson')
        save(trace,rows,dict(workload='W1',role='fixed-arrival observer/affinity calibration',
                             seed=81933,rate=8.,arrival='poisson'))
    observations=[]
    for block in range(5):
        order=[('token','8,9'),('light','8,9'),('light','8')]
        order=order[block%3:]+order[:block%3]
        for mode,affinity in order:
            os.sched_setaffinity(0,{8,9} if affinity=='8,9' else {8})
            directory=root/f'block-{block}-{mode}-cpu{affinity.replace(",","_")}'
            info=run_cell('S1000',trace,directory,mode='open-loop',collector_mode=mode,settle_seconds=0)
            if info['status']!='complete':raise RuntimeError(info)
            actual=Path(info.get('directory',directory))
            result=audit_run(actual)
            assert result['status']=='valid',result['issues']
            observation={k:result[k] for k in ('directory','output_tok_s','itl_server_ms_p99',
                         'itl_client_ack_ms_p99','tpot_server_ms_p95','loop_lag_p99_ms',
                         'send_lag_ms_p99','client_cpu_fraction','client_timing_clean')}
            observation.update(block=block,mode=mode,affinity=affinity)
            observations.append(observation)
            atomic(root/'progress.json',observations)
    os.sched_setaffinity(0,{8,9})
    token={r['block']:r for r in observations if r['mode']=='token'}
    light={r['block']:r for r in observations if r['mode']=='light' and r['affinity']=='8,9'}
    single={r['block']:r for r in observations if r['affinity']=='8'}
    observer=paired_ci([(token[i],light[i]) for i in range(5)],'itl_server_ms_p99',latency=True)
    affinity=paired_ci([(light[i],single[i]) for i in range(5)],'itl_server_ms_p99',latency=True)
    capacity=json.loads((OUT/'preflight/collector-calibration-light/acceptance.json').read_text())
    cost=capacity['collector']['output_tok_s']
    affinity_capacity=capacity['client_affinity']['output_tok_s']
    passed=(abs(cost['median_improvement_pct'])<=1 and
            abs(observer['median_improvement_pct'])<=2 and
            abs(affinity['median_improvement_pct'])<=3 and
            abs(affinity_capacity['median_improvement_pct'])<=3 and
            all(r['client_timing_clean'] for r in observations))
    receipt=dict(passed=bool(passed),protocol='fixed-arrival server-tail + closed-loop capacity',
              observer_server_itl_p99=observer,affinity_server_itl_p99=affinity,
              saturated_capacity_light_vs_off=cost,saturated_capacity_affinity=affinity_capacity,
              observations=observations,
              limitations='Token observer itself is required to define host emit time. Fully-off client ACK P99 is not substituted for server ITL. Closed-loop tail checks are retained as confounded diagnostics, not accepted evidence.',
              threshold=dict(capacity_bias_pct=1,server_tail_increment_pct=2,affinity_main_metric_pct=3))
    atomic(root/'acceptance.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='observations'}),flush=True)


if __name__=='__main__':main()
