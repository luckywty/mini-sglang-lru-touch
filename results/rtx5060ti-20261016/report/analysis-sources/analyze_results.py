"""Audit raw streams first, then calculate run-level and paired statistics."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import pickle

import numpy as np


def quantiles(values, prefix):
    a=np.asarray([v for v in values if v is not None],dtype=float)
    if not len(a):
        return {prefix+'_'+k:None for k in ('p50','p95','p99','max','mean')}
    return {prefix+'_'+k:float(v) for k,v in zip(('p50','p95','p99','max','mean'),
             [*np.quantile(a,[.5,.95,.99],method='linear'),np.max(a),np.mean(a)])}


def audit_run(directory):
    directory=Path(directory)
    info=json.loads((directory/'run.json').read_text())
    if not (directory/'measurement.pkl').exists() or not (directory/'client.json').exists():
        return dict(directory=str(directory),status=info['status'],variant=info['variant'],
                    experiment=info.get('experiment'),seed=info.get('seed'),profile=info.get('profile'),
                    trace_sha256=info.get('trace_sha256'),selected_for_primary=False,
                    error=info.get('error'),issues=[info.get('error','missing measurement')])
    trace=[json.loads(x) for x in Path(info['trace']).read_text().splitlines()]
    trace_map={x['request_id']:x for x in trace}
    client=json.loads((directory/'client.json').read_text())
    server=pickle.loads((directory/'measurement.pkl').read_bytes())
    tokens=np.array(server['tokens'],dtype=np.int64).reshape(-1,5)
    by_uid=defaultdict(list)
    for row in tokens:
        by_uid[int(row[0])].append(row)
    queues={x[0]:x for x in server['queued']}
    admissions={x[0]:x for x in server['admissions']}
    issues=[]
    metrics=defaultdict(list)
    success=[]
    merged=[]
    for r in client['requests']:
        expected=trace_map[r['request_id']]
        uid=r['uid']
        observed=by_uid.get(uid,[])
        queue=queues.get(uid)
        admission=admissions.get(uid)
        errors=[]
        expected_timeout=r['status']=='timeout'
        if r['status']!='success':
            errors.append(r['status'])
        if queue is None or queue[2]!=expected['input_tokens'] or queue[4]!=expected['input_ids']:
            errors.append('input_ids_or_length_mismatch')
        count=len(observed)
        if expected['ignore_eos'] and count!=expected['max_output_tokens']:
            errors.append('generated_length_mismatch')
        if len(r['ack_ns'])!=count:
            errors.append('sse_ack_token_mismatch')
        if count and (list(int(t[1]) for t in observed)!=list(range(count)) or
                      sum(int(t[4]) for t in observed)!=1 or observed[-1][4]!=1):
            errors.append('ordinal_or_terminal_mismatch')
        if not count:
            errors.append('no_generated_tokens')
        s_times=np.array([x[3] for x in observed],dtype=np.int64)
        ack=np.asarray(r['ack_ns'],dtype=np.int64)
        if len(ack)==count and count and np.any(ack<s_times):
            errors.append('clock_domain_or_uid_mismatch')
        if errors and not expected_timeout:
            issues.append({'request':r['request_id'],'uid':uid,'errors':errors})
        if expected_timeout and count:
            if [int(t[1]) for t in observed]!=list(range(count)) or sum(int(t[4]) for t in observed)>1:
                issues.append({'request':r['request_id'],'errors':['corrupt_partial_token_stream']})
        actual=dict(**r,input_tokens=queue[2] if queue else None,generated_tokens=count,
                    cached_tokens=admission[2] if admission else None,
                    queue_ns=queue[1] if queue else None,admission_ns=admission[1] if admission else None,
                    waiting_passes=admission[3] if admission else None,
                    actual_terminal_count=sum(int(x[4]) for x in observed),audit_errors=errors,
                    **{'class':expected['class']})
        if not errors:
            success.append(actual)
            if r['first_text_ns'] is not None:
                metrics['ttft_text_ms'].append((r['first_text_ns']-r['send_ns'])/1e6)
                metrics[expected['class']+'_ttft_text_ms'].append((r['first_text_ns']-r['send_ns'])/1e6)
            metrics['ttft_ack_ms'].append((ack[0]-r['send_ns'])/1e6)
            metrics['ttft_server_queue_ms'].append((s_times[0]-queue[1])/1e6)
            metrics['itl_server_ms']+=list(np.diff(s_times)/1e6)
            metrics['itl_client_ack_ms']+=list(np.diff(ack)/1e6)
            if count>1:
                metrics['tpot_server_ms'].append((s_times[-1]-s_times[0])/1e6/(count-1))
                metrics['tpot_client_ack_ms'].append((ack[-1]-ack[0])/1e6/(count-1))
                metrics['request_itl_server_p95_ms'].append(float(np.quantile(np.diff(s_times)/1e6,.95)))
            e2e=(r['terminal_ns']-r['send_ns'])/1e6
            metrics['e2e_ms'].append(e2e)
            metrics['arrival_to_done_ms'].append((r['terminal_ns']-r['planned_ns'])/1e6)
            slo=2000 if expected['class']=='long' else 500
            ttft=(r['first_text_ns']-r['send_ns'])/1e6 if r['first_text_ns'] else np.inf
            actual['goodput_client']=bool(ttft<=slo and e2e<=slo+50*(count-1)+500)
            actual['goodput_server']=bool((s_times[0]-queue[1])/1e6<=slo and
                  (count<=1 or ((s_times[-1]-s_times[0])/1e6/(count-1)<=50 and np.quantile(np.diff(s_times)/1e6,.95)<=100)))
        merged.append(actual)
    uid_expected={r['uid']:trace_map[r['request_id']] for r in client['requests'] if r['uid'] is not None}
    # First-admission waits use every admitted request, including a request that
    # later times out. Success-only waiting distributions hide starvation.
    for uid,h,*_ in server['admissions']:
        if uid in queues:
            wait=(h-queues[uid][1])/1e6
            metrics['wait_ms'].append(wait)
            metrics[uid_expected.get(uid,{}).get('class','unknown')+'_wait_ms'].append(wait)
    metrics['send_lag_ms']=[max(0,(r['send_ns']-r['planned_ns'])/1e6) for r in client['requests']]
    duration=(max(x['terminal_ns'] for x in client['requests'])-client['start_ns'])/1e9
    total_out=sum(x['generated_tokens'] for x in success)
    total_in=sum(x['input_tokens'] for x in success)
    input_admitted=sum(queues[x[0]][2]-1 for x in server['admissions'])
    cached=sum(x[2] for x in server['admissions'])
    resource=server['resources']
    if resource['pending'] or resource['running'] or resource['protected']:
        issues.append('resources_still_active_at_dump')
    if resource['free_tables']!=resource['table_capacity'] or resource['unique_free_tables']!=resource['free_tables']:
        issues.append('table_leak_or_double_free')
    if resource['free_pages']+resource['evictable']!=resource['num_pages']:
        issues.append('page_accounting_mismatch')
    if resource.get('unique_free_pages',resource['free_pages'])!=resource['free_pages']:
        issues.append('duplicate_free_pages')
    if server['dropped_events']:
        issues.append('dropped_collector_events')
    budgets=server['budgets']
    for _,n,b,u,t,f,dynamic in budgets:
        expected_b=min(256,u,max(f,t-n)) if dynamic else min(256,u)
        if b!=expected_b:
            issues.append('budget_formula_mismatch')
    rows=list(csv.DictReader((directory/'gpu-telemetry.csv').open()))
    def column(key):
        result=[]
        for row in rows:
            for k,v in row.items():
                if k and key==k.strip().split(' [')[0]:
                    try:
                        result.append(float(v.strip().split()[0]))
                    except ValueError:
                        pass
        return result
    free=column('memory.free')
    min_free=min(free) if free else None
    if min_free is not None and min_free<1024:
        issues.append('less_than_1GiB_free')
    vals=dict(directory=str(directory),status='valid' if not issues else 'invalid',
              variant=info['variant'],profile=info['profile'],trace_sha256=info['trace_sha256'],
              collector_sha256=info['collector_sha256'],source_manifest_sha256=info['source_manifest_sha256'],
              workload=json.loads(Path(info['trace']).with_suffix('.manifest.json').read_text()).get('workload',Path(info['trace']).name.split('-')[0])
                       if Path(info['trace']).with_suffix('.manifest.json').exists() else Path(info['trace']).name.split('-')[0],
              experiment=info.get('experiment',next((part for part in directory.parts if part.startswith('E') and len(part)>1 and part[1].isdigit()),directory.parent.parent.name)),
              seed=info.get('seed',next((int(part[5:]) for part in directory.parts if part.startswith('seed-')),None)),
              offered=len(trace),success=len(success),failed=len(trace)-len(success),
              request_status_counts=dict(Counter(r['status'] for r in client['requests'])),
              timeout_rate=sum(r['status']=='timeout' for r in client['requests'])/len(trace),
              duration_s=duration,injection_s=max(x['arrival_s'] for x in trace),
              drain_s=duration-max(x['arrival_s'] for x in trace),
              input_tokens=total_in,generated_tokens=total_out,cached_tokens=cached,
              output_tok_s=total_out/duration,total_logical_tok_s=(total_in+total_out)/duration,
              success_req_s=len(success)/duration,
              goodput_req_s=sum(x.get('goodput_client',False) for x in success)/duration,
              goodput_server_req_s=sum(x.get('goodput_server',False) for x in success)/duration,
              prefix_token_hit_ratio=cached/input_admitted if input_admitted else 0,
              prefix_request_hit_ratio=sum(x[2]>0 for x in server['admissions'])/max(1,len(server['admissions'])),
              starvation_rate=sum((x[1]-queues[x[0]][1])>5e9 for x in server['admissions'])/max(1,len(trace)),
              unadmitted=len(trace)-len(server['admissions']),
              server_interval_count=sum(max(0,len(x)-1) for x in by_uid.values()),
              loop_lag_p99_ms=client['loop_lag_p99_ms'],client_cpu_seconds=client['cpu_seconds'],
              client_cpu_fraction=client['cpu_seconds']/client['wall_seconds'],
              sampled_min_free_mib=min_free,sampled_max_used_mib=max(column('memory.used'),default=None),
              peak_allocated_mib=server['memory']['peak_allocated']/2**20,
              peak_reserved_mib=server['memory']['peak_reserved']/2**20,
              gpu_temperature_median=float(np.median(column('temperature.gpu'))) if column('temperature.gpu') else None,
              gpu_power_median=float(np.median(column('power.draw'))) if column('power.draw') else None,
              graph_capture_list=server['graph_capture_list'],thermal_stable=info.get('thermal_stable'),
              budget_count=len(budgets),budget_values=sorted(set(x[2] for x in budgets)),
              budget_decode_values=sorted(set(x[1] for x in budgets)),
              mechanism_counts=server['counts'],issues=issues)
    prime_groups=set()
    if info.get('prime_trace'):
        prime_groups={r['prefix_group'] for r in map(json.loads,Path(info['prime_trace']).read_text().splitlines())}
    completed=sorted((rows[0][3],uid_expected[uid].get('prefix_group','')) for uid,rows in by_uid.items() if uid in uid_expected)
    seen=set(prime_groups);index=0;recompute=compulsory=0
    for uid,h,cached,*_ in sorted(server['admissions'],key=lambda x:x[1]):
        while index<len(completed) and completed[index][0]<h:
            seen.add(completed[index][1]);index+=1
        expected=uid_expected.get(uid,{})
        shared=expected.get('shared_prefix_tokens',0)
        missed=max(0,shared-min(shared,cached))
        if expected.get('prefix_group','') in seen:recompute+=missed
        else:compulsory+=missed
    batches=server.get('batches',[])
    vals['shared_prefix_recomputed_tokens']=recompute
    vals['compulsory_shared_prefix_tokens']=compulsory
    vals['evicted_tokens']=server['counts'].get('evicted_tokens',0)
    vals['prefill_batches']=sum(b[1]=='prefill' for b in batches)
    vals['decode_batches']=sum(b[1]=='decode' for b in batches)
    vals['prefill_forward_tokens']=sum(b[7] for b in batches if b[1]=='prefill')
    vals['decode_forward_tokens']=sum(b[2] for b in batches if b[1]=='decode')
    vals['graph_replay_batches']=sum(b[11] for b in batches)
    vals['eager_batches']=sum(not b[11] for b in batches)
    vals['competing_decode_submissions']=sum(b[1]=='decode' and b[4]>0 and b[5]>0 for b in batches)
    vals['protected_attempts']=len(budgets)
    vals['max_admitted_waiting_passes']=max((a[3] for a in server['admissions']),default=0)
    vals['actual_prefill_tokens_values']=sorted(set(b[7] for b in batches if b[1]=='prefill'))
    measured_batches=[b for b in batches if client['start_ns']<=b[0]<max(r['terminal_ns'] for r in client['requests'])]
    if measured_batches:
        occupancy=[(resource['num_pages']-b[8])/resource['num_pages'] for b in measured_batches]
        weights=np.diff([b[0] for b in measured_batches]+[max(r['terminal_ns'] for r in client['requests'])])
        vals['kv_occupancy_time_weighted_mean']=float(np.average(occupancy,weights=weights))
        vals['kv_occupancy_max']=max(occupancy)
    for name,values in metrics.items():
        vals.update(quantiles(values,name))
    decisions=np.array(server['decision_prepare_ns']).reshape(-1,2)
    vals.update(quantiles(decisions[:,0]/1000 if len(decisions) else [],'scheduler_us'))
    vals.update(quantiles(decisions[:,1]/1000 if len(decisions) else [],'prepare_us'))
    itl_median=vals.get('itl_server_ms_p50')
    vals['client_timing_limit_ms']=min(1,.05*itl_median) if itl_median else 1.
    vals['client_timing_clean']=bool((info.get('mode')=='closed-loop' or vals.get('send_lag_ms_p99',1e9)<1) and
                                    vals['loop_lag_p99_ms']<vals['client_timing_limit_ms'])
    formal='measure' in directory.parts
    pollution=[]
    if formal and not vals['client_timing_clean']:pollution.append('client_lag_exceeds_frozen_limit')
    if formal and info.get('thermal_stable') is not True:pollution.append('thermal_window_unstable')
    vals['performance_pollution']=pollution
    vals['selected_for_primary']=bool(info.get('selected_for_primary',not pollution))
    (directory/'requests.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in merged))
    tokens.tofile(directory/'tokens.bin')
    (directory/'summary.json').write_text(json.dumps(vals,indent=2,ensure_ascii=False))
    return vals


def paired_ci(pairs, metric, latency=False):
    # Resample paired *runs*, never requests/tokens. The same seed is one block.
    values=np.array([(1-b[metric]/a[metric]) if latency else (b[metric]/a[metric]-1)
                     for a,b in pairs],dtype=float)
    rng=np.random.default_rng(31723)
    samples=np.median(values[rng.integers(0,len(values),(10000,len(values)))],axis=1)
    return dict(n=len(values),median_improvement_pct=float(np.median(values)*100),
                ci95_low_pct=float(np.quantile(samples,.025)*100),
                ci95_high_pct=float(np.quantile(samples,.975)*100),
                paired_improvements_pct=list(values*100))


def analyze(root,output):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    runs=[]
    for path in sorted(root.rglob('run.json')):
        if 'attempt-' in str(path):
            # Attempts remain in the error ledger as independent receipts.
            pass
        try:
            runs.append(audit_run(path.parent))
        except Exception as e:
            runs.append(dict(directory=str(path.parent),status='analysis_failed',error=repr(e)))
    (output/'runs.json').write_text(json.dumps(runs,indent=2,ensure_ascii=False))
    fields=sorted({k for r in runs for k,v in r.items() if not isinstance(v,(dict,list))})
    with (output/'runs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(runs)
    (output/'errors.json').write_text(json.dumps([r for r in runs if r['status']!='valid' or r.get('performance_pollution') or r.get('failed',0)>0],indent=2,ensure_ascii=False))
    groups=defaultdict(list)
    for r in runs:
        if r['status']=='valid' and r.get('selected_for_primary',True):
            groups[(r['experiment'],r['variant'],r['profile'])].append(r)
    aggregates=[]
    for key,values in groups.items():
        agg=dict(experiment=key[0],variant=key[1],profile=key[2],n=len(values))
        for metric in ('output_tok_s','goodput_req_s','ttft_text_ms_p95','ttft_text_ms_p99','itl_server_ms_p99',
                       'itl_client_ack_ms_p99','tpot_server_ms_p95','long_ttft_text_ms_p95','cold_wait_ms_p95',
                       'cold_wait_ms_max','prefix_token_hit_ratio','starvation_rate','scheduler_us_mean'):
            a=[r[metric] for r in values if isinstance(r.get(metric),(int,float))]
            if a:
                med=float(np.median(a))
                agg[metric]=dict(median=med,iqr=float(np.quantile(a,.75)-np.quantile(a,.25)),
                                 mad=float(np.median(np.abs(np.array(a)-med))),min=min(a),max=max(a))
        aggregates.append(agg)
    (output/'aggregates.json').write_text(json.dumps(aggregates,indent=2))
    comparisons=[]
    for exp in sorted(set(r.get('experiment') for r in runs if r.get('experiment'))):
        valid=[r for r in runs if r['status']=='valid' and r.get('selected_for_primary',True) and r['experiment']==exp]
        variants=sorted(set(r['variant'] for r in valid))
        for a in variants:
            for b in variants:
                if a>=b:
                    continue
                first={r['trace_sha256']:r for r in valid if r['variant']==a}
                second={r['trace_sha256']:r for r in valid if r['variant']==b}
                keys=first.keys()&second.keys()
                if len(keys)<5:
                    continue
                pairs=[(first[k],second[k]) for k in sorted(keys)]
                # Pairing requires the same resource and observation contract.
                assert all(x['profile']==y['profile'] and x['collector_sha256']==y['collector_sha256'] for x,y in pairs)
                for m,latency in [('output_tok_s',False),('itl_server_ms_p99',True),
                                  ('long_ttft_text_ms_p95',True),('cold_wait_ms_p95',True)]:
                    p=[(x,y) for x,y in pairs if x.get(m) and y.get(m)]
                    if len(p)>=5:
                        comparisons.append(dict(experiment=exp,reference=a,variant=b,metric=m,
                                                evidence_scope='exploratory ITL: global tail calibration failed' if m=='itl_server_ms_p99' else 'conditional fixed-length measurement',
                                                **paired_ci(p,m,latency)))
    (output/'paired-ci.json').write_text(json.dumps(comparisons,indent=2))
    return runs,aggregates,comparisons


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--results',required=True)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    r,ag,c=analyze(a.results,a.output_dir)
    print(json.dumps({'runs':len(r),'statuses':dict(Counter(x['status'] for x in r)),
                      'groups':len(ag),'paired_comparisons':len(c)}))
