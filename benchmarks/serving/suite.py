"""Resumable phase driver. Pilot choices are frozen before formal replay."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from analyze_results import audit_run, analyze, paired_ci
from generate_workloads import Factory, save
from run_experiments import OUT, ROOT, run_cell, atomic, sha


def variant_order(variants,block):
    # Rotation plus reversal cancels for two variants. Alternate AB/BA there;
    # preserve the existing permutations for all already-started larger blocks.
    if len(variants)==2:return list(variants) if block%2==0 else list(reversed(variants))
    order=variants[block%len(variants):]+variants[:block%len(variants)]
    return list(reversed(order)) if block%2 else order


def validated_cell(variant,trace,directory,**kwargs):
    directory=Path(directory)
    formal='measure' in directory.parts
    metadata={}
    if formal:
        metadata=dict(experiment=next(p for p in directory.parts if p.startswith('E') and p[1].isdigit()),
                      seed=next(int(p[5:]) for p in directory.parts if p.startswith('seed-')))
        config=json.loads((OUT/'frozen-config.json').read_text())
        metadata['tail_calibration_passed']=config.get('tail_calibration_passed',False)
        metadata['protocol_version']=config.get('protocol_version','initial')
        metadata['evidence_scope']=config.get('evidence_scope','pending calibration')
        kwargs['metadata']=metadata
    # Locate a successful retry without rerunning completed work.
    receipts=sorted(directory.rglob('run.json')) if directory.exists() else []
    quality_failures=0
    last_quality_failure=None
    for receipt in receipts:
        info=json.loads(receipt.read_text())
        if info.get('superseded_by_protocol') or info['trace_sha256']!=sha(trace):continue
        if formal and info.get('formal_outcome_recorded') and info['status']!='complete':
            print('RESUME recorded failed outcome',receipt.parent,flush=True)
            return audit_run(receipt.parent)
        if info['status']=='complete':
            result=audit_run(receipt.parent)
            if formal and info.get('formal_outcome_recorded') and result['status']!='valid':return result
            if result['status']=='valid' and (not formal or not result.get('performance_pollution')):
                print('RESUME VALID',receipt.parent,flush=True)
                return result
            if formal and result['status']=='valid' and result.get('performance_pollution'):
                quality_failures+=1;last_quality_failure=result
    if formal and quality_failures>=3:
        print('RESUME quality budget exhausted',directory,'retaining all attempts',flush=True)
        return last_quality_failure
    for attempt in range(quality_failures,3 if formal else 1):
        target=directory if attempt==0 else directory/f'quality-retry-{attempt}'
        info=run_cell(variant,trace,target,**kwargs)
        actual=Path(info.get('directory',target))
        if not (actual/'run.json').exists():actual=target
        if info['status']!='complete':
            if not formal:raise RuntimeError(f'run failed: {actual}, see original logs')
            result=audit_run(actual)
            info.update(formal_outcome_recorded=True,selected_for_primary=False)
            atomic(actual/'run.json',info)
            result.update(metadata,selected_for_primary=False)
            return result
        # Older receipts may live in an attempt subdirectory.
        if not (actual/'measurement.pkl').exists():
            paths=[p.parent for p in target.rglob('run.json') if json.loads(p.read_text()).get('status')=='complete']
            actual=max(paths,key=lambda p:(p/'run.json').stat().st_mtime)
        result=audit_run(actual)
        if result['status']!='valid':
            if not formal:raise RuntimeError(f'correctness/resource/measurement audit failed: {actual}: {result["issues"]}')
            # A failed experimental configuration is an outcome. Restart for the
            # next frozen seed; never continue a potentially corrupt server.
            record=json.loads((actual/'run.json').read_text())
            record.update(formal_outcome_recorded=True,selected_for_primary=False)
            atomic(actual/'run.json',record)
            result.update(metadata,selected_for_primary=False)
            return result
        if formal:
            receipt=actual/'run.json';record=json.loads(receipt.read_text())
            record.update(metadata,selected_for_primary=not bool(result['performance_pollution']),
                          formal_outcome_recorded=True,quality_attempt_ordinal=attempt)
            atomic(receipt,record)
            result=audit_run(actual)
            if result['performance_pollution']:
                print('POLLUTED',actual,result['performance_pollution'],'retained; fixed-rule retry',flush=True)
                continue
        return result
    return result


def correctness_traces():
    factory=Factory(seed=91001)
    path=OUT/'traces/W0-acceptance.jsonl'
    if not path.exists():
        rows,_=factory.make('W0',32,1,'closed')
        # Previously completed exact prefixes are deliberately revisited.
        for i in range(16):
            row=copy.deepcopy(rows[i]);row['request_id']=f'w0-repeat-{i}'
            rows.append(row)
        save(path,rows,dict(role='correctness',cache='mixed unique and repeated prompts'))
    natural=OUT/'traces/W0-natural.jsonl'
    if not natural.exists():
        rows=[]
        prompts=['What is 2 + 2? Answer with only the number.',
                 'What is the capital of France? Answer with one word.',
                 'Translate the English word cat into Spanish. Answer with one word.',
                 'Continue: Monday, Tuesday, Wednesday,']
        for i,text in enumerate(prompts):
            prompt=factory.tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,
                         add_generation_prompt=True,enable_thinking=False)
            ids=factory.tok.encode(prompt)
            rows.append(dict(schema_version=1,request_id=f'natural-{i}',arrival_s=0.,prompt=prompt,
                      input_ids=ids,input_tokens=len(ids),max_output_tokens=64,prefix_group='',
                      shared_prefix_tokens=0,**{'class':'natural'},temperature=0.,top_k=1,top_p=1.,
                      ignore_eos=False,timeout_s=120))
        save(natural,rows,dict(role='natural EOS correctness',thinking=False))
    cache=OUT/'traces/W0-cache.jsonl'
    if not cache.exists():
        rows,_=Factory(seed=91002).make('W3',16,1,'closed')
        for r in rows:r['max_output_tokens']=8
        for i in range(16):
            row=copy.deepcopy(rows[i]);row['request_id']=f'cache-repeat-{i}';rows.append(row)
        save(cache,rows,dict(role='correctness',cache='partial and full prefix reuse',concurrency=1))
    return path,natural,cache


def token_sequences(summary):
    path=Path(summary['directory'])
    client=json.loads((path/'client.json').read_text())
    snapshot=pickle.loads((path/'measurement.pkl').read_bytes())
    mapping={r['uid']:r['request_id'] for r in client['requests']}
    result=defaultdict(list)
    for uid,index,token,t,done in np.asarray(snapshot['tokens']).reshape(-1,5):
        result[mapping[int(uid)]].append(int(token))
    return dict(result)


def preflight():
    path,natural,cache=correctness_traces()
    # The first startup also proves first-use native FI/JIT; it must finish first.
    initial=validated_cell('H0000',OUT/'traces/W0-smoke.jsonl',OUT/'preflight/H-G0-C1',
                           profile='G0',mode='closed-loop',concurrency=1,settle_seconds=0)
    runs=[initial]
    for profile,c,trace,variants in [
        ('G0',4,OUT/'traces/W0-smoke.jsonl',['H0000']),
        ('G0-graph',8,path,['H0000']),
        ('G1',16,path,['H0000','S1000','S1111']),
        ('G1',1,natural,['H0000','S1000','S1111']),
        ('G1',1,cache,['H0000','S1000','S1111']),
    ]:
        family='natural' if trace==natural else ('cache' if trace==cache else 'fixed')
        for variant in variants:
            runs.append(validated_cell(variant,trace,OUT/f'preflight/{variant}-{profile}-C{c}-{family}',
                            profile=profile,mode='closed-loop',concurrency=c,settle_seconds=0))
    groups=defaultdict(list)
    for result in runs:
        groups[result['trace_sha256']].append(result)
    checks=[]
    for key,group in groups.items():
        reference=next((r for r in group if r['variant']=='H0000' and r['profile']=='G1'),group[0])
        expected=token_sequences(reference)
        for result in group:
            actual=token_sequences(result)
            different={k:dict(reference=expected.get(k),actual=v) for k,v in actual.items() if expected.get(k)!=v}
            check=dict(reference=reference['directory'],candidate=result['directory'],differences=different)
            checks.append(check)
    atomic(OUT/'preflight/greedy-comparison.json',checks)
    bad=[x for x in checks if x['differences']]
    if bad:
        decision=OUT/'preflight/numerics/decision.json'
        if not decision.exists() or not json.loads(decision.read_text()).get('optimizer_fixed_input_equal_to_upstream'):
            raise RuntimeError('Unexplained greedy output differences; inspect preflight/greedy-comparison.json')
        # No arbitrary text-similarity waiver: matched-shape full-vocabulary
        # logits are exactly equal, and H alone reproduces shape sensitivity.
        for x in bad:
            if 'cache' in x['candidate'] or 'natural' in x['candidate']:
                raise RuntimeError('New cache/EOS semantic discrepancy requires independent diagnosis')
    atomic(OUT/'preflight/acceptance.json',dict(status='passed',runs=len(runs),
                      strict_greedy_cross_shape_equal=not bool(bad),
                      numerical_scope='Matched-shape equivalence to upstream; strict HF absolute tolerance failed and remains reported.',
                      numerical_decision='preflight/numerics/decision.json',
                      minimum_free_mib=min(r['sampled_min_free_mib'] for r in runs),
                      source_and_collector_hashes=[(r['source_manifest_sha256'],r['collector_sha256']) for r in runs]))


def pilot():
    acceptance=OUT/'preflight/acceptance.json'
    assert acceptance.exists() and json.loads(acceptance.read_text())['status']=='passed'
    frozen=OUT/'frozen-config.json'
    if frozen.exists():
        print('RESUME frozen pilot choices',frozen,flush=True)
        return
    factory=Factory(seed=31733)
    measurements={}
    for w,arrival in [('W4','closed'),('W6','closed'),('W3','closed')]:
        path=OUT/f'pilot/{w}-capacity.jsonl'
        if not path.exists():
            rows,prime=factory.make(w,96,1,arrival)
            save(path,rows,dict(workload=w,seed=31733,role='capacity anchor',mode='closed-loop'))
            if prime:save(path.with_suffix('.prime.jsonl'),prime,{'role':'prime'})
        primepath=path.with_suffix('.prime.jsonl')
        result=validated_cell('H0000',path,OUT/f'pilot/{w}-capacity-H',profile='G1',
                     mode='closed-loop',concurrency=16,settle_seconds=0,
                     prime=primepath if primepath.exists() else None)
        measurements[w]=dict(capacity_req_s=result['success_req_s'],raw=result['directory'])
    # Confirm 80% / 105% empirical anchors with independent arrival traces.
    for w in ('W4','W6','W3'):
        reference=measurements[w]['capacity_req_s']
        measures=[]
        for factor in (.8,1.05):
            rate=round(reference*factor,4)
            path=OUT/f'pilot/{w}-open-{factor}.jsonl'
            factory=Factory(seed=31734)
            arrival='burst' if w in ('W3','W6') else 'poisson'
            if not path.exists():
                rows,prime=factory.make(w,128,rate,arrival)
                save(path,rows,dict(workload=w,seed=31734,role='open-loop validation',rate=rate,factor=factor))
                if prime:save(path.with_suffix('.prime.jsonl'),prime,{'role':'prime'})
            primepath=path.with_suffix('.prime.jsonl')
            result=validated_cell('H0000',path,OUT/f'pilot/{w}-open-{factor}-H',profile='G1',
                         prime=primepath if primepath.exists() else None,settle_seconds=0)
            measures.append(dict(factor=factor,rate=rate,raw=result['directory'],
                            success=result['success'],duration_s=result['duration_s'],
                            send_lag_p99_ms=result['send_lag_ms_p99'],loop_lag_p99_ms=result['loop_lag_p99_ms']))
        measurements[w]['open_checks']=measures
    # W5 waves are compared with both fixed64 and matched fixed24; no parameter
    # search for the best-looking dynamic result is performed.
    w5=OUT/'pilot/W5-waves.jsonl'
    if not w5.exists():
        rows,prime=Factory(seed=31735).make('W5',108,1,'poisson',cycle_s=6.,background_output=256)
        save(w5,rows,dict(workload='W5',seed=31735,cycle_s=6.,background_output=256,role='pilot'))
    for v in ('H0000','S1010','fixed24','S1011'):
        validated_cell(v,w5,OUT/f'pilot/W5-{v}',settle_seconds=0)
    config=dict(schema_version=1,profile='G1',requests=1000,seeds=list(range(31723,31728)),
                extension_seeds=list(range(31728,31733)),
                model='/home/wangwei/ai-infra/models/Qwen3-0.6B',
                pilot_anchors=measurements,
                rates={w:round(m['capacity_req_s']*1.05,4) for w,m in measurements.items()},
                additional_rate_W4=round(measurements['W4']['capacity_req_s']*.8,4),
                W5=dict(cycle_s=6.,background_output=256,requests=1008),
                slo=dict(short_ttft_ms=500,long_ttft_ms=2000,e2e_extra_ms=500,
                         output_interval_allowance_ms=50,server_tpot_ms=50,
                         server_request_itl_p95_ms=100,starvation_ms=5000),
                extension_rule='At n=5 append to n=10 if run-paired 95% CI for registered primary crosses zero, signs differ, or primary run CV>5%; preserve all first five.',
                primary={'E01':'output_tok_s','E02':'itl_server_ms_p99','E03':'cold_wait_ms_p95',
                         'E04':'output_tok_s','E05':'itl_server_ms_p99'},
                thermal_rule='Before measurement, 60-second sampled range <=2C; otherwise record pollution and extend up to five windows.',
                client_rule='send and loop lag P99 <1ms and loop lag <5% server median ITL; compare affinity and dummy SSE before formal claims',
                frozen_wall=time.time())
    atomic(frozen,config)
    print('FROZEN',json.dumps(config,ensure_ascii=False),flush=True)


def measure():
    config=json.loads((OUT/'frozen-config.json').read_text())
    assert config.get('finalized') and config.get('W5_stability_verified'), 'Complete pilot finalization before formal runs'
    calibration=json.loads((OUT/'preflight/fixed-arrival-calibration/acceptance.json').read_text())
    if not calibration['passed']:
        print('TAIL CALIBRATION FAILED: retained. ITL effects remain exploratory and cannot become resume gains; throughput/failure/waiting data will still be collected.',flush=True)
    experiments=[('E02','W4',['H0000','S1000','S1010']),
                 ('E03','W6',['S1000','S1L00','S1100']),
                 ('E04','W3',['H0000','S1000','S1L00','S1100']),
                 ('E05','W5',['S1010','fixed24','S1011']),
                 ('E01','W1',['H0000','S1000']),
                 ('E02-low','W4',['S1000','S1010']),
                 ('E06','W4-X',['H0000','S1000','S1100','S1010','S1011','S1110','S1111'])]
    for experiment,w,variants in experiments:
        results=[]
        for block in range(10):
            if block==5:
                primary=config['primary'][experiment]
                by=defaultdict(dict)
                for r in results:
                    if r['status']=='valid' and r.get('selected_for_primary',True):by[r['variant']][r['trace_sha256']]=r
                comparison={'E01':('H0000','S1000'),'E02':('S1000','S1010'),
                            'E02-low':('S1000','S1010'),'E03':('S1L00','S1100'),
                            'E04':('S1000','S1L00'),'E05':('fixed24','S1011'),
                            'E06':('H0000','S1111')}[experiment]
                a,b=comparison
                pairs=[(by[a][h],by[b][h]) for h in by[a].keys()&by[b].keys()]
                if len(pairs)<5:
                    atomic(OUT/f'measure/{experiment}/extension-decision.json',dict(primary=primary,
                         extend_to_10=False,valid_paired_runs=len(pairs),
                         reason='Fewer than five quality-valid pairs; preserve unsupported/failed configuration, do not manufacture a CI.'))
                    break
                ci=paired_ci(pairs,primary,latency=primary!='output_tok_s')
                cv=max(float(np.std([p[i][primary] for p in pairs])/np.mean([p[i][primary] for p in pairs])) for i in (0,1))
                signs=[np.sign(v) for v in ci['paired_improvements_pct']]
                extend=(ci['ci95_low_pct']<=0<=ci['ci95_high_pct'] or len(set(signs))>1 or cv>.05)
                atomic(OUT/f'measure/{experiment}/extension-decision.json',dict(primary=primary,
                       reference=a,variant=b,ci=ci,extend_to_10=bool(extend)))
                if not extend:break
            seed=31723+block
            trace=OUT/f'traces/{w}-{experiment}-s{seed}.jsonl'
            rate=config['rates'].get(w,1.)
            if experiment=='E02-low':rate=config['additional_rate_W4']
            arrival='burst' if w in ('W3','W6') else ('closed' if w=='W1' else 'poisson')
            if not trace.exists():
                rows,prime=Factory(seed=seed).make(w,config['W5']['requests'] if w=='W5' else config['requests'],rate,arrival,
                            config['W5']['cycle_s'],config['W5']['background_output'])
                save(trace,rows,dict(workload=w,experiment=experiment,seed=seed,rate=rate,arrival=arrival,
                                    frozen_config_sha256=sha(OUT/'frozen-config.json')))
                if prime:save(trace.with_suffix('.prime.jsonl'),prime,{'role':'prime'})
            primepath=trace.with_suffix('.prime.jsonl')
            order=variant_order(variants,block)
            for v in order:
                result=validated_cell(v,trace,OUT/f'measure/{experiment}/seed-{seed}/{v}',
                           profile=config['profile'],mode='closed-loop' if w=='W1' else 'open-loop',
                           concurrency=16,prime=primepath if primepath.exists() else None,settle_seconds=60)
                receipt=Path(result['directory'])/'run.json'
                info=json.loads(receipt.read_text());info.update(experiment=experiment,seed=seed)
                atomic(receipt,info)
                result=audit_run(receipt.parent);results.append(result)
            atomic(OUT/'PROGRESS.json',dict(phase='measure',experiment=experiment,completed_block=block,
                                           last_seed=seed,last_update=time.time()))
            analyze(OUT/'measure',OUT/'report')
    analyze(OUT/'measure',OUT/'report')
    atomic(OUT/'PROGRESS.json',dict(phase='measure complete',updated=time.time()))


def finalize_config():
    frozen=OUT/'frozen-config.json'
    config=json.loads(frozen.read_text())
    if config.get('finalized') and config.get('W5_stability_verified'):return
    trace=OUT/'pilot/W4-X-capacity.jsonl'
    if not trace.exists():
        rows,prime=Factory(seed=31736).make('W4-X',96,1,'closed')
        save(trace,rows,dict(workload='W4-X',role='capacity anchor',seed=31736))
        save(trace.with_suffix('.prime.jsonl'),prime,{'role':'prime'})
    result=validated_cell('H0000',trace,OUT/'pilot/W4-X-capacity-H',mode='closed-loop',concurrency=16,
                 prime=trace.with_suffix('.prime.jsonl'),settle_seconds=0)
    rate=round(result['success_req_s']*1.05,4)
    confirmation=OUT/'pilot/W4-X-open.jsonl'
    if not confirmation.exists():
        rows,prime=Factory(seed=31737).make('W4-X',128,rate,'poisson')
        save(confirmation,rows,dict(workload='W4-X',role='open-loop validation',seed=31737,rate=rate))
        save(confirmation.with_suffix('.prime.jsonl'),prime,{'role':'prime'})
    validated_cell('H0000',confirmation,OUT/'pilot/W4-X-open-H',
                   prime=confirmation.with_suffix('.prime.jsonl'),settle_seconds=0)
    # The 6s W5 pilot accumulated long-prefill backlog under small budgets.
    # Increase wave spacing, keeping every input/output length unchanged.
    w5=OUT/'pilot/W5-waves-cycle12.jsonl'
    if not w5.exists():
        rows,prime=Factory(seed=31738).make('W5',108,1,'poisson',cycle_s=12.,background_output=256)
        save(w5,rows,dict(workload='W5',role='stability pilot',seed=31738,cycle_s=12.,background_output=256))
    stability=[]
    for v in ('H0000','S1010','fixed24','S1011'):
        r=validated_cell(v,w5,OUT/f'pilot/W5-cycle12-{v}',settle_seconds=0)
        stability.append(dict(variant=v,directory=r['directory'],duration_s=r['duration_s'],
                              long_ttft_p95=r.get('long_ttft_text_ms_p95'),
                              n_decode=r['budget_decode_values'],budget=r['budget_values']))
    atomic(OUT/'pilot/frozen-config-before-composite.json',config)
    config['rates']['W4-X']=rate
    config['W5']['cycle_s']=12.
    config['W5_stability_pilots']=stability
    config['W5_stability_verified']=True
    config['primary'].update({'E02-low':'itl_server_ms_p99','E06':'output_tok_s'})
    config['pilot_anchors']['W4-X']=dict(capacity_req_s=result['success_req_s'],raw=result['directory'])
    config['finalized']=True;config['finalized_wall']=time.time()
    config['source_basis']='Pinned H/S plus shared host-terminal correctness patch and common HTTP keepalive=300s'
    config['numerical_limitation']='Strict cross-shape greedy and HF absolute thresholds not met; matched-shape H/S logits bitwise equal and upstream itself reproduces BF16 shape sensitivity.'
    atomic(frozen,config)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['preflight','pilot','measure','all'],default='all')
    a=p.parse_args()
    if a.phase in ('preflight','all'):preflight()
    if a.phase in ('pilot','all'):
        pilot();finalize_config()
    if a.phase in ('measure','all'):measure()
