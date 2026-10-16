"""Offline deliverables; never launches a model or changes the measurement gate.

Run after suite.py, or with --partial for a clearly labelled checkpoint. Original
requests, receipts and primary formulas remain intact. Supplemental accounting
is written separately so an offline reporting repair cannot change run selection.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np

from run_experiments import OUT, atomic, sha
from make_report import main as render, table, LABELS, get_pairs
from analyze_results import paired_ci

MATRIX={
    'E02':['H0000','S1000','S1010'],
    'E03':['S1000','S1L00','S1100'],
    'E04':['H0000','S1000','S1L00','S1100'],
    'E05':['S1010','fixed24','S1011'],
    'E01':['H0000','S1000'],
    'E02-low':['S1000','S1010'],
    'E06':['H0000','S1000','S1100','S1010','S1011','S1110','S1111'],
}
EXPECTED={'dtype':'torch.bfloat16','attention_backend':'fi','page_size':'1',
          'max_running_req':'16','max_seq_len_override':'2304',
          'num_page_override':'16384','max_extend_tokens':'256',
          'cuda_graph_max_bs':'16','cache_type':'radix','use_dummy_weight':'False'}


def identity(info):
    p=Path(info['directory'])
    exp=info.get('experiment',next((x for x in p.parts if x in MATRIX),None))
    seed=info.get('seed',next((int(x[5:]) for x in p.parts if x.startswith('seed-')),None))
    return exp,seed


def supplemental(info):
    p=Path(info['directory'])
    if not (p/'measurement.pkl').exists() or not (p/'requests.jsonl').exists():return None
    s=pickle.loads((p/'measurement.pkl').read_bytes())
    c=json.loads((p/'client.json').read_text())
    reqs=[json.loads(x) for x in (p/'requests.jsonl').read_text().splitlines()]
    trace=[json.loads(x) for x in Path(info['trace']).read_text().splitlines()]
    request_ids=[r['request_id'] for r in reqs]
    trace_ids=[r['request_id'] for r in trace]
    uids=[r['uid'] for r in reqs if r['uid'] is not None]
    mapping_ok=(len(request_ids)==len(trace_ids) and len(set(request_ids))==len(request_ids)
                and set(request_ids)==set(trace_ids) and len(uids)==len(set(uids)))
    mismatches={k:{'expected':v,'actual':s['config'].get(k)} for k,v in EXPECTED.items() if s['config'].get(k)!=v}
    if s['graph_capture_list']!=[1,2,4,8,16]:mismatches['actual_graphs']=s['graph_capture_list']
    if info.get('client_cpus')!=[8,9]:mismatches['client_cpus']=info.get('client_cpus')
    result={'directory':str(p),'experiment':identity(info)[0],'seed':identity(info)[1],
            'variant':info['variant'],'resource_config_mismatches':mismatches,
            'request_id_coverage_and_uid_uniqueness':mapping_ok,
            'trace_hash_matches':sha(info['trace'])==info['trace_sha256'],
            'requested_output_tokens':sum(r['max_output_tokens'] for r in trace),
            'all_server_generated_tokens':sum(r['generated_tokens'] for r in reqs),
            'all_client_ack_tokens':sum(len(r['ack_ns']) for r in reqs),
            'successful_generated_tokens':sum(r['generated_tokens'] for r in reqs if not r['audit_errors']),
            'partial_or_failed_generated_tokens':sum(r['generated_tokens'] for r in reqs if r['audit_errors']),
            'abort_events':s.get('aborts',[]),
            'client_cpu_fraction':c['cpu_seconds']/c['wall_seconds'],
            'resource_release':s['resources'],
            'input_length_counts':dict(Counter(r['input_tokens'] for r in trace)),
            'requested_output_length_counts':dict(Counter(r['max_output_tokens'] for r in trace)),
            'class_accounting':{},'slo_sensitivity':[]}
    for cls in sorted({r['class'] for r in reqs}):
        a=[r for r in reqs if r['class']==cls]
        censored=[r for r in a if r['admission_ns'] is None]
        result['class_accounting'][cls]={'offered':len(a),'complete_and_audited':sum(not r['audit_errors'] for r in a),
            'timeouts':sum(r['status']=='timeout' for r in a),'unadmitted':len(censored),
            'unadmitted_wait_lower_bounds_ms':[(r['terminal_ns']-(r['queue_ns'] or r['send_ns']))/1e6 for r in censored],
            'admitted_wait_above_5s':sum(r['admission_ns'] is not None and r['queue_ns'] is not None and r['admission_ns']-r['queue_ns']>5e9 for r in a)}
    # Observed cache occupancy at submission boundaries, clipped to the actual
    # client interval; post-timeout abort drain can otherwise give negative dt.
    end=max(r['terminal_ns'] for r in reqs);start=c['start_ns']
    bs=[b for b in s['batches'] if start<=b[0]<end]
    if bs:
        dt=np.diff([b[0] for b in bs]+[end])
        occ=[(s['resources']['num_pages']-b[8])/s['resources']['num_pages'] for b in bs]
        result['submission_boundary_cache_occupancy_mean']=float(np.average(occ,weights=dt))
        result['submission_boundary_cache_occupancy_scope']='Measured client interval; first submission to final client terminal; cached and protected pages included; no interpolation of hidden GPU peaks.'
    duration=(end-start)/1e9
    for scale in (.5,1.,2.):
        count=0
        for r in reqs:
            if r['audit_errors'] or r['first_text_ns'] is None:continue
            slo=(2000 if r['class']=='long' else 500)*scale
            ttft=(r['first_text_ns']-r['send_ns'])/1e6
            e2e=(r['terminal_ns']-r['send_ns'])/1e6
            count+=ttft<=slo and e2e<=slo+50*(r['generated_tokens']-1)+500
        result['slo_sensitivity'].append({'ttft_threshold_scale':scale,'slo_success_requests':int(count),'goodput_req_s':count/duration})
    return result


def errors_and_receipts():
    ledger=[];receipts=[]
    for phase in ('preflight','pilot','measure'):
        for path in sorted((OUT/phase).rglob('run.json')):
            info=json.loads(path.read_text());info['directory']=str(path.parent)
            summary_path=path.parent/'summary.json'
            summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
            entry={'phase':phase,'directory':str(path.parent),'variant':info.get('variant'),
                   'experiment':identity(info)[0],'seed':identity(info)[1],
                   'run_status':info['status'],'audit_status':summary.get('status'),
                   'failures':summary.get('failed'),'request_status_counts':summary.get('request_status_counts'),
                   'issues':summary.get('issues',[]),'pollution':summary.get('performance_pollution',[]),
                   'error':info.get('error'),'started_wall':info.get('started_wall'),
                   'finished_wall':info.get('finished_wall')}
            if 'W0-abort' in info.get('trace',''):entry['failure_intent']='Intentional short-deadline resource correctness check; not a performance failure rate.'
            client_path=path.parent/'client.json'
            if client_path.exists() and not summary:
                cr=json.loads(client_path.read_text())['requests']
                entry['raw_client_status_counts']=dict(Counter(r['status'] for r in cr))
                entry['raw_client_ack_count']=sum(len(r['ack_ns']) for r in cr)
                entry['server_token_audit']='unavailable: no measurement snapshot, do not substitute ACK count as generated count'
            receipts.append((info,summary))
            if info['status']!='complete' or summary.get('status')=='invalid' or summary.get('failed',0) or summary.get('performance_pollution'):
                ledger.append(entry)
    for path in sorted((OUT/'preflight').rglob('acceptance.json')):
        obj=json.loads(path.read_text())
        if obj.get('passed') is False:ledger.append({'phase':'gate','file':str(path),'status':'failed','evidence':obj})
    for name in ('environment/compiler-failures.json','preflight/numerics/decision.json',
                 'preflight/abort-gpu-incident.json','preflight/natural-hf-reference.json'):
        path=OUT/name
        if path.exists():ledger.append({'phase':'environment_or_numerics','file':str(path),'evidence':json.loads(path.read_text())})
    atomic(OUT/'report/error-ledger.json',ledger)
    return receipts,ledger


def coverage(receipts):
    records=[]
    for exp,variants in MATRIX.items():
        for v in variants:
            matching=[(i,s) for i,s in receipts if identity(i)[0]==exp and i['variant']==v]
            attempted=sorted({identity(i)[1] for i,s in matching})
            completed=sorted({identity(i)[1] for i,s in matching if i['status']=='complete'})
            eligible=sorted({identity(i)[1] for i,s in matching if s.get('status')=='valid' and s.get('selected_for_primary')})
            records.append({'experiment':exp,'variant':v,'attempts':len(matching),'attempted_seeds':attempted,
                            'completed_seeds':completed,'eligible_seeds':eligible,
                            'minimum_five_attempted':len(attempted)>=5,'minimum_five_eligible':len(eligible)>=5})
    atomic(OUT/'report/coverage.json',records)
    return records


def interactions(rows):
    # Each log-ratio difference is computed inside one shared seed/trace block.
    data={v:{r['trace_sha256']:r for r in rows if r['experiment']=='E06' and r['variant']==v} for v in MATRIX['E06']}
    results=[]
    for name,variants in [('B_x_C_given_A',['S1000','S1100','S1010','S1110']),
                          ('B_x_D_given_A_C',['S1010','S1110','S1011','S1111'])]:
        keys=set.intersection(*(set(data[v]) for v in variants))
        if len(keys)<5:
            results.append({'interaction':name,'n':len(keys),'status':'unsupported: fewer than five complete legal paired blocks'});continue
        zero_keys=[k for k in keys if any(data[v][k]['output_tok_s']<=0 for v in variants)]
        if zero_keys:
            results.append({'interaction':name,'n':len(keys),'status':'undefined log ratio: a measured block has zero throughput; do not remove its failure outcome',
                            'zero_throughput_trace_hashes':zero_keys});continue
        values=[]
        for k in sorted(keys):
            a,b,c,d=[data[v][k]['output_tok_s'] for v in variants]
            values.append(np.log(d/c)-np.log(b/a))
        arr=np.asarray(values);rng=np.random.default_rng(31723)
        boots=np.median(arr[rng.integers(0,len(arr),(10000,len(arr)))],axis=1)
        results.append({'interaction':name,'n':len(values),'status':'measured conditional interaction',
                        'metric':'output_tok_s','definition':'log(d/c)-log(b/a), variants=[a,b,c,d]',
                        'variants':variants,'paired_log_ratios':values,'median_log_ratio':float(np.median(arr)),
                        'ci95_log_ratio':[float(x) for x in np.quantile(boots,[.025,.975])]})
    atomic(OUT/'report/conditional-interactions.json',results)
    return results


def tradeoffs(rows):
    comparisons=[('E02','S1000','S1010'),('E03','S1L00','S1100'),('E04','S1000','S1L00'),
                 ('E05','S1010','S1011'),('E05','fixed24','S1011'),('E06','H0000','S1111'),
                 ('E06','S1011','S1111'),('E06','S1110','S1111')]
    metrics=['output_tok_s','goodput_req_s','itl_server_ms_p99','long_ttft_text_ms_p95',
             'cold_wait_ms_p95','prefix_token_hit_ratio','failed','scheduler_us_mean']
    records=[]
    for exp,a,b in comparisons:
        for metric in metrics:
            pairs=get_pairs(rows,exp,a,b,metric)
            if len(pairs)<5:continue
            latency=metric in ('itl_server_ms_p99','long_ttft_text_ms_p95','cold_wait_ms_p95','failed','scheduler_us_mean')
            records.append({'experiment':exp,'reference':a,'variant':b,'metric':metric,
                            'reference_median':float(np.median([x[metric] for x,y in pairs])),
                            'candidate_median':float(np.median([y[metric] for x,y in pairs])),
                            'scope':'exploratory: tail calibration failed' if metric=='itl_server_ms_p99' else 'conditional fixed-length evidence',
                            **paired_ci(pairs,metric,latency)})
    atomic(OUT/'report/paired-tradeoffs.json',records)
    return records


def interpretations(records):
    descriptions=[('E02','S1000','S1010','阶段调度'),('E03','S1L00','S1100','Aging公平性'),
                  ('E04','S1000','S1L00','Prefix复用'),('E05','S1010','S1011','动态预算相对固定64'),
                  ('E05','fixed24','S1011','动态预算相对调优固定24'),('E06','H0000','S1111','整机净效果')]
    lines=[]
    for exp,a,b,label in descriptions:
        found=[r for r in records if (r['experiment'],r['reference'],r['variant'])==(exp,a,b)]
        lines.append(f'#### {exp}：{label}\n')
        if not found:
            lines.append('没有至少5对且参考指标非零的同trace入选记录，不能估计该对照的配对改善。实际失败、污染和样本数见覆盖表及台账。\n');continue
        for r in found:
            if r['metric'] not in ('output_tok_s','itl_server_ms_p99','long_ttft_text_ms_p95','cold_wait_ms_p95','prefix_token_hit_ratio'):continue
            lo,hi=r['ci95_low_pct'],r['ci95_high_pct']
            direction='区间支持该测量条件下改善' if lo>0 else ('区间支持退化' if hi<0 else '区间跨0，改善方向不稳定')
            if r['metric']=='itl_server_ms_p99':direction='探索性观察；全局尾延迟校准失败，不能升级为已验证收益'
            lines.append(f"- {r['metric']}：{r['reference_median']:.3f}→{r['candidate_median']:.3f}，{r['n']}对，改善率中位数{r['median_improvement_pct']:+.2f}%，95%配对区间[{lo:+.2f},{hi:+.2f}]%；{direction}。")
        if exp=='E03':lines.append('\n等待分布包含所有已准入请求；未准入是删失，不以成功子集或有限trace证明绝对无饥饿。')
        if exp=='E04':lines.append('\n真实cached_len、重算与淘汰计数用来解释复用机制；命中与吞吐的相关性不等于全部差异的因果归因。')
        if exp=='E05':lines.append('\n必须同时读取两个固定预算对照及长TTFT/吞吐代价。当前G1 running≤16、T=32，公式预算至少16，floor8在该资源档位结构性不可达；不宣称其GPU边界已验收。即使观测ITL区间在0以上，未通过的校准仍限制动态预算收益结论。')
        if exp=='E06':lines.append('\nH*包含共有正确性补丁，Full差异只表示固定混合负载上的净效果；不把该百分比拆成四个独立优化相加。')
        lines.append('')
    return '\n'.join(lines)+'\n'


def main(partial=False):
    progress=json.loads((OUT/'PROGRESS.json').read_text()) if (OUT/'PROGRESS.json').exists() else {}
    complete=progress.get('phase')=='measure complete'
    if not partial and not complete:raise RuntimeError('Measurement is not complete; use --partial for an honest checkpoint')
    render()
    receipts,ledger=errors_and_receipts();cover=coverage(receipts)
    extras=[]
    for info,s in receipts:
        if identity(info)[0] in MATRIX and s.get('status') in ('valid','invalid'):
            extra=supplemental(info)
            if extra:extras.append(extra)
    atomic(OUT/'report/supplemental-accounting.json',extras)
    rows=json.loads((OUT/'report/runs.json').read_text())
    valid=[r for r in rows if r.get('status')=='valid' and r.get('selected_for_primary')]
    interaction=interactions(valid)
    costs=tradeoffs(valid)
    semantic=[]
    hf_path=OUT/'preflight/natural-hf-reference.json'
    hf_rows={r['request_id']:r['output_text'] for r in json.loads(hf_path.read_text())['rows']} if hf_path.exists() else {}
    for v in ('H0000','S1000','S1111'):
        p=OUT/f'preflight/{v}-G1-C1-natural/client.json'
        if p.exists():
            for r in json.loads(p.read_text())['requests']:
                expected={'natural-0':'4','natural-1':'Paris','natural-2':'gato'}.get(r['request_id'])
                semantic.append({'variant':v,'request_id':r['request_id'],'actual_output':r['output_text'],
                                 'expected':expected,'semantic_passed':r['output_text'].strip().casefold()==expected.casefold() if expected else None,
                                 'matches_exact_input_hf_bf16_text':r['output_text']==hf_rows.get(r['request_id'])})
    atomic(OUT/'report/natural-semantic-audit.json',semantic)
    claims=json.loads((OUT/'report/primary-comparisons.json').read_text())
    config=json.loads((OUT/'frozen-config.json').read_text())
    calibration=json.loads((OUT/'preflight/fixed-arrival-calibration/acceptance.json').read_text())
    capacity_checks=[calibration[k] for k in ('saturated_capacity_light_vs_off','saturated_capacity_affinity')]
    resume_capacity_floor=max(abs(c[k]) for c in capacity_checks for k in ('ci95_low_pct','ci95_high_pct'))
    qualified=[]
    for c in claims:
        # Tail calibration failed: never promote a positive ITL observation.
        if c['metric']=='output_tok_s' and c['ci95_low_pct']>resume_capacity_floor and all(x>0 for x in c['paired_improvements_pct']):qualified.append(c)
    text='# 简历证据审核\n\n'
    text+=('完整冻结矩阵已执行；' if complete else '**矩阵尚在执行，本文件是阶段性快照；**')
    text+='严格跨形状数值一致性与尾延迟校准未通过；自然问题2+2在H*/S中均回答2，语义正确性不通过。不使用ITL改善百分比，不宣称质量无损，也不把H*称原样上游。\n\n'
    text+=f'当前归档{len(rows)}次正式尝试，{len(valid)}次入选；模型Qwen3-0.6B BF16，RTX5060Ti8GB，G1资源固定。\n\n'
    text+=f'简历另用保守编辑筛选：吞吐改善区间下界必须超过已观察capacity/affinity校准区间的最大绝对边界{resume_capacity_floor:.2f}%，且各配对方向一致。它不改变正式统计、不能使失败的尾延迟校准变为通过，也不是两个区间的联合置信区间。所有较小实测差异仍完整保留在报告中。\n\n'
    text+='建议可核验的工程表述：\n\n“建立8GB单卡Serving实测与消融框架，固定源码、模型和请求trace，完成SSE/token对账、资源回收和开销审计，定位并修复overlap终态计数错误；保留未通过的数值一致性、采集校准和优化负结果。”\n\n'
    text+='以下吞吐比较有至少5对、改善区间大于0且各对方向一致；仍受H*基线和数值行为范围限制，正式运行尚未结束时不建议写入简历：\n\n' if qualified else '尚无满足全部筛选条件的稳定吞吐百分比，暂不填写收益数字。\n'
    for c in qualified:
        text+=f"- {c['experiment']} {LABELS[c['reference']]}→{LABELS[c['variant']]}：{c['n']}对，输出吞吐改善中位数{c['median_improvement_pct']:.2f}%，95%配对CI[{c['ci95_low_pct']:.2f},{c['ci95_high_pct']:.2f}]%；绝对吞吐{c['reference_median']:.2f}→{c['candidate_median']:.2f} token/s。\n"
    (OUT/'RESUME_EVIDENCE.md').write_text(text)
    section='\n## 覆盖、补充核算及未测项\n\n'
    section+=table(['实验','配置','总尝试','已尝试seed','完成seed','入选seed'],[[r['experiment'],LABELS[r['variant']],r['attempts'],len(r['attempted_seeds']),len(r['completed_seeds']),len(r['eligible_seeds'])] for r in cover])
    section+='\n完成seed与通过质量检查的seed是不同口径；不足5个入选seed明确限制对应对照。失败或超时不会因性能慢而被丢弃。全部错误和失败验收见[完整台账](report/error-ledger.json)。\n\n'
    section+='### 同一配对下的收益与代价\n\n'
    if costs:
        section+=table(['实验/条件对照','指标','n','参考/候选中位数','改善率/95%配对CI'],[
            [c['experiment']+' '+LABELS[c['reference']]+'→'+LABELS[c['variant']],c['metric'],c['n'],
             f"{c['reference_median']:.3f} / {c['candidate_median']:.3f}",
             f"{c['median_improvement_pct']:+.2f}% [{c['ci95_low_pct']:+.2f},{c['ci95_high_pct']:+.2f}]%"] for c in costs])
    else:section+='尚无至少5对入选记录，不生成代价CI。\n'
    section+='\nITL均为探索性；吞吐/命中率/Goodput按越大越好，等待/失败数/CPU耗时按越小越好计算改善。参考值为0时比值未定义，绝对失败数仍完整保留在逐run与补充核算中；不能把未定义比值解释成没有失败差异。该表不是多重比较校正后的显著性筛选。\n\n'
    section+=interpretations(costs)
    section+='**自然输出语义检查未通过：** 2+2在H*/S1000/S1111均生成“2”而不是“4”；相同input IDs的HF BF16/SDPA参考也输出“2”，四个自然问题完整文本与mini一致。其余Paris/gato通过，星期续写仅作输出一致性检查。逐token一致、参考一致、终态完整与语义正确是不同验收。证据见[自然问题语义审计](report/natural-semantic-audit.json)和[HF原始参考](preflight/natural-hf-reference.json)，不能写“全部正确性通过”。本轮未修改权重、精度、prompt或输出长度去掩盖问题；它限制质量结论。\n\n'
    section+='首个C64正式尝试940成功/60超时后触发Abort-overlap页重复归还；真实CPU状态机在H/S均复现。共有修正只在Abort释放后将对象加入现有已释放结果集合，过滤其尚未消费的结果，不新增正常token热路径操作。旧正式数据和旧源码快照保留并明确标记superseded；最终比较统一采用修复后的H*/S版本。\n\n'
    section+='[补充核算](report/supplemental-accounting.json)逐run列出请求输出预算、服务端实际生成、客户端ACK、成功生成及失败/部分生成量，按类列出未准入的删失等待下界、Abort与资源状态，另给固定0.5×/1×/2×TTFT SLO敏感性。原始主指标与入选规则未改变。缓存占用补充核算将权重截在实际client终止边界，避免测量后Abort drain的负时间权重。\n\n'
    section+='[合法条件交互](report/conditional-interactions.json)只估计B×C|A与B×D|A,C的配对log ratio。A0隔离实现、A×C、非法B无A/D无C均未执行、不填预测值。W2无复用完整GPU sweep、W5低KV/G0资源压力扫、A-ISO CPU微基准与D floor8边界不属于本轮已测正式矩阵，不宣称已完成。\n\n'
    section+='Cold表示该run测量trace未prime；warmup使用独立prompt但仍会留下少量公共前缀，所以不宣称测量起点radix树绝对为空。Warm表示W3/W6/W4-X按冻结prime文件预热。W6在任何E03测量前修正cold-first，每个B32先放该burst的cold配额；旧pilot与新pilot以及未受影响W4字节hash均保留。\n\n'
    section+='简历表述见[证据审核](RESUME_EVIDENCE.md)，无稳定动态预算收益时按负结果报告。报告状态与当前运行记录见[PROGRESS.json](PROGRESS.json)。\n'
    with (OUT/'REPORT.md').open('a') as f:f.write(section)
    html_path=OUT/'report/index.html'
    if html_path.exists():
        import html
        body=html_path.read_text()
        appendix='<section><h2>完整分析与覆盖记录</h2><p>下方正文保留Markdown原文，包含条件对照、失败与限制；可另打开REPORT.md。</p><pre>'+html.escape((OUT/'REPORT.md').read_text())+'</pre></section>'
        html_path.write_text(body.replace('</html>',appendix+'</html>'))
    checks={'measurement_complete':complete,'all_cells_five_seeds_attempted':all(r['minimum_five_attempted'] for r in cover),
            'all_cells_five_seeds_eligible':all(r['minimum_five_eligible'] for r in cover),
            'all_completed_resource_configs_match':all(not r['resource_config_mismatches'] and r['trace_hash_matches'] and r['request_id_coverage_and_uid_uniqueness'] for r in extras),
            'tail_calibration_passed':config.get('tail_calibration_passed',False),
            'cross_shape_strict_numeric_acceptance':False,'natural_semantic_acceptance':all(r['semantic_passed'] is not False for r in semantic),
            'natural_reference_text_equivalence':bool(semantic) and all(r['matches_exact_input_hf_bf16_text'] for r in semantic),
            'formal_attempts':len(rows),'eligible_runs':len(valid),
            'global_ledger_entries':len(ledger),'created_wall':time.time(),
            'deliverable_scope':'Actual fixed-length benchmark with explicit failed acceptance gates; not an unqualified optimization success.'}
    atomic(OUT/'report/delivery-audit.json',checks)
    source_dir=OUT/'report/analysis-sources';source_dir.mkdir(exist_ok=True)
    source_hashes={}
    for name in ('analyze_results.py','make_report.py','finalize_campaign.py'):
        f=Path(__file__).with_name(name);shutil.copyfile(f,source_dir/name);source_hashes[name]=sha(f)
    atomic(OUT/'report/analysis-manifest.json',source_hashes)
    if complete:
        state_path=OUT/'CAMPAIGN_STATE.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        state.update(phase='measurement complete; report generated; rendered review pending',
                     updated_wall=time.time(),delivery_audit='report/delivery-audit.json',
                     tail_gate_passed=False,numerical_gate_passed=False)
        atomic(state_path,state)
    # Hash immutable/raw evidence only after execution; never hash growing logs.
    if complete:
        files={str(p.relative_to(OUT)):sha(p) for root in ('environment','patches','traces','preflight','pilot','measure') for p in sorted((OUT/root).rglob('*')) if p.is_file()}
        atomic(OUT/'report/artifact-sha256.json',files)
    print(json.dumps(checks,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
