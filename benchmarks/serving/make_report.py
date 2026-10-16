"""Render only measured data. Static SVG/PNG figures and a source-linked report."""
from collections import defaultdict
from datetime import datetime
import csv
import html
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from analyze_results import analyze, paired_ci
from run_experiments import OUT

LABELS={'H0000':'H*','S1000':'FCFS','S1L00':'LPM','S1100':'Aging',
        'S1010':'C64','fixed24':'C24','S1011':'D','S1110':'B+C','S1111':'Full'}
COLORS={'H0000':'#64748b','S1000':'#2563eb','S1L00':'#ca8a04','S1100':'#059669',
        'S1010':'#ea580c','fixed24':'#9333ea','S1011':'#db2777','S1110':'#0d9488','S1111':'#111827'}
CAPTION='Qwen3-0.6B BF16 · RTX 5060 Ti 8GB · G1/FI/page1/KV16384/graph16 · median + IQR; independent restarts · ITL exploratory: calibration failed'
FIGURES=[]


def save(fig,name,title,caption,source_rows):
    target=OUT/'report/figures';target.mkdir(exist_ok=True)
    fig.suptitle(title,fontsize=13,fontweight='bold')
    groups=defaultdict(set)
    workloads=sorted({r['workload'] for r in source_rows if r.get('workload')})
    for r in source_rows:
        if r.get('variant'):groups[r['variant']].add(r.get('directory',str(r)))
    counts=', '.join(f'{LABELS.get(v,v)}={len(paths)}' for v,paths in sorted(groups.items()))
    cache=('hot prefixes primed' if workloads and set(workloads)<=set(['W3','W6','W4-X']) else
           'unprimed measurement trace' if workloads and set(workloads)<=set(['W4','W5','W1']) else 'per-workload cache state')
    detail=('Workload '+','.join(workloads)+'; '+cache+'; source runs: '+counts) if source_rows else 'No eligible data for this family yet; no values imputed'
    fig.text(.02,.027,CAPTION,fontsize=7,color='#475569')
    fig.text(.02,.008,detail,fontsize=6.5,color='#475569')
    fig.tight_layout(rect=(0,.07,1,.95))
    fig.savefig(target/(name+'.png'),dpi=180)
    fig.savefig(target/(name+'.svg'))
    plt.close(fig)
    (target/(name+'-source.json')).write_text(json.dumps(source_rows,indent=2))
    FIGURES.append(dict(id=name,title=title,caption=caption,svg=f'figures/{name}.svg',
                        png=f'figures/{name}.png',source=f'figures/{name}-source.json',
                        data_status='measured observations' if source_rows else 'unavailable: no eligible observations',
                        source_run_count=len({r.get('directory',str(r)) for r in source_rows})))


def points(ax,rows,variants,metric,normalize=None):
    rng=np.random.default_rng(31723)
    for i,v in enumerate(variants):
        selected=[r for r in rows if r['variant']==v and isinstance(r.get(metric),(int,float))]
        if normalize:
            reference={r['trace_sha256']:r for r in rows if r['variant']==normalize}
            selected=[r for r in selected if r['trace_sha256'] in reference]
            values=[r[metric]/reference[r['trace_sha256']][metric] for r in selected]
        else:values=[r[metric] for r in selected]
        if not values:
            ax.text(i,.5,'no valid run',rotation=90,ha='center',va='center',transform=ax.get_xaxis_transform(),fontsize=7,color='#94a3b8')
            continue
        med=float(np.median(values));q1,q3=np.quantile(values,[.25,.75])
        ax.bar(i,med,color=COLORS.get(v,'#94a3b8'),alpha=.22,width=.65)
        ax.errorbar(i,med,yerr=[[med-q1],[q3-med]],fmt='o',color=COLORS.get(v,'#111827'),capsize=4)
        ax.scatter(i+rng.uniform(-.13,.13,len(values)),values,color=COLORS.get(v,'#111827'),s=14,alpha=.8)
        ax.annotate(f'n={len(values)}',(i,med),xytext=(0,9),textcoords='offset points',ha='center',fontsize=7)
    ax.set_xticks(range(len(variants)),[LABELS.get(v,v) for v in variants])
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)


def f1(rows):
    data=[r for r in rows if r['experiment']=='E06' and r['variant'] in ('H0000','S1000','S1111')]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    variants=['H0000','S1000','S1111']
    points(axes[0],data,variants,'output_tok_s');axes[0].set_ylabel('Generated output token/s')
    points(axes[1],data,variants,'goodput_req_s');axes[1].set_ylabel('Client SLO goodput (successful req/s)')
    save(fig,'F1-throughput','Whole-system comparison on the same W4-X trace',
         'E06，同一绝对到达率、每次1000请求。H*为官方固定提交加共有正确性补丁；不是原样上游。误差线是运行间IQR。',data)


def f2(rows,config):
    data=[r for r in rows if r['experiment'] in ('E02','E02-low')]
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    panels=[('ttft_text_ms_p99','Client text TTFT P99 (ms)'),
            ('itl_server_ms_p99','Server emit ITL P99 (ms)'),
            ('tpot_server_ms_p95','Per-request server TPOT P95 (ms)'),
            ('itl_client_ack_ms_p99','Client ACK ITL P99 (ms)')]
    rates={'E02':config['rates']['W4'],'E02-low':config['additional_rate_W4']}
    for ax,(metric,label) in zip(axes.flat,panels):
        for v in ['H0000','S1000','S1010']:
            x=[];y=[];lo=[];hi=[];ns=[]
            for exp in ('E02-low','E02'):
                a=[r[metric] for r in data if r['variant']==v and r['experiment']==exp and r.get(metric) is not None]
                if a:
                    med=np.median(a);q=np.quantile(a,[.25,.75]);x.append(rates[exp]);y.append(med);lo.append(med-q[0]);hi.append(q[1]-med);ns.append(len(a))
            if x:ax.errorbar(x,y,yerr=[lo,hi],marker='o',color=COLORS[v],label=LABELS[v],capsize=3)
            for xx,yy,n in zip(x,y,ns):ax.annotate(f'n={n}',(xx,yy),xytext=(5,6),textcoords='offset points',fontsize=6,color=COLORS[v])
        ax.set_xlabel('Offered arrival rate (req/s)');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0,0].legend()
    save(fig,'F2-tail-latency','Stage scheduling: tail latency and its timing boundaries',
         'E02。TTFT-text从HTTP调用前开始；server ITL从既有D2H后的host emit采集；TPOT是每请求平均间隔后求分位数。ACK经逐请求与服务端token数量/序号对账。',data)


def f3(rows):
    data=[r for r in rows if r['experiment']=='E06']
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    variants=['H0000','S1000','S1100','S1010','S1011','S1110','S1111']
    points(axes[0],data,variants,'output_tok_s',normalize='H0000');axes[0].set_ylabel('Output throughput / paired H*');axes[0].axhline(1,ls='--',color='#64748b',lw=.8)
    points(axes[1],data,variants,'goodput_req_s');axes[1].set_ylabel('Client goodput (req/s)')
    points(axes[2],data,variants,'itl_server_ms_p99');axes[2].set_ylabel('Server ITL P99 (ms)')
    for ax in axes:ax.tick_params(axis='x',rotation=30)
    save(fig,'F3-ablation','Legal ablations on a shared workload (E06)',
         'A0+C、A0+C+D尚未移植；A0+B和C0+D受依赖阻断，均未填数。B依赖A，D依赖C。H→S1000只表示默认实现净差异，不作A的独立归因。',data)


def f2_classes(rows):
    data=[r for r in rows if r['experiment']=='E02']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,cls in zip(axes,['short','long']):
        points(ax,data,['H0000','S1000','S1010'],cls+'_ttft_text_ms_p95')
        ax.set_ylabel(cls.capitalize()+' client TTFT P95 (ms)')
    save(fig,'F2b-class-ttft','Stage scheduling: short and long request TTFT costs',
         '与F2相同的E02高rate，按实际trace class分别统计；long cohort不足1000，不以其P99作为主结论。',data)


def f4(rows):
    data=[r for r in rows if r['experiment']=='E04']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    variants=['H0000','S1000','S1L00','S1100']
    for v in variants:
        a=[r for r in data if r['variant']==v]
        axes[0].scatter([r['prefix_token_hit_ratio']*100 for r in a],[r['output_tok_s'] for r in a],label=LABELS[v],color=COLORS[v],s=32,alpha=.8)
    axes[0].set_xlabel('Admitted prefix token hit ratio (%)');axes[0].set_ylabel('Generated output token/s');axes[0].legend(fontsize=8)
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=3))
    axes[0].ticklabel_format(axis='x',style='plain',useOffset=False)
    points(axes[1],data,variants,'shared_prefix_recomputed_tokens');axes[1].set_ylabel('Previously computed shared-prefix\ntokens recomputed')
    points(axes[2],data,variants,'evicted_tokens');axes[2].set_ylabel('Actually evicted cache tokens')
    save(fig,'F4-cache','Cache competition: reuse, recomputation and throughput',
         'W3/B32，hot已prime、cold工作集32组。命中长度来自实际首次准入handle；重算只计此前已完整计算/prime的共享区，首次必要计算单列。散点关联不等于完整因果归因。',data)


def representative(rows,experiment,variant,metric):
    candidates=[r for r in rows if r['experiment']==experiment and r['variant']==variant and isinstance(r.get(metric),(int,float))]
    if not candidates:return None
    med=np.median([r[metric] for r in candidates])
    return min(candidates,key=lambda r:(abs(r[metric]-med),r['seed']))


def f5(rows):
    r=representative(rows,'E05','S1011','itl_server_ms_p99')
    if r is None:
        fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
        ax.text(.5,.5,'No eligible E05 dynamic-budget run; timeline unavailable',ha='center')
        save(fig,'F5-timeline','Dynamic-budget timeline unavailable',
             '没有通过完整性与运行质量检查的E05/D记录，留空，不以pilot替代正式结果。',[])
        return
    p=Path(r['directory']);server=pickle.loads((p/'measurement.pkl').read_bytes());client=json.loads((p/'client.json').read_text())
    start=client['start_ns'];tokens=np.asarray(server['tokens']).reshape(-1,5)
    fig,axes=plt.subplots(5,1,figsize=(12,10),sharex=True)
    # Fixed display window: first two W5 cycles, never select by largest benefit.
    limit=24.
    for uid in sorted(set(tokens[:,0])):
        t=tokens[tokens[:,0]==uid,3];x=(t[1:]-start)/1e9;y=np.diff(t)/1e6;mask=x<=limit
        axes[0].scatter(x[mask][::4],y[mask][::4],s=2,color='#db2777',alpha=.18)
    batches=server['batches'];x=np.array([(b[0]-start)/1e9 for b in batches]);mask=x<=limit
    phase=np.array([int(b[1]=='prefill') for b in batches])
    axes[1].scatter(x[mask],phase[mask],s=4,c=np.where(phase[mask], '#ea580c','#2563eb'))
    axes[1].set_yticks([0,1],['Decode','Prefill'])
    pf=[b for b in batches if b[1]=='prefill' and (b[0]-start)/1e9<=limit]
    axes[2].plot([(b[0]-start)/1e9 for b in pf],[b[7] for b in pf],'.',color='#ea580c',ms=3)
    axes[3].step(x[mask],np.array([b[4] for b in batches])[mask],where='post',color='#059669',lw=.8)
    budgets=[b for b in server['budgets'] if (b[0]-start)/1e9<=limit]
    axes[4].plot([(b[0]-start)/1e9 for b in budgets],[b[2] for b in budgets],'.-',color='#9333ea',ms=2,lw=.6)
    for ax,label in zip(axes,['Server token gap (ms)','Submitted phase','Actual prefill tokens','Live decode requests','Protected budget (tokens)']):
        ax.set_ylabel(label);ax.grid(alpha=.15)
    axes[-1].set_xlabel('Monotonic wall time since measurement start (s)')
    save(fig,'F5-timeline',f'Dynamic-budget execution trace (paired seed {r["seed"]})',
         '代表run按S1011 ITL P99最接近该配置运行中位数选定，同分取较小seed。固定展示前两个12秒周期，散点每4点展示1点；轨迹保留部分失败token，主统计只计校验成功请求并保留失败分母。',
         [dict(directory=r['directory'],seed=r['seed'],variant='S1011',workload='W5',
               full_run_offered=r['offered'],full_run_server_interval_count=r['server_interval_count'],
               selection_metric=r['itl_server_ms_p99'],display_window_s=[0,24])])


def f6(rows):
    data=[r for r in rows if r['experiment']=='E03']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    base={r['trace_sha256']:r for r in data if r['variant']=='S1L00'}
    for v in ['S1000','S1L00','S1100']:
        a=[r for r in data if r['variant']==v and r['trace_sha256'] in base]
        axes[0].scatter([r['output_tok_s']/base[r['trace_sha256']]['output_tok_s'] for r in a],[r['cold_wait_ms_p95'] for r in a],label=LABELS[v],color=COLORS[v])
    axes[0].set_xlabel('Output throughput / paired LPM');axes[0].set_ylabel('Cold first-admission wait P95 (ms)');axes[0].legend()
    points(axes[1],data,['S1000','S1L00','S1100'],'cold_wait_ms_max');axes[1].set_ylabel('Maximum admitted cold wait (ms)')
    points(axes[2],data,['S1000','S1L00','S1100'],'starvation_rate');axes[2].set_ylabel('Admitted wait >5s / all offered')
    save(fig,'F6-fairness','Cold fairness versus throughput',
         'W6，hot/cold长度相同。cold每run约200，主报告P95/max与超5s比例，cold P99仅探索；未准入/超时另外列出。有限trace不能证明绝对无饥饿。',data)


def f7(rows):
    data=[r for r in rows if r['experiment']=='E05']
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    variants=['S1010','fixed24','S1011']
    for v in variants:
        a=[r for r in data if r['variant']==v]
        axes[0].scatter([r['long_ttft_text_ms_p95'] for r in a],[r['itl_server_ms_p99'] for r in a],
                        s=[max(20,r['output_tok_s']/10) for r in a],color=COLORS[v],label=LABELS[v],alpha=.75)
    axes[0].set_xlabel('Long request TTFT-text P95 (ms)');axes[0].set_ylabel('Server ITL P99 (ms)');axes[0].legend()
    points(axes[1],data,variants,'output_tok_s');axes[1].set_ylabel('Generated output token/s')
    rep=representative(rows,'E05','S1011','itl_server_ms_p99')
    s=pickle.loads((Path(rep['directory'])/'measurement.pkl').read_bytes()) if rep else {'budgets':[]}
    pairs=sorted(set((b[1],b[2]) for b in s['budgets']))
    axes[2].scatter([a for a,b in pairs],[b for a,b in pairs],color=COLORS['S1011'],label='Observed D')
    x=np.arange(1,17);axes[2].plot(x,np.minimum(64,np.maximum(8,32-x)),ls='--',color='#64748b',label='min(64,max(8,32-n))')
    axes[2].axhline(24,color=COLORS['fixed24'],ls=':',label='Fixed24')
    axes[2].set_xlabel('Actual live decode requests');axes[2].set_ylabel('Protected prefill budget (tokens)');axes[2].legend(fontsize=7)
    if rep is None:axes[2].text(.5,.5,'No eligible observed D run',ha='center',transform=axes[2].transAxes)
    save(fig,'F7-dynamic','Dynamic budget versus both fixed64 and matched fixed24',
         '相同K=4、相同冻结W5波次与O256/O32。气泡面积表示输出吞吐。公式只在有Decode竞争的保护性Prefill执行；G1 running≤16且T32使floor8结构性不可达，本轮不验收该边界。',data)


def f6_cdf(rows):
    variants=['S1000','S1L00','S1100']
    by={v:{r['trace_sha256']:r for r in rows if r['experiment']=='E03' and r['variant']==v} for v in variants}
    common=set.intersection(*(set(by[v]) for v in variants))
    fig,axes=plt.subplots(1,2,figsize=(12,4));sources=[]
    if common:
        # Fixed rule: first shared seed, not the seed with the largest benefit.
        key=min(common,key=lambda k:by['S1000'][k]['seed'])
        for v in variants:
            r=by[v][key];reqs=[json.loads(x) for x in (Path(r['directory'])/'requests.jsonl').read_text().splitlines()]
            for ax,cls in zip(axes,['hot','cold']):
                cohort=[q for q in reqs if q['class']==cls]
                waits=sorted((q['admission_ns']-q['queue_ns'])/1e6 for q in cohort if q['admission_ns'] is not None and q['queue_ns'] is not None)
                if waits:ax.step(waits,np.arange(1,len(waits)+1)/len(waits),where='post',label=f'{LABELS[v]} {len(waits)}/{len(cohort)} admitted',color=COLORS[v])
                sources.append(dict(directory=r['directory'],seed=r['seed'],variant=v,workload='W6',cohort=cls,
                                    offered=len(cohort),admitted=len(waits),wait_ms=waits,
                                    timeouts=sum(q['status']=='timeout' for q in cohort)))
    for ax,cls in zip(axes,['hot','cold']):
        ax.set_xlabel(cls.capitalize()+' first-admission wait (ms)');ax.set_ylabel('CDF conditional on admission')
        ax.set_xscale('symlog',linthresh=1);ax.set_ylim(0,1.02);ax.grid(alpha=.2)
        if common:ax.legend(fontsize=7)
        else:ax.text(.5,.5,'No eligible shared E03 seed',ha='center',transform=ax.transAxes)
    save(fig,'F6b-wait-cdf','Admitted hot and cold waiting-time distributions',
         '固定取三策略首个共有seed。分布含所有已准入请求（包括后来超时者），以已准入为条件；图例列准入/总请求数，逐类超时与删失下界见补充核算，不把未准入当作等待0。',sources)


def f9(rows):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    variants=['H0000','S1000','S1L00','S1100','S1010','fixed24','S1011','S1110','S1111']
    points(axes[0],rows,variants,'sampled_min_free_mib');axes[0].set_ylabel('Whole-card sampled minimum free VRAM (MiB)')
    axes[0].axhline(1024,color='#dc2626',ls='--',label='Hard minimum 1GiB')
    axes[0].axhline(1536,color='#ca8a04',ls=':',label='Target 1.5GiB');axes[0].legend(fontsize=7)
    points(axes[1],rows,variants,'peak_reserved_mib');axes[1].set_ylabel('Worker measurement peak reserved (MiB)')
    for ax in axes:ax.tick_params(axis='x',rotation=30)
    save(fig,'F9-memory','8GB feasibility and memory headroom',
         '包含正式目录全部有遥测的尝试，包括污染、失败、中断及superseded版本。整卡free含启动/预热，来自0.5秒采样，不能证明不存在瞬时峰值；worker峰值只在有测量snapshot时可得，缺失不是0。allocated与reserved不能相加。',rows)


def memory_evidence(runs):
    result=[]
    for row in runs:
        r=dict(row);p=Path(r['directory']);telemetry=p/'gpu-telemetry.csv'
        if not telemetry.exists():continue
        info=json.loads((p/'run.json').read_text())
        r.update(variant=info['variant'],superseded_by_protocol=info.get('superseded_by_protocol'),
                 experiment=info.get('experiment',r.get('experiment')),
                 workload=r.get('workload',Path(info['trace']).name.split('-')[0]))
        if r.get('sampled_min_free_mib') is None:
            values=[]
            for record in csv.DictReader(telemetry.open()):
                for key,value in record.items():
                    if key and key.strip().split(' [')[0]=='memory.free':
                        try:values.append(float(value.strip().split()[0]))
                        except (ValueError,AttributeError):pass
            if values:r['sampled_min_free_mib']=min(values)
        result.append(r)
    return result


def f8(rows,config):
    data=[r for r in rows if r['experiment'] in ('E02','E02-low')]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    rates={'E02':config['rates']['W4'],'E02-low':config['additional_rate_W4']}
    for v in ['H0000','S1000','S1010']:
        a=[r for r in data if r['variant']==v]
        axes[0].scatter([rates[r['experiment']] for r in a],[r['goodput_req_s'] for r in a],color=COLORS[v],label=LABELS[v],s=20,alpha=.75)
        axes[1].scatter([rates[r['experiment']] for r in a],[r['failed']/r['offered']*100 for r in a],color=COLORS[v],label=LABELS[v],s=20,alpha=.75)
    axes[0].set_ylabel('Client SLO goodput (req/s)');axes[1].set_ylabel('Failed or incomplete requests (%)')
    for ax in axes:ax.set_xlabel('Offered arrival rate (req/s)');ax.grid(alpha=.2)
    axes[0].legend()
    save(fig,'F8-goodput','SLO goodput and failure rate at both frozen arrival rates',
         '每点为一独立run，统一client TTFT短500ms/长2000ms及冻结E2E阈值；不同策略不各自调阈值。仅两个已测rate，不估计未测容量拐点。',data)


def get_pairs(rows,exp,a,b,metric):
    first={r['trace_sha256']:r for r in rows if r['experiment']==exp and r['variant']==a and r.get(metric) is not None}
    second={r['trace_sha256']:r for r in rows if r['experiment']==exp and r['variant']==b and r.get(metric) is not None}
    return [(first[k],second[k]) for k in sorted(first.keys()&second.keys()) if first[k][metric]>0]


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)


def main():
    runs,aggregates,comparisons=analyze(OUT/'measure',OUT/'report')
    rows=[r for r in runs if r['status']=='valid' and r.get('selected_for_primary',True)]
    assert rows,'No valid formal measurements exist; refusing to make a final report'
    config=json.loads((OUT/'frozen-config.json').read_text())
    cutoff=max(json.loads((Path(r['directory'])/'run.json').read_text()).get('finished_wall',0) for r in rows)
    cutoff_text=datetime.fromtimestamp(cutoff).astimezone().isoformat(timespec='seconds')
    memory_rows=memory_evidence(runs)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    FIGURES.clear()
    for fn in (f1,lambda x:f2(x,config),f2_classes,f3,f4,f5,f6,f6_cdf,f7,lambda x:f8(x,config)):fn(rows)
    f9(memory_rows)
    claims=[]
    primaries=[('E02','S1000','S1010','itl_server_ms_p99',True),
               ('E03','S1L00','S1100','cold_wait_ms_p95',True),
               ('E04','S1000','S1L00','output_tok_s',False),
               ('E05','fixed24','S1011','itl_server_ms_p99',True),
               ('E06','H0000','S1111','output_tok_s',False)]
    for exp,a,b,metric,latency in primaries:
        pairs=get_pairs(rows,exp,a,b,metric)
        if len(pairs)>=5:
            ci=paired_ci(pairs,metric,latency)
            claims.append(dict(experiment=exp,reference=a,variant=b,metric=metric,
                              scope='exploratory: tail calibration failed' if latency and not config.get('tail_calibration_passed') else 'conditional fixed-length timing',
                              reference_median=float(np.median([x[metric] for x,y in pairs])),
                              candidate_median=float(np.median([y[metric] for x,y in pairs])),**ci))
    (OUT/'report/primary-comparisons.json').write_text(json.dumps(claims,indent=2))
    (OUT/'report/figures.json').write_text(json.dumps(FIGURES,indent=2,ensure_ascii=False))
    brief=table(['实验 / 条件对照','指标','配对次数','参考绝对值','候选绝对值','改善率中位数 / 95%配对CI'],[
        [c['experiment']+' '+LABELS[c['reference']]+'→'+LABELS[c['variant']],c['metric'],c['n'],
         f"{c['reference_median']:.3f}",f"{c['candidate_median']:.3f}",
         f"{c['median_improvement_pct']:+.2f}% [{c['ci95_low_pct']:+.2f}, {c['ci95_high_pct']:+.2f}]%"] for c in claims])
    d=next((c for c in claims if c['experiment']=='E05'),None)
    dtext=('E05尚无至少5对入选记录，暂不发布动态预算边际收益。' if d is None else
           ('动态预算相对固定24未得到稳定的ITL尾部改善证据，保留负结果。' if d['ci95_low_pct']<=0 else
            '动态预算出现探索性ITL变化，但尾延迟校准未通过，不能认定为经校准的边际收益；需并列读取长TTFT、吞吐与失败代价。'))
    minfree=min(r['sampled_min_free_mib'] for r in memory_rows if r.get('sampled_min_free_mib') is not None)
    head=f'''# RTX 5060 Ti 8GB mini-SGLang 实测报告

本报告只使用正式目录的实际GPU/HTTP运行，不混入pilot、CPU替身或历史模拟。完整结果和失败记录均保留。

入选证据截止：{cutoff_text}。当前执行进度另见`PROGRESS.json`与`CAMPAIGN_STATE.json`，不将阶段性快照当作持续刷新的最终结果。

**尾延迟采集/客户端亲和性的预注册校准未通过。** ITL差异及其区间全部属于探索性观察，不能用作已验证的ITL优化或简历百分比；原始校准结果和失败判定均保留。正式矩阵状态以`PROGRESS.json`与覆盖表为准，未完成时本报告是阶段性快照。

共归档{len(runs)}次正式尝试，其中{len(rows)}次通过请求/记录/资源和逐run运行质量检查并入选。每个核心cell的次数见矩阵，不能把token间隔数当作独立重复；不足5对不发布主对照CI。

{brief}

{dtext}

## 必须先读的结论边界

- **H*不是原样上游。** 原样`9a91cfafe754aa85daee49998176275667eb58f2`在本机overlap中O32只交付31个ACK并发出两次终态，未通过验收。正式H*和S都使用同一份host长度终态修正、post-EOS/Abort结果过滤、观测adapter及HTTP边缘参数。共有补丁分别保存在`patches/`，原样失败未删除。
- **跨batch/chunk形状的greedy逐位相同及最初HF绝对误差门槛未通过。** 16个固定分歧前缀、2,430,976个logits在相同chunk下H/S逐位一致，51份推理数学源码一致；只改变H自身chunk就有7个argmax改变。严格HF门槛的失败未放宽。性能结论限于与上游相同数值行为的固定长度调度实验，不证明跨形状文本逐位一致或质量无损。
- BF16权重、tokenizer、TP1、FI(page1)、KV16384、context2304、running16、Prefill256和graph列表`[1,2,4,8,16]`固定。正式比较没有改变精度、权重或缩短输出。采样最低整卡free为{minfree:.0f} MiB；GPU轮询不能证明没有瞬时尖峰。
- TTFT-text、服务端D2H后host emit ITL、客户端token ACK ITL、每请求TPOT分别计算。所有正式ACK均先与服务端uid/ordinal/count/terminal核验。服务端时间不是纯GPU kernel完成时间。
- 主吞吐分母是冻结trace从首个计划到达到最后终止的完整时段，含drain与失败等待；只把完整校验成功请求的生成token计入成功吞吐。失败、超时和未准入另列。

## 环境与复现

Ubuntu26.04/WSL2，Core Ultra5 230F，10可见CPU，RTX5060Ti8151MiB，Windows驱动596.21；Python3.12.14、torch2.9.1/CUDA Runtime12.8、FlashInfer0.7.0.post1、Transformers4.57.3、sgl-kernel0.3.21。Toolkit13.2.86/GCC14用于JIT；Toolkit12.9的两次真实编译失败保留。模型以本地逐文件SHA256标识，不把规划中的HF revision当作已证明的权重身份。

客户端绑定CPU8,9，server绑定0–7；GPU/CPU遥测独立低频进程在CPU7。客户端预先tokenize/序列化、预热HTTP连接、禁用重试和测量段循环GC，绝对到达时间回放。采集开销、dummy SSE和亲和性结果在`preflight/`，污染标记按实际lag/温度列出。

冻结配置见[frozen-config.json](frozen-config.json)，模型/源码/依赖见[environment](environment/)，全部请求trace见[traces](traces/)。每个run保存run.json、client.json、measurement.pkl、tokens.bin、requests.jsonl、GPU/CPU遥测和stdout/stderr。

```bash
cd /home/wangwei/ai-infra/Promotion/mini-sglang-lru-touch
source benchmarks/serving/env.sh
taskset -c 8,9 python benchmarks/serving/suite.py --phase measure
python benchmarks/serving/analyze_results.py --results ../benchmark-results/measure --output-dir ../benchmark-results/report
python benchmarks/serving/make_report.py
```

已完成run由receipt核对后跳过；失败/中断保存到新attempt目录，不覆盖旧证据。可执行代码及独立源快照的hash保存在manifest中。

## 图表与证据

'''
    for f in FIGURES:
        head+=f"### {f['id']} — {f['title']}\n\n![{f['title']}](report/{f['png']})\n\n{f['caption']} [逐run绘图数据](report/{f['source']}) · [SVG](report/{f['svg']})\n\n"
    head+='''## 统计与失败解释

每次run独立计算分位数，重复间给中位数/IQR/MAD/min/max；误差线为IQR，95%区间以run为单位配对bootstrap10,000次。n=5区间不稳定；按冻结规则出现方向不稳、跨0或run内CV>5%时扩到10，保留前5次。cold每run约200，P99仅探索，公平性主读P95/max/超过5秒及未准入。完整统计见[逐次CSV](report/runs.csv)、[重复汇总](report/aggregates.json)、[全部配对CI](report/paired-ci.json)与[主对照CI](report/primary-comparisons.json)。

H→S1000包含默认实现的全部净差异；A没有独立关闭开关，A-ISO未实现。B依赖A，D依赖C，不能把四项当2^4独立因子。Full去B与去D只有在同W4-X trace下做条件比较。无V0010/V0011不估计A×C，不以naive cache伪造A0。

原样上游终态错误、CUDA编译错误、HTTP keepalive错误、跨形状数值分歧、pilot积压与所有正式失败均在错误台账中。W5的6秒pilot积压未并入正式；正式统一12秒周期，I/O长度保持不变。没有稳定改善的点不写简历收益。
'''
    (OUT/'REPORT.md').write_text(head)
    receipt=dict(formal_attempts=len(runs),audited_runs=len(rows),figures=FIGURES,primary_comparisons=claims,
                 evidence_cutoff=cutoff_text,
                 measurement_complete=(json.loads((OUT/'PROGRESS.json').read_text()).get('phase')=='measure complete' if (OUT/'PROGRESS.json').exists() else False),
                 numeric_limitations='preflight/numerics/decision.json',raw_error_ledger='report/errors.json')
    (OUT/'report/final-receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False))
    # A compact offline HTML companion; images remain standalone SVG artifacts.
    doc='<!doctype html><html lang="zh"><meta charset="utf-8"><title>mini-SGLang benchmark</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#172033}img{max-width:100%}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}section{margin:40px 0}</style>'
    doc+='<h1>mini-SGLang RTX 5060 Ti 8GB 实测</h1><p>完整正文、结论边界和复现命令见 <a href="../REPORT.md">REPORT.md</a>。H*含共有正确性补丁；严格跨形状数值一致性与尾延迟采集校准门槛均未通过。ITL变化为探索性观察，不作为简历收益。</p>'
    doc+=f'<p>归档{len(runs)}次正式尝试，{len(rows)}次入选。'+html.escape(dtext)+'</p>'
    if claims:
        doc+='<table><thead><tr><th>对照</th><th>指标</th><th>n</th><th>改善率中位数</th><th>95%配对CI</th></tr></thead><tbody>'
        for c in claims:
            cells=[c['experiment']+' '+LABELS[c['reference']]+'→'+LABELS[c['variant']],c['metric'],str(c['n']),f"{c['median_improvement_pct']:+.2f}%",f"[{c['ci95_low_pct']:+.2f}, {c['ci95_high_pct']:+.2f}]%"]
            doc+='<tr>'+''.join('<td style="padding:8px">'+html.escape(x)+'</td>' for x in cells)+'</tr>'
        doc+='</tbody></table>'
    else:doc+='<p>尚无至少5对入选记录，不发布主对照CI。</p>'
    for f in FIGURES:doc+=f'<section><h2>{html.escape(f["title"])}</h2><img src="{f["svg"]}"><p>{html.escape(f["caption"])}</p><a href="{f["source"]}">源数据</a></section>'
    doc+='</html>'; (OUT/'report/index.html').write_text(doc)
    print(json.dumps(dict(formal_attempts=len(runs),audited_runs=len(rows),figures=len(FIGURES)),ensure_ascii=False))


if __name__=='__main__':main()
