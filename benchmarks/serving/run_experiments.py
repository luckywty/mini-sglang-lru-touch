"""Owned process groups, frozen inputs, atomic receipts, resumable serial GPU runs."""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback
import uuid

import httpx

from generate_workloads import Factory, save
from replay import replay

ROOT = Path('/home/wangwei/ai-infra/Promotion')
OUT = ROOT/'benchmark-results'
PY = ROOT/'benchmark-env/bin/python'
MODEL = '/home/wangwei/ai-infra/models/Qwen3-0.6B'
BASE = ['--model-path',MODEL,'--dtype','bfloat16','--tp-size','1',
        '--attention-backend','fi','--cache-type','radix','--page-size','1',
        '--max-seq-len-override','2304','--max-prefill-length','256',
        '--memory-ratio','0.70','--num-tokenizer','0','--host','127.0.0.1','--port','1919']
VARIANTS = {
    'H0000': [],
    'S1000':['--schedule-policy','fcfs','--lpm-aging-tokens-per-pass','0','--fairness-max-wait-steps','0','--batch-schedule-policy','prefill-first'],
    'S1L00':['--schedule-policy','lpm','--lpm-aging-tokens-per-pass','0','--lpm-queue-threshold','128','--fairness-max-wait-steps','0','--batch-schedule-policy','prefill-first'],
    'S1100':['--schedule-policy','lpm-aging','--lpm-aging-tokens-per-pass','64','--lpm-queue-threshold','128','--fairness-max-wait-steps','0','--batch-schedule-policy','prefill-first'],
}
for name,policy,protected,dynamic in [('S1010','fcfs',64,False),('fixed24','fcfs',24,False),('S1011','fcfs',64,True),('S1110','lpm-aging',64,False),('S1111','lpm-aging',64,True)]:
    VARIANTS[name]=['--schedule-policy',policy,'--lpm-aging-tokens-per-pass',str(64 if policy=='lpm-aging' else 0),
                    '--lpm-queue-threshold','128','--fairness-max-wait-steps','0',
                    '--batch-schedule-policy','decode-first-bounded','--max-consecutive-decodes','4',
                    '--protected-prefill-tokens',str(protected)]
    if dynamic:
        VARIANTS[name] += ['--adaptive-prefill-budget','--target-batch-tokens','32','--min-chunk-tokens','8']
PROFILES = {'G0':(8192,8,0),'G0-graph':(8192,8,8),'G1':(16384,16,16),'G1-small':(8192,16,16)}


def atomic(path, obj):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(directory, action, label='measurement'):
    atomic(Path(directory)/'control.json',dict(id=uuid.uuid4().hex,action=action,label=label))


def stop_owned(process):
    if process is None:
        return
    try:
        os.killpg(process.pid,signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def rows_from(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def run_cell(variant, trace, directory, profile='G1', mode='open-loop', concurrency=16,
             prime=None, settle_seconds=60, collector_mode='light', server_cpus='0-7',metadata=None):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/'run.json'
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        summary=directory/'summary.json'
        audited_failure=summary.exists() and json.loads(summary.read_text()).get('status')=='invalid'
        if previous.get('status') in ('complete','invalid') and not audited_failure and not previous.get('superseded_by_protocol'):
            print('RESUME existing',directory,flush=True)
            return previous
        # Preserve partial/interrupted attempts, never replace evidence.
        directory=directory/('attempt-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
        directory.mkdir()
        receipt=directory/'run.json'
    pages,running,graph=PROFILES[profile]
    source=ROOT/('runtime-H' if variant=='H0000' else 'runtime-S')
    cmd=['taskset','-c',server_cpus,str(PY),'-m','minisgl']+BASE+[
         '--num-pages',str(pages),'--max-running-requests',str(running),
         '--cuda-graph-max-bs',str(graph)]+VARIANTS[variant]
    env=os.environ.copy()
    env.update(PYTHONPATH=str(source/'python'),CUDA_HOME='/usr/local/cuda-13.2',
               CC='/usr/bin/gcc-14',CXX='/usr/bin/g++-14',
               NVCC_PREPEND_FLAGS='--compiler-bindir=/usr/bin/g++-14',
               PATH=str(PY.parent)+':/usr/local/cuda-13.2/bin:'+env['PATH'],
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
               MAX_JOBS='2',FLASHINFER_CUDA_ARCH_LIST='12.0',TORCH_CUDA_ARCH_LIST='12.0',
               MINISGL_DISABLE_OVERLAP_SCHEDULING='0',MINISGL_BENCH_DIR=str(directory),
               MINISGL_BENCH_MODE=collector_mode)
    rows=rows_from(trace)
    info=dict(status='starting',directory=str(directory),variant=variant,profile=profile,trace=str(trace),trace_sha256=sha(trace),
              source=str(source),source_manifest_sha256=sha(OUT/'environment'/('runtime-H-files.json' if variant=='H0000' else 'runtime-S-files.json')),
              collector_sha256=sha(source/'python/minisgl/benchmark/collector.py'),
              command=cmd,environment={k:env[k] for k in ['CUDA_HOME','CC','CXX','NVCC_PREPEND_FLAGS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','FLASHINFER_CUDA_ARCH_LIST','TORCH_CUDA_ARCH_LIST','MINISGL_DISABLE_OVERLAP_SCHEDULING','MINISGL_BENCH_MODE','PYTHONPATH']},
              mode=mode,concurrency=concurrency,prime_trace=str(prime) if prime else None,
              transport='prewarmed raw HTTP1.1; keepalive=3600s; no request retries',
              tool_hashes={name:sha(Path(__file__).with_name(name)) for name in
                           ['run_experiments.py','replay.py','raw_http.py','cpu_telemetry.py','generate_workloads.py','suite.py']},
              prime_sha256=sha(prime) if prime else None,started_wall=time.time(),
              clock_domain='same-host CLOCK_MONOTONIC perf_counter_ns',
              client_cpus=sorted(os.sched_getaffinity(0)))
    if metadata:info.update(metadata)
    atomic(receipt,info)
    server=gpu=cpu_monitor=None
    try:
        gpu_log=(directory/'gpu-telemetry.csv').open('w')
        gpu=subprocess.Popen(['taskset','-c','7','nvidia-smi','--query-gpu=timestamp,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,clocks_event_reasons.active','--format=csv','-lms','500'],stdout=gpu_log,stderr=subprocess.STDOUT,start_new_session=True)
        stdout=(directory/'server.stdout.log').open('w')
        stderr=(directory/'server.stderr.log').open('w')
        server=subprocess.Popen(cmd,cwd=source,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        info['server_pid']=server.pid
        atomic(receipt,info)
        cpu_monitor=subprocess.Popen(['taskset','-c','7',str(PY),str(Path(__file__).with_name('cpu_telemetry.py')),
                                      '--pid',str(server.pid),'--output',str(directory/'cpu-telemetry.json')],
                                     stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        deadline=time.monotonic()+1200
        with httpx.Client(trust_env=False,timeout=1.) as client:
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'server exited during startup: {server.returncode}')
                try:
                    if client.get('http://127.0.0.1:1919/v1').status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic()>deadline:
                    raise TimeoutError('server startup/JIT deadline (1200s)')
                time.sleep(.5)
        info['ready_wall']=time.time()
        info['status']='warmup'
        atomic(receipt,info)
        print('READY',variant,profile,directory,flush=True)
        # The factory and all payload generation run outside measurement.
        factory=Factory(seed=99117)
        warm=[]
        for n,length in [(1,128),(4,512),(8,128),(16,128),(4,2048)]:
            subset=[factory.row(len(warm)+i,0,length,16,'warmup') for i in range(n)]
            for row in subset:
                row['timeout_s']=1200  # First kernel compilation is outside measurement.
            warm+=subset
            result=asyncio.run(replay(subset,'http://127.0.0.1:1919',min(n,running),'closed-loop'))
            if any(x['status']!='success' for x in result['requests']):
                raise RuntimeError('warmup request failed')
        save(directory/'warmup.jsonl',warm,{'role':'warmup'})
        sentinel=[factory.row(99999,0,128,1,'sentinel')]
        # Thermal rule checks the actual 60-second window; caller may skip for
        # functional smoke/pilot only. No time.sleep longer than 1s.
        settle_start=time.monotonic();stable=False;temperatures=[]
        if settle_seconds>=60:
            import csv
            # Cool the brief shape-warmup transient first, then require the
            # actual 60-second stable window specified in the frozen protocol.
            for _ in range(20):time.sleep(1)
            for window in range(5):
                print('SETTLE',directory,'window',window+1,flush=True)
                for _ in range(settle_seconds):time.sleep(1)
                data=list(csv.DictReader((directory/'gpu-telemetry.csv').open()))
                temperatures=[]
                for row in data[-120:]:
                    for key,value in row.items():
                        if key.strip()=='temperature.gpu':
                            try:temperatures.append(float(value))
                            except ValueError:pass
                stable=bool(temperatures) and max(temperatures)-min(temperatures)<=2
                if stable:break
                print('THERMAL WINDOW UNSTABLE',directory,'range',
                      max(temperatures)-min(temperatures) if temperatures else None,flush=True)
        else:
            for _ in range(max(settle_seconds,1)):time.sleep(1)
        info['thermal_stable']=stable if settle_seconds>=60 else None
        info['settle_seconds']=time.monotonic()-settle_start
        info['settle_temperature_range']=temperatures
        if prime:
            result=asyncio.run(replay(rows_from(prime),'http://127.0.0.1:1919',1,'closed-loop'))
            atomic(directory/'prime-client.json',result)
            if any(x['status']!='success' for x in result['requests']):
                raise RuntimeError('prime failed')
        info['status']='measuring'
        atomic(receipt,info)
        command(directory,'begin')
        print('MEASURE',directory,'offered',len(rows),'injection_s',round(max(r['arrival_s'] for r in rows),2),flush=True)
        result=asyncio.run(replay(rows,'http://127.0.0.1:1919',concurrency,mode))
        atomic(directory/'client.json',result)
        info['measurement_start_ns']=result['start_ns']
        info['measurement_end_ns']=max(x['terminal_ns'] for x in result['requests'])
        if any(r['status']!='success' for r in result['requests']):
            # Let the existing API disconnect/Abort path finish before checking
            # resource reclamation. This delay is outside measured latency/T.
            for _ in range(10):time.sleep(.1)
        # The HTTP parent can stay alive after a scheduler worker exception.
        # Check at the run boundary before sending a sentinel to a dead worker.
        failure_text=(directory/'server.stderr.log').read_text(errors='replace')
        if 'Traceback (most recent call last)' in failure_text:
            raise RuntimeError('Worker traceback before measurement dump; see server.stderr.log')
        # All measured requests have terminated before this host-only dump.
        command(directory,'dump')
        asyncio.run(replay(sentinel,'http://127.0.0.1:1919',1,'closed-loop'))
        deadline=time.monotonic()+30
        while not (directory/'measurement.pkl').exists() and time.monotonic()<deadline:
            time.sleep(.2)
        if not (directory/'measurement.pkl').exists():
            raise RuntimeError('scheduler dump missing')
        snapshot=pickle.loads((directory/'measurement.pkl').read_bytes())
        info['snapshot_resources']=snapshot['resources']
        info['memory']=snapshot['memory']
        info['status']='complete'
        info['finished_wall']=time.time()
        atomic(receipt,info)
        print('COMPLETE',directory,'seconds',round(result['wall_seconds'],2),'requests',len(result['requests']),flush=True)
        return info
    except BaseException as e:
        info.update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',
                    error=repr(e),traceback=traceback.format_exc(),finished_wall=time.time())
        atomic(receipt,info)
        print('FAILED',directory,repr(e),flush=True)
        if isinstance(e,KeyboardInterrupt):
            raise
        return info
    finally:
        stop_owned(server)
        if gpu:
            stop_owned(gpu)
        stop_owned(cpu_monitor)
        info['server_exit_code']=server.returncode if server else None
        info['gpu_telemetry_exit_code']=gpu.returncode if gpu else None
        info['cpu_telemetry_exit_code']=cpu_monitor.returncode if cpu_monitor else None
        atomic(receipt,info)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--trace',required=True)
    p.add_argument('--results-dir',required=True)
    p.add_argument('--profile',choices=PROFILES,default='G1')
    p.add_argument('--mode',choices=['open-loop','closed-loop'],default='open-loop')
    p.add_argument('--concurrency',type=int,default=16)
    p.add_argument('--prime')
    p.add_argument('--settle-seconds',type=int,default=60)
    p.add_argument('--collector-mode',choices=['full','light','token','off'],default='light')
    p.add_argument('--server-cpus',default='0-7')
    a=p.parse_args()
    result=run_cell(a.variant,a.trace,a.results_dir,a.profile,a.mode,a.concurrency,
                    a.prime,a.settle_seconds,a.collector_mode,a.server_cpus)
    sys.exit(0 if result['status']=='complete' else 1)
