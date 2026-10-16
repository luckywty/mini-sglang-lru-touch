"""A paced dummy SSE source tests the client separately from GPU inference."""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time
import os
import signal

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
import numpy as np

from replay import replay, sleep_until

app=FastAPI()
counter=0
TARGET_PER_STREAM=375.


@app.get('/v1')
async def ready():return {'status':'ok'}


@app.post('/v1/chat/completions')
async def completion(request:Request):
    global counter
    payload=await request.json();uid=counter;counter+=1
    count=payload['max_tokens']
    async def stream():
        start=time.perf_counter_ns()
        chunk={'id':f'cmpl-{uid}','choices':[{'delta':{'content':'a'},'finish_reason':None}]}
        encoded=('data: '+json.dumps(chunk)+'\n\n').encode()
        for i in range(count):
            await sleep_until(start+int(i/TARGET_PER_STREAM*1e9))
            yield encoded
        chunk['choices'][0]={'delta':{},'finish_reason':'stop'}
        yield ('data: '+json.dumps(chunk)+'\n\n').encode()
        yield b'data: [DONE]\n\n'
    return StreamingResponse(stream(),media_type='text/event-stream')


def measure(rate,output,startup_bursts=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['BENCH_DUMMY_PER_STREAM']=str(rate/32)
    log=(output/'dummy-server.log').open('w')
    proc=subprocess.Popen(['taskset','-c','6,7',sys.executable,__file__,'--serve'],
                          env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        with httpx.Client(trust_env=False,timeout=1) as client:
            for _ in range(100):
                try:
                    if client.get('http://127.0.0.1:1920/v1').status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(.1)
        rows=[dict(request_id=f'dummy-{i}',arrival_s=(i//32)*.1 if startup_bursts else 0.,
                   prompt='dummy source',max_output_tokens=4 if startup_bursts else 2048,
                   ignore_eos=True,timeout_s=60) for i in range(1000 if startup_bursts else 32)]
        result=asyncio.run(replay(rows,'http://127.0.0.1:1920',32,'open-loop'))
        (output/'client.json').write_text(json.dumps(result))
        assert all(r['status']=='success' and len(r['ack_ns'])==(4 if startup_bursts else 2048) for r in result['requests'])
        send_lag=[(r['send_ns']-r['planned_ns'])/1e6 for r in result['requests']]
        times=np.sort([t for r in result['requests'] for t in r['ack_ns']])
        receipt=dict(target_ack_s=rate,observed_ack_s=len(times)/((times[-1]-times[0])/1e9),
                     workload='1000 requests, B32 at100ms, O4' if startup_bursts else '32 streams, O2048',
                     token_count=len(times),loop_lag_p99_ms=result['loop_lag_p99_ms'],
                     send_lag_p99_ms=float(np.quantile(send_lag,.99)),client_cpu_seconds=result['cpu_seconds'],
                     client_cpu_fraction=result['cpu_seconds']/result['wall_seconds'])
        (output/'summary.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt),flush=True)
    finally:
        os.killpg(proc.pid,signal.SIGINT)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--serve',action='store_true')
    p.add_argument('--rate',type=float,default=12000);p.add_argument('--output')
    p.add_argument('--startup-bursts',action='store_true')
    a=p.parse_args()
    if a.serve:
        TARGET_PER_STREAM=float(os.environ['BENCH_DUMMY_PER_STREAM'])
        uvicorn.run(app,host='127.0.0.1',port=1920,access_log=False,loop='uvloop',timeout_keep_alive=300)
    else:measure(a.rate,a.output,a.startup_bursts)
