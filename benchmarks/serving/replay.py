"""HTTP/SSE collector. All timestamps monotonic, no retries or open-loop gate."""
import argparse
import asyncio
import json
import gc
from pathlib import Path
import resource
import time

import httpx
import numpy as np
import uvloop
from raw_http import RawHTTPClient

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

CLOCK = time.perf_counter_ns


async def sleep_until(target_ns):
    """Absolute host deadline with cooperative polling in the last millisecond.

    uvloop timers have millisecond resolution. Waking early and yielding instead
    of blocking reduces timer quantization without hiding HTTP parse contention.
    Its CPU cost remains included in the client process CPU measurement.
    """
    while (remaining := target_ns-CLOCK()) > 0:
        await asyncio.sleep((remaining-1_000_000)/1e9 if remaining>2_000_000 else 0)


async def replay(rows, base_url, concurrency=16, mode='open-loop', max_outstanding=4096,transport='raw'):
    assert rows, 'Cannot replay an empty trace'
    limits = httpx.Limits(max_connections=max_outstanding, max_keepalive_connections=max_outstanding)
    results, loop_lags = [], []
    finished = False
    semaphore = asyncio.Semaphore(concurrency)
    # Serialize before the measurement begins.
    bodies = {r['request_id']:json.dumps(dict(model='Qwen3-0.6B', prompt=r['prompt'], stream=True,
                max_tokens=r['max_output_tokens'], temperature=0., top_k=1, top_p=1.,
                ignore_eos=r['ignore_eos'])).encode() for r in rows}
    session=(RawHTTPClient(base_url,min(len(rows),max_outstanding)) if transport=='raw' else
             httpx.AsyncClient(limits=limits,timeout=120.,trust_env=False))
    async with session as client:
        await client.get(base_url+'/v1')
        # Avoid cyclic-GC pauses in the load generator. Objects still use normal
        # reference counting; memory/CPU usage and max outstanding are recorded.
        gc.collect()
        gc_was_enabled=gc.isenabled()
        gc.disable()
        # Prepare all arrival tasks before the first absolute deadline. Creating
        # 1000 tasks after t=0 would itself delay the first burst beyond 1ms.
        start = CLOCK()+100_000_000
        cpu0 = usage0 = None

        async def lag_monitor():
            # sleep_until absolute schedule, log scheduler jitter at 10ms cadence.
            await sleep_until(start)
            target = start+10_000_000
            last_progress=CLOCK()
            while not finished:
                await sleep_until(target)
                now = CLOCK()
                loop_lags.append(max(0,now-target)/1e6)
                target = now+10_000_000
                if now-last_progress>=15_000_000_000:
                    print('REPLAY',len(results),'/',len(rows),'completed',
                          'elapsed_s',round((now-start)/1e9,1),flush=True)
                    last_progress=now

        async def request_one(row):
            nonlocal cpu0,usage0
            planned = start + int(row['arrival_s']*1e9)
            if mode == 'open-loop':
                await sleep_until(planned)
            if mode == 'closed-loop':
                await sleep_until(start)
                await semaphore.acquire()
            if cpu0 is None:
                cpu0=time.process_time()
                usage0=resource.getrusage(resource.RUSAGE_SELF)
            result = dict(request_id=row['request_id'], planned_ns=planned, send_ns=CLOCK(),
                          uid=None, ack_ns=[], recv_batch=[], first_text_ns=None, last_text_ns=None,
                          output_text='', terminal_ns=None, status='unresolved', error=None,
                          finish_reason=None, sse_events=0)
            try:
                async with asyncio.timeout(row.get('timeout_s',120)), client.stream('POST',base_url+'/v1/chat/completions',
                                         content=bodies[row['request_id']],
                                         headers={'content-type':'application/json'},
                                         timeout=row.get('timeout_s',120)) as response:
                    response.raise_for_status()
                    if response.headers.get('x-bench-uid') is not None:
                        result['uid']=int(response.headers['x-bench-uid'])
                    buf = b''
                    batch = 0
                    done = False
                    async for part in response.aiter_bytes():
                        received = CLOCK()
                        batch += 1
                        buf += part
                        while b'\n\n' in buf:
                            raw,buf = buf.split(b'\n\n',1)
                            if not raw.startswith(b'data: '):
                                continue
                            raw = raw[6:]
                            result['sse_events'] += 1
                            if raw == b'[DONE]':
                                done = True
                                result['terminal_ns'] = CLOCK()
                                continue
                            item = json.loads(raw)
                            uid = int(item['id'].split('-')[-1])
                            assert result['uid'] in (None,uid)
                            result['uid'] = uid
                            choice = item['choices'][0]
                            if choice['finish_reason'] is not None:
                                result['finish_reason'] = choice['finish_reason']
                                continue
                            # Every data ACK, including empty content. Formal
                            # acceptance still requires server ID/ordinal audit.
                            result['ack_ns'].append(received)
                            result['recv_batch'].append(batch)
                            content = choice['delta'].get('content','')
                            if content:
                                if result['first_text_ns'] is None:
                                    result['first_text_ns'] = received
                                result['last_text_ns'] = received
                                result['output_text'] += content
                    result['status'] = 'success' if done and result['finish_reason'] else 'incomplete'
            except (httpx.TimeoutException,TimeoutError) as e:
                result['status'],result['error'] = 'timeout',repr(e)
            except Exception as e:
                result['status'],result['error'] = 'failure',repr(e)
            finally:
                if result['terminal_ns'] is None:
                    result['terminal_ns'] = CLOCK()
                results.append(result)
                if mode == 'closed-loop':
                    semaphore.release()

        monitor = asyncio.create_task(lag_monitor())
        # One task per immutable arrival; there is no semaphore in open loop.
        tasks = [asyncio.create_task(request_one(row)) for row in rows]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks),
                                   timeout=max(r['arrival_s']+r.get('timeout_s',120) for r in rows)+5)
        finally:
            finished = True
            await monitor
        end = CLOCK()
        cpu1=time.process_time()
        usage1 = resource.getrusage(resource.RUSAGE_SELF)
        if gc_was_enabled:gc.enable()
    return dict(start_ns=start,end_ns=end,requests=results,loop_lag_ms=loop_lags,
                loop_lag_p99_ms=float(np.quantile(loop_lags,.99)) if loop_lags else None,
                cpu_seconds=cpu1-cpu0, wall_seconds=(end-start)/1e9,
                voluntary_switches=usage1.ru_nvcsw-usage0.ru_nvcsw,
                involuntary_switches=usage1.ru_nivcsw-usage0.ru_nivcsw,
                maxrss_kib=usage1.ru_maxrss,transport=transport,cyclic_gc_during_measurement=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--trace',required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:1919')
    p.add_argument('--mode',choices=['open-loop','closed-loop'],default='open-loop')
    p.add_argument('--concurrency',type=int,default=16)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    rows=[json.loads(line) for line in Path(a.trace).read_text().splitlines()]
    result=asyncio.run(replay(rows,a.base_url,a.concurrency,a.mode))
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False))
