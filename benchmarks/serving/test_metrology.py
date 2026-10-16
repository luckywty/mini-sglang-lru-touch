"""Deterministic metrology fixtures. These are never benchmark results."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import pickle
import threading
from array import array

import pytest

from analyze_results import audit_run
from generate_workloads import Factory
from replay import replay


def test_sse_empty_ack_finish_and_done_are_distinct():
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):
            pass
        def do_GET(self):
            self.send_response(200);self.send_header('content-length','2');self.end_headers();self.wfile.write(b'{}')
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            events=[{'id':'cmpl-17','choices':[{'delta':d,'finish_reason':f}]} for d,f in
                    [({'role':'assistant'},None),({'content':'A'},None),({},'stop')]]
            body=''.join('data: '+json.dumps(x)+'\n\n' for x in events)+'data: [DONE]\n\n'
            self.send_response(200);self.send_header('content-type','text/event-stream')
            self.send_header('content-length',str(len(body)));self.end_headers()
            self.wfile.write(body.encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    row=dict(request_id='fixture',arrival_s=0.,prompt='hello',max_output_tokens=2,
             ignore_eos=True,timeout_s=2)
    try:
        result=asyncio.run(replay([row],f'http://127.0.0.1:{server.server_port}'))
        r=result['requests'][0]
        assert r['status']=='success' and r['uid']==17
        assert len(r['ack_ns'])==2 and r['sse_events']==4
        assert r['output_text']=='A' and r['first_text_ns']>=r['ack_ns'][0]
        assert r['send_ns']<=r['ack_ns'][0]<=r['terminal_ns']
    finally:
        server.shutdown();thread.join()


def fixture_run(tmp_path, extra_ack=False):
    trace=tmp_path/'W-fixture.jsonl'
    trace.write_text(json.dumps(dict(request_id='r',input_ids=[1,2],input_tokens=2,
                max_output_tokens=3,ignore_eos=True,arrival_s=0.,**{'class':'cold'}))+'\n')
    (tmp_path/'run.json').write_text(json.dumps(dict(status='complete',variant='fixture',profile='fixture',
                    trace=str(trace),trace_sha256='fixture',collector_sha256='fixture',
                    source_manifest_sha256='fixture',experiment='fixture')))
    start=1_000_000_000
    times=[start+20_000_000,start+30_000_000,start+120_000_000]
    ack=[t+1_000_000 for t in times]
    if extra_ack:
        ack.append(times[-1]+2_000_000)
    request=dict(request_id='r',uid=0,status='success',send_ns=start,planned_ns=start,
                 terminal_ns=start+130_000_000,ack_ns=ack,first_text_ns=ack[0])
    (tmp_path/'client.json').write_text(json.dumps(dict(requests=[request],start_ns=start,
              cpu_seconds=.001,wall_seconds=.13,loop_lag_p99_ms=.01)))
    snapshot=dict(tokens=array('q',[v for i,t in enumerate(times) for v in (0,i,10+i,t,int(i==2))]),
          queued=[(0,start+1_000_000,2,3,[1,2])],admissions=[(0,start+2_000_000,0,0,0,0)],
          resources=dict(pending=0,running=0,protected=0,free_tables=8,table_capacity=8,
                         unique_free_tables=8,free_pages=8192,evictable=0,num_pages=8192),
          dropped_events=0,budgets=[],counts={},graph_capture_list=[1,2,4,8],
          memory=dict(peak_allocated=1,peak_reserved=1),decision_prepare_ns=array('q',[1000,2000]))
    (tmp_path/'measurement.pkl').write_bytes(pickle.dumps(snapshot))
    (tmp_path/'gpu-telemetry.csv').write_text('memory.free [MiB], memory.used [MiB]\n2048 MiB, 6000 MiB\n')
    return audit_run(tmp_path)


def test_tpot_is_per_request_mean_and_itl_is_pooled_quantile(tmp_path):
    result=fixture_run(tmp_path)
    assert result['status']=='valid'
    assert result['tpot_server_ms_p95']==50
    assert result['itl_server_ms_p99']==pytest.approx(89.2)
    assert result['generated_tokens']==3


def test_ack_count_mismatch_invalidates_request_and_throughput(tmp_path):
    result=fixture_run(tmp_path,extra_ack=True)
    assert result['status']=='invalid'
    assert result['success']==0 and result['output_tok_s']==0


@pytest.mark.parametrize('workload',['W3','W6'])
def test_exact_shared_prefix_and_lengths(workload):
    factory=Factory(seed=31799)
    rows,prime=factory.make(workload,32,10,'burst')
    for row in rows+prime:
        assert len(row['input_ids'])==576
        assert factory.tok.encode(row['prompt'])==row['input_ids']
    for p in prime:
        matching=[r for r in rows if r['prefix_group']==p['prefix_group']]
        assert all(r['input_ids'][:512]==p['input_ids'][:512] for r in matching)


def test_w6_cold_requests_lead_every_burst():
    rows,_=Factory(seed=31723).make('W6',160,50.,'burst')
    assert sum(r['class']=='cold' for r in rows)==32
    for start in range(0,160,32):
        burst=rows[start:start+32]
        quota=(start+32)//5-start//5
        assert all(r['class']=='cold' for r in burst[:quota])
        assert all(r['class']=='hot' for r in burst[quota:])


def test_two_variant_order_is_counterbalanced_and_other_orders_preserved():
    from suite import variant_order
    assert [variant_order(['A','B'],i) for i in range(4)]==[['A','B'],['B','A'],['A','B'],['B','A']]
    for n in (3,4,7):
        variants=list(range(n))
        for block in range(10):
            old=variants[block%n:]+variants[:block%n]
            if block%2:old=list(reversed(old))
            assert variant_order(variants,block)==old


@pytest.mark.parametrize('previous_failures,expected_new_runs',[(1,2),(3,0)])
def test_resume_preserves_total_quality_retry_budget(tmp_path,monkeypatch,previous_failures,expected_new_runs):
    import suite
    monkeypatch.setattr(suite,'OUT',tmp_path)
    (tmp_path/'frozen-config.json').write_text(json.dumps({'tail_calibration_passed':False}))
    trace=tmp_path/'trace.jsonl';trace.write_text('{}\n');digest=suite.sha(trace)
    directory=tmp_path/'measure/E02/seed-31723/S1010'
    def receipt(p):
        p.mkdir(parents=True,exist_ok=True)
        obj={'directory':str(p),'status':'complete','variant':'S1010','trace_sha256':digest}
        (p/'run.json').write_text(json.dumps(obj));(p/'measurement.pkl').write_bytes(b'unit fixture only')
        return obj
    for i in range(previous_failures):receipt(directory if i==0 else directory/f'quality-retry-{i}')
    calls=[]
    def run(variant,trace,target,**kwargs):
        calls.append(str(target));return receipt(Path(target))
    def audit(p):
        return {'directory':str(p),'status':'valid','variant':'S1010',
                'performance_pollution':['unit client lag'],'selected_for_primary':False}
    monkeypatch.setattr(suite,'run_cell',run);monkeypatch.setattr(suite,'audit_run',audit)
    result=suite.validated_cell('S1010',trace,directory)
    assert len(calls)==expected_new_runs
    assert not result['selected_for_primary']
    if calls:assert calls[0].endswith('quality-retry-1') and calls[-1].endswith('quality-retry-2')

