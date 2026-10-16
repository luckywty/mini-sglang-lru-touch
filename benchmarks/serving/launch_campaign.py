"""Start the resumable GPU suite in a detached Linux session, once only.

Source env.sh before invocation. Existing live controller or occupied port1919
causes refusal; this launcher never kills or replaces a running experiment.
"""
from datetime import datetime
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import time

from run_experiments import OUT, PY, ROOT, MODEL, atomic, sha


def main():
    lock=(OUT/'campaign-launch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=OUT/'CAMPAIGN_STATE.json'
    if state_path.exists():
        previous=json.loads(state_path.read_text());pid=previous.get('driver_pid')
        if pid:
            try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            except FileNotFoundError:cmd=''
            if 'suite.py --phase measure' in cmd:raise RuntimeError(f'Controller {pid} is still live; do not launch a duplicate')
    with socket.socket() as s:
        s.settimeout(.5)
        if s.connect_ex(('127.0.0.1',1919))==0:raise RuntimeError('Port1919 is occupied; inspect existing owned server before resuming')
    config=json.loads((OUT/'frozen-config.json').read_text())
    if not config.get('measurement_frozen'):raise RuntimeError('Campaign configuration is not frozen')
    for label in ('H','S'):
        manifest=OUT/f'environment/runtime-{label}-files.json'
        if sha(manifest)!=config[f'runtime_{label}_manifest_sha256']:raise RuntimeError('Source manifest differs from frozen contract')
        for name,digest in json.loads(manifest.read_text()).items():
            if sha(ROOT/f'runtime-{label}'/name)!=digest:raise RuntimeError(f'Runtime source changed: {label}/{name}')
    model_manifest=OUT/'environment/model-files.json'
    if sha(model_manifest)!=config['model_manifest_sha256']:raise RuntimeError('Model manifest changed')
    for name,record in json.loads(model_manifest.read_text()).items():
        if sha(Path(MODEL)/name)!=record['sha256']:raise RuntimeError(f'Model file changed: {name}')
    repo=Path(__file__).resolve().parents[2]
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path=OUT/f'measure/suite-detached-{stamp}.log'
    with log_path.open('a') as log:
        driver=subprocess.Popen(['taskset','-c','8,9',str(PY),'-u',str(Path(__file__).with_name('suite.py').resolve()),'--phase','measure'],
            cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    # Record driver identity before spawning any additional helper.
    state={'phase':'measurement running in detached Linux session','driver_pid':driver.pid,
           'protocol_version':config.get('protocol_version'),'tail_gate_passed':False,
           'updated_wall':time.time(),'log':str(log_path.relative_to(OUT)),
           'resume_command':'source benchmarks/serving/env.sh; python benchmarks/serving/launch_campaign.py'}
    atomic(state_path,state)
    prior=OUT/'WATCHER_STATE.json'
    if prior.exists():
        archive=OUT/'environment/amendments';archive.mkdir(exist_ok=True)
        (archive/f'WATCHER_STATE-{stamp}.json').write_bytes(prior.read_bytes())
    with (OUT/f'report/watcher-{stamp}.log').open('a') as log:
        watcher=subprocess.Popen(['taskset','-c','8,9',str(PY),'-u',str(Path(__file__).with_name('watch_campaign.py').resolve()),'--pid',str(driver.pid)],
             cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state['watcher_pid']=watcher.pid;atomic(state_path,state)
    print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
