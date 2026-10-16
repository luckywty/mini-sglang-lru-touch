"""Detached checkpoint/finalization companion for an already running suite.

It observes one PID plus Linux start-time identity; it never kills processes,
starts a second GPU suite, or treats a dead controller as successful execution.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_experiments import OUT, PY, atomic


def process_identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        return stat[19]
    except (FileNotFoundError,ProcessLookupError):return None


def main(pid):
    ident=process_identity(pid)
    if ident is None:raise RuntimeError('No live controller to adopt')
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'suite.py --phase measure' not in cmd:raise RuntimeError('Refusing to watch unrelated PID')
    state={'watcher_pid':os.getpid(),'driver_pid':pid,'driver_start_ticks':ident,
           'driver_command':cmd,'status':'watching','started_wall':time.time()}
    target=OUT/'WATCHER_STATE.json'
    while process_identity(pid)==ident:
        state['heartbeat_wall']=time.time()
        progress=OUT/'PROGRESS.json'
        if progress.exists():state['progress']=json.loads(progress.read_text())
        atomic(target,state)
        time.sleep(5)
    progress=OUT/'PROGRESS.json'
    p=json.loads(progress.read_text()) if progress.exists() else {}
    if p.get('phase')!='measure complete':
        state.update(status='controller exited before completion; inspect receipts before resuming',finished_wall=time.time())
        atomic(target,state)
        return 1
    state.update(status='finalizing',heartbeat_wall=time.time());atomic(target,state)
    log=OUT/'report/finalization.log';log.parent.mkdir(exist_ok=True)
    with log.open('a') as f:
        result=subprocess.run(['taskset','-c','8,9',str(PY),str(Path(__file__).with_name('finalize_campaign.py'))],stdout=f,stderr=subprocess.STDOUT)
    state.update(status='reports generated; rendered review still required' if result.returncode==0 else 'finalization failed; inspect report/finalization.log',
                 report_exit_code=result.returncode,finished_wall=time.time())
    atomic(target,state)
    return result.returncode


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);a=p.parse_args();raise SystemExit(main(a.pid))
