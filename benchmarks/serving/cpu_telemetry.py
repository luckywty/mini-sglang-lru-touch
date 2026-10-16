"""Low-frequency CPU telemetry in an independent process, outside client GIL."""
import argparse
import json
from pathlib import Path
import signal
import time

import psutil

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();running=True;rows=[]
    def stop(*args):
        global running
        running=False
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    while running:
        item=dict(wall_time=time.time(),mono_ns=time.perf_counter_ns())
        try:
            parent=psutil.Process(a.pid);procs=[parent]+parent.children(recursive=True)
            item['processes']=[dict(pid=x.pid,cpu=x.cpu_times().user+x.cpu_times().system,
                                   rss=x.memory_info().rss,ctx=x.num_ctx_switches()._asdict()) for x in procs]
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        rows.append(item)
        time.sleep(1)
    Path(a.output).write_text(json.dumps(rows))
