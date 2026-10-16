"""Common host-only measurement adapter for H and S. No GPU work in event hooks.

Installed once after Scheduler I/O initialization. Token IDs already copied by the
engine are observed in send_result; buffers are written only on an explicit dump
command consumed before a new (sentinel) request. Never clear/reset prefix cache.
"""
from array import array
from collections import Counter
from pathlib import Path
import json
import os
import pickle
import time
from itertools import islice

import torch

CLOCK = time.perf_counter_ns


class Collector:
    def __init__(self, scheduler, config):
        self.s = scheduler
        self.directory = Path(os.environ['MINISGL_BENCH_DIR'])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mode = os.environ.get('MINISGL_BENCH_MODE', 'full')
        self.command_id = None
        self.active = False
        self.config = {k: str(v) for k, v in vars(config).items()}
        self.clear()

    def clear(self):
        self.tokens = array('q')  # uid, ordinal, token_id, host_emit_ns, finished
        self.ordinals = {}
        self.queued = []
        self.admissions = []
        self.batches = []
        self.prefills = []
        self.budgets = []
        self.aborts = []
        self.decisions = array('q')
        self.counts = Counter()
        self.dropped = 0
        self.prep_ns = 0
        self.batch_ordinal = 0
        self.started_ns = CLOCK()

    def command(self):
        # One small read per *request*, never per generated token.
        p = self.directory / 'control.json'
        if not p.exists():
            return
        obj = json.loads(p.read_text())
        if obj['id'] == self.command_id:
            return
        self.command_id = obj['id']
        if obj['action'] == 'begin':
            self.clear()
            torch.cuda.reset_peak_memory_stats(self.s.device)
            self.active = True
        elif obj['action'] == 'dump':
            self.active = False
            self.dump(obj['label'])

    def dump(self, label):
        s = self.s
        size = s.cache_manager.prefix_cache.size_info
        snapshot = {
            'clock': time.get_clock_info('perf_counter').__dict__,
            'pid': os.getpid(), 'config': self.config, 'mode': self.mode,
            'start_ns': self.started_ns, 'end_ns': CLOCK(),
            'tokens': self.tokens, 'queued': self.queued, 'admissions': self.admissions,
            'batches': self.batches, 'prefills': self.prefills, 'budgets': self.budgets,
            'aborts': self.aborts,
            'decision_prepare_ns': self.decisions, 'counts': dict(self.counts),
            'dropped_events': self.dropped,
            'graph_capture_list': s.engine.graph_runner.graph_bs_list,
            'memory': {'peak_allocated': torch.cuda.max_memory_allocated(s.device),
                       'peak_reserved': torch.cuda.max_memory_reserved(s.device),
                       'allocated': torch.cuda.memory_allocated(s.device),
                       'reserved': torch.cuda.memory_reserved(s.device),
                       'free_total': torch.cuda.mem_get_info(s.device)},
            'resources': {'free_tables': s.table_manager.available_size,
                          'table_capacity': s.table_manager._max_running_reqs,
                          'unique_free_tables': len(set(s.table_manager._free_slots)),
                          'free_pages': len(s.cache_manager.free_slots),
                          'unique_free_pages': len(torch.unique(s.cache_manager.free_slots)),
                          'num_pages': s.cache_manager.num_pages,
                          'evictable': size.evictable_size, 'protected': size.protected_size,
                          'pending': len(s.prefill_manager.pending_list),
                          'running': len(s.decode_manager.running_reqs)},
        }
        target = self.directory / (label + '.pkl')
        tmp = target.with_suffix('.tmp')
        tmp.write_bytes(pickle.dumps(snapshot, protocol=5))
        tmp.replace(target)

    def token_reply(self, replies):
        if not self.active or self.mode == 'off':
            return
        now = CLOCK()
        for msg in replies:
            idx = self.ordinals.get(msg.uid, 0)
            self.ordinals[msg.uid] = idx + 1
            if len(self.tokens) >= 5 * 4_000_000:
                self.dropped += 1
            else:
                self.tokens.extend((msg.uid, idx, msg.next_token, now, int(msg.finished)))


def install(s, config):
    if 'MINISGL_BENCH_DIR' not in os.environ:
        return
    from minisgl.message import UserMsg,AbortBackendMsg
    from minisgl.scheduler.prefill import PrefillAdder

    c = Collector(s, config)
    original_msg = s._process_one_msg

    def process(msg):
        if isinstance(msg, UserMsg):
            c.command()
        if isinstance(msg,AbortBackendMsg) and c.active:
            before=(s.table_manager.available_size,len(s.prefill_manager.pending_list),
                    len(s.decode_manager.running_reqs),s.cache_manager.prefix_cache.size_info.protected_size)
            t=CLOCK()
            result=original_msg(msg)
            after=(s.table_manager.available_size,len(s.prefill_manager.pending_list),
                   len(s.decode_manager.running_reqs),s.cache_manager.prefix_cache.size_info.protected_size)
            c.aborts.append((msg.uid,t,before,after))
            return result
        return original_msg(msg)
    s._process_one_msg = process
    if c.mode == 'off':
        return  # Retain only run-boundary control; no hot batch/token wrappers.

    original_send = s.send_result
    def send(replies):
        c.token_reply(replies)
        return original_send(replies)
    s.send_result = send

    original_add = s.prefill_manager.add_one_req
    def add(msg):
        if c.active and c.mode != 'off':
            c.queued.append((msg.uid, CLOCK(), len(msg.input_ids), msg.sampling_params.max_tokens,
                             msg.input_ids.tolist()))
        return original_add(msg)
    s.prefill_manager.add_one_req = add
    if c.mode == 'token':
        return  # Minimal reference: exact token timing + request input/queue data.

    original_allocate = PrefillAdder._try_allocate_one
    def allocate(adder, req):
        result = original_allocate(adder, req)
        if c.active and c.mode != 'off':
            if result is not None:
                c.admissions.append((req.uid, CLOCK(), result[0].cached_len,
                                     getattr(req, 'waiting_passes', 0),
                                     getattr(req, 'arrival_order', -1),
                                     getattr(s.prefill_manager, 'schedule_step', -1)))
            elif adder.table_manager.available_size == 0:
                c.counts['table_reject'] += 1
            else:
                c.counts['kv_reject'] += 1
        return result
    PrefillAdder._try_allocate_one = allocate

    if hasattr(s.prefill_manager, '_get_candidates'):
        original_candidates = s.prefill_manager._get_candidates
        def candidates(*a, **kw):
            result = original_candidates(*a, **kw)
            if c.active and c.mode != 'off':
                original=s.prefill_manager.pending_list
                chunks=0
                for r in original:
                    if r.chunked_req is None:break
                    chunks+=1
                new_count=len(original)-chunks
                reordered=(any(a is not b for a,b in zip(islice(original,chunks,None),islice(result,chunks,None)))
                           if s.prefill_manager.schedule_policy!='fcfs' and 2<=new_count<=s.prefill_manager.lpm_queue_threshold else False)
                c.counts['candidate_calls'] += 1
                c.counts['candidate_reorders'] += int(reordered)
                c.counts['queue_gt128'] += int(new_count > 128)
                c.counts['max_new_queue'] = max(c.counts['max_new_queue'], new_count)
            return result
        s.prefill_manager._get_candidates = candidates

    original_prefill = s.prefill_manager.schedule_next_batch
    def prefill(budget):
        n = len(s.decode_manager.running_reqs)
        queue = len(s.prefill_manager.pending_list)
        t0 = CLOCK()
        result = original_prefill(budget)
        if c.active and c.mode != 'off' and (queue or c.mode=='full'):
            c.prefills.append((t0, budget, n, queue, result is not None, CLOCK() - t0))
        return result
    s.prefill_manager.schedule_next_batch = prefill

    if hasattr(s, '_get_protected_prefill_budget'):
        original_budget = s._get_protected_prefill_budget
        def budget():
            result = original_budget()
            if c.active and c.mode != 'off':
                c.budgets.append((CLOCK(), len(s.decode_manager.running_reqs), result,
                                  s.protected_prefill_tokens, s.target_batch_tokens,
                                  s.min_chunk_tokens, s.adaptive_prefill_budget))
            return result
        s._get_protected_prefill_budget = budget

    original_prepare = s._prepare_batch
    def prepare(batch):
        start = CLOCK()
        result = original_prepare(batch)
        c.prep_ns += CLOCK() - start
        return result
    s._prepare_batch = prepare
    original_schedule = s._schedule_next_batch
    def schedule():
        start = CLOCK()
        c.prep_ns = 0
        result = original_schedule()
        elapsed = CLOCK() - start
        if c.active and c.mode != 'off' and (result is not None or s.prefill_manager.runnable):
            c.decisions.extend((elapsed - c.prep_ns, c.prep_ns))
        return result
    s._schedule_next_batch = schedule

    original_forward = s._forward
    def forward(inp):
        b = inp.batch
        if c.active and c.mode != 'off':
            c.batch_ordinal+=1
            size = s.cache_manager.prefix_cache.size_info
            c.batches.append((CLOCK(), b.phase, b.size, b.padded_size,
                              len(s.decode_manager.running_reqs),
                              len(s.prefill_manager.pending_list),
                              getattr(s, '_consecutive_decode_batches', 0),
                              sum(r.extend_len for r in b.reqs) if b.is_prefill else b.size,
                              len(s.cache_manager.free_slots), size.evictable_size,
                              size.protected_size,
                              bool(s.engine.graph_runner.can_use_cuda_graph(b)),
                              [(r.uid, r.cached_len, r.extend_len) for r in b.reqs]
                              if c.mode=='full' or b.is_prefill or c.batch_ordinal%32==0 else []))
        return original_forward(inp)
    s._forward = forward

    # Only CPU metadata and len() on tensors; never copy evicted IDs back from GPU.
    pc = s.cache_manager.prefix_cache
    for name in ('probe_prefix', 'refresh_probe', 'commit_match', 'match_prefix',
                 'touch_handle', 'lock_handle', 'evict'):
        if c.mode=='light' and name in ('probe_prefix','refresh_probe','lock_handle','touch_handle'):
            continue  # Detailed probe/LRU counts belong in diagnostic mode.
        if not hasattr(pc, name):
            continue
        original = getattr(pc, name)
        def counted(*a, _name=name, _fn=original, **kw):
            result = _fn(*a, **kw)
            if c.active and c.mode != 'off':
                c.counts[_name] += 1
                if _name == 'evict':
                    c.counts['evicted_tokens'] += len(result)
            return result
        setattr(pc, name, counted)

