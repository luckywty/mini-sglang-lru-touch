"""CPU batch scheduling benchmark with real admission, KV pages and request completion.

No model/attention/sampling/GPU runs. Batch gaps are not TTFT or GPU latency.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import torch

import minisgl.core as core
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import UserMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.table import TableManager


WORKLOADS = (
    "short", "long", "mixed", "memory-pressure", "table-pressure", "prefill-only", "decode-only"
)


@contextmanager
def cpu_transfers():
    empty = torch.empty
    tensor = torch.tensor

    def unpinned_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    def unpinned_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return tensor(*args, **kwargs)

    with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor), patch.object(
        torch, "empty", unpinned_empty
    ), patch.object(torch, "tensor", unpinned_tensor), patch(
        "minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1)
    ):
        yield


class CPUForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: object


def make_cpu_scheduler(
    policy="decode-first-bounded", cap=16, protected=256, budget=8192,
    page_size=1, num_pages=4096, table_slots=64, max_seq_len=4096,
    prefill_policy="fcfs", aging=0,
    adaptive=False, target=256, min_chunk=64,
):
    """Skip only GPU/model construction; execute actual scheduler lifecycle methods."""
    config = SchedulerConfig(
        "cpu-benchmark", DistributedInfo(0, 1), torch.float32,
        batch_schedule_policy=policy, max_extend_tokens=budget,
        max_consecutive_decodes=cap, protected_prefill_tokens=protected,
        adaptive_prefill_budget=adaptive, target_batch_tokens=target, min_chunk_tokens=min_chunk,
        schedule_policy=prefill_policy, lpm_aging_tokens_per_pass=aging,
    )
    core._GLOBAL_CTX = core.Context(page_size=page_size)
    s = object.__new__(Scheduler)
    page_table = torch.full((table_slots, max_seq_len), -1, dtype=torch.int32)
    s.table_manager = TableManager(table_slots, page_table)
    s.cache_manager = CacheManager(num_pages, page_size, page_table, type="radix")
    s.decode_manager = DecodeManager(page_size)
    s.prefill_manager = PrefillManager(
        s.cache_manager, s.table_manager, s.decode_manager,
        schedule_policy=prefill_policy, lpm_aging_tokens_per_pass=aging,
    )
    s.batch_schedule_policy = policy
    s.max_consecutive_decodes = cap
    s.protected_prefill_tokens = protected
    s.prefill_budget = budget
    s.adaptive_prefill_budget = config.adaptive_prefill_budget
    s.target_batch_tokens = config.target_batch_tokens
    s.min_chunk_tokens = config.min_chunk_tokens
    s._consecutive_decode_batches = 0
    s.token_pool = s.table_manager.token_pool
    s.device = torch.device("cpu")
    s.finished_reqs = set()
    s.eos_token_id = -1
    s.stream = None
    s.engine_stream_ctx = nullcontext()
    s.sent = []
    s.send_result = s.sent.extend
    s.receive_msg = lambda blocking: []

    def forward(batch, sample_args):
        tokens = torch.tensor([100000 + r.uid for r in batch.reqs], dtype=torch.int32)
        for req in batch.reqs:
            req.complete_one()
        return CPUForwardOutput(tokens, tokens, SimpleNamespace(synchronize=lambda: None))

    s.engine = SimpleNamespace(
        max_seq_len=max_seq_len,
        page_table=page_table,
        graph_runner=SimpleNamespace(
            pad_batch=lambda batch: setattr(batch, "padded_reqs", batch.reqs)
        ),
        attn_backend=SimpleNamespace(prepare_metadata=lambda batch: None),
        sampler=SimpleNamespace(prepare=lambda batch: None),
        forward_batch=forward,
        stream=SimpleNamespace(wait_stream=lambda stream: None),
    )
    return s


def enqueue(s, uid, ids, output_tokens=1):
    s._process_one_msg(UserMsg(uid, ids, SamplingParams(max_tokens=output_tokens, ignore_eos=True)))
    return s.prefill_manager.pending_list[-1]


def submit_one(s):
    forward_input = s._schedule_next_batch()
    if forward_input is None:
        return None
    output = s._forward(forward_input)
    s._process_last_data((forward_input, output))
    return forward_input.batch


def assert_resources(s, table_slots):
    assert not s.prefill_manager.runnable and not s.decode_manager.runnable
    assert s.table_manager.available_size == table_slots
    assert len(set(s.table_manager._free_slots)) == table_slots
    cache = s.cache_manager
    assert cache.prefix_cache.size_info.protected_size == 0
    cache.check_integrity()
    pages = list(cache.free_slots.tolist())
    stack = [cache.prefix_cache.root_node]
    while stack:
        node = stack.pop()
        assert node.ref_count == (1 if node.is_root() else 0)
        if not node.is_root():
            pages.extend(node.value[::cache.page_size].tolist())
        stack.extend(node.children.values())
    assert len(pages) == len(set(pages)) == cache.num_pages


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def simulate(
    policy, workload, args, cap=16, protected=256,
    adaptive=False, target=256, min_chunk=64,
):
    rng = random.Random(args.seed)
    arrivals_count = 0 if workload == "decode-only" else args.stream_steps * args.arrivals_per_pass
    initial_decodes = 0 if workload == "prefill-only" else args.initial_decodes
    slots = (
        max(1, initial_decodes) if workload == "table-pressure"
        else initial_decodes + arrivals_count + 1
    )
    if workload == "memory-pressure":
        # Each request is feasible alone, but the initial decode reservations can
        # temporarily prevent even the first cold request from being admitted.
        capacity = max(
            initial_decodes * (8 + args.decode_tokens + args.page_size),
            args.long_prompt_tokens + args.output_tokens + args.prefix_length,
        ) + 2 * args.page_size
    else:
        capacity = initial_decodes * (8 + args.decode_tokens + args.page_size) + args.prefix_length
        capacity += (
            arrivals_count * (args.long_prompt_tokens + args.output_tokens + args.page_size) + 32
        )
    s = make_cpu_scheduler(
        policy="prefill-first" if policy == "decode-first-unbounded" else policy,
        cap=cap, protected=protected, budget=args.prefill_budget, page_size=args.page_size,
        num_pages=math.ceil(capacity / args.page_size), table_slots=slots,
        max_seq_len=max(args.long_prompt_tokens + args.output_tokens, args.decode_tokens + 8,
                        args.prefix_length + 1 + args.output_tokens) + 16,
        prefill_policy=args.prefill_policy, aging=args.aging_tokens_per_pass,
        adaptive=adaptive, target=target, min_chunk=min_chunk,
    )
    for uid in range(initial_decodes):
        output_tokens = args.decode_tokens
        if getattr(args, "stagger_decode_completions", False) and output_tokens > 1:
            base = max(2, output_tokens // 2)
            output_tokens = base + (uid * 17) % (output_tokens - base + 1)
        enqueue(
            s, uid, torch.arange(1000 + uid * 100, 1008 + uid * 100, dtype=torch.int32),
            output_tokens,
        )
    if initial_decodes:
        # Bootstrap all decode sessions identically, outside timed/observed workload.
        batch = s.prefill_manager.schedule_next_batch(initial_decodes * 8)
        forward_input = s._prepare_batch(batch)
        s._process_last_data((forward_input, s._forward(forward_input)))
    bootstrap_finished = {m.uid for m in s.sent if m.finished}
    s.sent.clear()
    prefix = torch.arange(50000, 50000 + args.prefix_length, dtype=torch.int32)
    indices = s.cache_manager._page_to_token(
        s.cache_manager._allocate(args.prefix_length // args.page_size)
    )
    s.cache_manager.prefix_cache.insert_prefix(prefix, indices)
    s.cache_manager.prefix_cache.probe_prefix(prefix)  # Warm native matching before timing.
    if policy == "decode-first-unbounded":
        # Benchmark-only baseline; not a third production policy.
        def unbounded(self):
            return (self.decode_manager.schedule_next_batch()
                    or self.prefill_manager.schedule_next_batch(self.prefill_budget))
        s._select_next_batch = MethodType(unbounded, s)

    counters = {"prefill_attempts": 0, "rejected_prefill_attempts": 0,
                "protection_attempts": 0, "successful_protection_batches": 0}
    original_prefill = s.prefill_manager.schedule_next_batch
    budget_samples = []

    def tracked_prefill(budget):
        pending = s.prefill_manager.runnable
        protection = (policy == "decode-first-bounded" and pending and s.decode_manager.runnable
                      and s._consecutive_decode_batches >= cap)
        counters["prefill_attempts"] += int(pending)
        counters["protection_attempts"] += int(protection)
        batch = original_prefill(budget)
        counters["rejected_prefill_attempts"] += int(pending and batch is None)
        counters["successful_protection_batches"] += int(protection and batch is not None)
        if protection:
            actual = sum(r.extend_len for r in batch.reqs) if batch is not None else 0
            assert actual <= budget
            budget_samples.append({
                "decode_tokens": len(s.decode_manager.running_reqs),
                "budget": budget, "computed_tokens": actual, "accepted": batch is not None,
            })
        return batch

    s.prefill_manager.schedule_next_batch = tracked_prefill
    arrivals, first, prefill_done = {}, {}, {}
    completion = {uid: 0 for uid in bootstrap_finished}
    # Include the bootstrap output at batch 0, so prefill-first's initial
    # consecutive prefill burst is visible in the decode output gap metric.
    output_steps = {uid: [0] for uid in range(initial_decodes)}
    cached, prompt_lengths, durations, phases = {}, {}, [], []
    computed = 0
    streak = max_streak = 0
    max_queue = 0
    max_active_decode = len(s.decode_manager.running_reqs)
    chunk_batches = 0
    # Finite workload: every feasible request must finish; detect stalled/infeasible setup.
    bound = (args.stream_steps + args.decode_tokens + arrivals_count *
             (args.long_prompt_tokens + args.output_tokens)) * (cap + 2) + 32
    for step in range(1, bound + 1):
        if step <= args.stream_steps and workload != "decode-only":
            for offset in range(args.arrivals_per_pass):
                uid = initial_decodes + (step - 1) * args.arrivals_per_pass + offset
                cold = workload in ("long", "memory-pressure") or (
                    workload in ("mixed", "prefill-only") and rng.random() < 0.25
                )
                if cold:
                    start = 1000000 + uid * (args.long_prompt_tokens + 128)
                    ids = torch.arange(start, start + args.long_prompt_tokens, dtype=torch.int32)
                else:
                    ids = torch.cat([prefix, torch.tensor([2000000 + uid], dtype=torch.int32)])
                enqueue(s, uid, ids, 1 if workload == "prefill-only" else args.output_tokens)
                arrivals[uid], prompt_lengths[uid] = step, len(ids)
        had_prefill = s.prefill_manager.runnable
        max_active_decode = max(max_active_decode, len(s.decode_manager.running_reqs))
        max_queue = max(max_queue, len(s.prefill_manager.pending_list))
        start_time = time.perf_counter_ns()
        batch = s._select_next_batch()
        durations.append((time.perf_counter_ns() - start_time) / 1000)
        if batch is None:
            if s.prefill_manager.runnable or s.decode_manager.runnable:
                raise RuntimeError("Feasible workload stalled without a batch")
            if step >= args.stream_steps or workload == "decode-only":
                break
            continue
        phases.append(batch.phase)
        if batch.is_prefill:
            streak = 0
            chunk_batches += int(any(isinstance(r, ChunkedReq) for r in batch.reqs))
            computed += sum(r.extend_len for r in batch.reqs)
            for req in batch.reqs:
                if req.uid not in first:
                    first[req.uid], cached[req.uid] = step, req.cache_handle.cached_len
                if not isinstance(req, ChunkedReq):
                    prefill_done[req.uid] = step
        elif had_prefill:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
        forward_input = s._prepare_batch(batch)
        output = s._forward(forward_input)
        before_messages = len(s.sent)
        s._process_last_data((forward_input, output))
        for msg in s.sent[before_messages:]:
            output_steps.setdefault(msg.uid, []).append(step)
            if msg.finished:
                completion[msg.uid] = step
        if (step >= args.stream_steps or workload == "decode-only") and not (
            s.prefill_manager.runnable or s.decode_manager.runnable
        ):
            break
    else:
        raise RuntimeError("Workload did not drain within finite batch bound")
    assert set(first) == set(prefill_done) == set(arrivals)
    assert len(completion) == arrivals_count + initial_decodes
    assert_resources(s, slots)
    waits = [first[u] - arrivals[u] for u in arrivals]
    done_waits = [prefill_done[u] - arrivals[u] for u in arrivals]
    decode_gaps = [b - a for uid, times in output_steps.items() if uid < initial_decodes
                   for a, b in zip(times, times[1:])]
    return {
        "policy": policy, "workload": workload,
        "cap": cap if policy == "decode-first-bounded" else None,
        "protected_tokens": protected if policy == "decode-first-bounded" else None,
        "prefill_policy": args.prefill_policy, "requests": arrivals_count + initial_decodes,
        "adaptive_prefill_budget": adaptive, "target_batch_tokens": target,
        "min_chunk_tokens": min_chunk, "initial_decode_load": args.initial_decodes,
        "max_active_decode_reqs": max_active_decode, "chunk_batches": chunk_batches,
        "protection_budget_samples": budget_samples,
        "protection_budget_min": min((r["budget"] for r in budget_samples), default=None),
        "protection_budget_max": max((r["budget"] for r in budget_samples), default=None),
        "protection_budget_utilization": (
            sum(r["computed_tokens"] for r in budget_samples) /
            sum(r["budget"] for r in budget_samples) if budget_samples else None
        ),
        "prefill_first_wait_p95_batches": percentile(waits, 0.95),
        "prefill_first_wait_max_batches": max(waits, default=0),
        "prefill_completion_wait_p95_batches": percentile(done_waits, 0.95),
        "initial_decode_output_gap_p95_batches": percentile(decode_gaps, 0.95),
        "initial_decode_output_gap_max_batches": max(decode_gaps, default=0),
        "max_decode_streak_while_prefill_waits": max_streak,
        "prefill_batches": phases.count("prefill"), "decode_batches": phases.count("decode"),
        "computed_prefill_tokens": computed, "cached_tokens_at_admission": sum(cached.values()),
        "cache_hit_ratio": (
            sum(cached.values()) / sum(prompt_lengths.values()) if prompt_lengths else None
        ),
        "drain_batch": max(completion.values(), default=0), "max_queue": max_queue,
        "scheduling_median_us": statistics.median(durations),
        "scheduling_p95_us": percentile(durations, 0.95),
        "resource_integrity": "passed", **counters,
    }


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=31723)
    parser.add_argument("--repeats", type=positive, default=5)
    parser.add_argument("--decode-caps", type=positive, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument(
        "--protected-budgets", type=positive, nargs="+", default=[64, 128, 256, 512]
    )
    parser.add_argument("--workloads", choices=WORKLOADS, nargs="+", default=list(WORKLOADS))
    parser.add_argument("--prefill-policy", choices=["fcfs", "lpm", "lpm-aging"], default="fcfs")
    parser.add_argument("--aging-tokens-per-pass", type=int, default=8)
    parser.add_argument("--prefill-budget", type=positive, default=8192)
    parser.add_argument("--page-size", type=positive, default=1)
    parser.add_argument("--prefix-length", type=positive, default=64)
    parser.add_argument("--long-prompt-tokens", type=positive, default=1024)
    parser.add_argument("--decode-tokens", type=positive, default=128)
    parser.add_argument(
        "--stagger-decode-completions", action="store_true",
        help="Deterministically vary initial decode lengths to exercise live budget changes.",
    )
    parser.add_argument("--output-tokens", type=positive, default=8)
    parser.add_argument("--initial-decodes", type=positive, default=4)
    parser.add_argument("--stream-steps", type=positive, default=16)
    parser.add_argument("--arrivals-per-pass", type=positive, default=2)
    parser.add_argument("--compare-adaptive", action="store_true",
                        help="Compare fixed/adaptive bounded decode-first at each decode load.")
    parser.add_argument("--decode-loads", type=positive, nargs="+", default=[16, 64, 128, 192])
    parser.add_argument("--target-budgets", type=positive, nargs="+", default=[128, 256, 512])
    parser.add_argument("--min-chunks", type=positive, nargs="+", default=[32, 64, 128])
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/batch_scheduling/benchmark.json")
    )
    args = parser.parse_args()
    if args.prefix_length % args.page_size:
        parser.error("prefix-length must be page aligned")
    if args.aging_tokens_per_pass < 0:
        parser.error("aging-tokens-per-pass must be nonnegative")
    torch.set_num_threads(1)
    metadata = {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": next(
            (line.split(":", 1)[1].strip()
             for line in Path("/proc/cpuinfo").read_text().splitlines()
             if line.startswith("model name")), platform.processor()
        ),
        "logical_cpus": os.cpu_count(), "torch_threads": 1,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": "CPU real scheduler/allocator with simulated engine; no GPU TTFT/ITL/throughput",
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    cases = [(workload, policy, 16, 256) for workload in args.workloads
             for policy in ("prefill-first", "decode-first-unbounded")]
    cases += [(workload, "decode-first-bounded", cap, protected) for workload in args.workloads
              for cap in args.decode_caps for protected in args.protected_budgets]
    cases = [(*case, False, 256, 64, args.initial_decodes) for case in cases]
    if args.compare_adaptive:
        cases = [
            (workload, "decode-first-bounded", cap, protected, False, 256, 64, load)
            for workload in args.workloads for cap in args.decode_caps
            for protected in args.protected_budgets for load in args.decode_loads
        ]
        cases += [
            (workload, "decode-first-bounded", cap, protected, True, target, floor, load)
            for workload in args.workloads for cap in args.decode_caps
            for protected in args.protected_budgets for load in args.decode_loads
            for target in args.target_budgets for floor in args.min_chunks
            if floor <= min(target, args.prefill_budget, protected)
        ]
    runs = []
    rng = random.Random(args.seed)
    with cpu_transfers():
        for repeat in range(args.repeats):
            rng.shuffle(cases)
            for workload, policy, cap, protected, adaptive, target, floor, load in cases:
                case_args = SimpleNamespace(**{**vars(args), "initial_decodes": load})
                row = simulate(policy, workload, case_args, cap, protected, adaptive, target, floor)
                row["repeat"] = repeat
                runs.append(row)
            print(f"repeat={repeat + 1}: {len(cases)} cases drained, resources passed", flush=True)
    summary = []
    for workload, policy, cap, protected, adaptive, target, floor, load in sorted(cases):
        group = [
            r for r in runs if r["workload"] == workload and r["policy"] == policy and
            (policy != "decode-first-bounded" or
             (r["cap"] == cap and r["protected_tokens"] == protected)) and
            r["adaptive_prefill_budget"] == adaptive and r["target_batch_tokens"] == target and
            r["min_chunk_tokens"] == floor and r["initial_decode_load"] == load
        ]
        row = {k: v for k, v in group[0].items() if k != "repeat"}
        deterministic = [k for k in row if not k.startswith("scheduling_")]
        assert all(all(r[k] == row[k] for k in deterministic) for r in group)
        for key in ("scheduling_median_us", "scheduling_p95_us"):
            row[key] = statistics.median(r[key] for r in group)
        row["min_run_median_us"] = min(r["scheduling_median_us"] for r in group)
        row["max_run_median_us"] = max(r["scheduling_median_us"] for r in group)
        summary.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "summary": summary, "runs": runs}, indent=2) + "\n"
    )
    lines = [
        "# Batch 调度 CPU Benchmark", "",
        f"环境：{metadata['cpu']}；Python {metadata['python']}；"
        f"Torch {metadata['torch']}；1 个 Torch 线程。", "",
        "真实阶段选择、准入、KV 分配及完成释放；Engine 仅模拟进度和输出。"
        "Batch 间隔不是 GPU 延迟。", "",
        "| 负载 | 策略 | 初始 Decode | 自适应 | 目标/下限 | K | 保护预算 | 首次 Prefill 等待 P95（批） | "
        "Prefill 完成等待 P95（批） | 初始 Decode 输出最大间隔（批） | "
        "等待时最长连续 Decode | 调度中位数 µs | 保护成功/尝试 |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['workload']} | {r['policy']} | {r['initial_decode_load']} | "
            f"{r['adaptive_prefill_budget']} | {r['target_batch_tokens']}/{r['min_chunk_tokens']} | "
            f"{r['cap']} | {r['protected_tokens']} | "
            f"{r['prefill_first_wait_p95_batches']} | {r['prefill_completion_wait_p95_batches']} | "
            f"{r['initial_decode_output_gap_max_batches']} | "
            f"{r['max_decode_streak_while_prefill_waits']} | "
            f"{r['scheduling_median_us']:.2f} | "
            f"{r['successful_protection_batches']}/{r['protection_attempts']} |"
        )
    lines += ["", f"{len(runs)} 次运行排空并通过表槽唯一性、缓存引用、物理页守恒和唯一性检查。",
              "请求按固定 Batch 序号到达，各策略的到达序列一致；吞吐、TTFT、ITL 需要另做真实 GPU 实验。", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
