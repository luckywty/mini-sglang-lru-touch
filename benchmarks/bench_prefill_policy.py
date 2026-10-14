"""CPU scheduler benchmark: real radix matching/admission, model-free KV lifecycle.

Run with PYTHONPATH=python. No model, attention, sampling, or CUDA graph runs.
Scheduling microseconds and simulated admission passes are NOT GPU TTFT/throughput.
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
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

import minisgl.core as core
from minisgl.core import SamplingParams
from minisgl.message import UserMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
from minisgl.scheduler.table import TableManager


POLICIES = ("fcfs", "lpm", "lpm-aging")


@contextmanager
def cpu_transfers():
    """Replace only pinning in this benchmark; retain actual tensor copies/allocator."""
    empty = torch.empty

    def unpinned_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor), patch.object(
        torch, "empty", unpinned_empty
    ):
        yield


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


def make_manager(policy, prefix_len, page_size, aging, capacity, table_slots=4):
    core._GLOBAL_CTX = core.Context(page_size=page_size)
    page_table = torch.full((table_slots, prefix_len + 16), -1, dtype=torch.int32)
    cache = CacheManager(capacity, page_size, page_table, type="radix")
    manager = PrefillManager(
        cache, TableManager(table_slots, page_table), DecodeManager(page_size),
        schedule_policy=policy, lpm_aging_tokens_per_pass=aging,
        fairness_max_wait_steps=0, lpm_queue_threshold=128,
    )
    hot_prefix = torch.arange(prefix_len, dtype=torch.int32)
    indices = cache._page_to_token(cache._allocate(prefix_len // page_size))
    cache.prefix_cache.insert_prefix(hot_prefix, indices)
    return manager, hot_prefix


def hot_tokens(prefix, uid):
    return torch.cat([prefix, torch.tensor([2_000_000 + uid], dtype=torch.int32)])


def cold_tokens(prefix_len, uid):
    start = 1_000_000 + uid * (prefix_len + 1024)
    return torch.arange(start, start + prefix_len + 1, dtype=torch.int32)


def enqueue(manager, uid, ids):
    manager.add_one_req(UserMsg(uid, ids, SamplingParams(max_tokens=1)))
    return manager.pending_list[-1]


def microbenchmark(policy, prefix_len, queue_len, page_size, trials, seed):
    aging = max(1, prefix_len // 8)
    manager, prefix = make_manager(
        policy, prefix_len, page_size, aging, prefix_len // page_size + 64
    )
    rng = random.Random(seed)
    for uid in range(queue_len):
        ids = hot_tokens(prefix, uid) if rng.random() < 0.75 else cold_tokens(prefix_len, uid)
        req = enqueue(manager, uid, ids)
        req.waiting_passes = rng.randrange(16)
    # Warm native compare/module loading before measurements, including FCFS runs.
    warm = enqueue(manager, -1, hot_tokens(prefix, 0))
    manager.cache_manager.probe_req(warm)
    manager.abort_req(-1)
    for _ in range(10):
        manager._get_candidates({})
    durations = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        manager._get_candidates({})
        durations.append((time.perf_counter_ns() - start) / 1000)
    manager.cache_manager.check_integrity()
    return {
        "policy": policy, "prefix_tokens": prefix_len, "queue_len": queue_len,
        "page_size": page_size, "aging_tokens_per_pass": aging, "trials": trials,
        "median_us": statistics.median(durations), "p95_us": percentile(durations, 0.95),
        "sorting_probes_per_call": queue_len if policy != "fcfs" and 1 < queue_len <= 128 else 0,
    }


def simulate(policy, workload, args):
    prefix_len = args.stream_prefix
    requests = args.stream_steps * args.arrivals_per_pass + (workload == "mixed")
    capacity = math.ceil((prefix_len + 1) * (requests + 2) / args.page_size) + 16
    manager, prefix = make_manager(
        policy, prefix_len, args.page_size, args.aging_tokens_per_pass, capacity
    )
    warm = enqueue(manager, -1, hot_tokens(prefix, 0))
    manager.cache_manager.probe_req(warm)
    manager.abort_req(-1)
    arrivals, first_admission, completion = {}, {}, {}
    kinds, lengths, matched_at_admission = {}, {}, {}
    next_uid = 0
    if workload == "mixed":
        enqueue(manager, 0, cold_tokens(prefix_len, 0))
        arrivals[0], kinds[0], lengths[0] = 0, "cold", prefix_len + 1
        next_uid = 1
    durations, computed_tokens, completed_in_window, max_queue = [], 0, 0, 0
    # Every request is finite and all output is one token; protect against a stuck head.
    max_passes = args.stream_steps + math.ceil(requests * (prefix_len + 1) / args.budget) + requests
    for step in range(1, max_passes + 1):
        if step <= args.stream_steps:
            for _ in range(args.arrivals_per_pass):
                uid = next_uid
                next_uid += 1
                kind = "cold" if workload == "cold-only" else "hot"
                ids = cold_tokens(prefix_len, uid) if kind == "cold" else hot_tokens(prefix, uid)
                enqueue(manager, uid, ids)
                arrivals[uid], kinds[uid], lengths[uid] = step, kind, len(ids)
        max_queue = max(max_queue, len(manager.pending_list))
        start = time.perf_counter_ns()
        batch = manager.schedule_next_batch(args.budget)
        durations.append((time.perf_counter_ns() - start) / 1000)
        if batch is None:
            if manager.pending_list:
                raise RuntimeError("No progress in the model-free workload; check admission constraints")
            if step >= args.stream_steps:
                break
            continue
        # Physical page allocation and completion run outside the measured scheduling interval.
        manager.cache_manager.allocate_paged(batch.reqs)
        for req in batch.reqs:
            if req.uid not in first_admission:
                first_admission[req.uid] = step
                matched_at_admission[req.uid] = req.cache_handle.cached_len
            computed_tokens += req.extend_len
            req.cached_len = req.device_len
            if not isinstance(req, ChunkedReq):
                manager.cache_manager.cache_req(req, finished=True)
                manager.table_manager.free(req.table_idx)
                completion[req.uid] = step
                completed_in_window += step <= args.stream_steps
        if step >= args.stream_steps and not manager.pending_list:
            break
    else:
        raise RuntimeError("Workload did not drain within the finite-pass bound")
    assert len(completion) == len(arrivals) == requests
    assert manager.table_manager.available_size == 4
    assert manager.cache_manager.prefix_cache.size_info.protected_size == 0
    manager.cache_manager.check_integrity()
    hot_wait = [first_admission[u] - arrivals[u] for u in arrivals if kinds[u] == "hot"]
    cold_wait = [first_admission[u] - arrivals[u] for u in arrivals if kinds[u] == "cold"]
    return {
        "policy": policy, "workload": workload, "requests": requests,
        "drain_pass": max(completion.values()), "max_queue": max_queue,
        "cold_first_admission_pass": first_admission.get(0) if workload == "mixed" else None,
        "cold_completion_pass": completion.get(0) if workload == "mixed" else None,
        "hot_wait_p95_passes": percentile(hot_wait, 0.95),
        "cold_wait_p95_passes": percentile(cold_wait, 0.95),
        "completed_in_arrival_window": completed_in_window,
        "cached_tokens_at_admission": sum(matched_at_admission.values()),
        "prompt_tokens": sum(lengths.values()), "computed_prefill_tokens": computed_tokens,
        "cache_hit_ratio": sum(matched_at_admission.values()) / sum(lengths.values()),
        "scheduling_mean_us": statistics.mean(durations),
        "scheduling_median_us": statistics.median(durations),
        "scheduling_p95_us": percentile(durations, 0.95),
        "resource_integrity": "passed",
    }


def positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=31723)
    parser.add_argument("--trials", type=positive, default=100)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--prefix-lengths", type=positive, nargs="+", default=[64, 1024])
    parser.add_argument("--queue-lengths", type=positive, nargs="+", default=[8, 32, 128, 129, 256])
    parser.add_argument("--page-size", type=positive, default=1)
    parser.add_argument("--stream-prefix", type=positive, default=64)
    parser.add_argument("--stream-steps", type=positive, default=32)
    parser.add_argument("--arrivals-per-pass", type=positive, default=4)
    parser.add_argument("--budget", type=positive, default=4)
    parser.add_argument("--aging-tokens-per-pass", type=positive, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lpm_aging/benchmark.json"))
    args = parser.parse_args()
    if any(n % args.page_size for n in args.prefix_lengths + [args.stream_prefix]):
        parser.error("prefix lengths must be page aligned")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    metadata = {
        "platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
        "python": platform.python_version(), "torch": torch.__version__, "torch_threads": 1,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scope": "CPU scheduler and model-free allocator simulation; no GPU inference",
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    micro, runs = [], []
    with cpu_transfers():
        for prefix_len in args.prefix_lengths:
            for queue_len in args.queue_lengths:
                for policy in POLICIES:
                    micro.append(microbenchmark(
                        policy, prefix_len, queue_len, args.page_size, args.trials, args.seed
                    ))
            print(f"Measured candidate ordering: prefix={prefix_len}", flush=True)
        rng = random.Random(args.seed)
        cases = [(workload, policy) for workload in ("hot-only", "mixed", "cold-only")
                 for policy in POLICIES]
        for repeat in range(args.repeats):
            rng.shuffle(cases)  # Avoid measuring every policy in a fixed thermal/load order.
            for workload, policy in cases:
                row = simulate(policy, workload, args)
                row["repeat"] = repeat
                runs.append(row)
            print(f"Simulated all workloads/policies: repeat={repeat + 1}, resources passed", flush=True)
    summary = []
    for workload in ("hot-only", "mixed", "cold-only"):
        for policy in POLICIES:
            group = [r for r in runs if r["policy"] == policy and r["workload"] == workload]
            row = {k: v for k, v in group[0].items() if k != "repeat"}
            for key in ("scheduling_mean_us", "scheduling_median_us", "scheduling_p95_us"):
                row[key] = statistics.median(r[key] for r in group)
            row["min_run_mean_us"] = min(r["scheduling_mean_us"] for r in group)
            row["max_run_mean_us"] = max(r["scheduling_mean_us"] for r in group)
            row["repeats"] = args.repeats
            summary.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "metadata": metadata, "ordering_microbenchmark": micro,
        "simulation_summary": summary, "simulation_runs": runs,
    }, indent=2) + "\n")
    lines = [
        "# CPU 调度 Benchmark", "",
        f"环境：{metadata['cpu']}；Python {metadata['python']}；Torch {metadata['torch']}；1 个 Torch 线程。",
        "",
        "真实 RadixCache/PrefillManager/页分配器；CPU 拷贝不 pin；无模型、Attention、采样或 GPU 推理。",
        "调度耗时不包含页分配和模拟完成；轮数不是秒数或 GPU TTFT。固定种子，原生内核预热后测量。",
        "",
        "## 候选排序（中位数 / p95，微秒）", "",
        "| 前缀 token | 队列 | 策略 | 中位数 | p95 | 排序探测次数 |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in micro:
        lines.append(f"| {row['prefix_tokens']} | {row['queue_len']} | {row['policy']} | "
                     f"{row['median_us']:.2f} | {row['p95_us']:.2f} | {row['sorting_probes_per_call']} |")
    lines += [
        "", "## 有限请求流与真实缓存生命周期模拟", "",
        "| 负载 | 策略 | 冷请求首次准入轮 | 冷请求完成轮 | hot 等待 p95（轮） | "
        "总 Prefill token | 缓存命中率 | 调度均值（µs） | 各次运行均值范围（µs） |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(f"| {row['workload']} | {row['policy']} | {row['cold_first_admission_pass']} | "
                     f"{row['cold_completion_pass']} | {row['hot_wait_p95_passes']} | "
                     f"{row['computed_prefill_tokens']} | {100 * row['cache_hit_ratio']:.2f}% | "
                     f"{row['scheduling_mean_us']:.2f} | "
                     f"{row['min_run_mean_us']:.2f}–{row['max_run_mean_us']:.2f} |")
    lines += [
        "", f"{len(runs)} 个模拟运行均检查排空、表槽归还、protected_size=0 和缓存页总量守恒。",
        "所有有限请求流排空后才比较总计算量。混合负载中 Aging 提前服务冷请求，"
        "代价是 hot 请求可能等待 chunk 续算；不能据此声称总体吞吐提升。",
        "CPU 时间受宿主负载和频率变化影响，原始重复运行在 JSON 中；不报告 GPU 性能收益。", "",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
