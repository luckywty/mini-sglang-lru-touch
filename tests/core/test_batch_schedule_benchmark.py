from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import minisgl.core as core


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize(
    "workload", ["mixed", "memory-pressure", "table-pressure", "prefill-only", "decode-only"]
)
def test_benchmark_is_reproducible_and_drains(workload, page_size, monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    path = Path(__file__).resolve().parents[2] / "benchmarks/bench_batch_schedule_policy.py"
    spec = importlib.util.spec_from_file_location("batch_schedule_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    args = SimpleNamespace(
        seed=31723, stream_steps=4, arrivals_per_pass=1, initial_decodes=2,
        decode_tokens=16, output_tokens=2, prefix_length=8, long_prompt_tokens=17,
        prefill_budget=32, page_size=page_size, prefill_policy="fcfs", aging_tokens_per_pass=2,
    )
    with benchmark.cpu_transfers():
        first = benchmark.simulate("decode-first-bounded", workload, args, cap=2, protected=4)
        second = benchmark.simulate("decode-first-bounded", workload, args, cap=2, protected=4)
        unbounded = benchmark.simulate("decode-first-unbounded", workload, args)
    deterministic = [k for k in first if not k.startswith("scheduling_")]
    assert all(first[k] == second[k] for k in deterministic)
    assert first["resource_integrity"] == unbounded["resource_integrity"] == "passed"
    if workload == "mixed":
        assert first["prefill_first_wait_p95_batches"] < unbounded["prefill_first_wait_p95_batches"]
        assert first["max_decode_streak_while_prefill_waits"] <= 2
    if workload == "table-pressure":
        assert first["rejected_prefill_attempts"] > 0
        assert first["max_decode_streak_while_prefill_waits"] > 2
    if workload == "memory-pressure":
        assert first["rejected_prefill_attempts"] > 0
        assert first["max_decode_streak_while_prefill_waits"] > 2


def test_decode_gap_includes_initial_prefill_burst(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    path = Path(__file__).resolve().parents[2] / "benchmarks/bench_batch_schedule_policy.py"
    spec = importlib.util.spec_from_file_location("batch_schedule_gap_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    args = SimpleNamespace(
        seed=31723, stream_steps=4, arrivals_per_pass=1, initial_decodes=2,
        decode_tokens=16, output_tokens=2, prefix_length=8, long_prompt_tokens=17,
        prefill_budget=32, page_size=1, prefill_policy="fcfs", aging_tokens_per_pass=2,
    )
    with benchmark.cpu_transfers():
        prefill = benchmark.simulate("prefill-first", "short", args)
        bounded = benchmark.simulate("decode-first-bounded", "short", args, cap=2, protected=4)
    assert prefill["initial_decode_output_gap_max_batches"] == 5
    assert bounded["initial_decode_output_gap_max_batches"] == 2
