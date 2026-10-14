from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import minisgl.core as core


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("workload", ["long", "mixed", "memory-pressure", "table-pressure",
                                     "prefill-only", "decode-only"])
def test_fixed_adaptive_comparison_drains_and_repeats(page_size, workload, monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    path = Path(__file__).resolve().parents[2] / "benchmarks/bench_batch_schedule_policy.py"
    spec = importlib.util.spec_from_file_location("adaptive_prefill_benchmark", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    args = SimpleNamespace(
        seed=31723, stream_steps=4, arrivals_per_pass=1, initial_decodes=4,
        decode_tokens=16, output_tokens=2, prefix_length=8, long_prompt_tokens=17,
        prefill_budget=32, page_size=page_size, prefill_policy="lpm-aging", aging_tokens_per_pass=2,
    )
    with bench.cpu_transfers():
        fixed = bench.simulate("decode-first-bounded", workload, args, cap=2, protected=8)
        first = bench.simulate("decode-first-bounded", workload, args, cap=2, protected=8,
                               adaptive=True, target=8, min_chunk=2)
        repeat = bench.simulate("decode-first-bounded", workload, args, cap=2, protected=8,
                                adaptive=True, target=8, min_chunk=2)
    assert all(first[k] == repeat[k] for k in first if not k.startswith("scheduling_"))
    assert fixed["computed_prefill_tokens"] == first["computed_prefill_tokens"]
    assert first["resource_integrity"] == fixed["resource_integrity"] == "passed"
    samples = first["protection_budget_samples"]
    assert all(2 <= r["budget"] <= 8 and r["computed_tokens"] <= r["budget"] for r in samples)
    if workload in ("prefill-only", "decode-only"):
        assert not samples
        assert first["prefill_batches"] == fixed["prefill_batches"]
        assert first["decode_batches"] == fixed["decode_batches"]
    if workload == "long":
        assert samples and samples[0]["budget"] == 4


@pytest.mark.parametrize("page_size", [1, 4])
def test_natural_decode_completions_change_protection_budget(page_size, monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    path = Path(__file__).resolve().parents[2] / "benchmarks/bench_batch_schedule_policy.py"
    spec = importlib.util.spec_from_file_location("adaptive_completion_benchmark", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    args = SimpleNamespace(
        seed=31723, stream_steps=4, arrivals_per_pass=1, initial_decodes=4,
        decode_tokens=24, output_tokens=2, prefix_length=8, long_prompt_tokens=17,
        prefill_budget=32, page_size=page_size, prefill_policy="fcfs", aging_tokens_per_pass=2,
        stagger_decode_completions=True,
    )
    with bench.cpu_transfers():
        result = bench.simulate("decode-first-bounded", "long", args, cap=2, protected=8,
                                adaptive=True, target=8, min_chunk=2)
    samples = result["protection_budget_samples"]
    assert len({r["decode_tokens"] for r in samples}) > 1
    assert len({r["budget"] for r in samples}) > 1
    assert samples[0]["budget"] == 4 and max(r["budget"] for r in samples) > 4
    assert result["resource_integrity"] == "passed"
