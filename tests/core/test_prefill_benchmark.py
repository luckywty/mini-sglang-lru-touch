from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import minisgl.core as core


@pytest.mark.parametrize("page_size", [1, 4])
def test_benchmark_simulation_is_deterministic_and_preserves_resources(page_size, monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    path = Path(__file__).resolve().parents[2] / "benchmarks/bench_prefill_policy.py"
    spec = importlib.util.spec_from_file_location("prefill_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    args = SimpleNamespace(
        stream_prefix=8, stream_steps=8, arrivals_per_pass=1,
        page_size=page_size, aging_tokens_per_pass=2, budget=1,
    )
    with benchmark.cpu_transfers():
        rows = [benchmark.simulate(policy, "mixed", args) for policy in benchmark.POLICIES]
    assert [r["cold_first_admission_pass"] for r in rows] == [1, 9, 5]
    assert all(r["computed_prefill_tokens"] == 17 for r in rows)
    assert all(r["resource_integrity"] == "passed" for r in rows)
    assert len({r["cache_hit_ratio"] for r in rows}) == 1
