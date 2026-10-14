from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import minisgl.core as core
from minisgl.core import Batch
from minisgl.distributed import DistributedInfo
from minisgl.message import AbortBackendMsg
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.scheduler import Scheduler
from minisgl.server.args import parse_args


def config(**kwargs):
    return SchedulerConfig("test-model", DistributedInfo(0, 1), torch.float32, **kwargs)


def fake_scheduler(n=64, count=16, pending=True, adaptive=True, **kwargs):
    s = object.__new__(Scheduler)
    s.batch_schedule_policy = "decode-first-bounded"
    s.prefill_budget = 8192
    s.protected_prefill_tokens = 256
    s.adaptive_prefill_budget = adaptive
    s.target_batch_tokens = 256
    s.min_chunk_tokens = 64
    s.max_consecutive_decodes = 16
    s._consecutive_decode_batches = count
    s.prefill_manager = SimpleNamespace(
        runnable=pending,
        schedule_next_batch=Mock(return_value=Batch([], "prefill") if pending else None),
    )
    s.decode_manager = SimpleNamespace(
        runnable=bool(n), running_reqs=list(range(n)),
        schedule_next_batch=Mock(return_value=Batch([], "decode") if n else None),
    )
    for key, value in kwargs.items():
        setattr(s, key, value)
    return s


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    path = Path(__file__).resolve().parents[2] / "benchmarks/bench_batch_schedule_policy.py"
    spec = importlib.util.spec_from_file_location("adaptive_prefill_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.cpu_transfers():
        yield module


@pytest.mark.parametrize("n,expected", [(16, 240), (64, 192), (192, 64), (256, 64), (1000, 64)])
def test_budget_uses_live_decode_count(n, expected):
    s = fake_scheduler(n=n)
    assert s._get_protected_prefill_budget() == expected
    assert s._select_next_batch().is_prefill
    s.prefill_manager.schedule_next_batch.assert_called_once_with(expected)
    assert s._consecutive_decode_batches == 16


@pytest.mark.parametrize("kwargs,expected", [
    ({"target_batch_tokens": 512}, 256),
    ({"prefill_budget": 128}, 128),
    ({"protected_prefill_tokens": 128}, 128),
    ({"target_batch_tokens": 128, "min_chunk_tokens": 128}, 128),
])
def test_budget_caps_and_floor(kwargs, expected):
    assert fake_scheduler(**kwargs)._get_protected_prefill_budget() == expected


def test_budget_is_monotone_and_uses_fresh_snapshot():
    s = fake_scheduler(n=1)
    budgets = []
    for n in range(1, 257):
        s.decode_manager.running_reqs = list(range(n))
        budgets.append(s._get_protected_prefill_budget())
    assert budgets == sorted(budgets, reverse=True)
    assert min(budgets) == 64 and max(budgets) == 255
    assert s._consecutive_decode_batches == 16


def test_large_target_saturates_to_fixed_cap():
    for n in (1, 64, 128, 256):
        assert fake_scheduler(n=n, target_batch_tokens=1024)._get_protected_prefill_budget() == 256


def test_decode_count_is_not_future_kv_reservation():
    s = fake_scheduler(n=64)
    s.decode_manager.inflight_tokens = 100000
    s.decode_manager.padded_reqs = list(range(128))
    assert s._get_protected_prefill_budget() == 192


@pytest.mark.parametrize("count", [0, 15])
def test_no_prefill_or_budget_computation_before_k(count):
    s = fake_scheduler(count=count)
    s._get_protected_prefill_budget = Mock(side_effect=AssertionError("premature budget"))
    assert s._select_next_batch().is_decode
    s.prefill_manager.schedule_next_batch.assert_not_called()


@pytest.mark.parametrize("count", [0, 16, 99])
def test_no_decode_restores_normal_budget(count):
    s = fake_scheduler(n=0, count=count)
    s._get_protected_prefill_budget = Mock(side_effect=AssertionError("unneeded budget"))
    assert s._select_next_batch().is_prefill
    s.prefill_manager.schedule_next_batch.assert_called_once_with(8192)


def test_no_pending_prefill_stays_decode_only():
    s = fake_scheduler(pending=False)
    s._get_protected_prefill_budget = Mock(side_effect=AssertionError("unneeded budget"))
    assert s._select_next_batch().is_decode
    assert s._consecutive_decode_batches == 0


def test_protection_failure_keeps_counter_and_recomputes_next_budget():
    s = fake_scheduler(n=64)
    s.prefill_manager.schedule_next_batch.return_value = None
    batch = s._select_next_batch()
    assert batch.is_decode
    s.prefill_manager.schedule_next_batch.assert_called_once_with(192)
    s._record_batch_submission(batch)
    assert s._consecutive_decode_batches == 17
    s.decode_manager.running_reqs = list(range(192))
    assert s._select_next_batch().is_decode
    assert s.prefill_manager.schedule_next_batch.call_args.args == (64,)


@pytest.mark.parametrize("n", [1, 64, 192])
def test_disabled_budget_preserves_old_upper_even_with_small_configuration(n):
    s = fake_scheduler(n=n, adaptive=False, prefill_budget=8, protected_prefill_tokens=2)
    assert s._get_protected_prefill_budget() == 2
    assert s._select_next_batch().is_prefill
    s.prefill_manager.schedule_next_batch.assert_called_once_with(2)


def test_defaults_and_disabled_cross_constraints():
    c = config()
    assert (c.adaptive_prefill_budget, c.target_batch_tokens, c.min_chunk_tokens) == (False, 256, 64)
    c = config(max_extend_tokens=0, protected_prefill_tokens=2)
    assert not c.adaptive_prefill_budget
    assert not config(target_batch_tokens=1, min_chunk_tokens=64).adaptive_prefill_budget


@pytest.mark.parametrize("field", ["target_batch_tokens", "min_chunk_tokens"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "64"])
def test_invalid_positive_config(field, value):
    with pytest.raises(ValueError, match=field):
        config(**{field: value})


@pytest.mark.parametrize("value", [1, "false", None])
def test_invalid_switch_type(value):
    with pytest.raises(ValueError, match="adaptive_prefill_budget"):
        config(adaptive_prefill_budget=value)


@pytest.mark.parametrize("kwargs,message", [
    ({"batch_schedule_policy": "prefill-first"}, "requires decode-first-bounded"),
    ({"max_extend_tokens": 0}, "max_extend_tokens"),
    ({"max_extend_tokens": -1}, "max_extend_tokens"),
    ({"max_extend_tokens": True}, "max_extend_tokens"),
    ({"max_extend_tokens": 32}, "max_extend_tokens or protected_prefill_tokens"),
    ({"protected_prefill_tokens": 32}, "max_extend_tokens or protected_prefill_tokens"),
    ({"target_batch_tokens": 32}, "target_batch_tokens"),
])
def test_enabled_cross_constraints(kwargs, message):
    values = dict(batch_schedule_policy="decode-first-bounded", adaptive_prefill_budget=True)
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        config(**values)


def test_cli_configuration_composes_with_lpm_aging():
    common = ["--model-path", "test-model", "--dtype", "float32"]
    default, _ = parse_args(common)
    assert not default.adaptive_prefill_budget
    enabled, _ = parse_args(common + [
        "--batch-schedule-policy", "decode-first-bounded", "--adaptive-prefill-budget",
        "--target-batch-tokens", "128", "--min-chunk-tokens", "32",
        "--schedule-policy", "lpm-aging", "--lpm-aging-tokens-per-pass", "8",
    ])
    assert enabled.adaptive_prefill_budget and enabled.target_batch_tokens == 128
    assert enabled.min_chunk_tokens == 32 and enabled.lpm_aging_tokens_per_pass == 8
    with pytest.raises(ValueError, match="requires decode-first-bounded"):
        parse_args(common + ["--adaptive-prefill-budget"])


@pytest.mark.parametrize("flag", ["--target-batch-tokens", "--min-chunk-tokens"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_invalid_cli_numbers(flag, value):
    with pytest.raises(SystemExit) as error:
        parse_args(["--model-path", "test-model", "--dtype", "float32", flag, value])
    assert error.value.code == 2


def advance(s):
    fi = s._schedule_next_batch()
    computed = sum(r.extend_len for r in fi.batch.reqs)
    s._process_last_data((fi, s._forward(fi)))
    return fi.batch, computed


def drain(h, s, slots, limit=500):
    for _ in range(limit):
        if not (s.prefill_manager.runnable or s.decode_manager.runnable):
            break
        h.submit_one(s)
    h.assert_resources(s, slots)


@pytest.mark.parametrize("page_size", [1, 4])
def test_budget_changes_continue_same_chunk_and_preserve_aging(harness, page_size):
    h = harness
    s = h.make_cpu_scheduler(
        cap=2, protected=7, budget=32, page_size=page_size,
        adaptive=True, target=8, min_chunk=2, prefill_policy="lpm-aging", aging=2,
    )
    for uid in range(4):
        h.enqueue(
            s, uid, torch.tensor([1000 + uid * 10, 1001 + uid * 10], dtype=torch.int32), 16
        )
    assert h.submit_one(s).is_prefill
    pending = h.enqueue(s, 10, torch.arange(100, 125, dtype=torch.int32), 3)
    other = h.enqueue(s, 11, torch.tensor([200, 201], dtype=torch.int32), 1)
    for _ in range(2):
        assert h.submit_one(s).is_decode
    batch, used = advance(s)
    assert batch.is_prefill and used == 4 and pending.chunked_req.cached_len == 4
    table, handle = pending.chunked_req.table_idx, pending.chunked_req.cache_handle
    assert other.waiting_passes == 1
    for uid in (0, 1):
        s._process_one_msg(AbortBackendMsg(uid))
    for _ in range(2):
        h.submit_one(s)
    _, used = advance(s)
    assert used == 6 and pending.chunked_req.cached_len == 10
    assert pending.chunked_req.table_idx == table and pending.chunked_req.cache_handle is handle
    s._process_one_msg(AbortBackendMsg(2))
    for _ in range(2):
        h.submit_one(s)
    _, used = advance(s)
    assert used == 7 and pending.chunked_req.cached_len == 17
    assert other.waiting_passes == 1
    s._process_one_msg(AbortBackendMsg(3))
    batch, used = advance(s)
    assert batch.is_prefill and used == 10  # Remaining prompt 8 + next request 2; normal budget.
    assert pending.chunked_req is None
    drain(h, s, 64)


@pytest.mark.parametrize("page_size", [1, 4])
def test_budget_is_read_only_and_cached_prefix_is_not_charged(harness, page_size, monkeypatch):
    h = harness
    s = h.make_cpu_scheduler(cap=1, protected=5, adaptive=True, target=3, min_chunk=1,
                             page_size=page_size)
    h.enqueue(s, 0, torch.tensor([1000, 1001], dtype=torch.int32), 8)
    h.submit_one(s)
    prefix = torch.arange(100, 108, dtype=torch.int32)
    indices = s.cache_manager._page_to_token(s.cache_manager._allocate(8 // page_size))
    s.cache_manager.prefix_cache.insert_prefix(prefix, indices)
    for uid in (1, 2, 3):
        h.enqueue(s, uid, torch.cat([prefix, torch.tensor([200 + uid], dtype=torch.int32)]), 1)
    h.submit_one(s)
    tree = s.cache_manager.prefix_cache
    leaf = next(iter(tree.root_node.children.values()))
    before = (leaf.timestamp, leaf.ref_count, tree._structure_version, tree.size_info,
              s.cache_manager.free_slots.tolist(), s.table_manager.available_size)
    with monkeypatch.context() as m:
        m.setattr(s.cache_manager, "probe_req", Mock(side_effect=AssertionError("budget probes")))
        assert s._get_protected_prefill_budget() == 2
    assert before == (leaf.timestamp, leaf.ref_count, tree._structure_version, tree.size_info,
                      s.cache_manager.free_slots.tolist(), s.table_manager.available_size)
    fi = s._schedule_next_batch()
    assert [r.uid for r in fi.batch.reqs] == [1, 2]
    assert [r.cached_len for r in fi.batch.reqs] == [8, 8]
    assert sum(r.extend_len for r in fi.batch.reqs) == 2
    s._process_last_data((fi, s._forward(fi)))
    drain(h, s, 64)


@pytest.mark.parametrize("failure", ["kv", "table"])
@pytest.mark.parametrize("page_size", [1, 4])
def test_small_budget_cannot_bypass_admission(harness, failure, page_size):
    h = harness
    slots = 1 if failure == "table" else 4
    s = h.make_cpu_scheduler(
        cap=1, protected=3, budget=8, adaptive=True, target=4, min_chunk=1,
        num_pages=12 // page_size if failure == "kv" else 64,
        table_slots=slots, page_size=page_size,
    )
    h.enqueue(s, 0, torch.tensor([1000, 1001], dtype=torch.int32), 8)
    h.submit_one(s)
    pending = h.enqueue(s, 1, torch.arange(100, 106, dtype=torch.int32), 2)
    h.submit_one(s)
    tree = s.cache_manager.prefix_cache
    before = (tree._structure_version, tree.size_info, s.table_manager.available_size)
    batch = s._select_next_batch()
    assert batch.is_decode and pending.chunked_req is None
    assert before == (tree._structure_version, tree.size_info, s.table_manager.available_size)
    assert s._consecutive_decode_batches == 1
    fi = s._prepare_batch(batch)
    s._process_last_data((fi, s._forward(fi)))
    assert s._consecutive_decode_batches == 2
    drain(h, s, slots)


@pytest.mark.parametrize("overlap", [False, True])
def test_adaptive_budget_preserves_loop_submission_order(harness, overlap):
    h = harness
    s = h.make_cpu_scheduler(cap=2, protected=7, adaptive=True, target=3, min_chunk=1)
    h.enqueue(s, 0, torch.tensor([1000, 1001], dtype=torch.int32), 16)
    h.submit_one(s)
    h.enqueue(s, 1, torch.arange(100, 108, dtype=torch.int32), 1)
    original = s._forward
    phases, prefill_tokens, counts = [], [], []

    def record(fi):
        phases.append(fi.batch.phase)
        if fi.batch.is_prefill:
            prefill_tokens.append(sum(r.extend_len for r in fi.batch.reqs))
        return original(fi)

    s._forward = record
    last = None
    for _ in range(4):
        if overlap:
            last = s.overlap_loop(last)
        else:
            s.normal_loop()
        counts.append(s._consecutive_decode_batches)
    assert phases == ["decode", "decode", "prefill", "decode"]
    assert prefill_tokens == [2] and counts == [1, 2, 0, 1]
    if overlap:
        s._process_last_data(last)
    drain(h, s, 64)
