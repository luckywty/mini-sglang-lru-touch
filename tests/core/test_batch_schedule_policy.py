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
from minisgl.message import AbortBackendMsg, BatchBackendMsg, UserMsg
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.scheduler.scheduler import Scheduler
from minisgl.server.args import parse_args


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    path = Path(__file__).resolve().parents[2] / "benchmarks/bench_batch_schedule_policy.py"
    spec = importlib.util.spec_from_file_location("batch_schedule_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.cpu_transfers():
        yield module


def fake_scheduler(policy="decode-first-bounded", count=0, pending=True, running=True):
    s = object.__new__(Scheduler)
    s.batch_schedule_policy = policy
    s.max_consecutive_decodes = 3
    s.protected_prefill_tokens = 2
    s.prefill_budget = 8
    s.adaptive_prefill_budget = False
    s.target_batch_tokens = 256
    s.min_chunk_tokens = 64
    s._consecutive_decode_batches = count
    s.prefill_manager = SimpleNamespace(
        runnable=pending,
        schedule_next_batch=Mock(return_value=Batch([], "prefill") if pending else None),
    )
    s.decode_manager = SimpleNamespace(
        runnable=running,
        schedule_next_batch=Mock(return_value=Batch([], "decode") if running else None),
    )
    s._prepare_batch = Mock(side_effect=lambda batch: batch)
    return s


@pytest.mark.parametrize("policy", ["prefill-first", "decode-first-bounded"])
@pytest.mark.parametrize(
    "pending,running", [(False, False), (True, False), (False, True), (True, True)]
)
def test_phase_selection_and_default_call_order(policy, pending, running):
    s = fake_scheduler(policy, pending=pending, running=running)
    batch = s._schedule_next_batch()
    phase = (
        "prefill" if pending and (policy == "prefill-first" or not running)
        else "decode" if running else None
    )
    assert (batch.phase if batch else None) == phase
    if batch:
        s._prepare_batch.assert_called_once_with(batch)
    else:
        s._prepare_batch.assert_not_called()
    if phase == "prefill":
        s.prefill_manager.schedule_next_batch.assert_called_once_with(8)
        s.decode_manager.schedule_next_batch.assert_not_called()
    elif policy == "decode-first-bounded":
        s.prefill_manager.schedule_next_batch.assert_not_called()
    else:
        s.prefill_manager.schedule_next_batch.assert_called_once_with(8)


@pytest.mark.parametrize("count,expected", [(2, "decode"), (3, "prefill"), (4, "prefill")])
def test_cap_boundaries_do_not_count_selection(count, expected):
    s = fake_scheduler(count=count)
    assert s._schedule_next_batch().phase == expected
    assert s._consecutive_decode_batches == count
    if expected == "prefill":
        s.prefill_manager.schedule_next_batch.assert_called_once_with(2)


@pytest.mark.parametrize("normal,protected,expected", [(8, 2, 2), (2, 8, 2), (0, 2, 0)])
def test_protected_budget_is_capped_by_normal_budget(normal, protected, expected):
    s = fake_scheduler(count=3)
    s.prefill_budget, s.protected_prefill_tokens = normal, protected
    s._schedule_next_batch()
    s.prefill_manager.schedule_next_batch.assert_called_once_with(expected)


def test_failed_protection_keeps_counter_and_retries():
    s = fake_scheduler(count=3)
    s.prefill_manager.schedule_next_batch.return_value = None
    for expected in (4, 5):
        batch = s._schedule_next_batch()
        assert batch.is_decode
        s._record_batch_submission(batch)
        assert s._consecutive_decode_batches == expected
    assert s.prefill_manager.schedule_next_batch.call_count == 2
    assert s._prepare_batch.call_count == 2


def test_decode_unavailable_falls_back_without_preparing_a_discarded_batch():
    s = fake_scheduler()
    s.decode_manager.schedule_next_batch.return_value = None
    batch = s._schedule_next_batch()
    assert batch.is_prefill
    s.prefill_manager.schedule_next_batch.assert_called_once_with(8)
    s._prepare_batch.assert_called_once_with(batch)


@pytest.mark.parametrize("phase,pending,count,expected", [
    ("decode", True, 0, 1), ("decode", True, 3, 4),
    ("decode", False, 3, 0), ("prefill", True, 3, 0),
])
def test_submission_counter_rules(phase, pending, count, expected):
    s = fake_scheduler(count=count, pending=pending)
    s._record_batch_submission(Batch([object()] * 100, phase))
    assert s._consecutive_decode_batches == expected


def test_empty_waiting_queue_resets_historical_streak():
    s = fake_scheduler(count=99, pending=False)
    assert s._schedule_next_batch().is_decode
    assert s._consecutive_decode_batches == 0


def test_config_and_cli_defaults_and_independent_prefill_policy():
    config = SchedulerConfig("test-model", DistributedInfo(0, 1), torch.float32)
    assert (
        config.batch_schedule_policy, config.max_consecutive_decodes, config.protected_prefill_tokens
    ) == ("prefill-first", 16, 256)
    default, _ = parse_args(["--model-path", "test-model", "--dtype", "float32"])
    assert default.batch_schedule_policy == "prefill-first"
    configured, _ = parse_args([
        "--model-path", "test-model", "--dtype", "float32",
        "--batch-schedule-policy", "decode-first-bounded", "--max-consecutive-decodes", "4",
        "--protected-prefill-tokens", "128", "--schedule-policy", "lpm-aging",
        "--lpm-aging-tokens-per-pass", "8",
    ])
    assert (
        configured.batch_schedule_policy, configured.max_consecutive_decodes,
        configured.protected_prefill_tokens,
    ) == ("decode-first-bounded", 4, 128)
    assert configured.schedule_policy == "lpm-aging" and configured.lpm_aging_tokens_per_pass == 8


@pytest.mark.parametrize("field", ["max_consecutive_decodes", "protected_prefill_tokens"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_invalid_positive_integer_config(field, value):
    with pytest.raises(ValueError, match=field):
        SchedulerConfig("test-model", DistributedInfo(0, 1), torch.float32, **{field: value})


def test_unknown_batch_policy_is_rejected():
    with pytest.raises(ValueError, match="batch_schedule_policy"):
        SchedulerConfig(
            "test-model", DistributedInfo(0, 1), torch.float32, batch_schedule_policy="unknown"
        )


@pytest.mark.parametrize("flag,value", [
    ("--batch-schedule-policy", "unknown"), ("--max-consecutive-decodes", "0"),
    ("--max-consecutive-decodes", "-1"), ("--protected-prefill-tokens", "0"),
    ("--protected-prefill-tokens", "-1"),
])
def test_invalid_cli(flag, value):
    with pytest.raises(SystemExit) as error:
        parse_args(["--model-path", "test-model", "--dtype", "float32", flag, value])
    assert error.value.code == 2


def warm_decode(h, s, uid=0, output_tokens=16):
    h.enqueue(
        s, uid, torch.tensor([1000 + uid * 10, 1001 + uid * 10], dtype=torch.int32), output_tokens
    )
    assert h.submit_one(s).is_prefill


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("prefill_policy", ["fcfs", "lpm", "lpm-aging"])
def test_chunk_continuation_budget_and_release(harness, page_size, prefill_policy):
    h = harness
    s = h.make_cpu_scheduler(cap=2, protected=3, budget=32, page_size=page_size,
                             prefill_policy=prefill_policy, aging=2)
    warm_decode(h, s)
    pending = h.enqueue(s, 1, torch.arange(100, 108, dtype=torch.int32), 2)
    # Add a second candidate so LPM actually probes/sorts at admission opportunities.
    other = h.enqueue(s, 2, torch.arange(200, 202, dtype=torch.int32), 1)
    assert h.submit_one(s).is_decode
    assert h.submit_one(s).is_decode
    assert pending.waiting_passes == other.waiting_passes == 0
    batch = h.submit_one(s)
    assert batch.is_prefill and sum(r.cached_len for r in batch.reqs) == 3
    assert isinstance(pending.chunked_req, ChunkedReq)
    table_idx, handle = pending.chunked_req.table_idx, pending.chunked_req.cache_handle
    assert s._consecutive_decode_batches == 0
    assert other.waiting_passes == (1 if prefill_policy == "lpm-aging" else 0)
    for _ in range(2):
        assert h.submit_one(s).is_decode
    next_chunk = h.submit_one(s)
    assert next_chunk.is_prefill
    assert pending.chunked_req.table_idx == table_idx
    assert pending.chunked_req.cache_handle is handle
    assert pending.chunked_req.cached_len == 6
    assert other.waiting_passes == (1 if prefill_policy == "lpm-aging" else 0)
    for _ in range(100):
        if not (s.prefill_manager.runnable or s.decode_manager.runnable):
            break
        h.submit_one(s)
    h.assert_resources(s, 64)
    assert len([m for m in s.sent if m.finished]) == 3


@pytest.mark.parametrize("failure", ["kv", "table", "budget"])
def test_protection_failure_preserves_resources_then_recovers(harness, failure):
    h = harness
    s = h.make_cpu_scheduler(cap=1, protected=2, budget=8,
                             num_pages=12 if failure == "kv" else 64,
                             table_slots=1 if failure == "table" else 4)
    warm_decode(h, s, output_tokens=8)
    pending = h.enqueue(s, 1, torch.arange(100, 106, dtype=torch.int32), 2)
    assert h.submit_one(s).is_decode
    if failure == "budget":
        s.prefill_budget = 0
    tree = s.cache_manager.prefix_cache
    before = (tree._structure_version, tree.size_info, s.table_manager.available_size)
    batch = s._select_next_batch()
    assert batch.is_decode
    assert (tree._structure_version, tree.size_info, s.table_manager.available_size) == before
    assert pending.chunked_req is None and pending.waiting_passes == 0
    assert s._consecutive_decode_batches == 1
    forward_input = s._prepare_batch(batch)
    s._process_last_data((forward_input, s._forward(forward_input)))
    assert s._consecutive_decode_batches == 2
    s.prefill_budget = 8
    for _ in range(100):
        if not (s.prefill_manager.runnable or s.decode_manager.runnable):
            break
        h.submit_one(s)
    assert any(m.uid == 1 and m.finished for m in s.sent)
    h.assert_resources(s, 1 if failure == "table" else 4)


@pytest.mark.parametrize("chunked", [False, True])
def test_abort_and_new_arrival_do_not_inherit_streak(harness, chunked):
    h = harness
    s = h.make_cpu_scheduler(cap=1, protected=2)
    warm_decode(h, s)
    pending = h.enqueue(s, 1, torch.arange(100, 108, dtype=torch.int32), 2)
    h.submit_one(s)
    if chunked:
        h.submit_one(s)
        assert pending.chunked_req is not None
        h.submit_one(s)
    assert s._consecutive_decode_batches == 1
    # Abort and new arrival in one receive batch must observe the empty-queue boundary.
    new = UserMsg(2, torch.tensor([200, 201], dtype=torch.int32), core.SamplingParams(max_tokens=1))
    s._process_one_msg(BatchBackendMsg([AbortBackendMsg(1), new]))
    assert s._consecutive_decode_batches == 0
    assert [p.uid for p in s.prefill_manager.pending_list] == [2]
    for _ in range(100):
        if not (s.prefill_manager.runnable or s.decode_manager.runnable):
            break
        h.submit_one(s)
    h.assert_resources(s, 64)


def test_failed_forward_and_prepare_do_not_increment_counter(harness, monkeypatch):
    h = harness
    s = h.make_cpu_scheduler(cap=3)
    warm_decode(h, s)
    h.enqueue(s, 1, torch.tensor([10, 11], dtype=torch.int32), 1)
    forward_input = s._schedule_next_batch()
    assert s._consecutive_decode_batches == 0
    with monkeypatch.context() as m:
        m.setattr(s.engine, "forward_batch", Mock(side_effect=RuntimeError("submit failed")))
        with pytest.raises(RuntimeError, match="submit failed"):
            s._forward(forward_input)
    assert s._consecutive_decode_batches == 0
    with monkeypatch.context() as m:
        m.setattr(s, "_prepare_batch", Mock(side_effect=RuntimeError("prepare failed")))
        with pytest.raises(RuntimeError, match="prepare failed"):
            s._schedule_next_batch()
    assert s._consecutive_decode_batches == 0


def test_decode_priority_does_not_probe_prefill_before_protection(harness, monkeypatch):
    h = harness
    s = h.make_cpu_scheduler(cap=2, protected=2)
    warm_decode(h, s)
    pending = h.enqueue(s, 1, torch.arange(100, 108, dtype=torch.int32), 1)
    with monkeypatch.context() as m:
        m.setattr(s.cache_manager, "probe_req", Mock(side_effect=AssertionError("unneeded probe")))
        assert h.submit_one(s).is_decode
        assert h.submit_one(s).is_decode
    assert pending.waiting_passes == 0
    assert h.submit_one(s).is_prefill
    for _ in range(100):
        if not (s.prefill_manager.runnable or s.decode_manager.runnable):
            break
        h.submit_one(s)
    h.assert_resources(s, 64)


@pytest.mark.parametrize("page_size", [1, 4])
def test_protection_budget_is_batch_total_and_excludes_cached_prefix(harness, page_size):
    h = harness
    s = h.make_cpu_scheduler(cap=1, protected=2, page_size=page_size)
    warm_decode(h, s)
    prefix = torch.arange(100, 108, dtype=torch.int32)
    indices = s.cache_manager._page_to_token(s.cache_manager._allocate(8 // page_size))
    s.cache_manager.prefix_cache.insert_prefix(prefix, indices)
    for uid in (1, 2, 3):
        h.enqueue(s, uid, torch.cat([prefix, torch.tensor([200 + uid], dtype=torch.int32)]), 1)
    assert h.submit_one(s).is_decode
    batch = s._select_next_batch()
    assert batch.is_prefill
    assert [r.uid for r in batch.reqs] == [1, 2]
    assert [r.cache_handle.cached_len for r in batch.reqs] == [8, 8]
    assert sum(r.extend_len for r in batch.reqs) == 2
    assert [p.uid for p in s.prefill_manager.pending_list] == [3]
    forward_input = s._prepare_batch(batch)
    s._process_last_data((forward_input, s._forward(forward_input)))
    for _ in range(100):
        if not (s.prefill_manager.runnable or s.decode_manager.runnable):
            break
        h.submit_one(s)
    h.assert_resources(s, 64)


@pytest.mark.parametrize("overlap", [False, True])
def test_actual_loops_count_submissions_before_previous_results(harness, overlap):
    h = harness
    s = h.make_cpu_scheduler(cap=2, protected=2)
    warm_decode(h, s)
    h.enqueue(s, 1, torch.arange(100, 108, dtype=torch.int32), 1)
    phases = []
    original_forward = s._forward

    def track_forward(forward_input):
        phases.append(forward_input.batch.phase)
        return original_forward(forward_input)

    s._forward = track_forward
    last = None
    counts = []
    for _ in range(4):
        if overlap:
            last = s.overlap_loop(last)
        else:
            s.normal_loop()
        counts.append(s._consecutive_decode_batches)
    assert phases == ["decode", "decode", "prefill", "decode"]
    assert counts == [1, 2, 0, 1]
    if overlap:
        s._process_last_data(last)
    for _ in range(100):
        if not (s.prefill_manager.runnable or s.decode_manager.runnable):
            break
        h.submit_one(s)
    h.assert_resources(s, 64)


@pytest.mark.parametrize("prefill_policy", ["fcfs", "lpm-aging"])
def test_default_selection_matches_original_expression_over_lifecycle(harness, prefill_policy):
    h = harness

    def run(original):
        s = h.make_cpu_scheduler(
            policy="prefill-first", budget=3, prefill_policy=prefill_policy, aging=2
        )
        for uid, length in [(90, 8), (10, 2), (50, 5)]:
            h.enqueue(s, uid, torch.arange(uid * 100, uid * 100 + length, dtype=torch.int32), 3)
        trace = []
        for _ in range(100):
            if original:
                batch = (s.prefill_manager.schedule_next_batch(s.prefill_budget)
                         or s.decode_manager.schedule_next_batch())
            else:
                batch = s._select_next_batch()
            if batch is None:
                break
            trace.append((
                batch.phase, [(r.uid, r.cached_len, r.device_len, r.table_idx) for r in batch.reqs]
            ))
            forward_input = s._prepare_batch(batch)
            s._process_last_data((forward_input, s._forward(forward_input)))
        h.assert_resources(s, 64)
        return trace, [(m.uid, m.next_token, m.finished) for m in s.sent]

    assert run(True) == run(False)


def test_large_queue_and_equal_rank_inputs_have_deterministic_phase_choices(harness):
    h = harness
    traces = []
    for _ in range(2):
        s = h.make_cpu_scheduler(cap=2, protected=2, table_slots=140,
                                 prefill_policy="lpm-aging", aging=2)
        warm_decode(h, s)
        for uid in range(1, 130):
            h.enqueue(s, uid, torch.tensor([uid * 100, uid * 100 + 1], dtype=torch.int32), 1)
        trace = []
        for _ in range(3):
            batch = h.submit_one(s)
            trace.append((batch.phase, [r.uid for r in batch.reqs]))
        assert trace == [("decode", [0]), ("decode", [0]), ("prefill", [1])]
        traces.append(trace)
        for _ in range(1000):
            if not (s.prefill_manager.runnable or s.decode_manager.runnable):
                break
            h.submit_one(s)
        h.assert_resources(s, 140)
    assert traces[0] == traces[1]
