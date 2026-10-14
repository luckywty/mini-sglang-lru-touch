from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

import minisgl.core as core
from minisgl.core import Batch, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.kvcache.radix_cache import RadixTreeNode
from minisgl.message import UserMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import ChunkedReq, PrefillAdder, PrefillManager
from minisgl.scheduler.table import TableManager
from minisgl.server.args import parse_args


@pytest.fixture(autouse=True)
def cpu_context(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)


def make_manager(policy="fcfs", page_size=1, cache_type="radix", **kwargs):
    if core._GLOBAL_CTX is None:
        core.set_global_ctx(core.Context(page_size=page_size))
    page_table = torch.full((4, 64), -1, dtype=torch.int32)
    cache = CacheManager(64, page_size, page_table, type=cache_type)
    return PrefillManager(
        cache, TableManager(4, page_table), DecodeManager(page_size), schedule_policy=policy, **kwargs
    )


def cache_prefix(manager, input_ids, length):
    cache = manager.cache_manager
    assert length % cache.page_size == 0
    indices = cache._page_to_token(cache._allocate(length // cache.page_size))
    result = cache.prefix_cache.insert_prefix(input_ids[:length], indices)
    cache._free(indices[:result.cached_len])
    return result.handle


def enqueue(manager, uid, cached=0, output_len=1, input_ids=None):
    if input_ids is None:
        length = max(8, 3 * manager.cache_manager.page_size + 1)
        input_ids = torch.arange(uid * 1000, uid * 1000 + length, dtype=torch.int32)
    if cached:
        cache_prefix(manager, input_ids, cached)
    manager.add_one_req(UserMsg(uid, input_ids, SamplingParams(max_tokens=output_len)))
    return manager.pending_list[-1]


def uids(reqs):
    return [req.uid for req in reqs]


def finish_batch(manager, batch):
    manager.cache_manager.allocate_paged(batch.reqs)
    for req in batch.reqs:
        req.cached_len = req.device_len  # model-free completion of computed KV positions
        if not isinstance(req, ChunkedReq):
            manager.cache_manager.cache_req(req, finished=True)
            manager.table_manager.free(req.table_idx)


def snapshot(manager):
    cache = manager.cache_manager.prefix_cache

    def visit(node):
        return (
            id(node), node.timestamp, node.ref_count,
            None if node.is_root() else node._key.tolist(),
            None if node.is_root() else node.value.tolist(),
            [(key, visit(child)) for key, child in node.children.items()],
        )

    return (
        visit(cache.root_node), cache.size_info, cache._structure_version,
        RadixTreeNode.counter, manager.cache_manager.free_slots.tolist(),
    )


@pytest.mark.parametrize("policy", ["fcfs", "lpm", "lpm-aging"])
def test_policy_sorting_and_zero_aging(policy):
    manager = make_manager(policy)
    enqueue(manager, 90)
    enqueue(manager, 10, 6)
    expected = [90, 10] if policy == "fcfs" else [10, 90]
    assert uids(manager._get_candidates()) == expected
    assert uids(manager.pending_list) == [90, 10]
    assert [req.waiting_passes for req in manager.pending_list] == [0, 0]


def test_aging_composes_with_match_and_does_not_change_real_length():
    manager = make_manager("lpm-aging", lpm_aging_tokens_per_pass=2)
    older = enqueue(manager, 90, 4)
    enqueue(manager, 10, 6)
    older.waiting_passes = 2  # effective score 8, greater than 6
    assert uids(manager._get_candidates()) == [90, 10]
    assert older.cached_len_est == 4
    batch = manager.schedule_next_batch(4)
    assert uids(batch.reqs) == [90]
    assert batch.reqs[0].cached_len == 4  # never the virtual score of 8
    assert older.waiting_passes == 2
    finish_batch(manager, batch)
    manager.cache_manager.check_integrity()


@pytest.mark.parametrize("policy,aging", [("lpm", 100), ("lpm-aging", 0)])
def test_disabled_aging_does_not_bias_order_or_write_counters(policy, aging):
    manager = make_manager(policy, lpm_aging_tokens_per_pass=aging)
    cold = enqueue(manager, 90)
    hot = enqueue(manager, 10, 6)
    cold.waiting_passes = 100
    assert uids(manager._get_candidates()) == [10, 90]
    manager.schedule_next_batch(2)
    assert (cold.waiting_passes, hot.waiting_passes) == (100, 0)


@pytest.mark.parametrize("policy", ["lpm", "lpm-aging"])
def test_stable_ties_follow_arrival_order(policy):
    manager = make_manager(policy, lpm_aging_tokens_per_pass=2)
    first = enqueue(manager, 90, 4)
    second = enqueue(manager, 10, 6)
    if policy == "lpm-aging":
        first.waiting_passes = 1  # 4 + 2 == 6
    else:
        cache_prefix(manager, first.input_ids, 6)
    assert first.arrival_order < second.arrival_order
    assert uids(manager._get_candidates()) == [90, 10]


def test_reordered_admission_removes_only_selected_requests():
    manager = make_manager("lpm")
    enqueue(manager, 10)
    enqueue(manager, 20, 2)
    enqueue(manager, 30, 6)
    assert uids(manager.schedule_next_batch(2).reqs) == [30]
    assert uids(manager.pending_list) == [10, 20]
    enqueue(manager, 40, 7)
    assert uids(manager.schedule_next_batch(1).reqs) == [40]
    assert uids(manager.pending_list) == [10, 20]


@pytest.mark.parametrize("policy", ["lpm", "lpm-aging"])
@pytest.mark.parametrize("length", [128, 129])
def test_queue_fallback_boundary_and_no_sorting_probes(policy, length, monkeypatch):
    manager = make_manager(policy, lpm_aging_tokens_per_pass=2)
    for uid in range(length):
        enqueue(manager, uid)
    spy = Mock(wraps=manager.cache_manager.probe_req)
    monkeypatch.setattr(manager.cache_manager, "probe_req", spy)
    assert uids(manager._get_candidates()) == list(range(length))
    assert spy.call_count == (length if length == 128 else 0)


def test_fcfs_fallback_uses_original_order_after_lpm_reordering():
    manager = make_manager("lpm", lpm_queue_threshold=2)
    enqueue(manager, 90)
    enqueue(manager, 10, 6)
    assert uids(manager._get_candidates()) == [10, 90]
    enqueue(manager, 20, 7)
    assert uids(manager._get_candidates()) == [90, 10, 20]


def test_cold_request_overtakes_continuous_fresh_hot_arrivals():
    manager = make_manager("lpm-aging", lpm_aging_tokens_per_pass=2)
    cold = enqueue(manager, 90)
    for waiting in range(3):
        hot = enqueue(manager, waiting + 1, 6)
        assert cold.waiting_passes == waiting
        batch = manager.schedule_next_batch(2)
        assert uids(batch.reqs) == [hot.uid]
        assert cold.waiting_passes == waiting + 1
        assert hot.waiting_passes == 0
        finish_batch(manager, batch)
    enqueue(manager, 10, 6)
    assert uids(manager._get_candidates())[0] == cold.uid  # score 6, arrival tie-break
    batch = manager.schedule_next_batch(2)
    assert batch.reqs[0].uid == cold.uid
    assert isinstance(batch.reqs[0], ChunkedReq)
    assert batch.reqs[0].cached_len == 0
    assert cold.waiting_passes == 3


def test_counter_is_updated_once_after_an_effective_pass():
    manager = make_manager("lpm-aging", lpm_aging_tokens_per_pass=2)
    cold = enqueue(manager, 90)
    enqueue(manager, 10, 6)
    manager._get_candidates()
    manager._get_candidates()
    assert cold.waiting_passes == 0
    batch = manager.schedule_next_batch(2)
    assert cold.waiting_passes == 1
    manager._get_candidates()
    assert cold.waiting_passes == 1
    finish_batch(manager, batch)


def test_multiple_new_admissions_still_count_as_one_waiting_pass():
    manager = make_manager("lpm-aging", lpm_aging_tokens_per_pass=2)
    cold = enqueue(manager, 90)
    enqueue(manager, 10, 6)
    enqueue(manager, 20, 6)
    batch = manager.schedule_next_batch(4)
    assert uids(batch.reqs) == [10, 20]
    assert cold.waiting_passes == 1


def test_no_table_slots_skips_sorting_probes(monkeypatch):
    manager = make_manager("lpm-aging", lpm_aging_tokens_per_pass=2)
    enqueue(manager, 90)
    enqueue(manager, 10, 6)
    for _ in range(4):
        manager.table_manager.allocate()
    spy = Mock(wraps=manager.cache_manager.probe_req)
    monkeypatch.setattr(manager.cache_manager, "probe_req", spy)
    assert manager.schedule_next_batch(8) is None
    spy.assert_not_called()


@pytest.mark.parametrize("blocked", ["empty", "zero_budget", "table", "kv"])
def test_no_opportunity_does_not_age(blocked):
    manager = make_manager("lpm-aging", lpm_aging_tokens_per_pass=2)
    if blocked == "empty":
        assert manager.schedule_next_batch(8) is None
        return
    cold = enqueue(manager, 90, output_len=1000 if blocked == "kv" else 1)
    enqueue(manager, 10)
    if blocked == "table":
        for _ in range(4):
            manager.table_manager.allocate()
    for _ in range(4):
        assert manager.schedule_next_batch(0 if blocked == "zero_budget" else 8) is None
        assert cold.waiting_passes == 0


def test_sorting_overtake_does_not_guarantee_resource_admission():
    manager = make_manager("lpm-aging", lpm_aging_tokens_per_pass=2)
    cold = enqueue(manager, 90, output_len=1000)
    enqueue(manager, 10, 6)
    cold.waiting_passes = 100
    assert manager._get_candidates()[0] is cold
    assert manager.schedule_next_batch(8) is None  # required KV exceeds capacity
    assert cold.waiting_passes == 100
    assert uids(manager.pending_list) == [90, 10]


def test_chunk_only_pass_does_not_age_and_preserves_resources():
    manager = make_manager("lpm-aging", lpm_aging_tokens_per_pass=2)
    chunk_pending = enqueue(manager, 90)
    batch = manager.schedule_next_batch(2)
    finish_batch(manager, batch)
    arrival = (chunk_pending.arrival_order, chunk_pending.arrival_step)
    waiting = enqueue(manager, 10, 6)
    available_tables = manager.table_manager.available_size
    for _ in range(2):
        batch = manager.schedule_next_batch(2)
        assert uids(batch.reqs) == [90]
        assert waiting.waiting_passes == 0
        assert chunk_pending.waiting_passes == 0
        assert manager.table_manager.available_size == available_tables
        finish_batch(manager, batch)
    assert (chunk_pending.arrival_order, chunk_pending.arrival_step) == arrival
    batch = manager.schedule_next_batch(4)
    assert uids(batch.reqs) == [90, 10]
    finish_batch(manager, batch)
    assert not manager.pending_list
    assert manager.table_manager.available_size == 4
    assert manager.cache_manager.prefix_cache.size_info.protected_size == 0
    manager.cache_manager.check_integrity()


def test_abort_removes_aging_state_and_frees_chunk_via_existing_lifecycle():
    manager = make_manager("lpm-aging", lpm_aging_tokens_per_pass=2)
    old = enqueue(manager, 90)
    old.waiting_passes = 5
    assert manager.abort_req(90) is None
    fresh = enqueue(manager, 90)
    assert fresh.waiting_passes == 0 and fresh.arrival_order > old.arrival_order
    batch = manager.schedule_next_batch(2)
    finish_batch(manager, batch)
    chunk = manager.abort_req(90)
    assert isinstance(chunk, ChunkedReq)
    manager.cache_manager.cache_req(chunk, finished=True)
    manager.table_manager.free(chunk.table_idx)
    assert not manager.pending_list
    assert manager.table_manager.available_size == 4
    manager.cache_manager.check_integrity()


@pytest.mark.parametrize("policy", ["fcfs", "lpm", "lpm-aging"])
def test_single_request_and_naive_cache(policy):
    manager = make_manager(policy, cache_type="naive", lpm_aging_tokens_per_pass=2)
    req = enqueue(manager, 90)
    assert uids(manager.schedule_next_batch(8).reqs) == [90]
    assert req.waiting_passes == 0 and not manager.pending_list


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("kind", ["empty", "miss", "partial", "divergent", "full", "duplicate"])
def test_read_only_probe_matches_real_prefix_without_mutations(page_size, kind):
    manager = make_manager(page_size=page_size)
    ids = torch.arange(3 * page_size, dtype=torch.int32)
    cache_prefix(manager, ids, len(ids))
    if kind == "empty":
        query = ids[:0]
    elif kind == "miss":
        query = torch.tensor([999], dtype=torch.int32)
    elif kind == "partial":
        query = ids[:page_size + 1]
    elif kind == "divergent":
        query = torch.cat([ids[:2 * page_size], torch.tensor([999], dtype=torch.int32)])
    else:
        query = ids
    before = snapshot(manager)
    cache = manager.cache_manager.prefix_cache
    for _ in range(2):
        probe = cache.probe_prefix(query)
        assert cache.estimate_match_len(query.tolist()) == probe.cached_len
        assert snapshot(manager) == before
    assert cache.match_prefix(query).cuda_handle.cached_len == probe.cached_len


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("change", ["evict", "insert", "split"])
def test_stale_sorting_probe_is_revalidated_at_admission(page_size, change):
    manager = make_manager(page_size=page_size)
    pending = enqueue(manager, 90)
    cache = manager.cache_manager
    if change != "insert":
        cache_prefix(manager, pending.input_ids, 3 * page_size)
    pending.input_ids = pending.input_ids[:page_size + 1]
    probe = cache.probe_req(pending)
    if change == "evict":
        cache._free(cache.prefix_cache.evict(3 * page_size))
        expected = 0
    elif change == "insert":
        cache_prefix(manager, pending.input_ids, page_size)
        expected = page_size
    else:
        cache.prefix_cache.match_prefix(torch.arange(90000, 90000 + 2 * page_size, dtype=torch.int32))
        expected = page_size
    handle = cache.try_match_and_lock(pending, 0, probe=probe)
    assert handle.cached_len == expected
    cache.unlock(handle)
    cache.check_integrity()


@pytest.mark.parametrize("page_size", [1, 4])
def test_reference_count_changes_refresh_cost_without_comparing_tokens(page_size, monkeypatch):
    manager = make_manager(page_size=page_size)
    pending = enqueue(manager, 90, cached=2 * page_size)
    cache = manager.cache_manager
    probe = cache.probe_req(pending)
    other = cache.prefix_cache.commit_match(probe)
    cache.lock(other)
    pending.sampling_params.max_tokens = cache.available_size - (pending.input_len - probe.cached_len)

    def no_compare(*args):
        pytest.fail("Unchanged structure must reuse the token match")

    monkeypatch.setattr(RadixTreeNode, "get_match_len", no_compare)
    handle = cache.try_match_and_lock(pending, 0, probe=probe)
    assert handle is not None
    cache.unlock(handle)
    cache.unlock(other)
    cache.check_integrity()


def test_sorting_probes_are_reused_for_shared_prefix_admission(monkeypatch):
    manager = make_manager("lpm")
    ids = torch.arange(8, dtype=torch.int32)
    cache_prefix(manager, ids, 6)
    for uid in range(3):
        enqueue(manager, uid, input_ids=ids)
    original = RadixTreeNode.get_match_len
    calls = []

    def compare(node, query):
        calls.append(node.uuid)
        return original(node, query)

    monkeypatch.setattr(RadixTreeNode, "get_match_len", compare)
    batch = manager.schedule_next_batch(6)
    assert len(batch.reqs) == 3 and len(calls) == 3
    assert [req.cached_len for req in batch.reqs] == [6, 6, 6]


def legacy_fcfs(manager, budget):
    """Reference queue/slicing algorithm before policy ordering was introduced."""
    adder = PrefillAdder(budget, manager.decode_manager.inflight_tokens,
                         manager.cache_manager, manager.table_manager)
    reqs, chunks = [], []
    for pending in manager.pending_list:
        req = adder.try_add_one(pending)
        if req is None:
            break
        pending.chunked_req = req if isinstance(req, ChunkedReq) else None
        if pending.chunked_req is not None:
            chunks.append(pending)
        reqs.append(req)
    if not reqs:
        return None
    manager.pending_list = chunks + manager.pending_list[len(reqs):]
    return Batch(reqs, "prefill")


def test_default_fcfs_matches_legacy_batches_chunks_and_resources():
    managers = [make_manager(), make_manager()]
    for manager in managers:
        enqueue(manager, 90, 2)
        enqueue(manager, 10, 6)
        enqueue(manager, 20)
    for budget in [3, 3, 2, 8, 8]:
        actual = managers[0].schedule_next_batch(budget)
        expected = legacy_fcfs(managers[1], budget)
        assert (actual is None) == (expected is None)
        if actual is not None:
            describe = lambda batch: [(r.uid, r.cached_len, r.device_len, type(r)) for r in batch.reqs]
            assert describe(actual) == describe(expected)
            finish_batch(managers[0], actual)
            finish_batch(managers[1], expected)
        assert uids(managers[0].pending_list) == uids(managers[1].pending_list)
        assert managers[0].table_manager.available_size == managers[1].table_manager.available_size
        assert managers[0].cache_manager.prefix_cache.size_info == managers[1].cache_manager.prefix_cache.size_info
    for manager in managers:
        assert not manager.pending_list
        manager.cache_manager.check_integrity()


def test_identical_rank_inputs_have_identical_aging_decisions():
    managers = [make_manager("lpm-aging", lpm_aging_tokens_per_pass=2) for _ in range(2)]
    for manager in managers:
        cold = enqueue(manager, 90)
        enqueue(manager, 10, 6)
        cold.waiting_passes = 3
    assert [uids(m._get_candidates()) for m in managers] == [[90, 10], [90, 10]]


def test_unlock_refreshes_protection_cost_before_capacity_check():
    manager = make_manager()
    pending = enqueue(manager, 90, cached=6, output_len=57)
    cache = manager.cache_manager
    other = cache.prefix_cache.match_prefix(pending.input_ids[:-1]).cuda_handle
    cache.lock(other)
    probe = cache.probe_req(pending)
    assert probe.newly_protected_size == 0
    cache.unlock(other)
    before = snapshot(manager)
    assert cache.try_match_and_lock(pending, 0, probe=probe) is None
    assert snapshot(manager) == before


def test_split_by_first_admission_invalidates_later_sorting_probe():
    manager = make_manager("lpm")
    ids = torch.arange(12, dtype=torch.int32)
    cache_prefix(manager, ids, 12)
    enqueue(manager, 90, input_ids=ids[:5])
    enqueue(manager, 10, input_ids=ids[:9])
    batch = manager.schedule_next_batch(2)
    assert uids(batch.reqs) == [10, 90]
    assert [req.cached_len for req in batch.reqs] == [8, 4]
    assert [len(req.cache_handle.get_matched_indices()) for req in batch.reqs] == [8, 4]
    assert manager.cache_manager.prefix_cache.size_info.protected_size == 8
    finish_batch(manager, batch)
    manager.cache_manager.check_integrity()


def test_queue_fallback_ages_only_after_a_real_new_admission(monkeypatch):
    manager = make_manager("lpm-aging", lpm_aging_tokens_per_pass=2, lpm_queue_threshold=2)
    for uid in range(3):
        enqueue(manager, uid)
    spy = Mock(wraps=manager.cache_manager.probe_req)
    monkeypatch.setattr(manager.cache_manager, "probe_req", spy)
    batch = manager.schedule_next_batch(8)
    assert uids(batch.reqs) == [0]
    assert spy.call_count == 1  # admission probe only; no queue-wide sorting probe
    assert [req.waiting_passes for req in manager.pending_list] == [1, 1]


def test_optional_legacy_timeout_remains_available():
    manager = make_manager("lpm", fairness_max_wait_steps=2)
    enqueue(manager, 90)
    enqueue(manager, 10, 6)
    manager.schedule_step = 2
    assert uids(manager._get_candidates()) == [90, 10]


def test_cli_settings_and_defaults():
    common = ["--model-path", "test-model", "--dtype", "float32"]
    default, _ = parse_args(common)
    assert default.schedule_policy == "fcfs" and default.lpm_aging_tokens_per_pass == 0
    assert default.fairness_max_wait_steps == 0
    configured, _ = parse_args(common + [
        "--schedule-policy", "lpm-aging", "--lpm-aging-tokens-per-pass", "8",
        "--lpm-queue-threshold", "32", "--fairness-max-wait-steps", "0",
    ])
    assert (configured.schedule_policy, configured.lpm_aging_tokens_per_pass) == ("lpm-aging", 8)
    assert configured.lpm_queue_threshold == 32


@pytest.mark.parametrize("kwargs", [
    {"schedule_policy": "unknown"}, {"lpm_queue_threshold": 0},
    {"lpm_aging_tokens_per_pass": -1}, {"fairness_max_wait_steps": -1},
])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SchedulerConfig("test-model", DistributedInfo(0, 1), torch.float32, **kwargs)


@pytest.mark.parametrize("flag,value", [
    ("--schedule-policy", "unknown"), ("--lpm-queue-threshold", "0"),
    ("--lpm-aging-tokens-per-pass", "-1"), ("--fairness-max-wait-steps", "-1"),
])
def test_invalid_cli_is_rejected(flag, value):
    with pytest.raises(SystemExit) as error:
        parse_args(["--model-path", "test-model", "--dtype", "float32", flag, value])
    assert error.value.code == 2
