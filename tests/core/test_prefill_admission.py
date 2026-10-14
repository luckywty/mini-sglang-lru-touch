from __future__ import annotations

import pytest
import torch

import minisgl.core as core
from minisgl.core import SamplingParams
from minisgl.kvcache.radix_cache import RadixTreeNode
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.prefill import PrefillAdder
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq


@pytest.fixture(autouse=True)
def reset_global_ctx(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    # These tests exercise admission on CPU, without requiring a CUDA driver.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)


def _make_managers(page_size, cache_type="radix"):
    core.set_global_ctx(core.Context(page_size=page_size))
    page_table = torch.full((2, 32), -1, dtype=torch.int32)
    cm = CacheManager(4, page_size, page_table, type=cache_type)
    tm = TableManager(2, page_table)
    if cache_type == "radix":
        indices = cm._page_to_token(cm._allocate(3))
        cm.prefix_cache.insert_prefix(torch.arange(3 * page_size, dtype=torch.int32), indices)
    return cm, tm


def _pending(page_size, output_len):
    # The last prompt token is deliberately excluded from prefix matching.
    return PendingReq(
        uid=0,
        input_ids=torch.arange(2 * page_size + 1, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=output_len),
    )


def _snapshot(cache):
    def visit(node):
        return (
            id(node),
            node.timestamp,
            node.ref_count,
            None if node.is_root() else node._key.tolist(),
            None if node.is_root() else node.value.tolist(),
            [(key, visit(child)) for key, child in node.children.items()],
        )

    return visit(cache.root_node), cache.size_info, RadixTreeNode.counter


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("failure", ["current_capacity", "after_lock_capacity", "reserved", "table"])
def test_rejection_does_not_change_tree(page_size, failure):
    cm, tm = _make_managers(page_size)
    output_len = 4 * page_size if failure == "current_capacity" else 2 * page_size
    reserved_size = 0
    if failure == "reserved":
        output_len = 2 * page_size - 1
        reserved_size = 1
    if failure == "table":
        tm.allocate()
        tm.allocate()
    req = _pending(page_size, output_len)
    adder = PrefillAdder(32, reserved_size, cm, tm)
    before = _snapshot(cm.prefix_cache)
    available_tables = tm.available_size
    free_slots = cm.free_slots.clone()

    assert adder._try_allocate_one(req) is None

    assert _snapshot(cm.prefix_cache) == before
    assert tm.available_size == available_tables
    assert torch.equal(cm.free_slots, free_slots)


@pytest.mark.parametrize("page_size", [1, 4])
def test_accepted_partial_match_splits_locks_and_touches(page_size):
    cm, tm = _make_managers(page_size)
    cache = cm.prefix_cache
    original_leaf = next(iter(cache.root_node.children.values()))
    original_timestamp = original_leaf.timestamp
    req = _pending(page_size, 2 * page_size - 1)
    # Admission at the exact capacity boundary after protecting the prefix.
    adder = PrefillAdder(32, 0, cm, tm)

    handle, table_idx = adder._try_allocate_one(req)

    assert handle.cached_len == 2 * page_size
    assert handle.node.length == 2 * page_size
    assert handle.node.ref_count == 1
    assert handle.node.timestamp > original_timestamp
    assert original_leaf.parent is handle.node
    assert original_leaf.length == page_size
    assert original_leaf.ref_count == 0
    assert original_leaf.timestamp == original_timestamp
    assert cache.size_info.protected_size == 2 * page_size
    assert cm.available_size == 2 * page_size
    assert torch.equal(tm.page_table[table_idx, : handle.cached_len], handle.get_matched_indices())
    assert torch.equal(tm.token_pool[table_idx, : handle.cached_len], req.input_ids[:-1])
    cm.unlock(handle)
    cm.check_integrity()


@pytest.mark.parametrize("page_size", [1, 4])
def test_already_protected_ancestor_is_not_deducted_twice(page_size):
    cm, tm = _make_managers(page_size)
    cache = cm.prefix_cache
    shared_handle = cache.match_prefix(torch.arange(page_size, dtype=torch.int32)).cuda_handle
    cm.lock(shared_handle)
    before = _snapshot(cache)
    req = _pending(page_size, 2 * page_size - 1)

    probe = cm.probe_req(req)

    assert probe.cached_len == 2 * page_size
    assert probe.newly_protected_size == page_size
    assert _snapshot(cache) == before
    handle, _ = PrefillAdder(32, 0, cm, tm)._try_allocate_one(req)
    assert shared_handle.node.ref_count == 2
    assert handle.node.ref_count == 1
    assert handle.node.timestamp == shared_handle.node.timestamp
    assert cache.size_info.protected_size == 2 * page_size
    cm.unlock(handle)
    cm.unlock(shared_handle)
    cm.check_integrity()


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("matched_pages", [0, 1, 2, 3])
@pytest.mark.parametrize("protected", [False, True])
def test_probe_predicts_actual_locking_cost(page_size, matched_pages, protected, monkeypatch):
    cm, _ = _make_managers(page_size)
    cache = cm.prefix_cache
    if protected:
        shared = cache.match_prefix(torch.arange(page_size, dtype=torch.int32)).cuda_handle
        cm.lock(shared)
    input_ids = torch.arange(matched_pages * page_size, dtype=torch.int32)
    before = _snapshot(cache)
    probe = cache.probe_prefix(input_ids)
    assert _snapshot(cache) == before
    assert cache.estimate_match_len(input_ids.tolist()) == probe.cached_len
    assert _snapshot(cache) == before

    def unexpected_comparison(*args):
        pytest.fail("Committing a probe must not compare token prefixes again")

    monkeypatch.setattr(RadixTreeNode, "get_match_len", unexpected_comparison)
    available_before = cm.available_size
    handle = cache.commit_match(probe)
    cm.lock(handle)
    assert handle.cached_len == probe.cached_len == len(input_ids)
    assert available_before - cm.available_size == probe.newly_protected_size
    cm.unlock(handle)
    if protected:
        cm.unlock(shared)
    cm.check_integrity()


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("protected", [False, True])
def test_admission_matches_each_edge_only_once(page_size, protected, monkeypatch):
    cm, tm = _make_managers(page_size)
    cache = cm.prefix_cache
    if protected:
        shared = cache.match_prefix(torch.arange(page_size, dtype=torch.int32)).cuda_handle
        cm.lock(shared)
    comparisons = []
    original_compare = RadixTreeNode.get_match_len

    def track_comparison(node, input_ids):
        comparisons.append(node.uuid)
        return original_compare(node, input_ids)

    monkeypatch.setattr(RadixTreeNode, "get_match_len", track_comparison)
    handle, _ = PrefillAdder(32, 0, cm, tm)._try_allocate_one(
        _pending(page_size, 2 * page_size - 1)
    )

    assert len(comparisons) == (2 if protected else 1)
    assert len(comparisons) == len(set(comparisons))
    cm.unlock(handle)
    if protected:
        cm.unlock(shared)
    cm.check_integrity()


@pytest.mark.parametrize("cache_type", ["radix", "naive"])
def test_unmatched_request_is_admitted(cache_type):
    cm, tm = _make_managers(4, cache_type)
    req = PendingReq(0, torch.tensor([99, 98], dtype=torch.int32), SamplingParams(max_tokens=1))
    before = _snapshot(cm.prefix_cache) if cache_type == "radix" else None

    handle, _ = PrefillAdder(32, 0, cm, tm)._try_allocate_one(req)

    assert handle.cached_len == 0
    if cache_type == "radix":
        assert _snapshot(cm.prefix_cache) == before
    cm.unlock(handle)
    cm.check_integrity()
