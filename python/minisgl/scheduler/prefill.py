from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq, validate_prefill_policy

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.kvcache.base import MatchProbe
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    #继承自基础请求类 Req，
    # 专门表示“Prompt 太长、当前批次仅处理了部分 Token、尚未完成 Prefill”的中间状态请求
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager
#把巨大的 Prefill 工作拆成多个 chunk，使 Prefill 和 Decode 更容易进行资源共享






#PrefillManager 负责“管理等待队列”，
# PrefillAdder 负责“判断一个请求现在能不能塞进当前 Prefill Batch”
@dataclass
class PrefillAdder:
    token_budget: int   #本轮最多 Prefill 多少 token
    reserved_size: int  #已经被 Decode / in-flight 请求预留多少资源
    cache_manager: CacheManager #Prefix Cache
    table_manager: TableManager #Token Table 和 Page Table
    match_probes: Dict[int, MatchProbe] = field(default_factory=dict)

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.try_match_and_lock(
            req, reserved_size=self.reserved_size, probe=self.match_probes.pop(id(req), None)
        )
        if handle is None:
            return None
        cached_len = handle.cached_len

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        #决定本次到底 Prefill 多少 token
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler




        #将当前批次（Chunk）需要计算的 Token ID，
        # 从 CPU 内存高效、异步地拷贝到 GPU 预分配的全局显存池（token_pool）的对应槽位中
        _slice = slice(cached_len, cached_len + chunk_size) #仅加载一个chunk个
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)



        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)
    schedule_policy: str = "fcfs"
    lpm_queue_threshold: int = 128
    fairness_max_wait_steps: int = 0
    lpm_aging_tokens_per_pass: int = 0
    schedule_step: int = field(default=0, init=False)
    _next_arrival_order: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        validate_prefill_policy(
            self.schedule_policy,
            self.lpm_queue_threshold,
            self.fairness_max_wait_steps,
            self.lpm_aging_tokens_per_pass,
        )

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(
            PendingReq(
                req.uid,
                req.input_ids,
                req.sampling_params,
                arrival_order=self._next_arrival_order,
                arrival_step=self.schedule_step,
            )
        )
        self._next_arrival_order += 1

    def _get_candidates(
        self, match_probes: Dict[int, MatchProbe] | None = None, prefill_budget: int | None = None
    ) -> List[PendingReq]:
        # 已准入的 chunked 请求先继续执行；新请求的基础队列保留到达顺序。
        chunked_pending: List[PendingReq] = []
        new_pending: List[PendingReq] = []
        for req in self.pending_list:
            (chunked_pending if req.chunked_req is not None else new_pending).append(req)

        if (
            self.schedule_policy == "fcfs"
            or len(new_pending) > self.lpm_queue_threshold
            or len(new_pending) < 2
        ):
            return chunked_pending + new_pending
        if prefill_budget is not None and self.table_manager.available_size == 0:
            # Existing chunks can continue, but no new request can obtain a table slot.
            return chunked_pending + new_pending
        if prefill_budget is not None and sum(
            req.input_len - req.chunked_req.cached_len
            for req in chunked_pending if req.chunked_req is not None
        ) >= prefill_budget:
            # Continuations consume this entire pass, so ordering new admissions has no benefit.
            return chunked_pending + new_pending
        if self.fairness_max_wait_steps > 0 and any(
            self.schedule_step - req.arrival_step >= self.fairness_max_wait_steps
            for req in new_pending
        ):
            # 等待过久时整轮退回 FCFS，避免新请求持续越过老请求。
            return chunked_pending + new_pending

        for req in new_pending:
            probe = self.cache_manager.probe_req(req)
            req.cached_len_est = probe.cached_len
            if match_probes is not None:
                match_probes[id(req)] = probe
        aging_step = self.lpm_aging_tokens_per_pass if self.schedule_policy == "lpm-aging" else 0
        if aging_step:
            candidates = sorted(
                new_pending,
                key=lambda req: (
                    -(req.cached_len_est + aging_step * req.waiting_passes), req.arrival_order
                ),
            )
        else:
            candidates = sorted(new_pending, key=lambda req: (-req.cached_len_est, req.arrival_order))
        return chunked_pending + candidates

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        #在给定的 Token 算力预算内，
        # 将等待队列中的请求打包成一个批次（Batch），
        # 同时精准维护长 Prompt 的切块状态与队列优先级
        if len(self.pending_list) == 0:
            return None
        # 各 TP rank 在同一调度位置计数，不使用本机时间参与排序。
        self.schedule_step += 1
        if prefill_budget <= 0:
            return None
        # Probes live for this call only; admission refreshes them against current cache state.
        match_probes: Dict[int, MatchProbe] = {}
        candidates = self._get_candidates(match_probes, prefill_budget=prefill_budget)

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            match_probes=match_probes,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        scheduled_ids: set[int] = set()
        admitted_new = False
        for pending_req in candidates:
            was_new = pending_req.chunked_req is None
            if req := adder.try_add_one(pending_req):   #req：Req/ChunkedReq
                pending_req.chunked_req = None  #清除上一轮的状态
                if isinstance(req, ChunkedReq): #req会记录有多少cached_len,如果还有token没计算完成，那么就会将其加入pending_req.chunked_req = req
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
                scheduled_ids.add(id(pending_req))
                admitted_new |= was_new
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        # 候选已重排，按对象身份移除已调度项；不能按数量切掉原队列头部。
        self.pending_list = chunked_list + [
            req for req in self.pending_list if id(req) not in scheduled_ids
        ]
        # Effective waiting: a completed pass admitted at least one NEW request.
        # No aging for empty/zero-budget/all-rejected/chunk-only passes or admitted requests.
        if admitted_new and self.schedule_policy == "lpm-aging" and self.lpm_aging_tokens_per_pass > 0:
            for pending_req in self.pending_list:
                if pending_req.chunked_req is None:
                    pending_req.waiting_passes += 1
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
