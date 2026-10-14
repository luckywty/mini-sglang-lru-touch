from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    arrival_order: int = 0
    arrival_step: int = 0
    cached_len_est: int = 0
    waiting_passes: int = 0  # completed passes admitting another new request

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


def validate_prefill_policy(
    schedule_policy: str,
    lpm_queue_threshold: int,
    fairness_max_wait_steps: int,
    lpm_aging_tokens_per_pass: int = 0,
) -> None:
    if schedule_policy not in ("fcfs", "lpm", "lpm-aging"):
        raise ValueError("schedule_policy must be 'fcfs', 'lpm' or 'lpm-aging'")
    if lpm_queue_threshold <= 0:
        raise ValueError("lpm_queue_threshold must be positive")
    if fairness_max_wait_steps < 0:
        raise ValueError("fairness_max_wait_steps must be nonnegative (0 disables the fallback)")
    if lpm_aging_tokens_per_pass < 0:
        raise ValueError("lpm_aging_tokens_per_pass must be nonnegative")


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
