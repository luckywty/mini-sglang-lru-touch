from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig

from .utils import validate_prefill_policy


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    schedule_policy: str = "fcfs"
    lpm_queue_threshold: int = 128
    fairness_max_wait_steps: int = 0  # optional legacy FCFS timeout fallback
    lpm_aging_tokens_per_pass: int = 0
    batch_schedule_policy: str = "prefill-first"
    max_consecutive_decodes: int = 16
    protected_prefill_tokens: int = 256
    adaptive_prefill_budget: bool = False
    target_batch_tokens: int = 256
    min_chunk_tokens: int = 64

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    def __post_init__(self) -> None:
        validate_prefill_policy(
            self.schedule_policy,
            self.lpm_queue_threshold,
            self.fairness_max_wait_steps,
            self.lpm_aging_tokens_per_pass,
        )
        if self.batch_schedule_policy not in ("prefill-first", "decode-first-bounded"):
            raise ValueError(
                "batch_schedule_policy must be 'prefill-first' or 'decode-first-bounded'"
            )
        for name in (
            "max_consecutive_decodes", "protected_prefill_tokens",
            "target_batch_tokens", "min_chunk_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.adaptive_prefill_budget, bool):
            raise ValueError("adaptive_prefill_budget must be a bool")
        if self.adaptive_prefill_budget:
            if self.batch_schedule_policy != "decode-first-bounded":
                raise ValueError("adaptive_prefill_budget requires decode-first-bounded")
            if (
                not isinstance(self.max_extend_tokens, int)
                or isinstance(self.max_extend_tokens, bool)
                or self.max_extend_tokens <= 0
            ):
                raise ValueError("adaptive_prefill_budget requires positive integer max_extend_tokens")
            if self.min_chunk_tokens > self.target_batch_tokens:
                raise ValueError("min_chunk_tokens must not exceed target_batch_tokens")
            if self.min_chunk_tokens > min(self.max_extend_tokens, self.protected_prefill_tokens):
                raise ValueError(
                    "min_chunk_tokens must not exceed max_extend_tokens or protected_prefill_tokens"
                )

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
