from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


class BaseKVCachePool(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    cached_len: int  # length already in cache before insertion (should be freed)
    handle: BaseCacheHandle  # cache handle for the inserted prefix


class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
    # TODO: support HiCache


@dataclass(frozen=True)
class MatchProbe:
    cached_len: int
    newly_protected_size: int  # evictable matched tokens that locking would protect


class BasePrefixCache(ABC):
    @abstractmethod
    def probe_prefix(self, input_ids: torch.Tensor) -> MatchProbe:
        """Inspect the match and its locking cost without modifying cache state.

        The locking cost counts only matched tokens whose nodes are currently
        evictable, including only the matched part of a partially matched edge.
        """

    def refresh_probe(self, probe: MatchProbe, input_ids: torch.Tensor) -> MatchProbe:
        """Revalidate a speculative probe before admission, without cache mutations."""
        return self.probe_prefix(input_ids)

    @abstractmethod
    def commit_match(self, probe: MatchProbe) -> BaseCacheHandle:
        """Materialize a probe without repeating prefix matching.

        The cache must not change between probing and committing. This may
        split nodes, but does not lock the handle or refresh LRU timestamps.
        """

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """

    def touch_handle(self, handle: BaseCacheHandle) -> None:
        """Refresh the LRU timestamps of all nodes on the handle's matched path.

        Called only when the matched prefix is actually going to be used.
        Default is a no-op for caches without LRU eviction.
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation may split nodes, but does not refresh LRU timestamps.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            MatchResult: The match result containing the cache handles.
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            InsertResult: The result of the insertion.
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""

    @abstractmethod
    def check_integrity(self) -> None:
        """Check the integrity of the cache. Raise an error if the cache is corrupted."""
