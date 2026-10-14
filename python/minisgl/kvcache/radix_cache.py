from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchProbe, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor #保存当前压缩路径上包含的一维 Token ID 向量（形状如 [N]）。通过 set_key_value 显式写入。
        self._value: torch.Tensor   #KV 物理页/插槽索引。保存与 _key 一一对应的显存 Block/Page 索引向量
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:    #（子指向父，自底向上回溯）
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self  #把当前 Node 放进父节点的 children 字典

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:  #把一个 Radix Tree 节点从中间切开
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        #Python 的魔术方法，
        # 用于重载小于号（<）运算符。
        # 它规定两个树节点之间的大小比较仅取决于其访问时间戳 self.timestamp
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    #将沿途各节点分散存储的显存块索引，
    # 按时序还原为一条连续的物理页/槽位索引张量
    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


@dataclass(frozen=True)
class RadixMatchProbe(MatchProbe):
    node: RadixTreeNode
    matched_in_node: int
    structure_version: int


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self._structure_version = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        #当请求匹配或插入前缀后，
        # 底层显存块绝不能在模型正在使用时被 evict 回收。
        # 该方法负责在请求开始处理时加锁（锁定显存），
        # 并在推理结束或释放时解锁（允许淘汰）。
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def _touch_path(self, node: RadixTreeNode) -> None:
        #刷新 node -> root 整条路径的 LRU 时间戳。
        #必须整条路径都刷：evict() 先淘汰叶子，叶子删完后会把 parent 推入堆，
        #祖先节点时间戳过期会导致 LRU 判序错误。
        #node 为 root 时（cached_len == 0 / 完全未命中）是空操作。
        tic = time.monotonic_ns()
        while not node.is_root():
            node.timestamp = tic
            node = node.parent

    def touch_handle(self, handle: BaseCacheHandle) -> None:
        #仅在前缀被真正采纳（准入容量检查通过）后由调度器调用。
        assert isinstance(handle, RadixCacheHandle)
        self._touch_path(handle.node)

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        #内部调用 _tree_walk 从根节点开始自顶向下搜索
        #按照 page_size 的整数倍对齐检查匹配长度
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def estimate_match_len(self, token_ids: List[int]) -> int:
        # Scheduling estimates use the same read-only traversal as admission.
        input_ids = torch.tensor(token_ids, dtype=torch.int32)
        return self.probe_prefix(input_ids).cached_len

    def probe_prefix(self, input_ids: torch.Tensor) -> RadixMatchProbe:
        # 只读探测：不分裂、不刷新时间戳、不修改引用计数或容量。
        prefix_len = 0
        newly_protected_size = 0
        matched_in_node = 0
        indice_len = len(input_ids)
        node = self.root_node

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                break
            match_len = child_node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            if match_len == 0:
                break
            node = child_node
            matched_in_node = match_len
            prefix_len += match_len
            if node.ref_count == 0:
                # 部分命中时只扣匹配部分，未命中的后缀仍可淘汰。
                newly_protected_size += match_len
            if match_len != node.length:
                # partial edge: count fully-matched pages only, do not split
                break
        return RadixMatchProbe(
            prefix_len, newly_protected_size, node, matched_in_node, self._structure_version
        )

    def refresh_probe(self, probe: MatchProbe, input_ids: torch.Tensor) -> RadixMatchProbe:
        # 插入、分裂、淘汰会使节点位置失效；引用计数变化只需重算加锁成本。
        if not isinstance(probe, RadixMatchProbe) or probe.structure_version != self._structure_version:
            return self.probe_prefix(input_ids)
        node = probe.node
        matched_len = probe.matched_in_node
        newly_protected_size = 0
        while not node.is_root():
            if node.ref_count == 0:
                newly_protected_size += matched_len
            node = node.parent
            if not node.is_root():
                matched_len = node.length
        assert node is self.root_node, "Probe belongs to a different prefix cache"
        if newly_protected_size == probe.newly_protected_size:
            return probe
        return replace(probe, newly_protected_size=newly_protected_size)

    def commit_match(self, probe: MatchProbe) -> RadixCacheHandle:
        assert isinstance(probe, RadixMatchProbe)
        assert probe.structure_version == self._structure_version, "Refresh stale probe before commit"
        node = probe.node
        # 准入通过后直接使用探测记录的位置，不再从 root 遍历或比较 token。
        if not node.is_root() and probe.matched_in_node != node.length:
            node = node.split_at(probe.matched_in_node)
            self._structure_version += 1
        return RadixCacheHandle(probe.cached_len, node)

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        #为什么要 align_down 截断？

        # 在 PagedAttention 中，KV Cache 的显存回收与共享必须以整页（Page / Block）为最小单位。

        # 假设 page_size = 16，当前序列有 35 个 token，则只能插入前 32 个 token（2 个整页）。

        # 剩余未填满完整一页的 3 个 token 属于“脏数据”或“未固化页”，
        # 不能被当作公共前缀缓存，否则会导致跨请求的 KV 混淆
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        #本请求刚刚读/写了这段前缀的 KV，它就是"最后一次使用"。
        #必须在此就地刷新（而不是只靠准入时的 touch）：
        #准入 -> cache_req 之间路径一直被 lock（ref_count > 0），evict 看不到；
        #等 unlock 变回 evictable 时，若不刷新，时间戳会停在准入时刻，
        #落后真实使用时间一个完整的 prefill + decode 时长，导致 LRU 判序颠倒。
        self._touch_path(node)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self._structure_version += 1
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            self._structure_version += 1
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent) #系统立即将其动态推入最小堆

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        #清空整棵基数树，将前缀缓存系统完全重置为初始状态
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        #采用非递归的深度优先遍历（DFS）扫描整棵基数树，
        # 筛选出所有当前可以被安全淘汰的叶子节点
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        # 沿树匹配并按需切分节点，不刷新 LRU 时间戳。
        # 调度准入使用 probe_prefix + commit_match，检查通过后直接提交分裂。
        # insert_prefix 也使用此方法，并在 walk 后调用 _touch_path。

        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node.split_at(match_len) #返回的是分裂点之前（前半段）的那个节点
                self._structure_version += 1
                return node, prefix_len

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
#x[:page_size]：截取接下来的一整页 Token。

# .tolist()：将 Tensor 切片转为 Python list[int]。

# tuple(...)：将可变列表转换为不可变的 tuple。

#tuple 是不可变序列（Immutable），
# Python 天然支持对其进行哈希计算，
# 因此可以安全作为 children 字典的 Key
