# 变更记录：修改前（before） → 修改后（after）

- 修改前：`Serving_V1/mini-sglang` → 分支 `before`
- 修改后：`Serving_V1/Promotion/mini-sglang` → 分支 `after`
- 共同基线：`sgl-project/mini-sglang` @ `9a91cfa`，两个分支均在此基础上带有**完全相同**的本地改动。
- 差异范围：仅 4 个文件，原始补丁见 [`changes/before-after.patch`](changes/before-after.patch)。

```
python/minisgl/kvcache/base.py        |  7 +++++++
python/minisgl/kvcache/radix_cache.py | 45 +++++++++++++++++++++-------
python/minisgl/scheduler/cache.py     | 12 ++++++++--
python/minisgl/scheduler/prefill.py   |  3 +++
4 files changed, 60 insertions(+), 7 deletions(-)
```

## 核心意图

把「查询前缀」与「采纳前缀」解耦：

- **修改前**：`match_prefix()`（内部 `_tree_walk`）在**每次查询**时就刷新命中路径上节点的 LRU 时间戳。调度阶段的探测性匹配、最终未被采纳的请求也会提升前缀热度，造成 LRU 序失真（热度假象）。
- **修改后**：匹配过程变为**只读**（不改时间戳），只有当请求通过容量准入检查、前缀被真正占用时，才调用 `touch_handle()` 刷新整条路径的 LRU 时间戳。同时新增无副作用的 `estimate_match_len()`，供调度排序（如 LPM）在不产生任何副作用的前提下估算前缀命中长度。

## 逐文件差异

### 1. `python/minisgl/kvcache/base.py`（+7）

新增非抽象默认实现：

```python
def touch_handle(self, handle: BaseCacheHandle) -> None:
    """Refresh the LRU timestamps of all nodes on the handle's matched path.

    Called only when the matched prefix is actually going to be used.
    Default is a no-op for caches without LRU eviction.
    """
```

- 为 `BaseCacheManager` 抽象基类提供 `touch_handle` 钩子；无 LRU 淘汰策略的缓存实现保持默认空操作。

### 2. `python/minisgl/kvcache/radix_cache.py`（+39 / -6）

**(a) 新增 `RadixCache.touch_handle(handle)`**

```python
assert isinstance(handle, RadixCacheHandle)
tic = time.monotonic_ns()
node = handle.node
while not node.is_root():
    node.timestamp = tic
    node = node.parent
```

- 从 `handle.node` 沿 parent **一路刷新到 root**，而不是只刷最后一个匹配节点。
- 原因（代码注释）：`evict()` 先淘汰叶子，叶子删完后会把 parent 推入堆；若祖先节点时间戳未刷新而过期，会导致 LRU 判序错误。

**(b) `_tree_walk()` 改为只读**

- 删除 `tic = time.monotonic_ns()`；
- 删除两处 `node.timestamp = tic`（节点完全匹配后、以及部分匹配 split 之后）；
- `match_prefix()` 因此不再改变任何 LRU 状态（split 剖分节点的行为保留，仍属于匹配逻辑所需的结构操作）。

**(c) 新增 `estimate_match_len(token_ids: List[int]) -> int`**

- 只读的前缀匹配长度估计，用于调度排序（例如 LPM 按前缀长度排序）。
- 与 `_tree_walk` 的区别：**不 split、不刷新 timestamp、不改 ref_count**。
- 返回按 `page_size` 向下对齐的匹配 token 数；命中部分边时只计完整页，不剖分节点。

### 3. `python/minisgl/scheduler/cache.py`（+11 / -1）

```python
def estimate_req(self, req: PendingReq) -> int:
    input_len = req.input_len
    assert input_len > 0
    return self.prefix_cache.estimate_match_len(
        req.input_ids[: input_len - 1].tolist()   # 注意：estimate 收 List[int]
    )

def touch(self, handle: BaseCacheHandle) -> None:
    # 刷新该 handle 匹配路径上的 LRU 时间戳（仅在真正采纳该前缀后调用）
    self.prefix_cache.touch_handle(handle)
```

- `CacheManager` 暴露 `estimate_req()`（无副作用的命中长度估计）与 `touch()`（显式刷新）。
- 注意：`estimate_req` 目前**尚未被调用**，是为后续调度排序预留的 API。

### 4. `python/minisgl/scheduler/prefill.py`（+3）

准入检查通过后新增一次刷新：

```python
self.cache_manager.lock(handle)
if estimated_len + self.reserved_size > self.cache_manager.available_size:
    return self.cache_manager.unlock(handle)
# 双重检查通过、前缀被真正采纳后才刷新 LRU 时间戳
self.cache_manager.touch(handle)
```

- 刷新时机：`lock` → 容量双重检查 → **`touch`** → 分配 table/写 KV。
- 若容量不足直接 `unlock` 返回，**不刷新**时间戳——未被采纳的前缀不会被提升热度。

## 行为对比小结

| 场景 | before | after |
| --- | --- | --- |
| `match_prefix()` 查询 | 立即刷新命中路径时间戳 | 不刷新（只读） |
| 请求通过准入检查、真正占用前缀 | 已在匹配时刷新 | 此时调用 `touch_handle()` 刷新整条路径 |
| 容量不足、前缀未被采纳 | 时间戳已被错误提升 | 不刷新 |
| 调度阶段估算前缀命中长度 | 只能用会改状态的 `match_prefix` | 可用无副作用的 `estimate_match_len` / `estimate_req` |
| LRU 判序依赖的祖先时间戳 | 只刷到匹配末节点 | 刷到 root，避免 `evict()` 推堆时祖先过期 |

## 两个分支共同的本地改动（相对上游 `9a91cfa`）

以下文件在 before / after 中**完全一致**，不属于本次差异：

- `python/minisgl/core.py`
- `python/minisgl/message/backend.py`
- `python/minisgl/message/frontend.py`
- `python/minisgl/scheduler/scheduler.py`
- `tests/core/test_cache_allocate.py`
- 未跟踪文件：`docs/features_zh.md`、`docs/structures_zh.md`、`python/test/`、`uv.lock`
