# mini-sglang-lru-touch

mini-sglang 的两个版本快照：**LRU 时间戳刷新时机优化**（修改前 / 修改后），以及二者的差异记录。

## 分支

| 分支 | 内容 |
| --- | --- |
| `before` | 修改前版本（`Serving_V1/mini-sglang`） |
| `after` | 修改后版本（`Serving_V1/Promotion/mini-sglang`） |
| `main`（默认） | 本说明 + 变更记录 `CHANGES.md` + 原始差异补丁 `changes/before-after.patch` |

两个分支的项目文件都位于仓库根目录，均基于同一份上游代码：`sgl-project/mini-sglang` @ `9a91cfa`，且包含完全相同的本地改动（`core.py`、`message/backend.py`、`message/frontend.py`、`scheduler/scheduler.py`、`tests/core/test_cache_allocate.py`、`docs/features_zh.md`、`docs/structures_zh.md`、`python/test/`、`uv.lock` 等）。

二者的**唯一区别**是 4 个文件的改动，详见 [CHANGES.md](CHANGES.md)。

## 对比两个版本

```bash
git diff before after            # 在任意克隆中直接对比两个分支
git diff before after --stat     # 只看文件级统计
```

或使用本仓库携带的补丁：

```bash
git apply --stat changes/before-after.patch
git apply changes/before-after.patch   # 需在 after 分支工作区中执行
```

## 获取代码

```bash
git clone https://github.com/luckywty/mini-sglang-lru-touch.git
cd mini-sglang-lru-touch
git checkout before   # 或 after
```
