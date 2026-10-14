# Prefill LPM 与 Pass-based Aging

实现参考 [SGLang PR #31723](https://github.com/sgl-project/sglang/pull/31723)，
核对的 PR head 为 `622dcea784f1da8a8f90a1bd3baa89d5c7b853af`。
这里适配的是 mini-SGLang 的实际准入生命周期，而不是复制上游计数位置。

## 原有调用链与约束

```text
Scheduler._process_one_msg(UserMsg)
  → PrefillManager.add_one_req → PendingReq（等待队列）
Scheduler._schedule_next_batch
  → PrefillManager.schedule_next_batch
    → 本轮候选顺序
    → PrefillAdder.try_add_one
      → 新请求：_try_allocate_one → CacheManager.try_match_and_lock
        → probe/refresh → KV 容量检查 → commit_match → lock → touch
        → TableManager.allocate → 复制缓存索引与 prompt
      → 已准入 ChunkedReq：沿用 table_idx/cache_handle/cached_len
    → Batch(phase="prefill")
  → Prefill 没有 Batch 时才尝试 DecodeManager.schedule_next_batch
  → _prepare_batch → allocate_paged → 原有 Engine.forward
  → 请求完成/Abort：原有 cache_req + TableManager.free
```

工作区在开始时已有 PrefixCache 只读探测、延迟分裂和两种 FCFS 退化实现，
并有其他用户修改；这些文件已先备份和记录哈希。默认 FCFS 不进行排序探测。
新请求只追加到基础队列，LPM 生成单独的候选列表，基础新请求顺序仍按到达顺序维护。
候选重排后按对象身份移除成功准入项，不能再用 `pending_list[len(reqs):]`。
原有“一个候选不能接纳就 break”、Prefill/Decode 优先关系和预算算法保持不变。

## 配置与策略

| 配置 | 默认值 | 语义 |
|---|---|---|
| `schedule_policy` | `fcfs` | `fcfs`、`lpm`、`lpm-aging` |
| `lpm_aging_tokens_per_pass` | `0` | 非负整数；仅 `lpm-aging` 使用，0 关闭 Aging |
| `lpm_queue_threshold` | `128` | 待首次准入请求超过阈值时按到达顺序退回 FCFS |
| `fairness_max_wait_steps` | `0` | 保留先前的可选整轮 FCFS 等待阈值；0 关闭 |

本次将先前等待阈值的默认 64 调整为 0，以便纯 LPM 和 Pass-based Aging 单独工作；
显式传入 `--fairness-max-wait-steps 64` 仍可使用原来的兜底。
这项调整不影响默认 FCFS。阈值为正，Aging 和可选等待阈值非负，未知策略报错。

```bash
# 添加到原有服务启动命令；默认不改变现有 FCFS 行为
--schedule-policy lpm-aging --lpm-aging-tokens-per-pass 8 --lpm-queue-threshold 128
```

核心分数为 `cached_len_est + lpm_aging_tokens_per_pass * waiting_passes`，降序排序；
同分时按 `arrival_order` 升序。纯 LPM 或 Aging=0 使用原始匹配长度排序。
虚拟分数从不写入 `handle.cached_len`，也不参与实际页数或 Token Budget 计算。

## 有效等待轮次

新请求初始 `waiting_passes=0`。一次 `schedule_next_batch` 至少成功准入一个**新请求**，
本轮仍未准入的新请求才在调用末尾统一加 1。接纳多个新请求也只计一轮。
只调用排序函数不会增长计数。已准入请求（含新 ChunkedReq）不再增长。

空队列、零 Prefill Budget、没有表槽且无新准入、KV 容量全部拒绝、只有 chunk 续跑，
都不算有效等待轮次。计数保存在 PendingReq 中，Abort/移除无需维护额外状态表。
TP 使用相同逻辑事件计数和到达顺序，不读取各 rank 的本机时间；CPU 测试验证一致输入
会产生一致决策，未运行完整多 GPU 模型集成测试。

这与 PR 的“每次看见等待队列就计数”有所区别，避免 mini-SGLang 的 Decode-only/空转
尝试造成不合理增长。N>0、匹配上限有限时，冷请求可以超过后来的新 hot 请求；
这是排序保证，仍可能因表槽/KV/预算不足而不能接纳，不是严格服务时延保证。

## 只读前缀估算与安全复用

`CacheManager.probe_req` 与实际匹配都排除 prompt 的最后一个 token。
`RadixPrefixCache.probe_prefix` 按 page_size 对齐比较，只读取 children/key/ref_count，
不调用 split、lock、touch；timestamp、节点结构、引用计数、容量和 free_slots 不变。
部分命中只统计匹配段，已受保护的节点不重复计入新增保护量。

本轮排序结果保存为局部匹配计划，随调用结束释放。结构版本在插入新节点、分裂和淘汰时增长。
实际准入前 `refresh_probe` 执行：

1. 结构版本变化：重新只读匹配，避免使用失效节点位置或已淘汰索引。
2. 结构未变：复用 token 匹配，沿祖先路径重算新增保护量，处理先前请求 lock/unlock 的影响。
3. 用 `available_size - newly_protected_size` 检查真实容量，通过后才分裂、锁定和 touch。

即使 LPM 排序估算过时，也不会用旧分数或旧保护量分配 KV。
无排序收益时跳过探测：FCFS、队列过长、单请求、没有新表槽、chunk 已占满本轮预算。
结构稳定的共享前缀批次中，3 个请求只做 3 次边比较；分裂时保守地重新匹配。
该复用仅适用于相同请求输入及当前串行调度，不支持任意并发树修改。

## Chunked Prefill 与资源生命周期

已准入 chunk 始终排在新请求前面，继续沿用已有表槽、cache_handle 和进度。
它们不参与新请求 LPM 排序，也不累计等待轮次。
模型计算、Attention、Sampling、CUDA Graph、Decode 及原有页分配/释放流程均未修改。
CPU 生命周期测试真实执行 allocate_paged/cache_req，验证 chunk 完成和 Abort 后表槽归还、
protected_size 归零及缓存页总量守恒。测试只替换 CPU pinning，不伪造缓存资源计数。

## 验证与复现

最终结果：106 项 CPU 测试通过，0 失败、0 错误、0 跳过；既有 40 项测试包含在内。
另外以 Python 3.10 语法版本解析所有改动 Python 文件，通过；本次新增行无行尾空格。

在项目根目录执行（已有虚拟环境及 native radix 的 Ninja/C++ 编译依赖）：

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python -m pytest \
  tests/core tests/misc -q -o addopts='' --tb=short \
  --junitxml=artifacts/lpm_aging/tests.xml

PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_prefill_policy.py --seed 31723 --trials 200 --repeats 5 \
  --output artifacts/lpm_aging/benchmark.json
```

最终单元测试结果见 JUnit XML；详细 Benchmark 数据和每次重复见
[`benchmark.json`](../artifacts/lpm_aging/benchmark.json)，可读结果见
[`benchmark.md`](../artifacts/lpm_aging/benchmark.md)。

本次修改的实现文件为 `scheduler/{utils,prefill,cache,config,scheduler}.py`、
`server/args.py` 和 `kvcache/{base,radix_cache}.py`；新增两个策略/Benchmark 测试文件、
一个 Benchmark 脚本和本文档。任务开始时已有的其他代码/文档/锁文件保持原样。
[`changes.diff`](../artifacts/lpm_aging/changes.diff) 对比的是任务开始时的本地快照，
不是混入先前用户改动的整仓 Git diff；
[`verification.json`](../artifacts/lpm_aging/verification.json) 记录哈希、语法和测试摘要。

测试覆盖默认 FCFS 与原有切片算法逐轮对比、评分及同分、Aging=0、持续 hot 到达、
有效轮次及空转、128/129 边界、Abort、ChunkedReq、页尺寸 1/4、部分/无/重复匹配、
排序后插入/分裂/淘汰/lock/unlock 的变化、实际 KV 约束和 Benchmark 确定性。

## Benchmark 方法与结果解释

环境为 Ryzen 7 5800H、WSL2、Python 3.12、Torch 2.9.1+cu128，Torch CPU 线程为 1。
检测到 RTX 3060 Laptop 6GB，但没有加载模型；没有测量 GPU TTFT、TPOT 或推理吞吐。
排序微基准使用真实 CPU RadixCache，预热原生比较内核；队列 8/32/128/129/256，
前缀 64/1024 token，各 200 次。请求流模拟使用真实 PrefillManager、共享前缀、表槽、
页分配和缓存回收，仅把模型 forward 替换成确定的 KV 计算完成事件；输出长度均为 1。
页分配和模拟完成不计入调度 CPU 计时。5 次随机顺序重复保留运行间时间范围。

混合负载：初始一个 65-token 冷请求，持续 32 轮、每轮 4 个 hot 请求，
hot 命中 64 token，本轮预算 4 token，Aging N=8。冷请求的首次准入轮：
FCFS=1，LPM=33，LPM-Aging=9；完成轮分别 17/49/25。
这验证了排序超越，未把实际秒数的服务上界包装成公平性保证。

三种策略排空后均复用 8192 token，计算 193 个 Prefill token，命中率约 97.70%，
排空轮均为 49。Aging 提前计算冷请求的 65 个 token，使 hot 等待 p95 从纯 LPM 的 0
增长到 17 轮；同等总工作下没有推理吞吐提升证据。
hot-only/cold-only 负载的总计算量和完成进度也未因 Aging 改善。

1024-token 前缀、128 请求的候选排序中位数约为 LPM 1.06ms、LPM-Aging 1.08ms；
129 请求触发 FCFS 后约 3.3µs，并跳过全队列探测。
混合负载的整体调度 CPU 均值约 FCFS 157µs、LPM 159µs、LPM-Aging 374µs。
这不是公式本身变慢 2 倍：Aging 提前执行长 cold chunk 后，hot 队列积压，
续跑完成后的排序需要探测更大的队列。具体数字以 JSON/运行范围为准。

## 限制与后续方向

队列超过阈值只是 CPU 开销退化策略；Aging 只保证优先级增长，不保证资源受限时服务。
原有 break 会发生队首阻塞，不可准入的大请求仍需既有入口/资源策略处理。
没有增加模型级性能基准或实际多 GPU 测试；pinning 被替换的 CPU 模拟不能代表 GPU 拷贝成本。
全局结构版本是保守失效机制，不相关分支分裂也会重探测，可后续按实测成本决定是否细化。
排序和真实准入不会因为虚拟分数而放宽任何原有资源检查。

后续可在真实模型与固定请求轨迹上调优 Aging N/队列阈值，测 hot/cold TTFT 分位数、
调度 CPU 时间与缓存命中率，并研究 chunk 预算对公平性的影响；当前数据不支持宣称 GPU 加速。

简历表述：为 mini-SGLang 实现可配置 LPM 与按有效调度轮次老化的 Prefill 策略，
加入无副作用前缀探测、过长队列 FCFS 降级及缓存计划失效校验，完成 CPU 回归和可复现调度基准，
验证混合负载下冷请求排序/准入改善及资源守恒，并量化调度开销与 hot 请求等待的取舍。
