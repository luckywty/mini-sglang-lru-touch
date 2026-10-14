# 自适应 Chunk Budget

该功能只调整 `decode-first-bounded` 的 Prefill 保护轮次预算，默认关闭。正常阶段选择、连续 Decode 提交计数、Prefill 内部排序、KV 准入和请求资源回收均沿用现有流程。

## 配置

| 字段 | 默认值 | 作用 |
|---|---:|---|
| `adaptive_prefill_budget` | `False` | 是否启用自适应保护预算 |
| `target_batch_tokens` | 256 | 从中减去活跃 Decode 数量的目标 Token 额度 |
| `min_chunk_tokens` | 64 | 本轮整个 Prefill Batch 的计算预算下限 |
| `protected_prefill_tokens` | 256 | 保护轮次计算预算的上限，已有参数 |
| `max_extend_tokens` | 8192 | 正常 Prefill 计算预算，也限制保护预算，已有参数 |
| `max_consecutive_decodes` | 16 | 优先尝试 Prefill 前的连续 Decode Batch 提交数，已有参数 |

启动时追加参数：

```bash
--batch-schedule-policy decode-first-bounded \
--max-consecutive-decodes 16 \
--protected-prefill-tokens 256 \
--adaptive-prefill-budget \
--target-batch-tokens 256 \
--min-chunk-tokens 64
```

目标额度和最低预算必须为正整数，程序配置拒绝 bool、浮点或字符串冒充整数。开关必须为 bool。启用时必须使用 `decode-first-bounded`，`max_extend_tokens` 必须为正整数，并要求：

```python
min_chunk_tokens <= target_batch_tokens
min_chunk_tokens <= min(max_extend_tokens, protected_prefill_tokens)
```

关闭时不新增这些跨参数限制，原有正常预算为零或保护预算小于64的配置仍按原逻辑处理。默认 `prefill-first` 不受影响。

## 计算与调用位置

每次需要保护 Prefill 时，Scheduler 调用 `_get_protected_prefill_budget()`：

```python
upper = min(self.prefill_budget, self.protected_prefill_tokens)
if not self.adaptive_prefill_budget:
    return upper

decode_tokens = len(self.decode_manager.running_reqs)
return min(
    upper,
    max(self.min_chunk_tokens, self.target_batch_tokens - decode_tokens),
)
```

只有等待 Prefill 和活跃 Decode 同时存在且连续 Decode 提交数达到 K 时，调用该方法。预算计算一次后交给原有 `PrefillManager.schedule_next_batch(budget)`，同一 Batch 构造中固定使用这个预算。

| 状态 | 行为 |
|---|---|
| 自适应关闭 | 原固定保护预算 |
| 连续 Decode < K，两个阶段都有请求 | 优先 Decode，不计算 Prefill 预算 |
| 连续 Decode >= K，两个阶段都有请求 | 优先尝试动态预算 Prefill |
| 没有活跃 Decode | 使用正常 Prefill 预算 |
| 没有等待 Prefill | Decode 或空闲，连续计数清零 |
| 保护准入失败 | 回退 Decode，保留保护条件，下次重算当前预算 |
| Prefill 成功提交 | 连续 Decode 计数清零 |

当前标准 Decode 每个真实请求每批计算一个 Token，所以 `decode_tokens` 使用活跃 Decode 请求数量。等待 Prefill 和未完成的 ChunkedReq 不计入，CUDA Graph dummy 请求也不计入。`decode_manager.inflight_tokens` 是用于 KV 准入的未来资源预留，不能替代这里的计数。

以目标256、下限64、保护上限256为例，16/64/192个活跃 Decode 分别产生240/192/64的保护预算。请求完成、Abort 或新请求进入 Decode 后，下一次保护轮次会根据新数量重新计算。

## 正确性和性能边界

- 预算是整个 Prefill Batch 的 `sum(req.extend_len)` 上限，缓存前缀不占预算。实际待计算量不足最低预算时无需填满；该下限不是每个请求的最小 Chunk 长度。
- 计算方法只读取配置和 Decode 集合长度，不探测前缀、不修改 LRU、引用计数或树结构，不分配资源。
- ChunkedReq 复用原 Table 槽位、缓存 Handle 和真实计算进度，允许下一次使用不同预算。
- 不强制把 Token 预算按 page_size 取整，页对齐仍由 KV 分配器处理。
- KV/Table 准入仍检查完整剩余 Prompt、输出预留和正在运行请求的需求。计算预算变小不能解决显存或 Table 不足。
- LPM、有效等待轮次 Aging、超过128退化FCFS保持原定义。Chunk-only轮次不累计新请求等待计数；更多分块可能延迟新请求准入。
- normal/overlap继续按成功提交更新连续 Decode 计数，无新增时间判断或 GPU 同步。相同输入状态使用确定的整数预算。

当前 mini-SGLang 分别执行 Prefill 和 Decode Batch。此公式使用 Decode 数量作为 Prefill 计算量的负载信号，不表示同一 Batch 包含两阶段，也不能视为 GPU 时间预算。下限可使 `decode_tokens + budget` 超过目标额度。

目标过大可能始终命中保护上限。例如真实 Decode 数量最多256、目标1024、保护上限256时，动态保护预算仍然为256。较小预算也可能增加 Chunk次数、重复 KV 读取和首次输出等待。当前只验证 CPU 调度语义和资源生命周期，真实 TPOT、TTFT、吞吐收益需要 GPU实验。

## 测试与复现实验

项目根目录运行测试：

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python -m pytest \
  tests/core tests/misc -q -o addopts='' --tb=short \
  --junitxml=artifacts/adaptive_prefill_budget/tests.xml
```

扩展现有 Benchmark，使用 `--compare-adaptive` 对比相同 K、保护上限、请求输入及 Decode负载下的固定/动态预算：

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_batch_schedule_policy.py \
  --compare-adaptive --seed 31723 --repeats 5 \
  --decode-caps 16 --protected-budgets 256 \
  --decode-loads 16 64 128 192 \
  --target-budgets 128 256 512 --min-chunks 32 64 128 \
  --output artifacts/adaptive_prefill_budget/benchmark.json
```

实验覆盖短/长/冷热混合Prompt、KV/Table紧张、Prefill-only、Decode-only；同组输入由相同随机种子生成，请求按固定 Batch序号到达，运行顺序打乱，每组重复5次。无模型推理：Engine只模拟进度和输出，使用真实阶段选择、准入、页分配、结果处理及资源释放。原生 Radix比较器预热后计时，测量阶段选择/准入CPU时间，不含页分配和模拟Forward。

JSON保存每次保护的活跃 Decode 数、选定预算、实际计算Token和准入结果，以及Chunk批数、等待批数、Decode输出批间隔、缓存复用、CPU时间和资源守恒。预算利用率是实际Token/预算的逻辑比例，不是GPU利用率。固定Batch到达和批间隔不能替代真实请求到达时间与GPU延迟。

LPM-Aging和page_size=4补充实验：

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_batch_schedule_policy.py \
  --compare-adaptive --seed 31723 --repeats 5 \
  --decode-caps 16 --protected-budgets 256 --decode-loads 16 192 \
  --target-budgets 256 --min-chunks 64 \
  --prefill-policy lpm-aging --aging-tokens-per-pass 8 --page-size 4 \
  --output artifacts/adaptive_prefill_budget/benchmark_lpm_aging_page4.json
```

以前的阶段调度实验文件保留。当前结果、分析和本次相对于执行前工作区的差异记录保存于 `artifacts/adaptive_prefill_budget/`。

为覆盖同一次运行中 Decode 数量自然减少的情况，增加了可选实验参数 `--stagger-decode-completions`，把初始请求的输出长度确定地分布在最大输出长度的一半到最大值之间，默认关闭。交错完成实验命令：

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_batch_schedule_policy.py \
  --compare-adaptive --stagger-decode-completions --seed 31723 --repeats 5 \
  --decode-caps 16 --protected-budgets 256 --decode-loads 16 192 \
  --target-budgets 256 --min-chunks 64 --workloads long mixed \
  --output artifacts/adaptive_prefill_budget/benchmark_decode_churn.json
```

最终测试249项通过（原有175项、新增74项），主实验1400次、页大小/LPM-Aging补充实验140次、交错完成实验40次，均排空并通过资源守恒检查。结果分析见 `artifacts/adaptive_prefill_budget/analysis.md`。
