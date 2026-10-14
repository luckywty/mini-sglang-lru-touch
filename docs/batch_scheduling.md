# 有次数限制的 Decode 优先与小预算 Prefill 保护

默认阶段策略仍为 `prefill-first`。新增 `decode-first-bounded` 在存在活跃 Decode 时优先推进生成；有 Prefill 等待且连续提交 K 个 Decode Batch 后，下一次阶段选择优先尝试小预算 Prefill。

## 两层配置

| 配置 | 默认值 | 作用 |
|---|---|---|
| `batch_schedule_policy` | `prefill-first` | Prefill/Decode 阶段选择，另一个选项为 `decode-first-bounded` |
| `max_consecutive_decodes` | 16 | 等待期间连续 Decode Batch 提交数，达到后优先尝试 Prefill |
| `protected_prefill_tokens` | 256 | 保护轮次整个 Prefill Batch 的新增计算 Token 总预算 |
| `schedule_policy` | `fcfs` | Prefill 内部的 FCFS / LPM / LPM-Aging 排序，语义保持原样 |

示例服务参数：

```bash
--batch-schedule-policy decode-first-bounded \
--max-consecutive-decodes 16 \
--protected-prefill-tokens 256 \
--schedule-policy lpm-aging \
--lpm-aging-tokens-per-pass 8
```

两个整数参数必须为正整数。配置对象与 CLI 均拒绝非法参数；未知阶段策略报错。16 和 256 是实验起点，不代表适用于所有模型和硬件的最佳值。

## 实际调用链

```text
normal_loop / overlap_loop
  → _schedule_next_batch
      → _select_next_batch：选择并构造一个 Batch
      → _prepare_batch：仅准备这个最终 Batch 的 KV 页和元数据
  → _forward：提交计算、写入 Token Pool、更新 Decode 集合
      → _record_batch_submission：更新连续 Decode 计数
  → _process_last_data：处理采样结果、缓存前缀、释放完成请求资源
```

默认分支保留原表达式：先调用正常预算 Prefill，再在返回 `None` 时调用 Decode。

启用新策略后的行为：

| 状态 | 下一批 |
|---|---|
| 没有等待 Prefill | Decode 或 `None`，计数清零 |
| 有 Prefill、没有活跃 Decode | 正常预算 Prefill |
| 两阶段都有且计数 < K | Decode 优先，无法构造 Decode 时回退正常预算 Prefill |
| 两阶段都有且计数 >= K | 优先尝试预算 `min(max_extend_tokens, protected_prefill_tokens)` 的 Prefill |
| 保护尝试无法构造 Prefill Batch | 回退 Decode；计数不清零，下一次继续保护尝试 |

一个 Decode Batch 包含多少会话都只计一次。计数只在 `_forward` 成功路径中更新；选择 Batch、准备 Batch、空转和失败提交不会增加计数。成功提交任意 Prefill Batch（包括 ChunkedReq 续算）后清零。等待队列清空后清零；Abort 清空后紧接新到达也不会继承旧计数。

计数使用提交顺序，不使用 GPU 完成时间，不增加 CUDA 同步。在 overlap 循环中，当前 Batch 先提交，上一批结果随后处理，所以计数不能放到结果处理位置。相同输入状态使用确定的整数决策；CPU 测试验证相同状态下的决策一致性，未执行多 GPU TP 实验。

## 预算、缓存与生命周期

保护预算限制整个 Batch 的 `sum(req.extend_len)`，已命中的前缀不占预算。例如 256 个 Token 可以用于多个短请求，也可以用于一个长请求的部分 Prompt。没有活跃 Decode 时恢复正常预算。

PrefillManager、PrefillAdder、DecodeManager 和缓存实现无需为该策略改动：

- 已准入 ChunkedReq 仍然优先续算，复用 Table 槽位、缓存 Handle 和真实计算进度。
- 新请求沿用完整剩余 Prompt、预期输出和 `decode_manager.inflight_tokens` 的资源检查。缩小计算预算不能绕过 KV 准入限制。
- Prefix Cache 的只读探测、计划刷新、准入后提交、引用计数与 LRU 更新均沿用现有代码。
- 实际选定 Batch 才进入 `_prepare_batch`，不准备并丢弃另一个阶段的 Batch。
- 完成和 Abort 沿用现有资源回收，测试检查物理页总量、唯一性、表槽唯一性和缓存引用归还。
- Decode-only 轮次不增加 Prefill 内部的 `waiting_passes`；LPM-Aging 和可选 FCFS 等待退化仍按原有 Prefill 调度语义工作。

## 保证的边界

K 限制到下一次优先尝试 Prefill 的间隔。KV/Table 等资源不足时，可以连续 Decode 超过 K；后续持续尝试保护，直到可准入。保护机会也可能被已有 ChunkedReq 消耗，不能把阶段保护视为每个新请求的准入保证。

Token 预算不等于 GPU 时间预算；长上下文与 Attention 后端可能改变耗时。小块续算会增加 Batch 数和元数据处理开销，需要结合模型实验调参。本实现不包含运行中 Kernel 抢占、KV 抢占/重算、混合 Prefill/Decode Batch、墙钟超时或自适应参数。

提交失败与模型异常沿用现有异常处理；新增计数不会将异常视为服务，不增加新的计算或分配事务回滚机制。

## CPU 测试与 Benchmark

在项目根目录执行：

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python -m pytest \
  tests/core tests/misc -q -o addopts='' --tb=short \
  --junitxml=artifacts/batch_scheduling/tests.xml

PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_batch_schedule_policy.py \
  --seed 31723 --repeats 5 \
  --output artifacts/batch_scheduling/benchmark.json
```

Benchmark 比较 Prefill-first、仅在 Benchmark 内实现的无次数限制 Decode-first，以及新策略。K 扫描 1/4/8/16/32，保护预算扫描 64/128/256/512。覆盖短 Prompt、长 Prompt、混合缓存冷热、KV 紧张、Table 紧张、Prefill-only 和 Decode-only。新策略结合 LPM-Aging 及 page_size=4 的补充实验命令：

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_batch_schedule_policy.py \
  --seed 31723 --repeats 5 --decode-caps 16 --protected-budgets 256 \
  --prefill-policy lpm-aging --aging-tokens-per-pass 8 --page-size 4 \
  --output artifacts/batch_scheduling/benchmark_lpm_aging_page4.json
```

使用真实 Scheduler 阶段选择、准入、物理页分配、`Req.complete_one`、结果处理和缓存释放。CPU Engine 模拟输出和进度，移除 pinning 并提供单 rank CPU 上下文；不加载模型，不运行 Attention、Sampler 或 CUDA Graph。每组原生 Radix 比较器预热后测量阶段选择耗时，KV 页分配和模型进度模拟不在该计时区间内。

请求按固定 Batch 序号到达，种子和到达序列固定，各策略运行顺序打乱；重复运行的调度语义和资源统计必须一致。统计 Prefill 首次执行/完成等待批数、初始 Decode 输出间隔批数、保护成功/拒绝、计算量、缓存复用和 CPU 耗时。Batch 数不是秒数；结果不能证明真实 TTFT、ITL 或吞吐提升。

Decode 输出间隔从启动阶段 Batch 0 的输出开始记录，包含其后第一段等待。KV 紧张负载保证每个请求单独可行，同时让初始 Decode 预留阻止冷请求立即准入，用于验证保护失败后的行为。

真实 GPU 验证需固定本地模型、采样配置、到达时间序列及负载，测量 TTFT、ITL 的 P50/P95/P99、吞吐和显存峰值，并分别测试 normal/overlap。当前交付的 GPU 指标标记为未执行。

结果保存于 `artifacts/batch_scheduling/`；本次修改相对于执行前已有本地改动的差异保存在 `changes.diff`，审计摘要保存在 `verification.json`。

本次执行结果：175 项相关 CPU 测试通过，770 次主实验和 105 次补充实验全部排空并通过资源检查。取舍与限制见 `artifacts/batch_scheduling/analysis.md`。
