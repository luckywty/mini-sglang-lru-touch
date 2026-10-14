# 实施和结果分析

本次生产改动限于 `scheduler/scheduler.py`、`scheduler/config.py`、`server/args.py`。执行前保存了三个文件快照和全部 25 个已有本地改动文件的 SHA-256；任务范围外的已有文件内容保持一致。修改相对于执行前工作区的差异见 `changes.diff`，不是相对于 Git HEAD 的全部历史改动。

## 实现

- 默认 `prefill-first` 保留原来的 Prefill 调度表达式和预算。
- 新增 `decode-first-bounded`：等待队列存在时连续成功提交 K 个 Decode Batch 后，优先尝试一次小预算 Prefill。
- 保护预算为 `min(max_extend_tokens, protected_prefill_tokens)`，是整个 Batch 的新增计算 Token 总预算；命中前缀不占该预算。
- 计数在 `_forward` 成功路径更新，成功 Prefill 清零，等待队列清空清零。Abort 清空后同批新到达不会继承历史计数。保护失败不会清零或停止 Decode。
- 选择一个最终 Batch 后才准备 KV 页和计算元数据。ChunkedReq、Table 槽位、缓存 Handle、Decode 资源预留、实际准入和缓存锁定流程均复用已有实现。
- Prefill 内部 LPM-Aging 仍按原有效等待轮次累计，Decode 轮次不增加 `waiting_passes`。

## 验证

相关 CPU 测试最终结果：**175 passed，0 failed，0 error，0 skipped**，耗时 10.46 秒。包含原有 106 项和新增 69 项。命令及策略语义见 `docs/batch_scheduling.md`；JUnit 结果在 `tests.xml`。

新增测试覆盖策略默认值和非法参数、K-1/K/K+1、每个 Batch 只加一次、选择或失败提交不计数、保护失败重试、预算上限、缓存前缀不占计算预算、多个请求共享预算、无竞争恢复正常预算、KV/Table/预算不足、ChunkedReq 跨 Decode 续算、Abort 清空后新请求、normal/overlap 提交顺序、超过 128 个候选和确定性阶段选择。默认策略与原表达式执行完整请求生命周期的 Batch/输出轨迹一致。

首次 CPU 测试调试中发现模拟环境缺少 TP 上下文、`torch.tensor` 的 pinning 替换和 Engine 的 page_table 字段，均在测试/Benchmark 支架中修复。沙箱限制阻止原生 Radix 缓存锁写入时，停止该次运行并获准在可写编译缓存的环境重新执行。最终测试无失败项。

新增 Python 文件和生产修改使用 Python 3.10 语法解析验证；格式整理前后 AST 完全一致。GPU Kernel 测试及 `test_scheduler.py` 中需模型的手动集成脚本未执行。

`git diff --check` 对整个已修改的 Scheduler 文件报告 3 处执行前已存在的行尾空白，保持原有内容；相对于本次执行前快照的新增差异和全部新文件无行尾空白。

## Benchmark 方法

环境：AMD Ryzen 7 5800H，WSL2，4 个可见逻辑 CPU，Python 3.12.14，Torch 2.9.1+cu128，1 个 Torch 线程。

实际使用 Scheduler、PrefillManager、DecodeManager、RadixCache、TableManager、KV 分配和请求完成释放。CPU Engine 仅模拟进度与输出；无模型、Attention、真实采样或 GPU 推理。统计的 CPU 时间只覆盖阶段选择和准入，不含 `_prepare_batch` 与模拟 Forward。

请求按固定 Batch 序号到达，所有策略使用相同的种子和输入序列。默认实验初始 4 个 Decode 请求，各生成 128 个 Token；16 个到达批次，每批到达 2 个新请求；新请求输出 8 Token，冷热 Prompt 分别为 65/1024 Token。热前缀预热 64 Token。正常 Prefill 预算 8192。Decode 输出间隔从启动时 Batch 0 的第一个输出开始计算，包含 Prefill-first 初期连续 Prefill 导致的等待。

主实验：7 个负载 ×（2 个基线 + 5 个 K × 4 个保护预算）× 5 次重复 = **770 次运行**。补充实验使用 LPM-Aging、N=8、page_size=4、K=16、保护预算 256：7 个负载 × 3 个策略 × 5 次重复 = **105 次运行**。

**875 次运行全部排空**，通过 Table 槽位归还及唯一性、非根缓存节点引用为零、protected_size=0、物理页总量和唯一性检查。重复实验的非计时指标一致。原始数据分别在 `benchmark.json` 和 `benchmark_lpm_aging_page4.json`。

## 主要结果

以下为主实验，Prefill 内部为 FCFS；新策略固定 K=16、保护预算 256。所有等待和输出间隔均以 Batch 计数，不能解释为真实毫秒或 TTFT。

| 负载 | 策略 | 首次 Prefill 等待 P95（批） | Prefill 完成等待 P95（批） | 初始 Decode 输出最大间隔（批） | 等待期间最大连续 Decode | CPU 调度中位数（µs） |
|---|---|---:|---:|---:|---:|---:|
| 短 Prompt | Prefill-first | 0 | 0 | 17 | 0 | 4.19 |
| 短 Prompt | 无限 Decode-first | 127 | 127 | 1 | 127 | 2.80 |
| 短 Prompt | 有限 Decode-first | 16 | 16 | 2 | 16 | 2.76 |
| 长 Prompt | Prefill-first | 0 | 0 | 17 | 0 | 4.23 |
| 长 Prompt | 无限 Decode-first | 139 | 139 | 1 | 127 | 2.85 |
| 长 Prompt | 有限 Decode-first | 145 | 145 | 2 | 16 | 3.16 |
| 混合冷热 | Prefill-first | 0 | 0 | 17 | 0 | 4.23 |
| 混合冷热 | 无限 Decode-first | 127 | 127 | 1 | 127 | 2.75 |
| 混合冷热 | 有限 Decode-first | 131 | 132 | 2 | 16 | 3.16 |

短请求实验中，阶段保护减少相对于无限 Decode-first 的 Prefill 等待，并降低相对于 Prefill-first 的最大 Decode 输出批间隔。该结果证明阶段调度的取舍，不能证明 GPU 延迟或吞吐改善。

长 Prompt 和 FCFS 混合负载中，已有长请求的 ChunkedReq 先续算，保护轮次可能仅推进其一部分 Prompt；后续请求仍然排队。K=16/256 虽然限制了阶段之间的 Decode 连续批数，首次 Prefill 等待 P95 却可以高于无限 Decode-first。全部请求排空后的 Prefill 计算量相同：长 Prompt 32768 Token；混合负载 9239 Token，实际复用缓存 1472 Token。额外分块改变阶段交错与 Batch 数，不减少模型所需计算量。

混合负载参数扫描的两个例子：K=1/预算512 的首次等待 P95 为 19 批；K=1/预算64 为 238 批。参数对阶段机会和续算速度的作用不同，不能简单认为 K 越小或块越小越公平，也不能把 16/256 当作通用最优值。

LPM-Aging/page_size=4 补充实验中，短 Prompt 的首次等待 P95 为 16 批，混合负载为 130 批，阶段最大连续 Decode 均为 16。该补充实验用于验证组合运行和页大小兼容性；由于同时改变了页大小和内部策略，不据此归因某一项性能收益。

## 资源受限及开销

| 负载 | 策略 | 首次 Prefill 等待 P95（批） | 最大连续 Decode | 整批 Prefill 尝试失败次数 | CPU 调度中位数（µs） |
|---|---|---:|---:|---:|---:|
| KV 紧张 | 无限 Decode-first | 352 | 127 | 0 | 2.58 |
| KV 紧张 | 有限 Decode-first | 352 | 127 | 111 | 3.02 |
| Table 紧张 | 无限 Decode-first | 169 | 127 | 0 | 2.80 |
| Table 紧张 | 有限 Decode-first | 169 | 127 | 111 | 10.58 |

资源不足时，K=16 后的 111 次保护尝试均失败；直到已有 Decode 释放必要资源后请求才获得服务。排序和小预算无法产生额外 KV 或 Table 槽位。这些结果直接验证 K 只提供优先尝试的机会，不能构成资源受限情况下的服务上界。

失败后的逐批重试有 CPU 开销，Table 紧张场景相对于无限 Decode-first 的调度中位数从约 2.80 µs 增至 10.58 µs。当前保持简单、确定的重试语义；后续可以在明确资源变化事件后重试，需同时处理缓存可用量、表槽释放和多 rank 一致性。

Prefill-only 负载使用正常预算，32 个请求在 16 个 Prefill Batch 中完成；Decode-only 负载为 127 个 Decode Batch，所有阶段策略轨迹一致。默认兼容性另由完整生命周期测试验证。

## 限制和后续工作

- 没有可用本地模型权重，本次未执行真实 GPU TTFT/ITL/吞吐实验；CPU Batch 间隔不能替代这些指标。
- 未执行多 GPU TP 运行；决策只使用确定的整数状态，CPU 测试验证相同输入状态下的结果一致。
- 保护成功可能只是 Chunk 续算，不能保证某个新请求获得准入。原有队首资源阻塞语义保留。
- Token 预算不是 GPU 时间预算。后续应结合模型、上下文长度、真实请求到达时间、normal/overlap 模式调参。
- 失败重试节流、墙钟等待超时、混合阶段 Batch、实际 KV 抢占和自适应预算尚未实现。

可用于项目记录：在 mini-SGLang 中实现可配置的有限 Decode 优先与小预算 Prefill 保护，保留原有默认调度和缓存准入语义；新增 69 项 CPU 测试并完成 875 次可复现实验，量化阶段公平性、分块续算与资源受限重试的取舍。
