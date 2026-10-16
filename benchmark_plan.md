# mini-SGLang：8GB 单卡 Benchmark 设计

设计日期：2026-10-14（Asia/Shanghai）。状态：**代码审计与实验规划，尚无本次 GPU 性能结果**。目标设备为另一台 RTX 5060 Ti 8GB，已由用户确认。未运行完整 Benchmark、未下载模型、未修改引擎或实现下文建议工具。

配套：[负载](workload_design.md)、[消融](ablation_matrix.md)、[执行](benchmark_execution.md)、[成果与图表](resume_metrics_plan.md)。这套方案重新依据源码制定；已有审计文档作为待核验线索，已有 Benchmark 方案作为参考，不作为真实性证明。

## 1. 审计范围与基线身份

项目根目录：`/home/wangwei/ai-infra/Promotion/mini-sglang-lru-touch`。

| 项目 | 本次核对结果 |
|---|---|
| 原版基线 H | 本地上游仓库 HEAD `9a91cfafe754aa85daee49998176275667eb58f2`，remote 为 `sgl-project/mini-sglang` |
| 当前实现 S | H 加未提交工作区；13 个 tracked 文件变化，687 行新增、72 行删除；另有未跟踪 tests、benchmarks、artifacts、docs |
| tracked diff SHA256 | `405adc646024b672c0bb10643c369bc94c309ddd47b068e69c0c2630f7b039fc`，由 `git diff --binary HEAD` 原始字节计算 |
| 源码集合指纹 | `62723b0c224c09f6d07fddebb8be1066cd70f2d381fae61e0fd91fe8275f7811`；120 个文件，算法见执行文档 |
| 差异性质 | core/message 部分差异只是注释；不能将所有 diff 行数算作优化工作量 |
| 已有证据 | 审计报告记录历史 CPU 回归与模拟；本次未重跑这些实验。`benchmarks/` 使用替身 forward，不是 GPU Serving 性能 |

本次直接读取当前实现及 `git show HEAD:<path>`，核对 PrefillAdder、Radix tree walk、阶段选择及预算公式。完整来源：[原审计](docs/mini_sglang_four_optimizations_review.md)、[prefill.py](python/minisgl/scheduler/prefill.py)、[scheduler.py](python/minisgl/scheduler/scheduler.py)、[cache.py](python/minisgl/scheduler/cache.py)、[radix_cache.py](python/minisgl/kvcache/radix_cache.py)、[CLI](python/minisgl/server/args.py)。

**H 与 S-default 必须分开。** S-default 虽为 FCFS / Prefill-first / adaptive=False，仍然运行新的两阶段缓存准入。H 才是原版 mini-SGLang。正式运行前固定 H 的完整 SHA，将 S 及所有未跟踪实验源码保存为不可变快照或独立实验 commit；单独记录 measurement patch 的 hash。当前主工作区不做 reset，不覆盖历史 artifacts。

## 2. 四项真实机制、假设与代价

| 代号 | 真实代码路径与控制方式 | 可检验假设 | 代价与适用边界 |
|---|---|---|---|
| A：Radix 两阶段准入 | `probe_req → refresh_probe → available_after_lock 检查 → commit_match → lock → touch`；**无现成关闭开关** | 在部分命中、KV 拒绝频繁时减少无效 split/touch；可能减少比较与调度成本 | 增加 probe 对象、父链遍历和版本刷新；不保证净 CPU 收益；不是新增 Radix Cache |
| B：Prefix 排序与有效轮 Aging | `fcfs / lpm / lpm-aging`；score=`cached_len_est + N*waiting_passes`，稳定 arrival order | 有排队与缓存竞争时改善复用；Aging 相对纯 LPM 改善 cold 等待 | B 使用 A 的只读 probe；chunks 始终优先；候选拒绝即 break；队列新请求数 >128 退回 FCFS，不能将退化点收益归给 LPM |
| C：Decode-first-bounded | `batch_schedule_policy`；Prefill 等待时累计 Decode **提交批次**，达到 K 尝试保护性 Prefill | 长 Prefill 与 Decode 竞争时降低 Decode ITL 尾部 | 可能增加新请求 TTFT；K 不是秒数，也不保证准入成功；不抢占正在执行的 GPU batch |
| D：自适应保护预算 | 只在 C 的保护性 Prefill 中启用；`b=min(U,max(floor,target-n_decode))`，`U=min(max_extend_tokens,protected_prefill_tokens)` | 活跃 Decode 数变化时改善 ITL/TTFT/吞吐取舍 | D 依赖 C；它是 Prefill-only batch 总新增 token 预算，不是混合 batch，不是每请求最小 chunk，也不改变完整未来 KV 预留 |

A 的旧路径会先匹配并改变树/LRU，再做两次容量检查；新路径先按锁后容量预测再采用。不能把当前 `match_req()` 直接当旧实现：当前 `_tree_walk` 的 touch 行为已变化。B 的 Aging 只在本轮确实准入至少一个 NEW 请求后，给仍等待首次准入的请求加一；仅续算 chunk、无预算、全拒绝不加。`fairness_max_wait_steps` 是另一项 raw 调度轮 FCFS fallback，核心矩阵固定为 0，避免混淆。

收益假设的反例也必须报告：无复用时 LPM 可能只增加 CPU 排序；低并发无排队时 B 可能无效；无阶段竞争时 C/D 不触发；大 target 在 16 活跃请求范围内可能令 D 恒等于 U；小动态 chunk 可能增加准备和 launch 成本。

## 3. 实际环境检查与目标机验证边界

本机只用于审计。沙箱内 NVML 被阻止；经批准在沙箱外进行只读探测后得到：

| 项目 | 本机观测（不能冒充目标机结果） |
|---|---|
| OS | Ubuntu 24.04.4 LTS，WSL2，kernel `6.18.40.1-microsoft-standard-WSL2` |
| CPU | AMD Ryzen 7 5800H；WSL 可见 4 逻辑 CPU |
| GPU / Driver | RTX 3060 Laptop GPU，6144 MiB；Driver 577.00；CC 8.6 |
| NVML 快照 | used 1339 MiB / free 4658 MiB；42°C、12.11W、SM 210MHz（空闲瞬时值） |
| PyTorch CUDA 内存快照 | free 5,370,806,272 B / total 6,441,926,656 B；与 NVML 不同时刻、不同可见口径，不能互相替代 |
| Python / PyTorch | `.venv/bin/python` 3.12.14；torch `2.9.1+cu128`，CUDA available=True |
| CUDA 工具链 / Runtime | `nvcc` 12.5.82；`torch.version.cuda=12.8`。二者不同；`nvidia-smi` 的 CUDA 标签也不是已安装 Runtime |
| torch 编译架构 | 包含 `sm_120`；沙箱内返回的空架构列表不能据此判定 wheel 不支持 GPU |
| FlashInfer | `0.6.16.post3`，沙箱外 import 成功；未安装独立 flashinfer-cubin / flashinfer-jit-cache distribution；不代表所有 kernel 已编译 |
| 其他版本 | transformers 4.57.3、triton 3.5.1、sgl-kernel 0.3.21、apache-tvm-ffi 0.1.14.post1、openai 3.24.0 |
| 模型 | 本地存在未量化 Qwen3-0.6B；另有 W4A16 目录，**不纳入对比**；未进行加载/forward |

目标机 Driver、Toolkit、可用显存、CPU 与 FlashInfer kernel 可执行性均待检查。RTX 5060 Ti 的计算能力为 12.0；CUDA 12.8 增加了 SM_120 编译支持，因此目标机不应沿用本机 12.5 nvcc 作为原生 sm_120 JIT 编译方案。选择经目标机 smoke 验证的 12.8 或更新兼容工具链，再锁版本；不凭 import 成功推断 kernel 可运行。[NVIDIA GPU 表](https://developer.nvidia.com/cuda/gpus)、[CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)。

**显式指定 `--attention-backend fi --page-size 1`。** 源码 `is_sm100_supported()` 使用 capability >=(10,0)，因此 sm_120 在 `auto` 下会选 trtllm，并可能把 page size 改成 64；不能把该架构判定等同于 trtllm 在本模型上的兼容性。`FIMetadata.__post_init__` 明确只接受 page size=1；FI 内部 wrapper 固定 backend="fa2"。这些是本仓库选择，不代表上游 FlashInfer 的全部能力。

## 4. 8GB 显存预算与正式规模

默认唯一模型为 **Qwen3-0.6B，BF16，TP=1**。本地 HF snapshot revision 为 `c1899de289a04d12100db370d81485cdf75e47ca`。本地另一副本的 config hash 相同，不意味着权重 hash 已相同；正式实验逐文件校验。

配置为 28 层、8 KV heads、head_dim=128。单 token KV：

`2(K,V) × 28 × 8 × 128 × 2(BF16 bytes) = 114688 B = 112 KiB`。

| 档位 | max running / graph max | KV token 容量（page=1） | KV 池近似大小 | seq override / max Prefill tokens | 用途 |
|---|---:|---:|---:|---:|---|
| G0 | 8 / 0，然后 8 | 8192 | 896 MiB = 0.875 GiB | 2304 / 256 | 目标机 smoke、编译、正确性、小规模显存验证 |
| G1 | 16 / 16 | 16384 | 1792 MiB = 1.75 GiB | 2304 / 256 | 主实验；client 并发 1/4/8/16/32，后两者按负载选择 |
| G1-small | 16 / 16 | 8192 | 0.875 GiB | 2304 / 256 | 如 G1 余量不足，整组对比共同切换；单独命名配置 |
| G2（可选） | 16 / 16 | 24576 | 2.625 GiB | 4352 / 256 或 512 | 仅在逐级验证后测 4096 输入；不是默认正式矩阵 |

实际额外分配一个 dummy KV page。权重文件 header 显示 BF16 payload：751,632,384 个存储元素，文件 1,503,300,328 B，约 1.40 GiB；含 tied 权重存储因素，不能将文件元素数当模型独立参数数或运行时权重占用。预留模型权重约 1.1–1.5 GiB **仅为估算**，另有加载临时张量、激活、logits、FI 的 128 MiB workspace、metadata、CUDA context、graph pool 和桌面占用。记录完整启动峰值与测量阶段峰值。

显存策略：目标 **整卡 free ≥1.5 GiB**，正式验证期间最低不得小于 1.0 GiB；NVML/WSL 不可见项明确标 NA，并用 `torch.cuda.mem_get_info()` 与进程 allocator 统计补充。轮询不能证明没有瞬时尖峰。若启动或负载峰值不满足余量要求，则降低整个比较组的 KV 容量/graph/并发重新校准，不为单一优化偷偷放宽预算。

用 `--num-pages` 固定容量，不依赖自动 `memory_ratio`：该比例相对启动前 free memory，扣加载模型估算后确定 KV，且在 attention workspace/graphs 创建前运行；显式 num-pages 时比例不再约束容量。默认 graph 可能到160；必须覆盖。CLI `--cuda-graph-max-bs 0` 才明确禁用；当前 helper 在 max=1/2 时仍构造 `[1,2,4]`，记录实际捕获列表，不能只看参数值。

I 指 tokenizer 后的真实输入 token 数。所有正式请求须 `I+O≤max_seq_len`；超长会被丢弃或缩短 max_tokens，不能计为正常完成。KV 准入预留完整剩余输入和最大输出，小 chunk **不等于**低未来 KV 需求。开放环 backlog 可在 CPU 排队，max running 限制 GPU 活跃槽，不能将 client 并发32解释为 GPU batch32。

目前审计报告已记录 page>1 时多个新请求尾页余量累计可触发分配断言；本次核对发现对应 token 估算与逐请求向上取整仍存在，未重跑复现。主方案 page=1 与已知失败条件不同，仍须做资源压力与完整性验收；不能宣称已修复。紧容量压力超出资源时必须记录拒绝/超时，不捕获 allocator 断言后继续沿用污染进程。

Qwen3-1.7B 等第二模型不进入最低成本计划；只有固定同精度后权重加载、graph、最坏请求负载均留足余量，才建立独立实验系列。禁止跨模型、量化、精度、后端或权重版本计算收益。

## 5. 指标的公共计时约定

客户端使用 monotonic `perf_counter_ns()`；服务端各进程在同一主机使用 monotonic clock，启动验证时钟域，跨机器只比较各自时长。通过 request uid 与 token index 关联，不相减不同时钟域的绝对时间。

每请求 i 保存：计划到达 `a_i`、client 真正发送前 `c0_i`、首个非空文本 `cf_i`、最后有效文本 `cl_i`、读到终止的 `ce_i`；server API 接收 `r_i`、scheduler 入队 `q_i`、首次准入 `h_i`、各 chunk/Decode 提交 `u_ik`、首/各输出可处理时刻 `s_ij`。`s` 放在现有 D2H completion 同步之后，表示后端 token ready/emit，含 CPU 结果处理，不命名为纯 GPU 完成时刻。

`I_i` 为实际 server tokenization 长度，`O_i` 为服务端产生的真实输出 token 数，另存可见文本对应计数、EOS/特殊 token 数。固定长度性能轨道使用 ignore_eos=True，仍核验最后 EOS 被 detokenizer 隐去等计数差异；输出吞吐主口径是生成 token，不能混用可见文本重分词数。O=1 的 TPOT/ITL 为 NA。

主报告采用**固定 trace 全部排空**：`T=max(ce 或失败终止时刻)-min(a)`，首个 a=0，对所有策略固定 N、到达序列和超时。成功集 S 为收到终止、输出数量/正确性校验通过的请求。失败的部分输出另外汇报，不混入成功吞吐。包含预定首发等待、尾部 drain 及失败等待，报告 injection 时长与 drain 时长。补充稳态时间窗报告必须独立：token 吞吐按窗口内事件计数、request 吞吐按窗口内完成数；到达 cohort 的延迟排空后另算，不与主报告拼接。

## 6. Serving 指标字典

分位数统一使用 NumPy `quantile(method="linear")`。下表的 P50/P95/P99 每次运行独立计算；失败另列，不当作零延迟。

| 指标 | 精确定义 / 单位 | 采集位置与统计口径 | 优化关联 / 简历适用性 |
|---|---|---|---|
| Output Throughput | `Σ(i∈S) O_i / T`，generated token/s | server token 计数关联 client 成功状态；分母为完整测量时段 | 核心；说明是否含 EOS/特殊 token；适合 |
| Total Token Throughput | `Σ(i∈S)(I_i+O_i)/T`，logical token/s | 实际输入长度，缓存命中仍是逻辑输入；**不是 GPU 实算 token/s** | B 的复用场景需同时报 output；可用但不宜单独写简历 |
| 实算工作量（补充） | `Σprefill 新增 tokens + Σdecode 真实请求步数`；除 T 得 model-forward token/s | 提交时统计、不含图 dummy；首输出由 Prefill 产生，不将 Prefill tokens 与全部 O 再相加 | 解释缓存节省，单位 token 非 FLOP；不直接作为用户吞吐 |
| Request Throughput | `N_success/T`，successful req/s | client 完整请求终态 | 同请求长度才可比较；适合 |
| Client TTFT-text | `cf_i-c0_i`，ms，按请求 P50/P95/P99 | 发起 HTTP 前开始；忽略 role-only、空 delta、finish、DONE；无文本则 NA 并记录 | 用户体感，含 tokenizer/排队/计算/传输/客户端解析；适合，注明文本口径 |
| Client TTFT-token-ack | `c_i,1-c0_i`，ms，按请求 P50/P95/P99 | ack/token一一对应验证后，以首个生成token的数据ack计时，允许其文本为空；与首个非空文本时间分别列出 | 用于标准生成首token与用户可见首文本的区分；适合但须注明边界 |
| Server TTFT-token | `s_i1-r_i` 或 `s_i1-q_i`，ms，二者分别命名 | API 接收/调度入队到后端首 token ready；不能与 TTFT-text 混在一列 | 用于定位 C/D 的 TTFT 代价；适合辅助说明 |
| ITL-server | `s_ij-s_i,j-1`，j=2…O，ms | 主报告 pool 所有有效 token 间隔的 P50/P95/P99；另报每请求 P95 ITL 的分布，避免长输出权重掩盖短请求 | C/D 主指标；称后端 token 间隔，不称网络到达 ITL；适合 |
| ITL-client-token-ack | 验证每个token ack一一对应后，`c_ij-c_i,j-1`，ms，pool P50/P95/P99及每请求分布 | 当前源码每个DetokenizeMsg对应一个UserReply，API每ack发一个data event（可空文本），再另发finish/DONE；校验计数/顺序后可按ack测量。不按非空文本块数计token；将来若合并ack则不可复原逐token到达 | C/D的网络与CPU交付效果；必须先用server日志对账，不能直接沿用现有tics |
| TPOT-server | `(s_i,O-s_i,1)/(O_i-1)`，ms/token | 每请求一个均值，再对这些均值求 P50/P95/P99；另存 token-weighted mean | 不是 pooled ITL 的分位数；C/D 需与 ITL 尾部并列；适合 |
| Client TPOT-token-ack | `(c_i,O-c_i,1)/(O-1)`，ms/token，每请求均值再求分位数 | 仅在ack/token映射对账通过后报告；否则`(cl-cf)/(O-1)`只叫文本交付归一化时长 | 与server TPOT分开；不以retokenization伪造 |
| E2E latency | `ce_i-c0_i`，s 或 ms（列名固定），请求 P50/P95/P99 | client 从请求前到消费完整终止；可另报最后文本时刻延迟 | 衡量整体公平和 TTFT/Decode 代价；适合 |
| Arrival-to-done | `ce_i-a_i`，ms | 显示 client 自身迟发开销；与 E2E 并列 | 诊断同机 CPU 竞争，不能把迟发藏起来 |
| Request Goodput-client | `Σ 1[success ∧ TTFT-text≤S_T(i) ∧ E2E≤S_E(i)]/T`，req/s | 预注册相同 SLO；终止/无文本失败均不达标 | 最适合 Serving 业务表达 |
| Goodput-server（补充） | `Σ 1[success ∧ TTFT-server≤S_ST(i) ∧ TPOT-server≤S_P ∧ request_P95_ITL≤S_I]/T`，req/s | 单独标明后端 SLO，不能称 client SLO；同时报符合条件请求的 output token/s | C/D 与公平权衡；适合但要讲清指标 |

预设 SLO 候选：short/cached TTFT-text 500ms，long 2000ms；E2E=`对应 TTFT 上限 + 50ms*(O-1)+500ms`；server TPOT≤50ms/token、每请求 P95 ITL≤100ms。**这是实验门槛候选，不是硬件性能预测。** 目标机 pilot 可在正式实验前统一修订一次并冻结；同时保留一组公共严格 SLO，报告 sensitivity 曲线，不能每个策略各定一套标准。Server TTFT SLO 若采用，单独预注册；不能照搬 client 门槛又称测量边界相同。

## 7. Scheduling 指标与最小采集点

| 指标 | 定义 / 单位 / 统计 | 现状与最小新增点 | 价值 |
|---|---|---|---|
| 首次 Prefill Waiting | `h_i-q_i`，ms，请求 P50/P95/P99/max，hot/cold 分组 | `add_one_req` 记 q；`_try_allocate_one` 成功记 h；现无时间戳 | B/C 公平；适合辅助简历 |
| Chunk 等待 | 已准入 chunk 两次可运行机会之间、扣除该 chunk 自身执行的等待；每轮 ready→submit 的 host wait 单列 | chunk 放回 pending 时记录 ready，下次提交记录；GPU ready 需 event 证据，host wait 不称纯设备空闲 | 防止把首次准入当完整 Prefill 完成 |
| Decode Waiting | host-ready→下一次 decode submit，逐请求逐步 ms；若提交早于上一步 GPU 完成，不能定义负等待 | `filter_reqs`/结果 ready 与 `_forward` 记录；精确设备可运行等待采用 `max(0,gpu_start_next-gpu_end_prev)` 仅诊断采样 | C/D；不是 ITL 同义词，也不能用 CPU enqueue gap 代替 GPU 间隙 |
| Scheduler Decision Overhead | `_select_next_batch` 入口至返回的 CPU wall time，µs/batch，mean/P95/P99；包括失败尝试 | `perf_counter_ns`，不包含 `_prepare_batch`、GPU forward 或结果同步；另存排序/probe 子区间 | A/B，适合 CPU 工程成果 |
| 准备与结果处理成本 | `_prepare_batch` wall time、既有 D2H 等待时间分别记 µs | 采样到预分配缓冲；异步 CUDA API 时间不是 kernel 耗时 | 解释小 chunks 的成本 |
| Prefill/Decode 次数与比例 | 实际提交 `N_P,N_D,N_P/(N_P+N_D)`；另记 attempted/failed protected Prefill | `_record_batch_submission`、保护尝试分支；H 同边界采集 | C 的路径证据；不把一次尝试当服务 |
| Chunk size | 每 Prefill batch `Σ real_req.extend_len`；每请求 chunk size 另报 histogram/P50/P95；tokens | 在 forward 前快照，不能在 `complete_one()` 后读变更字段 | D；min_chunk 是 batch budget floor，实际单请求可更小 |
| Starvation | 等待首次准入超过预注册 `W*=5s` 的比例；测量结束仍未准入比例；资源拒绝次数；cold max/p99 wait | 由 q/h/终态/队列事件统计；未完成样本记至少已等待时间，下界或删失 | B/Aging 主证据；有限 trace 无法证明无限时间无饥饿 |
| 最大等待 | 所有完成等待的 max 与所有未完成 elapsed max 分开，另存有效 waiting_passes/raw steps | 不能把删失样本当已完成 quantile | 高负载 fairness 核心 |

服务端排队=`q→首次准入`，完整 Prefill 墙钟包含后续 chunk 排队、metadata、GPU 和 CPU；实际计算延迟只用 CUDA events/profiler 表示。由于 overlap，不能简单把 CPU/GPU 区间相加还原 E2E，也不能用 `client TTFT - scheduler wait` 命名为纯计算。

## 8. Cache、显存和失败指标

| 指标 | 定义 / 单位 | 可观测性与所需最小计数器 |
|---|---|---|
| Prefix token hit ratio | `Σ C_i / Σ(I_i-1)`，%，C 为成功**首次准入**时真实复用长度；同时报 `ΣC/ΣI` | handle.cached_len 已有，增加 admission 累计；排序 probe 值不能算命中；按已准入集合报告，不受最终成功筛选偏差影响 |
| Prefix request hit ratio | `#admitted(C_i>0)/#admitted`，% | 同上；少量公共模板产生的微小命中也算 request hit，需同时看 token ratio |
| Cached / uncached input | `C_i`；`I_i-C_i`，token；实际 chunk extends 总和与后者核对 | 首次 admission 和每次 chunk submit 计数；chunks 不重复算 cache hits |
| Shared-prefix recompute | 对 trace 指定的可复用公共区长度 L_i，`max(0,L_i-min(C_i,L_i))`，tokens | 需要 workload 的真实 token LCP 标签；首次出现/未预热的 compulsory compute 单列；`I-C` 不能全叫 recompute |
| KV physical occupancy | `(num_pages-len(free_slots))/num_pages`，% ，时间加权 mean/max/P95 | CPU 管理状态已有；包含 active 私有页和 cache，排除 dummy。tree 的 evictable+protected 不代表全部物理占用 |
| Cache 可淘汰/受保护量 | `size_info.evictable_size/protected_size`，token | 已有 CPU 元数据；与活跃私有 KV 分开解释，避免共享页按引用次数重复计算 |
| Eviction | evict API 调用次数、实际移除 node 数、实际 evicted token/page 数 | 在 `evict` 已有循环增加整数；一次调用可能移除多个节点且超额释放，不能只计请求 size |
| Probe 成本与副作用 | probes、version invalidations、token compare 次数、split/touch/锁定数；拒绝时树/LRU 改变量 | 诊断模式采集；全树快照只在 CPU/少量 correctness probe，不能每请求打印或遍历全树 |
| Peak GPU memory | 进程 peak allocated/peak reserved，MiB；整卡 sampled max used/min free，MiB | PyTorch stats 在 worker 读；NVML 外部0.5–1s。allocated/reserved 有包含关系，不相加；启动与 measurement 分开 reset/报告 |
| OOM / failure / timeout | OOM run 数/attempted runs；failed requests/全部 offered；timeout/全部 offered | client 和 server 错误台账新增；GPU OOM 与逻辑 KV allocator assertion 分开；无请求终态也列失败 |

## 9. 统计、控制变量与验收

每个核心配置至少 **5 次独立重启重复**，seed block 为 31723…31727，同 block 各策略使用同一 trace 文件及到达时间。每组记录单次值、中位数、IQR/MAD、min/max；尾延迟分位数与运行间方差是不同统计层次。以整次运行为单位配对 bootstrap，给 throughput ratio / latency reduction 的95% CI；n=5 的 CI 不稳定，预先约定若方向不稳或 CI 跨0则追加到10次并保留前5次。不能将大量相关 token 间隔当独立重复。

每个 block 随机平衡策略顺序，5次可采用轮转/反向顺序；每次重启到同样 cache 条件，预热后温度处于预定稳定区间。缓存、温度、随机种子不可与策略永久绑定。正式 trace 与调参 trace 分离，留出 seeds31733…用于 pilot；负载强度只按 H 的 pilot 统一确定，冻结后不按各优化自身饱和点对比。

核心 tail 实验尽量每 run ≥1000 请求；hot/cold 分组尾部各 ≥1000 才将该组 P99 作为主要结论。低成本组至少500：尾部只有约5个点，只能探索性报告；按预注册增加 cohort 或更换主指标为P95，不用合并5次伪装1000独立请求。ITL 需同时给请求数、间隔数与每请求分布，不能靠一条超长输出满足尾部证据。

所有模型、tokenizer、dtype、backend、page/KV容量、graph捕获列表、max running、seq限额、请求输入、采样参数、timeout和SLO一致。性能轨道 greedy/ignore_eos=True；另用少量真实文本、ignore_eos=False 做正确性验收。采样精度与权重从不作为调度优化变量。

吞吐提升=`(optimized/baseline-1)*100%`；延迟降低=`(1-optimized/baseline)*100%`。对每个配对 run 算改善率并给中位数/CI，同时显示两组绝对值。不同 workload、负载、长度、显存档位、cache 条件或计时域不得交叉相除；组合提升不累加单项百分比。

client/server 同机时，预先 tokenization、关闭逐 token tqdm、复用 HTTP 连接、禁用 SDK 自动重试，记录 event-loop lag、发送延迟 `c0-a`、进程 CPU 与上下文切换。以静态/模拟 SSE 服务验证 generator 能承受预计峰值2倍；在固定策略/trace上比较 client 核亲和设置或第二客户端主机。预设 client lag P99<1ms且<目标最小 ITL 的5%（取更严者），CPU持续占满或换亲和后主指标变化>3%则判为 client-bound，修正环境后整组重跑。不得通过 semaphore 把开放环变成闭环后仍宣称同一 offered load。

GPU 每0.5–1s记录温度、功耗、SM/显存频率、利用率、显存和 throttle 信息；WSL不可见字段写NA，并记录桌面/宿主负载。不要求消费卡强锁频；用相同电源/散热设置、热稳定与配对随机顺序控制。失败/超时/OOM/热不稳均保留台账；可按预注册规则标记污染运行并追加替代，但原记录不能删除。

## 10. Instrumentation 的最小范围

现有引擎没有完整 request metrics。下阶段只建议 benchmark collector、CPU计数器/时间戳、离线聚合，不在本阶段实现。所有 H/S 变体使用同一观测补丁，在 run 边界落盘。

1. API 请求 uid 与客户端返回的 `cmpl-{uid}` 建映射；若需自定义 request ID，在 API 边缘加映射即可，不改采样链。
   当前`tokenizer/server.py`中DetokenizeMsg→UserReply为一一映射，API首个role字段与第一条token ack同事件发送，没有独立role-only前导事件。collector可将`finish_reason is None`的数据事件作为token ack候选（即使content为空），先与server实际token序列对账；排除最终finish/DONE。非空文本事件仍单独用于TTFT-text。文本一次flush多个token的字符不等于ack本身包含多个新token；未来改协议时重新验映射。
2. scheduler 用预分配数组/有界 ring buffer 记录 q/h、batch phase、真实 sizes、预算、n_decode、K状态、拒绝原因和既有 D2H 完成后的 token index/timestamp；记录 dropped events，发生丢失则详细指标无效。
3. token记录放在现有已完成的 CPU 结果路径，不新加 `.item()`、`.cpu()`、`torch.cuda.synchronize()`。只计 CPU metadata；实际 token IDs 可复用现有结果，不新增 GPU→CPU 拷贝。
4. CUDA events 仅单独诊断运行采样，例如每32批记录一对，末尾统一查询；不每步 synchronize。主 Serving 比较不用 profiler 计时。
5. worker 在测量边界 reset/read allocator peak；系统遥测由独立低频进程收集。进程重启才能可靠定义 cold cache，当前 `RadixPrefixCache.reset()` 未实现。
6. 在固定 H 或 S-default 上对 collector关闭/仅汇总/详细模式各做配对开销校准。若吞吐偏差>1%或目标尾延迟偏差>2%，降采样、缓冲或把详细诊断另跑；保留校准数据。精确 ITL 仍需完整 token 时间戳，不能用抽样还原其P99。

优先输出三条证据链：**阶段调度→Decode尾部与Prefill代价；缓存排序/Aging→复用与cold公平；动态预算→相对固定小预算的增量权衡**。A 的独立工程证据作为补充；没有测出GPU收益时，不把它包装成吞吐提升。
