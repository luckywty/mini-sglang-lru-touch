# 四项优化的 A/B Benchmark 方案

依据：[审计报告](mini_sglang_four_optimizations_review.md)。以下分为已有CPU实验的复现命令与尚未执行的GPU实验计划。没有把计划写成实测结果。本次审计不新增压测代码或业务埋点。

## 1. 先统一实验条件与指标

保存Git HEAD、完整dirty diff、相关文件SHA、模型权重/Tokenizer版本、设备型号/显存、驱动/CUDA/PyTorch版本、TP数、Attention backend、page_size、num_pages、max_running_req、正常预算、所有策略参数与随机种子。当前四项优化未提交，只有HEAD不能唯一表示实验实现。

优先在能容纳完整权重与目标KV的单GPU模型上运行，例如当前项目支持的Qwen3-0.6B，但**尚未在本任务下载模型或运行GPU**；本机小显存的结果不能推广到大模型/多卡。先TP1，再以同权重TP2验证状态一致性；保持正常/overlap配置各组一致，最后另做normal/overlap专项。

测量流程：模型/Kernel/Graph预热 → 使用一致的显式prefix预热 → 重放固定请求trace → 收集所有请求完成/超时/错误 → 检查资源回收。每组至少5个seed，AB顺序随机；不要让前一组的缓存残留成为后一组优势。冷启动、warm-cache、cache-churn分开统计；冷启动模型加载不混入稳态TTFT。

开放环按相同真实时间发送Poisson/固定trace，设到达速率为baseline稳定容量的0.5/0.8/0.95/1.1倍；1.1倍组固定窗口并报告未完成请求，不靠排空结束时间掩盖过载。闭环固定并发另做吞吐实验，不能拿策略相关完成速度自动改变到达流的闭环结果证明排队公平性。

| 指标 | 推荐定义/采样方式 |
|---|---|
| TTFT | 客户端发起请求时间 → 首个有效内容Token到达；记录连接/传输及server队列分解 |
| TPOT | 每请求`(最后Token时间-首Token时间)/(实际输出Token数-1)`，输出<2时不计算；再做请求级分位数 |
| ITL | 相邻真实输出Token的时间差；逐请求p95/max及全体分布，不用一次SSE chunk强行当一个Token |
| Throughput | 同一稳态窗口内完成requests/s、实际输出tokens/s；输出实际长度而非max_tokens或事件条数 |
| Goodput | 预先声明TTFT/TPOT SLO，窗口内满足SLO且成功完成的requests/s |
| GPU阶段耗时 | CUDA事件/Profiler测Prefill、Decode分别的持续时间，记录上下文与计算Token；用一致埋点，避免每批同步改变overlap |
| Scheduler overhead | CPU候选生成、probe/refresh、准入、prepare、结果处理分别计时；报告p50/p95/p99和每实际计算Token成本 |
| Prefix/KV | 真实准入cached_len、匹配率、prefix复用比例、free/protected/evictable页、重复尾页/碎片、evict次数、峰值占用 |
| Fairness | hot/cold分别的首次准入与TTFT分位、最大等待、完整Prompt完成、未服务/超时比例；同时记录被拒原因、effective passes、K状态 |

现有`python/minisgl/benchmark/client.py`可参考`benchmark_trace:L287`实现定时trace，但**不能直接把原日志口径当本方案的TTFT/TPOT**：`benchmark_one:L235`的起始tics在`await chat.completions.create`返回后设置，对所有流事件记时间；`process_benchmark_results:L320`将首流间隔叫TTFT，后续事件间隔汇总叫TPOT，未保证每条事件都是一个Token。正式实验需以发送前时间和有效内容/实际Token计数重核口径；本次未修改该客户端。

## 2. 通用 workload 矩阵

| 负载 | 并发/初始Decode | 输入Token | 输出Token | Prefix重复设置 | 用途 |
|---|---|---|---|---|---|
| short-only | 16/64/128/192 | 128或512 | 16/128 | 0%、50%、90%请求共享前缀 | 小请求CPU成本/阶段响应 |
| long-only | 16/64/128 | 4096或8192 | 128 | 0%、50% | Chunk与Prefill突刺 |
| hot/cold混合 | 32/64/128；待准入数8/32/128 | hot=前缀512/2048+唯一suffix32；cold=1024/4096 | 16/128 | hot请求占25%/50%/75%/90% | LPM复用与cold公平 |
| 持续fresh hot | 起始1个cold+稳定Decode；每秒连续hot | cold4096；hot共享2048+suffix32 | 16 | hot到达占90%，缓存预热 | cold长期等待与Aging超越 |
| Decode-heavy | 初始Decode16/64/128/192，周期加入Prefill | Prefill1024/4096/8192 | 初始Decode512/1024；新请求16/128 | 0%、50%、90% | K保护与自适应预算 |
| Queue边界 | 同时待准入127/128/129/256 | 512/2048 | 16 | 0%、90% | FCFS降级与CPU成本 |
| 内存/Table压力 | active逼近Table上限；KV余量10%/2% | 5/17/1024/4096 | 1/16/128 | 共享4/8/512；page1/4 | 拒绝安全与尾页舍入 |
| Decode自然完成 | 初始16/192 | 新请求1024，另做mixed | 初始输出64–128交错 | 50% | 同次运行live count变化 |

“90%重复”指90%请求共享预定义前缀组；另报告`matched_tokens/全部Prompt tokens`，不能把请求重复率当Token命中率。输入经过服务端chat template/Tokenizer后核验实际长度与共同Token前缀，唯一suffix避免整个Prompt意外全重复；Prefix长度按page对齐。

真实活跃请求不能超过Table容量。默认max_running_req256，初始192时仅剩有限新槽；要隔离计算保护与资源压力，宽资源组单独增加可运行槽与KV容量并记录，不能默默更换模型/内存。队列129测试要确保是“待首次准入”129，不能用总HTTP并发129代替。先验证所有单请求input+output<=模型max_seq_len。

## 3. 优化一：两阶段匹配的独立归因

| 项目 | A baseline | B optimized |
|---|---|---|
| 匹配路径 | HEAD风格match→两次容量检查→lock/unlock | 当前probe/refresh→预测锁后容量→commit/lock/touch |
| 其他策略 | FCFS、Prefill-first、adaptive关闭、同预算 | 完全相同 |
| 资源条件 | 宽容量、KV拒绝80%、Table满、partial match、共享路径 | 完全相同 |
| 输入/并发 | Prompt513/2049，复用前缀512/2048；输出16/128；队列8/32/128 | 完全相同 |

当前没有两阶段关闭参数。严格单因素实验需**后续另建最小ablation分支或测试适配器**，只替换匹配准入路径，并记录patch。直接比较HEAD与当前全部功能会混入其他变更；`--cache-type naive`去掉整个Prefix Cache，也不是“两阶段”baseline。本次没有实现ablation。

CPU先测：拒绝前后树快照、split/touch/ref变化次数、Token边比较次数、父链访问次数、probe/refresh/commit/lock/索引复制时间；覆盖一批第一个partial split令后续计划失效与不相交分支结构变化。Benchmark计时应包含不同路径等价的工作范围。

GPU再测TTFT、CPU调度尾延迟、前缀实际命中与后续evict模式；该项主要目标是副作用正确性与CPU工作控制，不应只凭吞吐波动认定“更快”。异常容量场景先复现/验证拒绝，不能吞掉断言后继续算吞吐。

## 4. 优化二：LPM与Aging

依次做三个对照，保持阶段Prefill-first、adaptive关闭、normal预算8192、Table/KV宽容量、raw阈值0、page1；然后page4与紧缓存复验。

| 对照 | Baseline参数 | 优化参数 | 验证贡献 |
|---|---|---|---|
| LPM | `--schedule-policy fcfs` | `--schedule-policy lpm` | Prefix排序是否提高有eviction负载下的复用/吞吐，及CPU成本 |
| Aging | `--schedule-policy lpm-aging --lpm-aging-tokens-per-pass 0` | N=8/32/128，其他相同 | 分数老化对cold等待/hot代价；N=0应等同LPM |
| 队列FCFS退化 | 阈值2048（实验中关闭过长退化） | 阈值128 | queue127/128/129/256时开销与命中/fairness折中 |
| raw兜底 | `--fairness-max-wait-steps 0` | 16/64 | 与effective Aging分离的原始调用步FCFS退化 |

先在小于阈值的持续hot负载测N，避免Aging结果被队列FCFS掩盖。设置cold长度与资源足够可独立接纳，再增加Table/KV压力区分“排序已超越”和“实际准入失败”。缓存热度稳定组可以验证N超越公式；churn组另量化排名估算与准入重验证的差异。

按hot/cold分组报告TTFT p50/p95/p99/max、首次准入/effective passes、完成率、最长未准入时间；同时报告hot退化、Prefix实际命中、CPU probe数/耗时和总tokens/s。不能仅选cold改善组，不报告hot/吞吐损失。

## 5. 优化三：有限次数Decode优先

保持内部FCFS、adaptive关闭、normal8192、资源充足，比较默认Prefill-first与bounded。K取1/4/8/16/32，保护上限64/128/256/512；随后固定已选参数再用LPM-Aging组合复验，避免一次同时改变两层策略。

| 组 | 配置 | 说明 |
|---|---|---|
| A | `--batch-schedule-policy prefill-first` | 默认production基线 |
| B | `--batch-schedule-policy decode-first-bounded --max-consecutive-decodes 16 --protected-prefill-tokens 256` | 实际可运行新策略；扫描K/预算 |
| U | 无上限Decode-first | 现有CPU Benchmark-only基线；production CLI没有该策略。GPU若要运行需后续独立实验适配，不能编造启动参数 |

Decode-heavy和long burst负载主要检查ITL p99/max与新Prefill TTFT；短/长Prompt分别报告。设持续固定到达窗口，保留所有未服务样本以发现真正长期等待。

同时记录counter轨迹、达到K后的prefill尝试/成功/失败、失败原因、实际连续Decode最大值。资源宽时验证下一选择在K后尝试保护；资源紧时允许超过K并解释准入拒绝，不把阈值当服务保证。Abort清空、首个新到达、chunk-only保护和实际overlap分别专项。

## 6. 优化四：自适应保护预算

只对比bounded的adaptive on/off，固定K16、normal8192、保护上限256、内部FCFS、raw0；on加入`--adaptive-prefill-budget --target-batch-tokens 256 --min-chunk-tokens 64`，off省略开关。GPU阶段选择和内部排序完全相同。

活跃Decode负载取16/64/128/192，target128/256/512，floor32/64/128；排除floor>target或>保护/normal的非法组合。另用target1024检验上限饱和的无变化对照。无Decode与纯Prefill验证normal预算恢复；纯Decode验证不额外探测。

同一运行里交错初始Decode输出长度64–128，让live count真实下降，记录每次保护`(count,budget,actual_extend_tokens,chunk_num,duration)`。有GPU后建立“context×Decode数量×Prefill Token数→阶段耗时”响应曲线，判断简单count是否足够。

核心报告为ITL/TPOT尾部收益对TTFT/throughput/更多chunk的代价；同时看初始Decode的per-request平均与长尾，不能仅用平均掩盖突刺。若GPU时延无下降、TTFT升高，就如实报告该参数点无收益；不要把budget缩小直接写成性能提升。

## 7. 配置示例与组合ablation

以下是服务启动**计划示例**，未执行；使用已有入口和实际参数名。`/path/to/local/model`需替换为已验证可用的本地权重，不触发本任务下载。

```bash
# 默认基线
PYTHONPATH=python .venv/bin/python -m minisgl --model /path/to/local/model \
  --cache-type radix --page-size 1 --max-prefill-length 8192 \
  --schedule-policy fcfs --batch-schedule-policy prefill-first

# 完整组合
PYTHONPATH=python .venv/bin/python -m minisgl --model /path/to/local/model \
  --cache-type radix --page-size 1 --max-prefill-length 8192 \
  --schedule-policy lpm-aging --lpm-aging-tokens-per-pass 8 \
  --lpm-queue-threshold 128 --fairness-max-wait-steps 0 \
  --batch-schedule-policy decode-first-bounded --max-consecutive-decodes 16 \
  --protected-prefill-tokens 256 --adaptive-prefill-budget \
  --target-batch-tokens 256 --min-chunk-tokens 64
```

注意正常预算CLI为`--max-prefill-length`或`--max-extend-length`，字段为max_extend_tokens；现有help中的“max-extend-tokens”不是实际CLI选项。来源：`server/args.py:L178–185`。normal-loop组启动进程前设置`MINISGL_DISABLE_OVERLAP_SCHEDULING=1`，overlap组省略该环境变量，分别重复矩阵；不要跨组改变后混算数据。来源：`env.py:L51/L69`、`scheduler.py:L170`。

| Ablation组 | 缓存准入 | 排序 | 阶段 | 保护预算 |
|---|---|---|---|---|
| R0 | 旧路径（需后续adapter） | FCFS | Prefill-first | 不触发 |
| R1 | 两阶段 | FCFS | Prefill-first | 不触发 |
| R2 | 两阶段 | LPM | Prefill-first | 不触发 |
| R3 | 两阶段 | LPM-Aging | Prefill-first | 不触发 |
| R4 | 两阶段 | FCFS | bounded | 固定 |
| R5 | 两阶段 | FCFS | bounded | 动态 |
| R6 | 两阶段 | LPM-Aging | bounded | 固定 |
| R7 | 两阶段 | LPM-Aging | bounded | 动态 |

比较R0/R1隔离副作用；R1/R2隔离LPM；R2/R3隔离Aging；R1/R4隔离阶段；R4/R5和R6/R7隔离动态；R3/R6观察阶段与请求公平互动。该矩阵不要求所有功能收益可相加。

## 8. 已有CPU模拟的复现命令

在项目根目录执行。输出写`/tmp`新文件，保留仓库历史结果。下面命令是复现方法，本次未重跑全矩阵；本次回归会执行相关小规模排空/重复测试。

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_prefill_policy.py --seed 31723 --trials 200 --repeats 5 \
  --output /tmp/minisgl-audit-lpm-benchmark.json

PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_batch_schedule_policy.py --seed 31723 --repeats 5 \
  --output /tmp/minisgl-audit-stage-benchmark.json

PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_batch_schedule_policy.py --compare-adaptive --seed 31723 --repeats 5 \
  --decode-caps 16 --protected-budgets 256 --decode-loads 16 64 128 192 \
  --target-budgets 128 256 512 --min-chunks 32 64 128 \
  --output /tmp/minisgl-audit-adaptive-benchmark.json

PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_batch_schedule_policy.py --compare-adaptive \
  --stagger-decode-completions --seed 31723 --repeats 5 \
  --decode-caps 16 --protected-budgets 256 --decode-loads 16 192 \
  --target-budgets 256 --min-chunks 64 --workloads long mixed \
  --output /tmp/minisgl-audit-decode-churn-benchmark.json
```

page4/LPM-Aging补充在相应阶段/动态命令增加`--page-size 4 --prefill-policy lpm-aging --aging-tokens-per-pass 8`；初始前缀64是4的整数倍。默认CPU Benchmark的`--decode-tokens`是**初始Decode输出长度**，不是生产预算helper的`decode_tokens=len(running_reqs)`；不同命名位置不要混淆。

## 9. 结果验收与后续资源验证

每组保存每请求trace、CPU/GPU阶段trace、策略状态、错误/超时、请求完成数、实际输出Token、资源检查、配置信息；报告5次的中位和离散范围，条件允许以seed配对bootstrap给置信区间。公平指标按hot/cold和Prompt长度分组，超时/未服务作为删失或失败样本明确列出，不能只统计完成项。

先做审计报告P1复现：page4、8页、1Decode、共享前缀4、7个Prompt5/输出1的hot新请求、保护预算7。当前固定/动态都触发物理页不足断言，这是已知未修复限制。正式紧内存性能结果必须先保证该资源配置合法且错误显式统计；后续修复的验收应是安全拒绝或正确预留，而非捕获断言继续。

此外专项验证：Table满后请求完成恢复；缓存在probe与准入间insert/evict/lock/unlock；partial split多计划；多个不同非页对齐Prompt同批；预算0、单请求、空请求队列、Abort chunk、准备/提交异常；真实TP各rank一致的到达、候选、阶段与分配。异常注入与回滚属于未来实现任务，本次仅列方案。

性能接受标准要预先写定，例如在相同offered load下冷请求TTFT尾部下降、Decode ITL达SLO，且吞吐损失在声明范围；保留代价而不事后选最漂亮的单指标。当前能交付的是语义验证、CPU成本和逻辑轮数取舍，GPU结论需要上述实测。
