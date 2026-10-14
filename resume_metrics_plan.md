# 成果指标、结果图与面试证据计划

本文件提前设计结果表达，**所有改善数值均为占位符**。尚未在目标RTX 5060 Ti 8GB上运行正式实验。模型、原版身份和指标边界见 [benchmark_plan.md](benchmark_plan.md)，实验ID见 [ablation_matrix.md](ablation_matrix.md)。

## 1. 最值得优先测量的三组结果

### 第一组：混合负载下的阶段调度收益

测E02：S1000与S1010，在同一W4 trace、相同offered load和固定预算下对照，另补同trace H。主成果是Decode ITL P99/每请求尾部的改善；同时报TPOT、长请求TTFT、output吞吐与Goodput。

这组最直接体现Serving工程判断：Prefill-first会让正在decode的请求受长Prefill影响；bounded阶段调度能改变竞争，但K只限制提交次数后的尝试，不能给绝对延迟上界。已有decode不能在执行中的GPU kernel中被抢占，过大的保护chunk仍会造成突刺。

预注册可接受取舍候选：output吞吐不低于基线95%、长请求TTFT P95增加不超过20%，且ITL尾部改善的区间稳定；这些是实验判定门槛，不是预期结果。若条件未同时满足，改为报告Pareto取舍，不能只摘有利ITL。

### 第二组：Prefix复用与cold公平性的可解释取舍

测E03/E04：FCFS、纯LPM、LPM-Aging三路。W3说明在cache工作集压力和排队条件下，真实cached tokens/recompute如何对应吞吐；W6说明Aging相对纯LPM的cold等待、超时/未服务风险及hot代价。

这组最能抵御“只是把hot请求插队”的追问。应同时展示命中、冷请求等待、每类SLO达标率，解释为何等长hot/cold仍有资源阻塞。cold P99样本不足时用P95/max/超5s比例作主要成果，不能用总体请求数掩盖cold尾部样本不足。

预注册取舍候选：Aging降低cold等待或超阈值比例，同时output吞吐不低于纯LPM95%；如果纯LPM没有吞吐优势，则只报告排序成本或公平机制效果，不强求“提高吞吐”的故事。

### 第三组：动态预算相对静态调优的边际收益

测E05：同C/K下固定64、固定24、target32/floor8三路，用真实n_decode与budget轨迹证明动态机制有执行。主结果是动态策略在Decode数量变化时，对ITL、TTFT和output吞吐的整体Pareto位置。

G1只有16活跃槽，动态预算20–28的变化远小于数据中心上百decode的场景，因此这一组**有研究价值，但收益不确定性最大**。若只胜过固定64而没有胜过固定24，表述为“保护预算调节改善取舍”，不能说“自适应优于固定策略”。如果D无稳定收益，保留负结果；简历第三条可改写为A两阶段准入的独立CPU成本/副作用验证与实验工程，不编造第三个GPU提升。

成本控制：先做W0与短pilot，再执行E02一个高rate、E03/E04/E05及E01，均5次重复；首轮70runs。需要原版缓存对照、第二个rate或更有说服力cold P99时，再按矩阵补15runs或扩大N。第二模型、全长上下文、全参数扫描和非法LOO均不属于第一轮。

## 2. 最终结果表的必需列

```text
experiment_id / workload / trace_hash / seed / variant / code_hash
model_revision / tokenizer_hash / dtype / backend / resource_profile
cache_condition / arrival_mode / offered_rate / concurrency
offered / admitted / success / failed / timeout / unresolved
input_tokens / generated_tokens / visible_tokens / cached_tokens
output_tok_s / total_logical_tok_s / success_req_s / goodput_req_s
ttft_text_p50,p95,p99 / ttft_token_ack_p50,p95,p99
itl_server_p50,p95,p99 / itl_client_ack_p50,p95,p99
tpot_server_p50,p95,p99 / tpot_client_ack_p50,p95,p99
e2e_p50,p95,p99 / cold_wait_p95,max / starvation_rate
kv_occupancy / evicted_tokens / peak_allocated / peak_reserved / min_free
client_send_lag_p99 / client_loop_lag_p99 / gpu_temp,power,clocks
mechanism_activation / collector_dropped_events / run_status
```

TTFT-token-ack仅在API数据ack与后端生成token对账成立后计算；否则NA。文本TTFT、后端token ready、client token ack的计时边界分别列出，不把不同域拼成一个“Token Latency”。报告每次run绝对值、5次中位数/IQR、配对改善及CI；附完整失败台账，不只展示成功行。

## 3. 图表规格

正式结果用静态SVG/PNG，所有图标注模型、GPU8GB、资源档位、到达模式、cache条件、样本数、重复数与错误条含义。数据来自聚合CSV，可回溯到run目录；不用现在构造示例性能数字。

| 图 | X / Y 轴 | 使用数据与分组 | 支持什么结论 / 对应优化 | README / 简历 |
|---|---|---|---|---|
| F1 原版vs优化吞吐 | X=同一workload下H/S1000/S1111；Y=generated token/s，旁图success req/s或Goodput | E06，同绝对rate；每点5run中位与IQR，失败另标 | 整机净效果；不能单凭此图归因A/B/C/D | README首图；简历项目说明适合 |
| F2 Tail Latency | X=绝对offered req/s；Y=TTFT-text P99、server TPOT P95、server/client-ack ITL P99，分别子图和单位 | E02各rate，按短/长分组，5run统计 | C降低Decode突刺及Prefill TTFT代价；不得把TPOT分位数当ITL | README核心；简历选一个主指标 |
| F3 Ablation | X=合法variant；Y=output吞吐相对同trace H与Goodput，tail latency独立面板 | E06及合法LOO，缺失cell留空并标依赖 | 独立/条件边际及组合；A0/B1等非法点不补数 | README、项目详细说明 |
| F4 Cache与吞吐 | X=真实token hit ratio；Y=output token/s，点色=FCFS/LPM/Aging，形状=工作集，标offer rate | E04与W2；同rate形成比较组，另图shared recompute与evicted tokens | B的机制关联；单纯相关不能证明所有吞吐差都由hit导致 | README很适合；简历文字给固定条件 |
| F5 Prefill/Decode时间轨迹 | X=真实墙钟相对秒；Y1=server token gap或每请求ITL，Y2=phase、actual Prefill tokens、n_decode；分面 | E02/W5单个**预先规则选定**代表run（如主指标最接近median的run） | C保护尝试及D预算变化对应突刺；不是挑最好一条曲线 | README解释原理；简历可作附图 |
| F6 Fairness–Throughput | X=output吞吐相对纯LPM；Y=cold P95等待或超5s比例；点色N，标未准入率 | E03，另图hot/cold等待CDF与删失比例 | B/Aging的公平取舍；不存在资源约束下的绝对无饥饿证明 | README核心；简历项目说明适合 |
| F7 动态vs静态预算 | X=方案fixed64/fixed24/D，或TTFT P95；Y=ITL P99，output吞吐用点大小/旁图 | E05同trace；加n_decode→budget散点，公式曲线作机制验算 | D是否优于调优静态点；区分小预算效应与动态效应 | README技术含量高；只在稳定有效时写简历 |
| F8 SLO Goodput容量 | X=offered req/s；Y=满足同SLO的req/s；旁图error/timeout率 | E02或E06，公共SLO，不能用各策略自定阈值 | 从总吞吐转到可用Serving容量 | README/简历都适合 |
| F9 显存与可复现性 | X=run时间或variant；Y=整卡used/min free、worker peak reserved；另图温度/SM频率 | 全部run遥测，启动与measurement分段；缺字段NA | 证明8GB真实可运行且未以更大KV预算制造收益 | README附录；简历只写硬件条件 |
| F10 A的CPU证据 | X=queue长度/partial-reject比例；Y=decision µs、token compare/split/touch次数 | E09 H/A-ISO，CPU微基准，至少5重复 | A减少拒绝副作用的工程效果及实际CPU净成本 | README附录；简历必须标CPU而非GPU吞吐 |

不要把不同单位放在同一无解释双Y轴；F5的phase/budget/latency优先用对齐的多面板。柱状图尾部小样本不能遮住原始run点；CDF包含完成请求的条件分布时，旁边明确展示timeout/删失比例。所有图同时给绝对值，normalized百分比只作辅助。

## 4. 简历成果模板（只填真实配对结果）

- “基于 mini-SGLang 实现有界 Decode 优先调度，在 RTX 5060 Ti 8GB 单卡、Qwen3-0.6B BF16 的【输入/输出分布】【到达率】混合负载下，相对【原版commit/兼容基线】将【后端/client-ack】P99 ITL降低【XX%】，输出吞吐维持在基线【XX%】，长请求P95 TTFT变化【XX%】；基于【N】次配对重复验证。”
- “实现只读Prefix探测与LPM-Aging调度，在【缓存工作集与复用率】【并发/到达率】下，将真实Prefix token命中率由【X%】提高至【Y%】、输出吞吐提高【XX%】；相对纯LPM将cold请求【P95等待/超阈值比例】降低【XX%】，对应吞吐代价【XX%】。”
- “基于活跃Decode数设计保护性Chunk预算，在【动态负载】下相对调优固定预算【B】将【ITL尾部/Goodput】改善【XX%】，同时将【TTFT/吞吐】控制在【明确边界】；通过预算轨迹、固定小预算对照与留一消融验证边际效果。”
- 若D无稳定GPU收益：“建立适配8GB单卡的Serving Benchmark与消融框架，固定代码/模型/trace、区分端到端与后端token计时，通过【N】次重复识别调度策略在【场景】下的【尾延迟/公平性/CPU成本】取舍，并定位【资源边界问题】。”
- A的独立成果：“将Radix匹配改为只读探测与条件提交，在【CPU微基准条件】下将【调度耗时/Token比较次数】变化【XX%】，消除已验证拒绝路径的【split/LRU touch】副作用；真实GPU收益以独立Serving实验报告。”

没有达到统计证据门槛时不写“显著”；没有测到有效吞吐增长不写“提升QPS”；最大观察等待降低不写“保证无饥饿”。原版对照与兼容配置对照各自标明，不能把相对纯LPM的公平改善百分比改写成相对原版。

## 5. 如何解释单项、边际与组合收益

先交代触发条件，再给机制轨迹，再给同条件对照，最后给代价。例如：“保护尝试在有Prefill与Decode竞争时触发；batch gap与真实chunk变小；在固定到达率下ITL尾部下降；TTFT与吞吐代价也在同表报告。”没有机制激活就不为噪声编解释。

B先用FCFS→LPM解释缓存选择，再用LPM→Aging解释公平；A不能被描述为从无Prefix Cache升级到有Prefix Cache。D先对固定64，再对固定24；如果同样小的静态预算已取得相同效果，自适应不是独立收益来源。

交互用合法2²块的差中差或log ratio，按run配对；不将“单项提高X%、另一项提高Y%”相加当Full提高X+Y%。Full去A或去C受依赖阻断，联合删除两个机制只回答联合贡献。没有完整实验的交互项就留空，不用理论预测填数。

## 6. 面试追问与应准备的证据

| 追问 | 回答要点与可出示材料 |
|---|---|
| 你真正在5060Ti8GB上测了吗？ | 展示目标机GPU UUID/name/VRAM、驱动、run manifest与峰值显存；本机3060审计记录单独保留 |
| Baseline是否真是原版？ | H完整commit、隔离源码路径、import origin、公共measurement patch；S1000仍有A，不能代替H |
| 这四项是独立的吗？ | B依赖A、D依赖C；合法嵌套矩阵与两个2²块；非法LOO明确未测 |
| TTFT从哪一刻开始？ | HTTP调用前；另报文本、token ack、server ready的差异；不从create返回后开始 |
| TPOT和ITL有什么区别？ | TPOT每请求平均间隔；ITL逐token gap分布；TPOT的P99不是ITL的P99 |
| SSE一个事件就是一个token吗？ | 本仓库ack链当前一一对应但final/empty要分类；用server uid/index计数对账；文本字符flush不等于新token数量 |
| 为什么用server时间戳，不用CUDA synchronize？ | 复用既有D2H完成点取host token ready；GPU纯计算用单独event诊断，主路径不加同步 |
| 为什么不用total throughput作为唯一指标？ | 缓存命中使逻辑input吞吐上升但GPU工作减少；output、Goodput、尾延迟与failure并列 |
| K是否保证Prefill在K步内开始？ | K是Decode批次提交后的保护尝试；KV/Table不足、chunk优先、拒绝即break均可继续延迟 |
| Aging是不是按时间？ | 不是；按发生new admission的有效pass；以512/N说明评分追平条件，不给秒数保证 |
| 为什么8GB上可能看不到D收益？ | n≤16时预算变化小；target过大恒U；小chunk开销可能抵消竞争收益；展示负控制与静态matched对照 |
| 小chunk为什么仍可能KV不足？ | 新请求预留完整剩余Prompt+输出；预算控制本轮计算，不是完整显存承诺；page>1另有已知舍入缺口 |
| 为什么显式fi/page1？ | 本仓库FI限制、auto按capability≥10可能选trtllm及page64；避免不同后端/页粒度混杂 |
| CPU客户端是不是瓶颈？ | 实际send lag/loop lag、CPU利用率、dummy SSE校准与亲和/远端交叉验证，记录同机竞争 |
| Warmup会不会污染Prefix？ | warmup与测试prefix隔离；cold-test-prefix与整树空不同；warm组prime trace固定、每run重启 |
| P99有多少样本？ | 给请求与每类计数、interval计数；cold不足1000则以P95/max为主或追加实验 |
| 5次重复足够吗？ | 给run中位、IQR/MAD和配对CI，承认n5区间局限；按预注册规则扩到10，不把token数当重复数 |
| 输出正确性怎么保证？ | 同模型/权重/精度/greedy；token序列、实际长度、终态/uid完整性；差异先定位，不能删样本 |
| 没有性能收益的点去哪了？ | 完整cell与失败台账、OOM/timeout/热不稳记录；没有稳定改善则不写提升 |
| 能推广到H100/多卡吗？ | 结论只覆盖单卡消费GPU与这些负载；TP、不同memory bandwidth/CPU比例另做实验 |

## 7. 发布结果前的结论边界

README应包含：环境与源码manifest、原版/优化差异、冻结trace与生成命令、请求完整性、失败台账、指标字典、主图和代价、未测项与已知限制。简历压缩为1–2条有条件的量化结果，链接到README证据。

优先保证第一组和第二组结论牢靠，再决定是否为第三组扩大实验。第三组若收益很小，报告明确的适用边界与负结果，比把CPU模拟、逻辑轮数或不等参数对照包装成GPU加速更能说明推理系统工程能力。
