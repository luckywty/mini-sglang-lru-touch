# Workload 构造与回放协议

本文件为实验设计，尚未实现 generator 或执行负载。模型与显存配置统一引用 [benchmark_plan.md](benchmark_plan.md) 的 Qwen3-0.6B BF16 / G1；实验变体定义见 [ablation_matrix.md](ablation_matrix.md)。所有 I/O 长度是实际 token 数，不是字符数。

## 1. 先冻结输入，再切换策略

采用本地固定 tokenizer，将文本模板编码后存为 trace；seed 只控制生成，正式 replay 不再随机造输入。保存 tokenizer文件hash、文本、实际input_ids、I、O、期望共享前缀长度L、prefix_group、hot/cold标签、arrival timestamp、采样参数、timeout、trace SHA256。每个对照读取同一文件。

建议核心在线测试通过现有 `/v1/chat/completions` 提交 **`prompt` 字符串**（该端点源码支持 prompt），明确不传 messages；这样避开额外 chat template 的公共前缀和长度不确定性。使用 server 同样的 `tokenizer.encode(prompt)`，检验 decode→encode round trip，最终以 server I 为准。不能用目前未被 API request schema 接受的 `input_length_override` 强行声明长度。真实聊天补充组才传 messages，并按 `apply_chat_template(..., tokenize=False, add_generation_prompt=True)` 再 encode 的实际路径计算 I；不在两组间改 thinking 模板。

随机 ID 只用于离线正确性/CPU微基准；在线主负载优先本地自然文本块拼接，避开不可见/特殊token和无意义乱码。用冻结 tokenizer 对拼接边界重编码，确认精确长度；无法得到目标长度时重新生成文本而非裁切 Unicode 字符。公共字符串拼接不保证公共 token 边界，必须逐对验证 token LCP。

无复用组在最早可用 token 位置放入不同的合法文本标记，并检查实际 LCP；messages 路径的共享模板不能当零缓存命中。缓存组的请求应在共享前缀后加入独有 token，保证 C 的上界可解释；缓存匹配不包含 prompt 最后一个 token。

## 2. Trace 文件与采样

拟新增 `benchmarks/serving/generate_workloads.py`，每行一条请求，示意：

```json
{"schema_version":1,"request_id":"s31723-0001","arrival_s":0.125,"prompt":"<真实文本>","input_ids":[101,102,103],"input_tokens":3,"max_output_tokens":32,"prefix_group":"g0","shared_prefix_tokens":2,"class":"hot","temperature":0.0,"top_k":1,"top_p":1.0,"ignore_eos":true,"timeout_s":120}
```

上述3个token只是字段示例，不是可直接运行的工作负载。正式输入用真实 tokenizer 构造并存 sidecar manifest。manifest 记录总请求数、各类数目、长度直方图、到达模型、rate、seed、是否有 prime、prime trace hash、SLO版本和选择负载强度的规则。

主性能轨道 greedy、ignore_eos=True、固定O；保证每请求生成长度一致。自然结束轨道单列，ignore_eos=False，记录实际O和EOS，不能与固定长度轨道混算提升。性能输入也须做正确性抽检：冻结 baseline greedy输出token_ids；差异时查长度、cache、chunk、采样与数值路径，不用删除差异请求解决。

## 3. Arrival 模式与压力标定

| 模式 | 构造 | 测量用途与限制 |
|---|---|---|
| Closed loop | 保持 C 个 outstanding 请求，完成一个立即补一个，C=1/4/8/16/32 | 低并发延迟和饱和吞吐；具有反馈，不能声称固定到达率下的公平性或避免 coordinated omission |
| Open loop Poisson | `Δt~Exp(λ)`，预生成全部绝对到达时间，不等待前请求完成 | 主要用于 SLO/尾部；记录 `actual_send-planned_send`，不因排队降速 |
| 固定 bursts | 每周期在固定 timestamp 注入 B=16或32个请求，顺序冻结 | 制造 Prefill 候选竞争与cache工作集；按相同B/周期比较 |
| Mixed / churn | 固定时刻发送长输出背景请求，再按独立墙钟时间注入长Prefill；请求自然结束 | 验证阶段和动态预算；不能以“第N个batch”触发下一波到达 |

用 **H原版** 在相同 workload、cache条件、G1 下的短 pilot 找到容量锚点 λ_ref：闭环稳定输出折算req/s并经开放环确认，最大可持续点须无持续backlog增长、失败且满足预注册条件。若pilot只能给近似容量，明确记为经验锚点，不称理论最大QPS。

主强度为 `0.8 λ_ref` 和 `1.05 λ_ref`；敏感性可加 `0.5/0.95/1.2`。rate 使用绝对值写入manifest，所有策略相同；不能对每个策略重新计算0.8×自身容量。各 workload 可有自己的 λ_ref，但不能跨 workload 算收益。矩阵中的“高负载”必须落到实际req/s。

正式trace通常 `N=max(1000,ceil(λ*120s))`；低成本探索可降为500并以P95为主。混合/公平组按类别分别检查样本量；组内不足1000时该类P99仅探索，不能借总体N替代。超载trace必须有限，注入完后排空或到120s每请求deadline，另设固定最大drain120s；到终止仍无结果的请求明确记失败/删失。触发client内存保护阈值（建议4096 outstanding）则停止该run并记录overload-invalid，不能悄悄阻塞到达。

## 4. Workload Matrix

表内 W0…W8 是冻结的负载家族；只扩展必要维度，不把所有长度、rate与消融做笛卡尔积。

| ID | 输入 / 输出 tokens | 构造与到达 | 缓存条件 | 优化目标 / 主指标 | 优先级 |
|---|---|---|---|---|---|
| W0 正确性/显存 | 128/32、512/128、2048/128；每类8–16请求 | C=1→4→8→16；总计约32–64，先G0再G1 | 独有prompt及精确partial prefix | 所有变体：输出token、终止、峰值显存、页/槽回收；不报告性能收益 | P0门槛 |
| W1 无复用控制 | 512/128 | C=1/8/16；主选C16，N≥1000 | 无刻意共享，记录真实微小LCP | A/B额外CPU成本、output吞吐、TTFT、failure | P0最小控制 |
| W2 缓存稳定工作集 | 576/32，L=512，suffix64 | 4个prefix groups，hot80%/cold20%，Poisson或B32 bursts | 先prime4组；工作集能驻留 | B的路径与overhead控制；全部已命中时LPM可能无吞吐优势 | P1控制 |
| W3 缓存竞争/再访问 | 576/32，L=512 | hot80%引用4个组；cold20%轮转32个独立组；B32，周期按λ_ref标定，N≥1000 | prime hot；cold工作集大于KV16384，未来再访问标记清楚 | A/B：token hit、shared recompute、eviction、output吞吐、Goodput、hot/cold TTFT | P0缓存主组 |
| W4 Prefill/Decode混合 | 80%短128/256，20%长2048/32 | Poisson，0.8/1.05λ_ref；长prompt与正在decode请求竞争 | 独有prompt；预热不prime测试prefix | C：ITL P95/P99、TPOT、长TTFT、E2E、Goodput；实际两阶段竞争比例 | P0阶段主组 |
| W5 Decode数量变化 | 背景128/256（后续可512输出）；干扰2048/32 | 背景波次使活跃Decode覆盖约4/8/12；间隔固定；同时少量长Prefill流；cycle重复 | 无意图复用；所有时间预生成 | D：budget(n)、真实chunk、chunk数、ITL、TTFT；相对固定小预算增量 | P0动态主组 |
| W6 公平性竞争 | hot/cold均576/32；hot L512，cold新prefix | 80%hot/20%cold，cold先进入每个burst且hot持续插入；open loop约1.05λ_ref，N≥1000 | prime hot；压力主要来自Table/完整KV预留 | B的Aging：cold P95/max等待、超5s比例、未准入率；hot与output代价 | P0公平主组 |
| W7 队列退化边界 | 576/32 | 单次burst128/129/256；记录真实new queue size，不保证HTTP接收次序就是scheduler次序 | 同W3 | 证明>128退回FCFS，排序成本；CPU即可先验证 | P2诊断 |
| W8 Context敏感性 | 背景128或1536/256；干扰2048/32 | 固定n≈8、相同预算，但改变已有context | 控制无复用 | D用count近似成本的局限；相同n不等于相同GPU时长 | P2诊断 |

W3中每个cold组首次出现属于 compulsory compute；只有已出现/prime过的共享prefix再被计算才叫shared recompute。`shared_prefix_tokens=512`指本族设计公共区，不是承诺缓存中一定有512。W6追求等长hot/cold，避免把长度优势误当缓存公平收益。

W3同时具有cache压力与排队，并不保证B改善命中：当前LRU、future KV reservation、chunk优先都可能令顺序收益消失。实际观测若所有方案token hit相同，不能以LPM名称解释吞吐差；转查CPU准备、批大小或噪声。

## 5. Mixed Load 的执行细节

W4先以全短请求pilot确认能形成持续decode，再用固定比例混合trace。短/长标签在生成时冻结，不根据运行时完成快慢重新采样。分别报告两类TTFT/E2E，Decode ITL按所有间隔与每请求分布双报；长Prefill自身最终产生一个token，不能把它当作背景Decode间隔样本。

C 对比保持 normal max Prefill budget=256、固定保护预算=64、K=4；仅切阶段策略。诊断可扫K=1/4/8，固定预算32/64/128，但pilot最多3个代表点，正式选择规则提前登记，避免大扫参挑漂亮结果。

每run验证实际存在 `pending prefill && running decode`，记录该状态持续时间、保护attempt和成功数。若两阶段从未竞争，标为机制未激活；该run仍保留，但不用于支持C收益。

## 6. 动态预算的可辨识性

G1最多16活跃请求；不要照搬`target=256`和上百Decode的服务器实验。主D候选为U=64、target=32、floor=8：

| 实际 n_decode | 固定U | D预算 b | 解释 |
|---:|---:|---:|---|
| 4 | 64 | 28 | 只在保护性Prefill阶段应用 |
| 8 | 64 | 24 | 后续以固定24作matched静态对照 |
| 12 | 64 | 20 | D变化范围有限，不能预设一定有性能收益 |
| 0 | — | 正常预算256 | `_select_next_batch`走无Decode分支，不调用保护预算 |

主对照为固定64、固定24、动态32/8；补充固定20/28说明调参公平性。若只比固定64，则结论只能是“改变保护预算的组合效果”，不能说负载自适应已优于静态调优。

在G1的n≤16范围内，主候选的floor8通常不触发；这是有意限制正式扫描成本，不表示验证了所有floor边界。floor/clamp分支通过已有CPU语义测试和下述pilot候选覆盖，主性能结论只覆盖实际观测到的预算区间。

仅pilot可试target16/floor4，n=4/8/12时预算12/8/4，说明更强适应也可能引入大量小chunk开销；不默认纳入正式矩阵。另设target256/floor8负控制：n≤16时预算恒为64，adaptive开关打开却预算不变，预期无机制收益，正好检查“开关即优化”的错误归因。

W5前导背景请求同样属于workload，不偷偷当warmup删掉。用n随时间轨迹核验波次是否达成；目标4/8/12是设计目标，不能人为阻塞scheduler固定这个数。若本机输出太快/太慢导致波次失效，只在pilot统一调节背景输出256→512或波次间隔，冻结后所有策略重用同trace。分析可按真实n分箱，但不同策略同n样本是运行结果，不能仅据这种条件切片作因果收益；主因果结论仍来自同arrival trace整体对照。

## 7. Fairness 的报告方式

N=64 tokens/effective pass为主Aging候选；L=512时，cold与一个新hot在资源可行且每轮有new准入的条件下约8个有效等待轮可达到同分，稳定arrival order再破同分。该推导既不是8个GPU batch，也不是8秒，更不是TTFT上界。chunk-only pass不老化，>128队列退化、首候选资源拒绝、Table满都会改变路径。

同时保存：q→首次准入、q→首token、完整E2E、raw schedule_step、effective waiting_passes、cold被越过次数、拒绝原因、最大等待与未服务比例。统计两个群体的SLO达标率和吞吐；Jain index可作为补充，对每类单位offered请求的达标率算`(Σx)^2/(kΣx²)`，全零时NA。不能在不同输出长度间比较每请求token数的“公平指数”。

若cold样本<1000，核心报告cold P95/max与超5s比例。需要在简历写cold P99时，扩展到至少1000个cold请求（20%比例时总N≥5000）并重复5次，单列追加成本。任何未准入请求都属于饥饿/失败风险，不能从分位数分母删除后称“公平性已保证”。

## 8. Cold/Warm Cache 与 Warmup 的分离

1. JIT/graph预热：每个run从新进程启动，完成预先规定的shape warmup；warmup prompt与测试prefix刻意不同。不计性能样本，但记录编译时间/启动峰值。
2. 冷启动cache：warmup后不prime测试prefix；此时树可能有warmup残留，称“测试prefix冷”，并核验C/eviction；若声称整树空，必须使用将来经过验证的安全clear接口或独立启动模式，现有reset不可用。
3. 主warm-cache组：执行冻结prime trace直到相关前缀确实可复用、全部结束；确认hot C约512。prime请求O建议1–2，最后token排除规则照常生效，实际命中必须验。
4. Warm动态稳态：正式trace会继续插入/淘汰；记录cache随时间变化，不宣称warm意味着100%命中。
5. 不同策略run必须重启并重放同warmup与prime；不能沿用上一个策略留下的树。H和S的LRU状态可能因实现不同，这是被测机制结果，输入历史必须一致。

## 9. 复用现有 Trace 的边界

`read_qwen_trace` / `read_mooncake_trace` 能复用timestamp和长度，但当前忽略hash_ids；`dummy=True`从同一长prompt截取，产生人造全共享前缀，不能据此宣称真实trace缓存命中提升。Qwen示例还会自动下载外部trace，并用偏大的默认规模，不作为本阶段执行命令。

最低成本先用W3/W4/W5/W6的可解释合成trace。真实业务trace只作P2外部有效性验证：事先本地准备、保留实际prefix关系或明确匿名化方法、裁剪到G1长度、冻结版本与hash，记录裁剪带来的分布偏移。
