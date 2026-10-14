# 消融矩阵、合法依赖与归因

本文件只规划实验，不实现开关。固定模型M0=Qwen3-0.6B/BF16/TP1，默认G1=KV16384 tokens、page1、FI、graph16、max running16、seq2304、normal Prefill256。长度/负载定义见 [workload_design.md](workload_design.md)，计量见 [benchmark_plan.md](benchmark_plan.md)。

## 1. 因子与合法配置空间

| 因子 | 0 / 关闭 | 1 / 开启 | 当前可否直接切换 |
|---|---|---|---|
| A 缓存两阶段准入 | H的旧match/split/touch及双重容量检查 | probe/refresh、先容量检查后commit/lock/touch | **不能**；当前无CLI，S-default始终为A1 |
| B Prefix公平调度 | FCFS | LPM-Aging，N64 | 可；另设B=L纯LPM用于拆分缓存排序与Aging |
| C 阶段调度 | Prefill-first | Decode-first-bounded，K4/U64 | 可 |
| D 动态预算 | 固定U | target32/floor8 | 可，但必须C1 |

工程依赖为 `B1 → A1`（现有B依赖只读probe和刷新计划），`D1 → C1`（配置直接禁止D1/C0）。因此**不是四个独立二元因子，不能机械地做2⁴，也不能虚构“D单独开启”**。若把B视为三水平（FCFS/LPM/LPM-Aging），还须明确不是纯二元全因子。

按两个嵌套链（A,B）∈{00,10,11}、（C,D）∈{00,10,11}，得到9种合法机制配置。当前S只能直接提供A1下的6种；A0/C1需未来在H隔离副本中只移植C/D，或验证过的旧路径adapter。为了低成本，先跑H+6个S配置，另用纯LPM拆分B；缺失项标“未实现”，不把它们当测过。

## 2. 基线与代码身份

- **H0000：原版基线。** 使用上游完整commit `9a91cfafe754aa85daee49998176275667eb58f2` 的隔离源码；不把当前工作区恢复掉。模型、依赖、benchmark collector、资源参数与S相同。common measurement patch 不改变调度、attention、sampling。
- **S1000：当前兼容基线。** FCFS、Prefill-first、adaptive关闭；运行A，同时仍有当前实现的队列元数据/计数等开销。H→S1000是“默认实现差异的净效果”，不能无检查地称为A纯收益。
- **A-ISO：纯A来源隔离。** 后续从H仅移植A的cache接口与PrefillAdder准入修改，不加入B/C/D bookkeeping；给独立源码hash。H→A-ISO才用于较干净的A机制归因，S1000→A-ISO用于检查附带bookkeeping影响。此变体本阶段未创建。
- 禁止用 `--cache-type naive` 当作A0：这会关闭整个Prefix Cache，改变研究问题。
- 禁止用 `lpm-aging,N=0` 当FCFS：它会退化为LPM。B0用明确 `schedule_policy=fcfs`。

## 3. 合法完整机制表

| Variant ID | A | B | C | D | 获得方式 / 状态 | 主要归因 |
|---|---:|---:|---:|---:|---|---|
| H0000 | 0 | 0 | 0 | 0 | 原版HEAD；已有源码可隔离还原 | 原版基线 |
| V0010 | 0 | 0 | 1 | 0 | 待仅移植C到H；当前无现成开关 | C独立开启，排除A交互 |
| V0011 | 0 | 0 | 1 | 1 | 待仅移植C+D到H | D在旧准入上的条件增量 |
| S1000 | 1 | 0 | 0 | 0 | 当前CLI可提供 | compatible baseline / A路径 |
| S1100 | 1 | 1 | 0 | 0 | 当前CLI可提供 | B独立于C/D的条件收益 |
| S1010 | 1 | 0 | 1 | 0 | 当前CLI可提供 | C在A上的条件收益 |
| S1011 | 1 | 0 | 1 | 1 | 当前CLI可提供 | D在A+C上的条件收益；Full去B |
| S1110 | 1 | 1 | 1 | 0 | 当前CLI可提供 | B+C组合；Full去D |
| S1111 | 1 | 1 | 1 | 1 | 当前CLI可提供 | All Optimizations |

额外S1L00/S1L10/S1L11：B=L纯LPM，分别搭配C/D关闭、C开启固定、C+D开启。首轮优先S1L00，在W6拆分“按缓存排序”与“Aging公平”。不将额外三水平当新增独立优化。

## 4. 真实运行开关映射

以下均叠加相同资源参数；H不接受新增策略flag，所以H启动时只传公共原版flag。解析后的实际config和实际路径计数都要保存。

| 配置部分 | 参数 |
|---|---|
| B0 | `--schedule-policy fcfs --lpm-aging-tokens-per-pass 0 --fairness-max-wait-steps 0` |
| B=L | `--schedule-policy lpm --lpm-aging-tokens-per-pass 0 --lpm-queue-threshold 128 --fairness-max-wait-steps 0` |
| B1 | `--schedule-policy lpm-aging --lpm-aging-tokens-per-pass 64 --lpm-queue-threshold 128 --fairness-max-wait-steps 0` |
| C0 | `--batch-schedule-policy prefill-first`，不传adaptive flag |
| C1/D0 | `--batch-schedule-policy decode-first-bounded --max-consecutive-decodes 4 --protected-prefill-tokens 64`，不传adaptive flag |
| C1/D1 | C1参数再加 `--adaptive-prefill-budget --target-batch-tokens 32 --min-chunk-tokens 8` |

A0不能靠不存在的`--disable-two-stage-cache`实现。后续如果增加benchmark入口的`--admission-impl legacy|two-stage`，它是**拟议接口**，必须显式调用不同factory/准入代码并做路径证明；不是现有server flag。

## 5. 优先实验表

每行涉及的变体集合采用同一trace、同一资源档位；rate取W对应H pilot的绝对值。成本为**预算估算，不是实测耗时**。`R`表示每个变体5次，`L`表示2个offered rates；pilot每次约30–60s，正式每run通常3–10min加启动/JIT/prime。N≥1000可能比10min更久，以实际H校准为准。

| Experiment ID | 开启优化 / 变体 | Workload | Model | I/O tokens | C / Arrival | 主要指标 | 对比Baseline | 预计运行成本 | 优先级 |
|---|---|---|---|---|---|---|---|---|---|
| E00 | H、S1000、S1111 | W0 | M0 | 128/32、512/128、2048/128 | C1→4→8→16 | 正确性、完整性、peak VRAM | H | 3变体×约32–64请求；不计收益 | P0门槛 |
| E01 | H vs S1000 | W1 | M0 | 512/128 | C16，N≥1000 | output吞吐、TTFT、scheduler µs | H | 2×R=10 runs | P0兼容控制 |
| E02 | S1000 vs S1010（A vs A+C） | W4 | M0 | 128/256、2048/32 | 0.8/1.05λ_W4 | ITL P99、TPOT、长TTFT、Goodput | S1000；同trace补H | 2×L×R=20 runs；H高rate另5 | P0核心1 |
| E03 | S1000、S1L00、S1100（A / A+LPM / A+B） | W6 | M0 | 576/32，hot L512 | 1.05λ_W6 | cold等待P95/max、超5s比例、output、hot代价 | S1000；Aging直接对S1L00 | 3×R=15 runs | P0核心2 |
| E04 | S1000、S1L00、S1100 | W3 | M0 | 576/32，L512 | B32，固定周期 | hit/recompute、output、Goodput、evict | S1000；H同trace另5 | 3×R=15 runs，H另5 | P0缓存收益 |
| E05 | A+C fixed64、fixed24、A+C+D | W5 | M0 | 128/256、2048/32 | n目标4/8/12的固定波次 | ITL、TTFT、chunk分布、forward/prepare成本 | fixed64及matched fixed24 | 3×R=15 runs | P0核心3 |
| E06 | H、S1000、S1100、S1010、S1011、S1110、S1111 | W4-X | M0 | W4+固定共享prefix子集 | 1.05λ_X，N≥1000 | output、Goodput、ITL、cold等待、各机制激活 | H；边际对照见下节 | 7×R=35 runs；允许复用完全相同已有cells | P1整机归因 |
| E07 | S1111、S1011、S1110 | W4-X | M0 | 同E06 | 同E06 | 合法LOO：去B / 去D | S1111 | E06已有，不额外运行 | P1 |
| E08 | S1111 vs S1100；S1111 vs V0011 | W4-X | M0 | 同E06 | 同E06 | 联合移除C+D / A+B | S1111 | S1100已有；V0011需实现后5runs | P2依赖删除 |
| E09 | H vs A-ISO vs S1000 | CPU cache/准入；GPU W3可选 | M0（CPU微基准不加载） | prefix512/2048，partial/reject | queue8/32/128 | decision µs、比较次数、拒绝split/touch、净吞吐 | H | CPU每case5重复；GPU最多15runs | P2独立A |
| E10 | H、V0010、V0011 | W4/W5 | M0 | 同主组 | 同主组绝对到达率 | C独立、D条件收益及与A交互 | H / V0010 | 后续source variants；最多30runs | P2补完整矩阵 |
| E11 | S1000/S1L00/S1100 | W2、W7 | M0 | 576/32 | C1；burst128/129 | 无争用负控制、退化路径、overhead | S1000 | CPU先测；GPU每负载只选必要cell | P2 |
| E12 | S1010/S1011 | W5负控制 | M0 | 同W5 | 同波次；D target256 | budget恒64时差异应接近噪声 | fixed64 | 2×R=10runs，可先pilot | P2 |

W4-X为组合负载：W4短请求中一半引用4个已prime的64-token公共前缀，另一半独有；长请求保持独有。实际I/O仍为W4的128/256和2048/32；保存专属trace与λ_X，不能拿W4普通组代替。若B没有在此负载产生足够候选重排，E06只能支持整机/C/D结论，B独立结论来自W3/W6。

P0成本：E01 10 + E02 25 + E03 15 + E04 20 + E05 15 = **85个正式runs**，另有smoke和pilot。按每run3–10min粗估为4.25–14.2小时，加重启/编译。最低先用E02单个高rate减少10runs，E04不补H减少5runs，形成70runs的首轮；需写原版缓存收益结论再补H。时间预算不足时降低实验家族数量，不把每个核心cell的重复数从5次降成1次。

E06–E07为P1，不加入首轮成本；完全相同的trace hash、seed、variant hash、环境和采集边界才能复用已有cell。新增W4-X不能复用W4数据。图表和摘要均显示完成/未完成矩阵，不填补未测cell。

## 6. 独立收益、边际收益与交互

### 6.1 核心条件对照

| 对照 | 可解释结论 | 不能解释为 |
|---|---|---|
| H→S1111 | 真实工作区全部优化的净系统收益（连同附带bookkeeping） | 四项收益可逐个相加 |
| H→A-ISO | 独立A的实现效果 | S1000默认差异全部来自A |
| S1000→S1L00 | 在A基础上的纯LPM排序效果 | Aging收益 |
| S1L00→S1100 | 在纯LPM基础上增加有效轮Aging | 整个Prefix Cache或全B收益 |
| S1000→S1100 | B整体条件收益 | A0上的B独立收益 |
| S1000→S1010 | A背景下C条件收益 | D效果 |
| S1010→S1011 | FCFS背景下D增量 | 所有小预算效应都是自适应效果 |
| S1110→S1111 | B+C背景下D增量 | C+D总效果 |
| S1011→S1111 | C+D背景下B边际效果 | B单独收益与此值相同 |
| fixed24→dynamic32/8 | 动态策略相对代表性调优静态预算的整体取舍 | 所有静态预算已被全面击败 |

### 6.2 可行的2²交互块

在A1、D0固定时，S1000/S1100/S1010/S1110构成完整B×C二因子块，适合评估请求排序与阶段公平交互。令同workload同load的原始指标为Y：

`I_BC = (Y1110-Y1100) - (Y1010-Y1000)`。

延迟的正负含义与吞吐相反，需用绝对单位说明；可以使用log吞吐ratio作乘性交互，但不能相加改善百分比。在C1下，S1010/S1011/S1110/S1111构成B×D块。D0/C0的无效值不参与回归估计D主效应。

未来补V0010可组成A×C（B0/D0）块；没有该cell就不能给A/C交互的定量结论。没有三个独立二元因子可合法跑完整2³，因此这里选择嵌套9配置和两个可行2²块，而不是伪造8格。

## 7. Leave-one-out 的完整解释

| 从Full S1111移除 | 理想编码 | 当前是否合法 | 本方案怎么处理 |
|---|---|---|---|
| B | S1011 | 合法 | 严格单项LOO，在同W4-X上测 |
| D | S1110 | 合法 | 严格单项LOO，在同W4-X上测 |
| A | 0111 | 现有B依赖A，不可直接运行 | 报“依赖阻断，未测”；可联合移除A+B得到V0011，明确这是两项删除 |
| C | 1101 | 配置禁止D1/C0 | 报“依赖阻断，未测”；联合移除C+D得到S1100 |

若后续确有研究必要，可把“只读排序probe基础设施”与“A的准入提交策略”分离，让B保留只读probe、A0使用完整旧准入；此时0111是**新定义的机制手术实验**，必须独立审核旧_tree_walk/touch/refresh行为并给源码hash。不能把它追溯为当前四项自然开关。D离开C没有执行入口，强造1101没有原始机制意义。本阶段不实施这些改动。

## 8. 执行路径验收，防止开关空转

| 机制 | 每run必须保留的路径证据 | 未满足时处理 |
|---|---|---|
| H旧准入 | 旧matching路径hit，new probe/commit计数为0；源码hash符合H+common measurement patch | 不标原版基线 |
| A | probe/refresh/commit/lock/touch的实际计数；拒绝case无split/touch（诊断快照） | 记录路径不符；修工具后整组重跑 |
| B | 实际new queue长度、候选重排次数、fallback原因、hot/cold准入序、waiting_passes增长 | 若全FCFS fallback或无重排则标未激活 |
| C | 竞争状态下实际Decode提交streak、达到K的保护attempt、成功与失败、成功Prefill后的reset | 不能仅凭K配置宣称有公平保护 |
| D | 每个protected batch的n、U、target、floor、预算、实际extends；与公式逐条核对 | 若budget恒U则只能作负控制；不宣称动态收益 |
| Graph | 实际capture list、replay/eager真实次数、真实与padded batch sizes | 不将dummy当请求/token；不同捕获设置另建系列 |

预算与extend并不相等：剩余prompt较短、资源或Table不足时真实chunk可更小。只增加计数、CPU时间戳和离线分析；不在热点路径做树全量快照、同步打印或额外GPU同步。
