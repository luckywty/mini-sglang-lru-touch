# mini-SGLang 四项优化面试准备

对应[完整审计报告](mini_sglang_four_optimizations_review.md)。每项6题，共24题；回答严格区分代码行为、CPU验证和GPU效果。

## 一、Radix Prefix Cache 两阶段匹配

### 1. 为什么不能直接用原match_prefix做LPM排序？

HEAD的`match_prefix → _tree_walk`会在部分命中时split并touch时间戳，排序未必接纳这些请求，却已改变缓存。当前调度使用只读probe，只有容量检查通过后才commit、lock、touch。旧公开match_prefix仍可split，不能把它当只读API。证据：`radix_cache.py:L164/L175/L320`，HEAD `L205–229`。

### 2. probe和commit之间传递什么？commit是否负责lock？

`MatchProbe`传可复用长度和新增保护Token量，`RadixMatchProbe`另传末节点、节点内命中长度和结构版本。Radix的commit只在必要时split并构造Handle，lock和LRU touch由`CacheManager.try_match_and_lock`调用。采用发生在准入，不是GPU完成时。证据：`base.py:L67`、`radix_cache.py:L106/L222`、`scheduler/cache.py:L40`。

### 3. 为什么要扣newly_protected_size？双重检查还在吗？

可用容量包括空闲页和可淘汰缓存；采纳前缀后原可淘汰匹配部分被加引用保护，不能再分配给新Token。于是检查`required <= available - newly_protected_size`，预测旧第二次锁后检查。在当前串行状态不变前提下，它蕴含更宽松的锁前检查，不需先split/lock再回退。证据：`cache.py:L53–60`。

### 4. 缓存变化后如何避免复用过期计划？是否完全避免重复遍历？

insert/split/evict改变结构版本，刷新时重新probe；只变引用计数而版本未变，沿父链重算保护成本，不比较Token。commit直接定位节点，但refresh、lock、touch、索引拼接和unlock仍走父链。一次split可能让本轮多个计划重新probe，因此只能说避免同版本下无条件双重Token匹配。证据：`radix_cache.py:L204/L222/L126/L149/L96`。

### 5. 如何验证probe无副作用？

对树节点/children/parent、timestamp、ref_count、evictable/protected计数做前后快照，比较probe长度与实际匹配；覆盖page1/4、部分命中、无匹配、重复前缀、拒绝准入和状态变化。现有测试还检查每条边Token比较次数。冻结dataclass只限制计划字段赋值，本身不是线程安全证明。证据：`test_prefill_admission.py:L58/L136/L166`、`test_prefill_policy.py:L312/L337/L361`。

### 6. 通过这些测试能保证不会OOM吗？

不能。审计额外复现：page4时7个短请求各需1私有页，Token准入通过但仅剩6页，prepare触发evict断言。加引用和Table已提交而无通用回滚。两阶段隔离匹配副作用不等于页级预算/异常生命周期完整安全；后续需要逐请求舍入余量及失败回滚测试。证据：报告“P1分页舍入”及`cache.py:L81`、`prefill.py:L84`。

## 二、FCFS / LPM / LPM-Aging

### 1. 什么时候用哪种策略？超过128一定Aging吗？

默认FCFS；显式lpm或lpm-aging时，new数≤阈值、至少2个、存在新准入机会且未触发可选等待退化，才做前缀探测排序。lpm只按缓存长度；lpm-aging仅N>0才使用有效等待分数。new>128按原始到达序退化，chunks始终先续算。证据：`prefill.py:L165–211`、`config.py:L21`。

### 2. N会随着每个请求到达而改变吗？这是HRRN吗？

N是统一配置`lpm_aging_tokens_per_pass`，运行中不自动调整；不同请求的waiting_passes不同，乘积才不同。评分是缓存长度加虚拟Token，不是等待时间/服务时间的HRRN，也不按秒Time Aging。虚拟分数不写入Handle实际缓存长度。证据：`prefill.py:L201–210`、`utils.py:L23`。

### 3. 为什么使用有效等待轮，而不是每调用一次就加一？

无资源、零预算、Decode-only或chunk-only没有新准入机会，简单调用数会让优先级凭空增长。当前只有本轮至少一个new成功准入，仍等待的new才统一加一；多个准入也只算一轮。计数在Batch构造末尾，先于物理页准备/GPU执行。证据：`prefill.py:L239–262`、测试`L196/L210/L233`。

### 4. 如何保证同分稳定和FCFS回退真正先来先服务？

入队只追加并分配arrival_order，LPM生成单独候选列表，用`(-score, arrival_order)`排序，基础新请求队列保持到达序。成功项按对象身份删除，而不是按成功数量切队头；回退使用基础列表，所以不会把以前的LPM顺序误当FCFS。证据：`prefill.py:L153/L205/L254`、测试`L143/L167`。

### 5. 等待8轮就一定获得服务吗？

若cold缓存0、fresh hot缓存64、N8，8个有效轮后同分，较早cold在下一排序优先；前提是比较fresh hot、缓存估算稳定、LPM-Aging分支生效。已有hot也在老化、chunks优先、Table/KV不足和候选头阻塞都可能改变服务结果，所以这是排序结论。证据：测试`test_prefill_policy.py:L176/L248`。

### 6. LPM-Aging实测更快吗？raw等待阈值是什么？

指定CPU模拟中cold首次准入从纯LPM33轮到9轮，但FCFS是1轮；hot等待p95从0到17轮，CPU平均从159.39到373.52μs，最终复用量相同。raw阈值是非空Prefill调度调用步，与Aging有效轮不同，也不是秒数。没有GPU TTFT/吞吐提升证据。证据：`artifacts/lpm_aging/benchmark.json`、`prefill.py:L189/L220`。

## 三、Decode-first-bounded

### 1. 原调度器绝对Prefill优先吗？新策略改变哪一层？

原表达式先尝试正常预算Prefill，无法构成Batch才Decode；因此有可接纳Prefill时优先，并不意味着Prefill无法接纳也会阻断Decode。新策略只选择阶段，内部FCFS/LPM/Aging独立配置，默认仍保留原表达式。证据：HEAD `scheduler.py:L219`、当前`L311`。

### 2. 连续Decode数是会话数还是Batch数？为什么提交后记？

是有Prefill等待时成功提交的连续Decode Batch数，每Batch只加一，不论会话数。overlap先提交当前批再处理上批结果，放到结果处理中会滞后选择周期；当前放在_forward成功路径，避免prepare失败/仅选择也计数，无额外CUDA同步。证据：`scheduler.py:L100/L340/L353`。

### 3. 达到K后会暂停所有Decode强制Prefill吗？

不会。达到K优先尝试保护预算Prefill，失败则继续Decode并保留保护条件；下一轮再次尝试。KV或Table长期不足时实际连续Decode可以超过K，因此只能限制尝试间隔。失败清零会让已经被拒的Prefill再等K个Batch。证据：`scheduler.py:L325–330/L340`。

### 4. 什么情况下用大预算Prefill？

默认Prefill-first总用正常预算；bounded有pending无active Decode时用正常预算；两个阶段竞争且不足K则先Decode，异常无Decode Batch回退也用正常预算；达到K用保护预算。任何Prefill成功提交（含chunk）后清计数。证据：`scheduler.py:L313–333/L348`。

### 5. 小保护预算如何改善Decode？会改变KV预留吗？

它减少一次Prefill Batch新增计算Token量，理论上降低长Prefill对Decode输出间隔的冲击；KV准入仍预留完整剩余Prompt和最大输出，不释放已有Decode KV。长context即使小Token数也可能耗时，所以需要GPU ITL测量。证据：`prefill.py:L79–84/L228`；真实效果未验证。

### 6. CPU Benchmark支持什么结论？

K16/保护256的short模拟中等待Prefill p95为16批，无上限Decode为127批；初始Decode最大输出批间隔为2批，Prefill-first为17批。long/mixed的p95未全面改善；内存/Table紧张时111次保护失败。数字为逻辑批数，不是毫秒TPOT/TTFT。证据：`artifacts/batch_scheduling/benchmark.json`。

## 四、Adaptive Chunk Budget

### 1. 请写出实际预算公式，原正常预算叫什么？

`upper=min(prefill_budget,protected_prefill_tokens)`；关闭返回upper，开启返回`min(upper,max(min_chunk_tokens,target_batch_tokens-len(running_reqs)))`。正常预算来自config.max_extend_tokens，默认8192，代码没有新建max_prefill_length字段。证据：`scheduler.py:L78/L299`。

### 2. decode_tokens与inflight_tokens有什么区别？

前者是活跃真实Decode请求数量，在标准Decode中每请求每批生成一个逻辑Token，作为计算负载信号；后者为剩余输出加分页余量的未来KV资源预留。图padding、pending和ChunkedReq不计入前者。不能拿未来输出总数代替当轮Decode计算量。证据：`decode.py:L14/L28`、`scheduler.py:L308`。

### 3. 为什么只在保护分支使用？

它针对两个阶段竞争时的Prefill突刺；没有活跃Decode不需要为Decode缩预算，所以用normal。K前不尝试保护Prefill；K后才算一次budget并传给原Adder。这样不改变默认策略、纯Prefill吞吐路径或请求内部排序。证据：`scheduler.py:L319–333`。

### 4. floor能保证每个请求至少做64Token吗？需要按页对齐吗？

不能。floor是整个保护Batch的计算预算下限，实际Prompt不足或资源拒绝会少做/不做；多个请求共享budget。计算Token预算不要求按页对齐，但物理KV由allocator舍入，必须保证页级容量；当前审计发现同批短请求尾页估算不足。证据：`prefill.py:L79/L83`、报告P1。

### 5. 它是不是Sarathi式混合Prefill/Decode？参数能自动学吗？

不是：mini-SGLang仍提交独立Prefill或Decode Batch，target减Decode数只是启发式，并无真实混合总Token约束；target/floor手动设置，没有GPU时间反馈或在线学习。负载高时仍保留floor，也不能推导总工作量≤target。证据：`core.py:L72`、`scheduler.py:L299`。

### 6. 有没有TPOT改善证据？如果预算变小效果更差怎么办？

没有GPU证据。mixed192模拟中保护预算256→64，chunk7→8，首次等待p95 131→133批，完成132→133批，CPU中位16.542→18.095μs。设计目标是换取Decode时间保护，是否值得取决于真实ITL/TPOT、TTFT、吞吐的Pareto曲线；应固定K、排序和workload做on/off A/B。证据：`artifacts/adaptive_prefill_budget/benchmark.json`和[Benchmark方案](mini_sglang_four_optimizations_benchmark_plan.md)。

## 一分钟口述

我把问题拆成缓存副作用、Prefill请求排序、Prefill/Decode阶段机会和保护计算预算四个层次。先用只读Radix计划支撑排序与容量预测，准入后再split和加引用；再用有效新准入轮做Aging，避免空转老化；阶段层按Decode提交批数提供Prefill保护机会；保护轮按活跃Decode数缩预算。默认行为保留，249项CPU回归和可复现模型外模拟验证了语义与取舍，但没有声称GPU性能收益。审计还发现了多短请求分页舍入的容量边界，说明计数/引用测试通过与全部资源边界安全是不同验收项。
