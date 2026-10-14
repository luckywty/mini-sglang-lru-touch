# mini-SGLang 项目简历描述

依据：[代码审计报告](mini_sglang_four_optimizations_review.md)。本文件描述当前已有实现；CPU模拟数据不转换为GPU TTFT、TPOT或吞吐提升。

## 总项目介绍

**mini-SGLang 推理调度与 Prefix KV Cache 优化**：基于现有 Continuous Batching、Radix Prefix Cache 和 Chunked Prefill，完成缓存两阶段匹配、LPM/有效轮Aging、有限次数Decode优先和自适应保护预算的局部改造。保持默认FCFS/Prefill-first配置与模型执行流程，建立CPU回归和模型外调度Benchmark，分析缓存复用、请求等待与Decode保护之间的取舍。

## 四项核心技术贡献

- **隔离匹配副作用**：设计只读Radix probe与延迟commit，记录匹配节点、页对齐长度、新增保护容量及结构版本；在容量准入前刷新计划，检查通过后才split、加引用和更新LRU，减少被拒请求的缓存扰动并支持排序计划复用。
- **兼顾前缀复用与请求公平**：实现FCFS/LPM/LPM-Aging可选策略，以`cached_len + N × 有效等待轮`排序；只在其他新请求成功准入后老化等待项，保持同分到达序、chunk续算优先和队列过长FCFS降级。
- **控制阶段服务机会**：实现可选Decode-first-bounded，按等待期间成功提交的Decode Batch计数，在K次后尝试小预算Prefill，资源拒绝时保留保护状态；计数位置适配normal与overlap的提交顺序。
- **按活跃Decode负载调整保护计算量**：增加受正常/保护双上限和最低预算约束的自适应Chunk Budget，仅影响保护轮次；复用原Prefill准入、Table和KV生命周期，补充动态负载、续算、Abort与资源压力测试。

对应代码分别是：`kvcache/{base,radix_cache,naive_cache}.py + scheduler/{cache,prefill}.py`；`scheduler/{utils,prefill,config,scheduler}.py + server/args.py`；`scheduler/{scheduler,config}.py + server/args.py`；`scheduler/{scheduler,config}.py + server/args.py`。详细行号与测试在审计报告。

## 可直接使用的精简版本

**mini-SGLang 推理调度优化｜Python / PyTorch / KV Cache**

- 基于Radix Prefix Cache实现只读匹配计划与延迟提交，增加结构失效重验证及锁后可用容量预测；支持LPM、有效准入轮Aging、稳定到达序和长队列FCFS退化。
- 实现有限次数Decode优先与活跃Decode负载驱动的保护性Prefill预算，保持默认调度行为，适配Chunked Prefill和overlap提交生命周期。
- 完成249项CPU回归，构建可复现模型外调度Benchmark；在指定冷热混合模拟中将cold首次准入从纯LPM的33轮提前至9轮，并量化hot等待和CPU开销代价。

若版面只有两行，可删除最后一条的轮数，保留“完成CPU回归与可复现模型外Benchmark，分析公平性与调度开销”。

## 面试中的证据与边界

轮数结果来自`artifacts/lpm_aging/benchmark.json`，配置为hot共享前缀64、预算4、N=8、每轮4个hot、32轮到达；不是毫秒TTFT。历史2,500次运行是多个不同配置的CPU模拟，不能宣传为GPU压测规模。

不能写“TPOT降低”“QPS提升”“GPU利用率提升”或“严格防饥饿”：没有真实GPU测量；K限制尝试、Aging限制排序，资源不够仍拒绝。不要写“完全杜绝泄漏/OOM”：审计已复现多短请求在page_size=4下的分页容量缺口，业务代码尚未修复。可以如实描述“审计定位Token准入与物理页舍入不一致的边界问题，给出复现与后续验证方案”。
