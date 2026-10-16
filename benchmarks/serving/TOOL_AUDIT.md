# 本机执行工具与计量核对

本次优化源码起点7cd28c7，官方基线9a91cfa。原仓库的offline/client工具可以提供请求和调用结构，但不能直接承担本次正式计量：旧客户端把SSE事件数与token数混用、缺少早期TTFT和完整失败终态账；offline默认资源不适合8GB。五份规划中列出的批量trace回放、服务端token对账、统一统计和开销审计在开始时尚无完整可执行闭环。

本目录补齐的实际实现：

| 职责 | 可执行文件 | 证据 |
| --- | --- | --- |
| 隔离官方/优化源码与共有补丁 | prepare_sources.py | environment/sources、runtime文件manifest、patches |
| 本机依赖与CUDA编译固定 | env.sh | environment版本、uv.lock、编译失败与成功的真实日志 |
| 精确token长度、共享前缀、冻结绝对到达trace | generate_workloads.py | traces JSONL与manifest；BPE往返验证 |
| HTTP1.1/SSE回放和客户端负载审计 | raw_http.py、replay.py | client.json，UID/ACK/空文本/finish/DONE分别记录 |
| 既有host token结果和调度事件观测 | collector.py | measurement.pkl、tokens.bin、requests.jsonl |
| 独立重启、预热、温度窗口、遥测、资源检查 | run_experiments.py、cpu_telemetry.py | 每次run.json、GPU/CPU遥测、stdout/stderr |
| pilot、冻结、配对矩阵和中断恢复 | suite.py、freeze_campaign.py | frozen-config、seed目录、PROGRESS、extension-decision |
| 输入/输出/终态/资源先审计再统计 | analyze_results.py | 逐run CSV/JSON、IQR/MAD、按run配对bootstrap |
| 最终图表、负结果和简历证据审核 | make_report.py、finalize_campaign.py | REPORT.md、静态SVG/PNG及绘图源数据、错误台账、覆盖表 |
| 控制器存活与退出后报告生成 | watch_campaign.py | WATCHER_STATE；不会把控制器退出当作实验完成 |

观测点没有加入逐token打印、额外GPU同步或GPU到CPU拷贝。输入ID来自原有CPU请求；token ID来自引擎原有拷贝结果。显存峰值只在run边界重置和读取。Light保留全部token时间、phase与budget，Decode请求详情每32个batch采一行；详细probe/touch/lock计数缺失不是零。

开关与实际路径对应：

- A是S源码默认两阶段probe/refresh/commit路径，无独立A0开关。H*→S1000只能解释默认实现净差异。
- B通过fcfs/lpm/lpm-aging候选选择和实际首次准入cached_len核对；Aging64与纯LPM分别运行。
- C通过真实Decode/Prefill提交、竞争队列、连续Decode与protected budget事件核对；decode-first-bounded才进入该路径。
- D只在C保护性Prefill路径计算`min(normal,U,max(floor,T-n_decode))`。与相同K=4固定64及固定24都比较，floor8未被实际达到时不宣称GPU边界已验证。
- B依赖A、D依赖C。不存在独立A0+B或C0+D；合法条件交互只使用同trace的完整区块。

原样上游在本机overlap存在真实终态错误，H*/S共用最小host终态修正；原始失败保留。严格跨形状greedy/HF误差门槛与最终固定到达尾延迟开销校准都没有通过。报告必须保留这些失败，不能通过运行更多配置宣称门槛已经通过。
