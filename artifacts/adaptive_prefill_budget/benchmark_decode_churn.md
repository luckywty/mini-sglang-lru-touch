# Batch 调度 CPU Benchmark

环境：AMD Ryzen 7 5800H with Radeon Graphics；Python 3.12.14；Torch 2.9.1+cu128；1 个 Torch 线程。

真实阶段选择、准入、KV 分配及完成释放；Engine 仅模拟进度和输出。Batch 间隔不是 GPU 延迟。

| 负载 | 策略 | 初始 Decode | 自适应 | 目标/下限 | K | 保护预算 | 首次 Prefill 等待 P95（批） | Prefill 完成等待 P95（批） | 初始 Decode 输出最大间隔（批） | 等待时最长连续 Decode | 调度中位数 µs | 保护成功/尝试 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| long | decode-first-bounded | 16 | False | 256/64 | 16 | 256 | 141 | 141 | 2 | 16 | 3.13 | 7/7 |
| long | decode-first-bounded | 192 | False | 256/64 | 16 | 256 | 145 | 145 | 2 | 16 | 15.85 | 7/7 |
| long | decode-first-bounded | 16 | True | 256/64 | 16 | 256 | 141 | 141 | 2 | 16 | 3.19 | 7/7 |
| long | decode-first-bounded | 192 | True | 256/64 | 16 | 256 | 145 | 146 | 2 | 16 | 15.49 | 7/7 |
| mixed | decode-first-bounded | 16 | False | 256/64 | 16 | 256 | 127 | 128 | 2 | 16 | 3.30 | 7/7 |
| mixed | decode-first-bounded | 192 | False | 256/64 | 16 | 256 | 131 | 132 | 2 | 16 | 16.12 | 7/7 |
| mixed | decode-first-bounded | 16 | True | 256/64 | 16 | 256 | 127 | 128 | 2 | 16 | 3.30 | 7/7 |
| mixed | decode-first-bounded | 192 | True | 256/64 | 16 | 256 | 133 | 133 | 2 | 16 | 16.34 | 7/7 |

40 次运行排空并通过表槽唯一性、缓存引用、物理页守恒和唯一性检查。
请求按固定 Batch 序号到达，各策略的到达序列一致；吞吐、TTFT、ITL 需要另做真实 GPU 实验。
