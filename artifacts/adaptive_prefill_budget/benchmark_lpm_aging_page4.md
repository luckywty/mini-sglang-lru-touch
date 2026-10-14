# Batch 调度 CPU Benchmark

环境：AMD Ryzen 7 5800H with Radeon Graphics；Python 3.12.14；Torch 2.9.1+cu128；1 个 Torch 线程。

真实阶段选择、准入、KV 分配及完成释放；Engine 仅模拟进度和输出。Batch 间隔不是 GPU 延迟。

| 负载 | 策略 | 初始 Decode | 自适应 | 目标/下限 | K | 保护预算 | 首次 Prefill 等待 P95（批） | Prefill 完成等待 P95（批） | 初始 Decode 输出最大间隔（批） | 等待时最长连续 Decode | 调度中位数 µs | 保护成功/尝试 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode-only | decode-first-bounded | 16 | False | 256/64 | 16 | 256 | None | None | 1 | 0 | 2.96 | 0/0 |
| decode-only | decode-first-bounded | 192 | False | 256/64 | 16 | 256 | None | None | 1 | 0 | 19.31 | 0/0 |
| decode-only | decode-first-bounded | 16 | True | 256/64 | 16 | 256 | None | None | 1 | 0 | 2.93 | 0/0 |
| decode-only | decode-first-bounded | 192 | True | 256/64 | 16 | 256 | None | None | 1 | 0 | 17.17 | 0/0 |
| long | decode-first-bounded | 16 | False | 256/64 | 16 | 256 | 145 | 145 | 2 | 16 | 3.20 | 7/7 |
| long | decode-first-bounded | 192 | False | 256/64 | 16 | 256 | 145 | 145 | 2 | 16 | 17.76 | 7/7 |
| long | decode-first-bounded | 16 | True | 256/64 | 16 | 256 | 145 | 145 | 2 | 16 | 3.28 | 7/7 |
| long | decode-first-bounded | 192 | True | 256/64 | 16 | 256 | 145 | 146 | 2 | 16 | 19.17 | 7/7 |
| memory-pressure | decode-first-bounded | 16 | False | 256/64 | 16 | 256 | 232 | 232 | 1 | 127 | 6.80 | 0/111 |
| memory-pressure | decode-first-bounded | 192 | False | 256/64 | 16 | 256 | 139 | 139 | 1 | 127 | 287.98 | 0/111 |
| memory-pressure | decode-first-bounded | 16 | True | 256/64 | 16 | 256 | 232 | 232 | 1 | 127 | 10.75 | 0/111 |
| memory-pressure | decode-first-bounded | 192 | True | 256/64 | 16 | 256 | 139 | 139 | 1 | 127 | 286.70 | 0/111 |
| mixed | decode-first-bounded | 16 | False | 256/64 | 16 | 256 | 130 | 130 | 2 | 16 | 3.25 | 7/7 |
| mixed | decode-first-bounded | 192 | False | 256/64 | 16 | 256 | 130 | 130 | 2 | 16 | 18.27 | 7/7 |
| mixed | decode-first-bounded | 16 | True | 256/64 | 16 | 256 | 130 | 130 | 2 | 16 | 3.27 | 7/7 |
| mixed | decode-first-bounded | 192 | True | 256/64 | 16 | 256 | 130 | 132 | 2 | 16 | 19.04 | 7/7 |
| prefill-only | decode-first-bounded | 16 | False | 256/64 | 16 | 256 | 0 | 0 | 0 | 0 | 134.20 | 0/0 |
| prefill-only | decode-first-bounded | 192 | False | 256/64 | 16 | 256 | 0 | 0 | 0 | 0 | 135.61 | 0/0 |
| prefill-only | decode-first-bounded | 16 | True | 256/64 | 16 | 256 | 0 | 0 | 0 | 0 | 138.64 | 0/0 |
| prefill-only | decode-first-bounded | 192 | True | 256/64 | 16 | 256 | 0 | 0 | 0 | 0 | 134.25 | 0/0 |
| short | decode-first-bounded | 16 | False | 256/64 | 16 | 256 | 16 | 16 | 2 | 16 | 3.00 | 1/1 |
| short | decode-first-bounded | 192 | False | 256/64 | 16 | 256 | 16 | 16 | 2 | 16 | 17.78 | 1/1 |
| short | decode-first-bounded | 16 | True | 256/64 | 16 | 256 | 16 | 16 | 2 | 16 | 3.12 | 1/1 |
| short | decode-first-bounded | 192 | True | 256/64 | 16 | 256 | 16 | 16 | 2 | 16 | 17.68 | 1/1 |
| table-pressure | decode-first-bounded | 16 | False | 256/64 | 16 | 256 | 127 | 127 | 1 | 127 | 11.67 | 0/111 |
| table-pressure | decode-first-bounded | 192 | False | 256/64 | 16 | 256 | 127 | 127 | 1 | 127 | 41.25 | 0/111 |
| table-pressure | decode-first-bounded | 16 | True | 256/64 | 16 | 256 | 127 | 127 | 1 | 127 | 11.89 | 0/111 |
| table-pressure | decode-first-bounded | 192 | True | 256/64 | 16 | 256 | 127 | 127 | 1 | 127 | 38.06 | 0/111 |

140 次运行排空并通过表槽唯一性、缓存引用、物理页守恒和唯一性检查。
请求按固定 Batch 序号到达，各策略的到达序列一致；吞吐、TTFT、ITL 需要另做真实 GPU 实验。
