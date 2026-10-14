# Batch 调度 CPU Benchmark

环境：AMD Ryzen 7 5800H with Radeon Graphics；Python 3.12.14；Torch 2.9.1+cu128；1 个 Torch 线程。

真实阶段选择、准入、KV 分配及完成释放；Engine 仅模拟进度和输出。Batch 间隔不是 GPU 延迟。

| 负载 | 策略 | K | 保护预算 | 首次 Prefill 等待 P95（批） | Prefill 完成等待 P95（批） | 初始 Decode 输出最大间隔（批） | 等待时最长连续 Decode | 调度中位数 µs | 保护成功/尝试 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode-only | decode-first-bounded | 16 | 256 | None | None | 1 | 0 | 3.00 | 0/0 |
| decode-only | decode-first-unbounded | None | None | None | None | 1 | 0 | 2.93 | 0/0 |
| decode-only | prefill-first | None | None | None | None | 1 | 0 | 5.34 | 0/0 |
| long | decode-first-bounded | 16 | 256 | 145 | 145 | 2 | 16 | 4.10 | 7/7 |
| long | decode-first-unbounded | None | None | 139 | 139 | 1 | 127 | 3.67 | 0/0 |
| long | prefill-first | None | None | 0 | 0 | 17 | 0 | 4.33 | 0/0 |
| memory-pressure | decode-first-bounded | 16 | 256 | 352 | 352 | 1 | 127 | 4.21 | 0/111 |
| memory-pressure | decode-first-unbounded | None | None | 352 | 352 | 1 | 127 | 3.31 | 0/0 |
| memory-pressure | prefill-first | None | None | 352 | 352 | 1 | 127 | 304.04 | 0/0 |
| mixed | decode-first-bounded | 16 | 256 | 130 | 130 | 2 | 16 | 4.10 | 7/7 |
| mixed | decode-first-unbounded | None | None | 127 | 127 | 1 | 127 | 3.22 | 0/0 |
| mixed | prefill-first | None | None | 0 | 0 | 17 | 0 | 5.53 | 0/0 |
| prefill-only | decode-first-bounded | 16 | 256 | 0 | 0 | 0 | 0 | 211.48 | 0/0 |
| prefill-only | decode-first-unbounded | None | None | 0 | 0 | 0 | 0 | 157.93 | 0/0 |
| prefill-only | prefill-first | None | None | 0 | 0 | 0 | 0 | 167.90 | 0/0 |
| short | decode-first-bounded | 16 | 256 | 16 | 16 | 2 | 16 | 3.31 | 1/1 |
| short | decode-first-unbounded | None | None | 127 | 127 | 1 | 127 | 2.85 | 0/0 |
| short | prefill-first | None | None | 0 | 0 | 17 | 0 | 5.56 | 0/0 |
| table-pressure | decode-first-bounded | 16 | 256 | 169 | 169 | 1 | 127 | 14.03 | 0/111 |
| table-pressure | decode-first-unbounded | None | None | 169 | 169 | 1 | 127 | 3.49 | 0/0 |
| table-pressure | prefill-first | None | None | 169 | 169 | 1 | 127 | 13.86 | 0/0 |

105 次运行排空并通过表槽唯一性、缓存引用、物理页守恒和唯一性检查。
请求按固定 Batch 序号到达，各策略的到达序列一致；吞吐、TTFT、ITL 需要另做真实 GPU 实验。
