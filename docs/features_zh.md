# Mini-SGLang 功能特性

## 在线服务

Mini-SGLang 支持在线服务，提供与 OpenAI 兼容的 API 服务器。它提供标准的 `/v1/chat/completions` 接口，可以与现有工具和客户端无缝集成。详细的命令行参数和配置选项，请运行 `python -m minisgl --help` 查看。

## 交互式 Shell 模式

为了便于演示和测试，Mini-SGLang 提供了交互式 Shell 模式。在该模式下，用户可以直接输入提示词，LLM 会实时生成回复。Shell 会自动缓存聊天历史以维持上下文。要清空对话历史并开始新会话，请使用 `/reset` 命令。

示例：

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## 分布式服务

为了在多 GPU 上扩展性能，Mini-SGLang 支持张量并行（Tensor Parallelism, TP）。你可以通过 `--tp n` 参数指定 GPU 数量来启用分布式服务，其中 `n` 为并行度。

## 支持的模型

当前框架支持以下稠密（Dense）模型架构：

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) 系列
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) 系列（包括 MoE）
- [`Qwen-2.5`](https://huggingface.co/collections/Qwen/qwen25) 系列

## 分块预填充（Chunked Prefill）

分块预填充（Chunked Prefill）是由 [Sarathi-Serve](https://arxiv.org/abs/2403.02310) 提出的一种技术，默认开启。该功能在预填充（Prefill）阶段将长提示词拆分为更小的块，显著降低峰值内存占用，避免长上下文服务中的内存溢出（OOM）错误。分块大小可通过 `--max-prefill-length n` 进行配置。注意，不建议将 `n` 设置为过小的值（例如 128），因为这可能会显著降低性能。

## 页面大小（Page Size）

你可以通过 `--page-size` 参数指定系统的页面大小。

## 注意力后端（Attention Backends）

Mini-SGLang 集成了高性能注意力内核，包括 [`FlashAttention`](https://github.com/Dao-AILab/flash-attention)（`fa`）、[`FlashInfer`](https://github.com/flashinfer-ai/flashinfer)（`fi`）和 [`TensorRT-LLM fmha`](https://github.com/NVIDIA/TensorRT-LLM)（`trtllm`）。它支持为预填充（Prefill）和解码（Decode）阶段使用不同的后端，以最大化效率。例如，在 NVIDIA Hopper GPU 上，默认预填充使用 `FlashAttention 3`，解码使用 `FlashInfer`。

你可以通过 `--attn` 参数指定后端。如果提供两个值（例如 `--attn fa,fi`），第一个指定预填充后端，第二个指定解码后端。注意，某些注意力后端可能会覆盖用户指定的页面大小（例如 `trtllm` 仅支持页面大小为 16、32、64）。

## CUDA 图（CUDA Graph）

为了最小化解码阶段的 CPU 启动开销，Mini-SGLang 支持 CUDA 图的捕获与回放。该功能默认开启。CUDA 图捕获的最大批量大小可通过 `--cuda-graph-max-bs n` 设置。将 `n` 设为 `0` 可禁用该功能。

## 基数缓存（Radix Cache）

Mini-SGLang 沿用了 [SGLang](https://github.com/sgl-project/sglang.git) 的原始设计，实现了基数缓存（Radix Cache）来管理键值（KV）缓存。这使得跨请求共享前缀的 KV 缓存可以被复用，减少冗余计算。该功能默认开启，可通过 `--cache naive` 切换为朴素的缓存管理策略。

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*基数注意力（Radix Attention）示意图，来自 [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/)。*

## 重叠调度（Overlap Scheduling）

为了进一步降低 CPU 开销，Mini-SGLang 采用了重叠调度技术，该技术由 [NanoFlow](https://arxiv.org/abs/2408.12757) 提出。该方法将 CPU 调度开销与 GPU 计算重叠，从而提升系统整体吞吐。

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*重叠调度示意图，来自 [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)。*
