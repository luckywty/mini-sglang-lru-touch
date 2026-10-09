# Mini-SGLang 结构

## 系统架构

Mini-SGLang 被设计为一个分布式系统，以高效地处理大语言模型（LLM）推理。它由多个协同工作的独立进程组成。

### 核心组件

- **API Server（API 服务器）**：用户入口。它提供与 OpenAI 兼容的 API（例如 `/v1/chat/completions`），用于接收提示词并返回生成的文本。
- **Tokenizer Worker（分词工作进程）**：将输入文本转换为模型可以理解的数字（Token）。
- **Detokenizer Worker（解码工作进程）**：将模型生成的数字（Token）转换回人类可读的文本。
- **Scheduler Worker（调度工作进程）**：核心工作进程。在多 GPU 配置中，每个 GPU 对应一个 Scheduler Worker（称为一个 **TP Rank**）。它管理该 GPU 上的计算与资源分配。

### 数据流

各组件之间使用 **ZeroMQ（ZMQ）** 传递控制消息，使用 **NCCL**（通过 `torch.distributed`）在 GPU 之间交换大体量的张量数据。

![Process overview diagram](https://lmsys.org/images/blog/minisgl/design.drawio.png)

**请求生命周期：**

1. **用户**向 **API Server** 发送请求。
2. **API Server** 将请求转发给 **Tokenizer**。
3. **Tokenizer** 将文本转换为 Token，并发送给 **Scheduler（Rank 0）**。
4. **Scheduler（Rank 0）** 将请求广播给其他所有的 Scheduler（多 GPU 场景下）。
5. **所有 Scheduler** 对请求进行调度，并触发各自本地的 **Engine** 计算下一个 Token。
6. **Scheduler（Rank 0）** 收集输出的 Token 并发送给 **Detokenizer**。
7. **Detokenizer** 将 Token 转换为文本，并回传给 **API Server**。
8. **API Server** 将结果流式返回给**用户**。

## 代码组织（`minisgl` 包）

源代码位于 `python/minisgl`。以下是面向开发者的模块说明：

- `minisgl.core`：提供核心数据类 `Req` 和 `Batch`（表示请求状态）、`Context` 类（保存推理上下文的全局状态），以及 `SamplingParams` 类（保存用户提供的采样参数）。
- `minisgl.distributed`：提供张量并行中的 all-reduce 和 all-gather 接口，以及数据类 `DistributedInfo`（保存 TP 工作进程的 TP 信息）。
- `minisgl.layers`：实现构建支持 TP 的 LLM 的基本模块，包括 linear、layernorm、embedding、RoPE 等。它们共享定义在 `minisgl.layers.base` 中的公共基类。
- `minisgl.models`：实现 LLM 模型，包括 Llama 和 Qwen3。同时定义了从 Hugging Face 加载权重并对权重做分片的工具函数。
- `minisgl.attention`：提供注意力（Attention）后端的接口，并实现了 `flashattention` 和 `flashinfer` 后端。它们由 `AttentionLayer` 调用，并使用存储在 `Context` 中的元数据。
- `minisgl.kvcache`：提供 KVCache 池和 KVCache 管理器的接口，并实现了 `MHAKVCache`、`NaiveCacheManager` 和 `RadixCacheManager`。
- `minisgl.utils`：提供常用工具集合，包括日志配置和对 zmq 的封装。
- `minisgl.engine`：实现 `Engine` 类，即单个进程上的 TP 工作进程。它管理模型、上下文、KVCache、注意力后端以及 CUDA 图回放。
- `minisgl.message`：定义 api_server、tokenizer、detokenizer 和 scheduler 之间（通过 zmq）交换的消息。所有消息类型均支持自动序列化与反序列化。
- `minisgl.scheduler`：实现 `Scheduler` 类，运行在每个 TP 工作进程上并管理对应的 `Engine`。Rank 0 的 scheduler 从 tokenizer 接收消息，与其他 TP 工作进程上的 scheduler 通信，并向 detokenizer 发送消息。
- `minisgl.server`：定义命令行参数和 `launch_server`（用于启动 Mini-SGLang 的所有子进程）。同时在 `minisgl.server.api_server` 中实现了一个基于 FastAPI 的前端服务器，提供 `/v1/chat/completions` 等接口。
- `minisgl.tokenizer`：实现 `tokenize_worker` 函数，处理分词与解码请求。
- `minisgl.llm`：提供 `LLM` 类，作为与 Mini-SGLang 系统交互的便捷 Python 接口。
- `minisgl.kernel`：实现自定义 CUDA 内核，基于 `tvm-ffi` 提供 Python 绑定和 JIT 接口。
- `minisgl.benchmark`：基准测试工具。
