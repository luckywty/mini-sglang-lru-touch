"""最简测试: cuda_graph_max_bs=0 走 eager 模式 (禁用 CUDA Graph).

运行方式 (在 mini-sglang 根目录下):
    python python/test/test1.py
"""

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    # cuda_graph_max_bs=0 -> GraphRunner.max_graph_bs=0, 全部走 eager, 不捕获/回放 CUDA graph
    llm = LLM(
        "/home/wty238388/ai-infra/models/Qwen3-0.6B",
        cuda_graph_max_bs=0,
    )

    prompts = [
        "Hello, who are you?",
        "1 + 1 = ?",
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

    outputs = llm.generate(prompts, sampling_params)

    for prompt, out in zip(prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Output: {out['text']}")
        print(f"Tokens: {out['token_ids']}")
        print("-" * 40)

    # 简单断言: 每个请求都生成了 token
    assert len(outputs) == len(prompts)
    assert all(len(out["token_ids"]) > 0 for out in outputs)
    print("test1 passed (eager mode, cuda_graph_max_bs=0).")


if __name__ == "__main__":
    main()
