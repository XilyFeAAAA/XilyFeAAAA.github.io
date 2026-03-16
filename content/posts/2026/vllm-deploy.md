---
title: vLLM 部署大模型
date: 2026-03-15T23:03:06+08:00
featuredImage: http://img.xilyfe.top/img/20260316120049695.png
authors:
  - Xilyfe
series:
  - 推理框架
tags:
  - Inference
  - 大模型
lastmod: 2026-03-16T10:48:53+08:00
---
## 概述

大模型推理有多种方式比如
- 最基础的 HuggingFace Transformers
- TGI
- vLLM
- Triton + TensorRT-LLM
- …

其中，热度最高的应该就是 vLLM，性能好的同时使用也非常简单，上一次分析了 vLLM 如何实现这么高的性能，这次记录一下如何使用 vLLM 来启动大模型推理服务。

## 部署

如果用的是 NVIDIA GPU，可以用 pip 直接安装 vLLM：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm
```

在 Mac 上虚拟克隆 vLLM 仓库自己编译：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install -e .
```

## 准备模型

vLLM 支持 Huggingface 格式存储的模型，里面应该包含必要文件如下：

```
 ├── config.json
 ├── model.safetensors
 ├── tokenizer.json
 ├── tokenizer_config.json
 ├── special_tokens_map.json
 └── generation_config.json
```

>需要注意 vLLM 仅支持大部分开源的模型框架，如果是它不支持的模型架构 vLLM 会报错 `Unsupported model architecture`

## 推理

vLLM 支持两种推理服务。

### 兼容 OpenAI 的服务器

vLLM 支持提供 OpenAI 格式的 API,启动命令如下：

```bash
modelpath=/models/Qwen1.5-1.8B-Chat

python3 -m vllm.entrypoints.openai.api_server \
        --model $modelpath \
        --trust-remote-code
```

输出如下

```python
INFO:     Started server process [614]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

对于多卡则是增加参数 `tensor-parallel-size` ，将该参数设置为 GPU 数量即可，vLLM 会启动 ray cluster 将模型切分到多个 GPU 上运行，对于大模型很有用。

```bash
python3 -m vllm.entrypoints.openai.api_server \
        --model $modelpath \
        --tensor-parallel-size 8 \
        --trust-remote-code
```

![image.png](http://img.xilyfe.top/img/20260316193432019.png)

发送 OpenAI 格式的请求就可以：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"}
        ]
    }'
```

用之前 RLHF 训练的模型测试，结果如下：

![image.png](http://img.xilyfe.top/img/20260316194654123.png)


### 离线批处理推理

```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(model="facebook/opt-125m")
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

if __name__ == "__main__":
    main()
```

{{< admonition type=warning title="聊天模板">}} 

需要注意：`generate` 方法默认不会将模型的聊天模板应用到输入提示上。因此，如果你使用 Instruct 模型或 Chat 模型，需要手动应用相应的聊天模板：

```python
# Using tokenizer to apply chat template
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/path/to/chat_model")
messages_list = [
    [{"role": "user", "content": prompt}]
    for prompt in prompts
]
texts = tokenizer.apply_chat_template(
    messages_list,
    tokenize=False,
    add_generation_prompt=True,
)

# Generate outputs
outputs = llm.generate(texts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Using chat interface.
outputs = llm.chat(messages_list, sampling_params)
for idx, output in enumerate(outputs):
    prompt = prompts[idx]
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

{{< /admonition >}}

{{< admonition type=warning title="BUG">}} 

![image.png](http://img.xilyfe.top/img/20260316204534963.png)

在调试过程中发现了一个问题，输出的文本提前被截断了。我在 LLM 的配置参数中设置了 `max_model_len=6000`，为什么还会出现这个问题呢？

![image.png](http://img.xilyfe.top/img/20260316205126628.png)

我通过断点查看了 output 里面输出结束的原因 `finish_reason` 显示是 `length`，说明确实是超过 max_length 被截断了。查了 GPT，原来 LLM 里面的参数 `max_model_len` 指的是 prompt + response 的总长度，我们可以在 `SamplingParams` 里面设置 `max_tokens`，这个设置的是 response 的最大 token 数，改到 256 之后就回复正常了。

{{< /admonition >}}

## 量化

vLLM 的量化分为两种：**权重量化** 和 **KVCache 量化**。

```python
llm = LLM(
    model="model",
    quantization="bitsandbytes",
	kv_cache_dtype="fp8", # KV cache量化  
	calculate_kv_scales=True
)
```

>实测 Qwen3-0.6B 可以减少模型一半的显存占用。

## Lora

要求是 LoRA 必须是 PEFT adapter 格式。vLLM 直接兼容 **PEFT** 保存的 LoRA 目录结构。目录里必须有两个关键文件：

```
adapter_config.json  
adapter_model.safetensors
```

其中 adapter_config 记录 LoRA rank、target modules 等配置；adapter_model 保存 A/B 矩阵权重。如果只有 `pytorch_model.bin` 或其他文件格式，vLLM 是识别不了的。

```python
llm = LLM(
    model="model/qwen2-7b",
    enable_lora=True,
    max_lora_rank=64,
    max_loras=2,
)

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("Adapter_Name", 1, Adapter_Path)
)
```

- `--enable-lora`：打开 LoRA 功能，否则 vLLM 不会加载 LoRA adapter。
- `--max-loras`：一个 batch 里最多支持多少个不同的 LoRA， vLLM 支持 不同请求用不同 LoRA。
- `--max-lora-rank`：LoRA 的 rank 上限，必须 ≥ 你所有 LoRA rank 的总和。

```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "base_model",
  "messages": [{"role": "user", "content": "你好"}],
  "lora": "my_lora"
}'
```

在请求里必须指定好用的哪个 LoRA。
