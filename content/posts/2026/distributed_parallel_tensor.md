---
title: 分布式训练技术 - 张量并行（待完成）
date: 2026-03-15T19:50:43+08:00
featuredImage: http://img.xilyfe.top/img/20260315231138396.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 分布式
lastmod: 2026-03-15T11:12:08+08:00
---

### 模型

在之前的 inference 中，我们调用的是 transformers 库的 `AutoModelForCausal` 来获得 huggingface 已经预留的模型。比如我们通过 `AutoModelForCausal.from_pretrained` 来调用一个预训练的Qwen3 模型，这时候 transformers 库会根据我们传入的路径中的模型信息，例如 `num_layers` 等，实例化它已经存储过的 Qwen3 Model，在把权重载入模型。

但是 vLLM 中没有像 Huggingface 这样预定义大量开源模型，让我们像 ，这就需要我们手动定义好模型。

```python
class Qwen3Attention(nn.Module)

class Qwen3MLP(nn.Module)

class Qwen3DecoderLayer(nn.Module)

class Qwen3Model(nn.Module)

class Qwen3ForCausalLM(nn.Module)
```

Qwen3 就是对 Llama 模型进行了一些修改，后面我也会开一个文章学习一下 Qwen3 的技术报告。这里我们需要关注两个东西，首先模型内部用的不是 torch 自带的 `nn.Linear` 或者 `nn.Embedding`，而是在此基础上封装的了一个张量并行（Tensor Parallelism）版，这个我后面也会开一个文章学习，暂且不谈。其次就是它的 Attention Module。

