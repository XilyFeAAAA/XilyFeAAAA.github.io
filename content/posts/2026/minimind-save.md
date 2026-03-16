---
title: MiniMind 学习指北(七)：模型存储
date: 2026-02-15T14:08:48+08:00
featuredImage: http://img.xilyfe.top/img/20260122134824760.png
authors:
  - Xilyfe
series:
  - minimind
tags:
  - 大模型
  - 深度学习
lastmod: 2026-03-16T06:10:33+08:00
---
>在之前的训练中，我们都是通过 `torch.save()` 来保存模型的权重，我们是可以通过训练脚本来加载并且评估，但是在推理框架上这种权重就无法直接使用，所以这篇记录一下如何保存和使用 Huggingface 格式的模型。

在深度学习中，模型通常会以不同的格式保存，不同框架或工具链会使用不同的标准。最基础的是 **PyTorch 原生格式**。这种方式通常通过 `torch.save` 保存模型权重或整个模型对象。例如我们之前就是通过 `model.state_dict()` 获得模型的权重，然后调用 `torch.save()` 保存。这种格式的特点是非常灵活，可以保存任何 Python 对象，例如模型权重、优化器状态、训练轮数等，例如：

```python
 resume_data = {
    "model": state_dict,
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "step": step,
    "wandb_id": wandb_id
}
torch.save(resume_data, "checkpoint.pth")
```

但缺点是缺乏统一标准，不同项目之间很难直接复用。

>保存的后缀名是 `.pt` 或者 `.pth` 没有影响，每个人习惯不同。


另一类是 **Hugging Face Transformers 标准格式**。这是目前大模型生态最常见的一种格式，由 Hugging Face 在其库 Transformers 中定义。模型通过 `save_pretrained` 保存，通过 `from_pretrained` 加载。这种格式的优点是统一规范，很多推理框架都可以直接使用，例如 vLLM。HuggingFace 格式的模型结构如下：

```
my_model/
 ├── config.json
 ├── model.safetensors
 ├── tokenizer.json
 ├── tokenizer_config.json
 ├── special_tokens_map.json
 └── generation_config.json
```

- **config.json**：这是模型的结构配置文件，包含模型架构信息，例如层数、隐藏层维度、attention head 数量等。加载模型时，Transformers 会先读取这个文件，再构建模型结构。
- **model.safetensors**：这是 Hugging Face 推出的新型权重格式，主要解决安全性和加载速度问题。很多模型仓库已经从 `pytorch_model.bin` 迁移到 `model.safetensors`。
- **tokenizer.json**：Tokenizer 的相关信息，涵盖了 `merges.json` 和 `vocab.json`

HuggingFace 中 `save_pretrained` 保存的模型包含了完整信息（结构+权重+tokenizer），各种推理框架可以直接加载。

```python
model.save_pretrained(PATH, safe_serialization=True)
config.save_pretrained(PATH)
tokenizer.save_pretrained(PATH)
```

{{< admonition type=info title="注意点">}} 
如果要让模型变成 Huggingface Transformers 兼容模型需要进行一定修改。

1. config 类需要继承 `transformers.PretrainedConfig`，model 需要继承 `transformers.PreTrainedModel`
2. model 内部必须定义类成员 `config_class`

同时我们需要注意，例如 vLLM、sgLang 的推理框架内部都事先存储了有名的开源模型结构，比如：Llama、Mistral 或者 Qwen 等等。如果 architecture 是我们自己定义的，我们需要给 vLLM 写 custom model loader。
{{< /admonition >}}