---
title: transformers 库提供的更方便的 Trainer
date: 2026-02-18T21:49:24+08:00
featuredImage: http://img.xilyfe.top/img/20260218215148609.png
authors:
  - Xilyfe
series:
  - Transformer相关
  - LLM
tags:
  - 大模型
lastmod: 2026-02-18T09:51:55+08:00
---
## SFTTrainer

### 函数调用

我们主要依赖 Huggingface 的 transformers 库以及 trl 库来进行自动化微调。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
```

SFTTrainer 负责模型的监督微调，SFTConfig 传入了训练时必须的参数，包括学习率、batch_size等等。

```python
sft_config = SFTConfig(
    output_dir=args.save_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.accumulation_steps,
    learning_rate=args.lr,
    logging_steps=args.log_interval,
    fp16=(args.dtype == "float16"),
    bf16=(args.dtype == "bfloat16"),
    num_train_epochs=args.epochs,
    save_steps=args.save_interval,
    optim="paged_adamw_32bit",
    max_length=args.max_length,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=custom_dataset,
)

trainer.train()
```

### 数据内容格式

#### Tensor Format

```python
return {
	"input_ids": tensor,
    "attention_mask": tensor,
    "labels": tensor
}	
```

当 SFTTrainer 检测到传入的 `train_dataset` 已经包含了 `input_ids` 等模型需要的张量字段时，它会跳过内部的格式化逻辑，直接把数据喂给模型。

#### Conversation Format

第二种方法是把数据预处理为通用的格式：

```python
return {
	"messages": [
        {"role": "user", "content": "你好，请介绍一下你自己。"},
        {"role": "assistant", "content": "你好！我是由 AI 训练的大语言模型。"}
    ]
}
```

**一定要包含 `messages` 这个 key**，SFTTrainer 会自动读取模型内置的 `tokenizer.chat_template` 来拼接对话。

#### Raw Text Format

第三种方法是用纯文本格式，我们需要在数据预处理阶段把所有 Prompt 和 Answer 拼成了一个长字符串，存在 `text` 字段中。

```python
return {"text": "User: 1+1等于几？ Assistant: 等于2。"}
```

在 SFTConfig 中我们需要手动指定 `dataset_text_field`，标志了文本存在哪个字段。

#### Instruction Format

如果我们监督微调的数据是 instruction 类型的，例如 “把句子翻译成英文：今天天气很好”，那我们就可以采用这种内容格式：

```python
return {
    "instruction": "把句子翻译成英文",
    "input": "今天天气很好",
    "output": "The weather is nice today."
}
```

这种方式我们需要明确传入 `formatting_func` 来组合 instruction、input 和 output，例如：

```python
def my_template(instruction, input_text, output_text):
    return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n###Response:\n{output_text}"

def formatting_prompts_func(examples):
    output_texts = []
    for example in range(examples):
        text = my_template(
            example['instruction'],
            example['input'],
            example['output']
        )
        output_texts.append(text)
    return output_texts
```

### 数据化载体

1. 直接传 `list`，SFTTrainer 内部会偷偷把 `list` 转换成 `datasets.Dataset` 处理
2. `torch.utils.data.Dataset`  就是我们之前继承实现的自定义数据集，在 `__getitem__` 里面可以灵活处理复杂的逻辑。
3. `datasets.load_dataset` 好处是支持磁盘映射、多进程预处理、自动缓存。
