---
title: Dataset 预处理
date: 2026-03-25T23:13:25+08:00
featuredImage: http://img.xilyfe.top/img/20260325231453933.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 数据集
lastmod: 2026-06-09T02:41:55+08:00
---
{{< admonition type=info title="Summary">}} 
这一篇笔记主要记录一下 Huggingface 库提供的 Trainer 接受的数据集格式。
{{< /admonition >}}

## Trainer 数据集处理

### text/纯文本

这一类型的数据一般用于**语言模型预训练、领域继续训练**，目标是预测下一个 token，我们通常采用 Huggingface 提供的 Trainer。由于数据集是单纯的文本不是对话数据，所以不需要 `apply_chat_template`。其次我们需要先明确一点，next-token-prediction 训练的过程中我们需要传给模型的是：

```json
{
	"input_ids": tensor,
	"attention_mask": tensor,
	"labels": tensor
}
```

>这里不需要手动对 labels 进行 shift，模型内部会自动偏移。

```python
def tokenize_fn(examples: dict[str, list]):
	inputs = tokenizer(
		examples["text"],
		truncation=True,
		padding="max_length",
		max_length=512,
	)
	inputs["labels"] = inputs["input_ids"].copy()
	return inputs

lm_dataset = load_dataset("json", data_files="").map(tokenize_fn, batched=True)
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./out", ...),
    train_dataset=lm_dataset["train"],
)
trainer.train()
```

这里可能会疑惑：刚刚不是说 Dataset 应该传张量出来吗，为什么这里返回的是 `list[int]`？这里就需要提到 Trainer 内部的 collator 这个东西。当我们没有手动指定 collator 时候，它会采用默认的 `DataCollatorWithPadding`，他有如下两个功能：
1. 根据 batch 进行动态 padding，所以如果我们没有指定 `padding="max_length",max_length=512` 也可以。
2. 把 `input_ids`、`attention_mask`、`labels` 从 `list[int]` 转为张量。

但是上面的代码存在一个问题，loss 计算交叉熵损失时候有一个 `ignore_index=-100` 的参数，它不会把 `label_id=ignore_index` 参与计算损失，而我们目前的代码没有处理 labels（把 pad token 对应的 labels 改成 -100）。要么我们手动对 labels 进行处理：

```python
def tokenize_padding_fn(examples: dict[str, str]):
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    inputs["labels"] = [
        [
            input_id if input_id != tokenizer.pad_token_id else -100
            for input_id in sample
        ]
        for sample in inputs["input_ids"]
    ]
    return inputs
```

但是这种方式效率比较低，静态 padding 会把每条样本补到固定长度，可能出现 20 → 512 的情况；而动态 padding 在组 batch 时再按当前 batch 的最长序列补齐，假如这个 batch 的最大样本长度为 128，那么所有样本都会 pad 补充到 128 的长度。我们可以通过 `DataCollatorForLanguageModeling` 这个 transformers 库提供的 collator 来实现。

```python
def tokenize_fn(examples: dict[str, list]):
	return tokenizer(
		examples["text"],
		truncation=True,
		max_length=512,
	)

lm_dataset = load_dataset("json", data_files="").map(tokenize_fn, batched=True)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./out", ...),
    train_dataset=lm_dataset["train"],
    data_collator=collator,
)
trainer.train()
```

这里就指定了 `max_length` 让它截断长度超过 512 的样本，然后 padding 就交给 collator 来实现。DataCollatorForLanguageModeling 会做以下几件事：
1. 动态 padding，自动对齐 batch 内最长序列。
2. 复制 `labels = input_ids.clone()`，并且把 labels 中 pad 的部分置为 `ignore_index`
3. 把 `input_ids`、`attention_mask`、`labels` 从 `list[int]` 转为张量。

>`DataCollatorForLanguageModeling` 中 `mlm=False` 用于 GPT / Qwen / LLaMA 等模型，它干的就是上述操作。`mlm=True` 用于 Bert，它会进行随机掩码。

{{< admonition type=info title="总结">}} 
通过 Trainer 进行预训练，我们需要通过 `mapping_func` 对数据集进行预处理，使其返回 `{input_ids, attention_mask}` 格式的数据。labels 的处理有两种路径：
- 手动处理：`padding="max_length"` + 手动把 pad 位置改为 `-100`，依赖默认 `default_data_collator` 转 tensor
- 交给 collator：不做 padding，使用 `DataCollatorForLanguageModeling` 实现动态 padding + 自动构造 labels
{{< /admonition >}}

### messages

messages 类型的数据集指的是 json 或者 jsonl 文件，数据格式为：

```json
[
	{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]},
	{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]},
]
```

其次这种对话数据集一般用于 sft 阶段，所以这里我用 SFTTrainer 举例，SFTTrainer 是 TRL 在 Trainer 上的封装。

```python
lm_dataset = load_dataset("json", data_files="", split="train")
training_args = SFTConfig(
    output_dir="sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_length=2048,
    packing=True,
    packing_strategy="bfd",
    assistant_only_loss=True,  # 只算 assistant 的 loss
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
```

如果数据集的是标准的 messages 列，SFTTrainer 内部就会自动调用 `apply_chat_template` 给对话应用模板，然后再 tokenize。需要注意 SFT 时候我们通常只计算 assistant 部分的损失，所以 `assistant_only_loss=True` 就会自动帮我们给 user 部分的 label mask 掉。

如果你的 dataset 不是标准格式，比如：

```json
[
	{"instruction": "...", "output": "..."},
	{"instruction": "...", "output": "..."}
]
```

我们有两种处理方式，第一种就是我们预处理数据集，使其返回标准的 messages 格式：

```python
def to_messages(example):
    return {"messages": [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]}
lm_dataset = load_dataset("json", data_files="instruction_output.jsonl").map(to_messages)
```

第二种方式就是我们可以用 `formatting_func` 把自定义格式转成 `apply_chat_template` 后的 **最终字符串**。

```python
def formatting_func(example):
	messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    return tokenizer.apply_chat_template(
	    messages, 
        tokenize=False,
        add_generation_prompt=False,
    )

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_func,
)
```

然后 SFTTrainer 是继承 Trainer 的子类，所以 Trainer 的方法他也支持，我们可以在 dataset 的 `mapping_func` 中进行预处理，返回 `{input_ids,attention_mask}`。但是这种方法就没办法设置 SFTTrainer 的 `assistant_only_loss` 参数，让它只计算 assistant 部分的损失了。

{{< admonition type=question title="为什么 assistant_only_loss 失效？">}} 
`assistant_only_loss=True` 的实现依赖于内部的 tokenize 步骤：
1. 调用 `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)` 来生成 assistant_masks。
2. 再用这个 mask 把 user/system 部分的 labels 设为 `ignore_index`，只保留 assistant 部分的 loss。

而你通过 mapping_fn 预先提供了 input_ids 时，这个 mask 生成步骤被完全跳过，所以 `assistant_only_loss=True` 被静默忽略了。官方文档明确说明：
>The trainer accepts datasets that already contain an input_ids field (tokenized). In this case the trainer skips the internal tokenization step and uses the provided input_ids directly.

{{< /admonition >}}



