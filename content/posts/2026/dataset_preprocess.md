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
lastmod: 2026-03-27T02:32:03+08:00
---
{{< admonition type=info title="Summary">}} 
这一篇笔记主要记录一下 Huggingface 库提供的 Trainer 接受的数据集格式，以及 Packing 和 Padding 两种对其策略。
{{< /admonition >}}

## 对其策略

| **Padding** | **Packing**       |                |
| ----------- | ----------------- | -------------- |
| 思路          | 每条样本补齐到固定长度       | 多条样本拼接填满 block |
| 计算效率        | 低（大量无效 token）     | 高（几乎无浪费）       |
| 实现复杂度       | 简单                | 较复杂            |
| 适用场景        | 样本长度均匀 / finetune | Pretrain 主流做法  |

### Padding

Pretrain 几乎不用纯 padding，但有时在 eval 或 SFT 阶段使用。它的思路就是用 pad_token 把每一个 sample 都补齐到相同的长度，因为不同长度的 sample 无法组成一个张量。但它存在的问题就是：为了补齐到相同长度，我们不得已加入无意义的 pad_token，可能导致某些 sample 的长度 10 → 512 的情况，浪费大量算力。

```python
def tokenize_padding_fn(examples: dict[str, str]):
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=block_size,
    )
    inputs["labels"] = [
        [
            input_id if input_id != tokenizer.pad_token_id else -100
            for input_id in sample
        ]
        for sample in inputs["input_ids"]
    ]
    return inputs

lm_datasets = dataset.map(
    tokenize_padding_fn,
    batched=True,
    num_proc=data_args.preprocessing_num_workers.
    remove_columns=list(dataset["train"].features()),
    desc="Running tokenizer on dataset",
)
```

{{< admonition type=warning title="ignore_index">}} 
通过 padding 进行对齐会出现大量无意义的 pad_token，而在计算损失时候这些 token 不应该参与计算。所以我们需要在 labels 里面把 pad_token 对应的位置置为 `ignore_index`，这样计算交叉熵损失时候才会忽略 `label=ignore_index` 的 token。我们也可以通过 collator 来自动添加 `ignore_index`，这样就不用手动处理了。
{{< /admonition >}}

### Packing

![image.png](http://img.xilyfe.top/img/20260326182310593.png)


packing 对齐的思想就是，假如每个样本长度各不同，我们需要通过 pad 补齐到相同长度，那是不是可以把下一个样本补上来，变成 \[sample1, eos_token, sample2, eos_token, sample3_truncated]，这样就可以最大化利用算力，不用 padding 填充了。 

```python
def tokenize_packing_fn(examples: dict[str, str]):
	inputs = tokenizer(examples["text"], add_special_token=True)
	all_input_ids = []
	all_attn_masks = []
	for ids, masks in zip(inputs["input_ids"], inputs["attention_mask"]):
	    all_input_ids.extend(ids)
	    all_attn_masks.extend(masks)
	num_blocks = len(all_input_ids) // block_size
	output = {
	    "input_ids": [
	        all_input_ids[i * block_size : (i + 1) * block_size]
	        for i in range(num_blocks)
	    ],
	    "attention_mask": [
	        all_attn_masks[i * block_size : (i + 1) * block_size]
	        for i in range(num_blocks)
	    ],
	}
	output["labels"] = output["input_ids"].copy()
	return output
```

使用了 packing 就需要考虑以下问题，attention mask 和 position embedding：
1. 位置编码本来每个 sample 都应该从 0 开始，但是现在把 sample1/2/3 packing 到一起，那 sample2/3 的位置编码就变了，不是从 0 开始了而是有了 offset。
2. attention mask 的目的是确保某个 token 只能看到上文不能看到下文，但是采用了 packing 就可能导致某个 token 看到上一个 sample，也就是说计算 attention 时候 sample2 可能注意到 sample1 的 K/V。
3. 如果 packing 导致一个 sample 被截断，那么在下一个 block 的后半部分 sample 计算 attention 时候就看不到上文信息了。

先说前两个问题，现在模型的 `forward` 方法都允许我们自己传入 `position_ids` 和 `attention_mask`：

![image.png](http://img.xilyfe.top/img/20260326185908975.png)

`position_ids` 我们传入 \[0,1,2,3,4,0,1,2,3,0,1]，保证每个 sample 都从 0 开始就可以了。然后 `attention_mask` 我们可以用分块上三角矩阵来实现 sample 之间的隔离。

![image.png](http://img.xilyfe.top/img/20260326190220470.png)

第三个问题我们看看 LlamaFactory 是怎么处理的：

```python
model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
knapsacks = greedy_knapsack(lengths, data_args.cutoff_len)
for knapsack in knapsacks:
    packed_input_ids, packed_attention_masks, packed_labels = [], [], []
    for i, length in enumerate(knapsack):
        index = length2indexes[length].pop()
        packed_input_ids += batch_input_ids[index]
        packed_labels += batch_labels[index]
        # 这里分为两种做法
        if data_args.neat_packing:
            # 将attention mask进行区分，不同文档使用不同的标识符，然后padding部分用0标识
            # 例如 [1,1,1,1,2,2,2,3,3,3,0,0]
            packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
        else:
            # 这里还是按照之前全部置为1
            packed_attention_masks += [1] * len(batch_input_ids[index])

    # 这里把padding位置的loss忽略掉，labels设置为IGNORE INDEX
    if len(packed_input_ids) < data_args.cutoff_len:
        pad_length = data_args.cutoff_len - len(packed_input_ids)
        packed_input_ids += [tokenizer.pad_token_id] * pad_length
        packed_labels += [IGNORE_INDEX] * pad_length
        if data_args.neat_packing:
            packed_attention_masks += [0] * pad_length
        else:
            packed_attention_masks += [1] * pad_length  # more efficient flash_attn

    if len(packed_input_ids) != data_args.cutoff_len:
        raise ValueError("The length of packed example should be identical to the cutoff length.")

    model_inputs["input_ids"].append(packed_input_ids)
    model_inputs["attention_mask"].append(packed_attention_masks)
    model_inputs["labels"].append(packed_labels)
```

LlamaFactory 采用了 Packing 和 Padding 结合的策略：它首先基于所有数据的长度进行检索，类似于背包问题，将其排序然后在截断长度之内贪心检索最合适的长度加入。例如排序之后我们得到如下数组：$[[2048],[1024,1023],[1000,1000,41],[500,500,500,20]]$。我们确保每一个 block 都尽可能容纳 sample，然后剩余的部分就进行 padding 补全，这样就不会出现 sample 被截断出现在两个 block，导致缺失上下文信息的问题。

### 如何选择

一般来说，Padding 适用于 SFT 和推理阶段，Packing 适用于 Pretrain 阶段。

Pretrain 的 loss 是单纯的 **next token prediction**，对整个 token 序列的每一个位置都预测下一个 token：

```
输入:  The  cat  sat  on  the  mat
预测:  cat  sat  on   the mat  <EOS>
loss:   ✓    ✓    ✓    ✓   ✓    ✓   ← 每个位置都算
```

pretrain 的目标是让模型学会"什么词后面跟什么词"，也就是语言的统计规律。这个目标**天然不需要样本边界**。自然语言本身就是一条流。互联网上的文本从来不是孤立存在的，一篇文章结束另一篇开始，这种"跨文档"的语言模式本来就是真实世界语言数据的样子。模型只需要在 EOS 处学会"这里是文档结束"，剩下的 token 全部参与 loss 完全合理。所以 packing 对 pretrain 是 **zero-cost 的**，不损失任何训练信号，反而把算力利用率提高了。

SFT 的目标完全不同，它要教模型**在给定 prompt 的条件下生成正确的 response**，Loss 只算 response 部分：

```
输入:  [INST] 帮我写诗 [/INST]  春风吹  绿了  江南岸  <EOS>
label:  -100    -100   -100  -100   春风吹  绿了  江南岸  <EOS>
loss:    ✗       ✗      ✗     ✗      ✓      ✓      ✓      ✓
```

在 SFT 阶段使用 packing 就会出现上文提到的 2/3 种情况，模型在生成问题 B 的回答时，它的"已知条件"不只是问题 B，还包含了回答 A。真实推理时每个请求是独立的，不会有"上一个用户的回答"混在 context 里。这就产生了**训练和推理的分布偏差**，模型学到了一些在推理时永远不会出现的 pattern，实际效果变差。其次 SFT 阶段的数据集大小比 Pretrain 小几个数量级，所以 padding 导致的算力浪费可以接受。

>Pretrain 的 loss 是无差别的 next token prediction，token 流可以任意拼接；SFT 的 loss 是有条件的、有边界的，样本之间必须隔离，推理时样本天然独立

{{< admonition type=question title="SFT 就完全不能用 packing 吗">}} 
可以用，但需要额外处理——加 **document attention mask**，让 packing 后的每条样本只 attend 自己内部的 token：

```python
# 每个 token 记录自己属于哪条样本
doc_ids = [0,0,0,0, 1,1,1,1,1, 2,2,2]  # 拼了3条样本

# attention 时只允许同一 doc_id 内的 causal attention
# 不同 doc_id 之间完全屏蔽
```

TRL 的 `SFTTrainer` 4.x 之后支持 `packing=True`，内部就是这么做的。但实现复杂，且 SFT 数据量通常不大，收益有限，大多数情况下直接 padding 更省心。
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



