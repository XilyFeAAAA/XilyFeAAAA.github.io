---
title: MiniMind 学习指北(五)：SFT
date: 2026-02-10T15:59:28+08:00
featuredImage: http://img.xilyfe.top/img/20260122134824760.png
authors:
  - Xilyfe
series:
  - minimind
tags:
  - 大模型
  - 深度学习
lastmod: 2026-02-10T11:50:43+08:00
---
前面我们已经进行了预训练，得到了一个只会续写的模型。这是因为我们预训练数据集的文本都是简单的一句话，然后通过加工我们得到了类似 "<|im_start|> 秦始皇的功绩包括统一货币、文字" 的文本，通过 Teacher-Forcing 它只能做到预测下一个 token，或者说只会机械接龙，不会对话。

SFT 全称是 Supervised-Finetune，也就是监督微调。我们通过 **对话文本** 的数据集在预训练的模型基础上进行训练，就能让模型学会对话。简单来说 SFT 和 Pretrain 者有以下区别：

1. Pretrain 的数据都是纯文本如 "今天天气很好..."，而 SFT 的数据集是对话如 {"user":"你好", "assistant": "你也好"}
2. Pretrain 会直接 tokenize 整个文本，而 SFT 会用 template 模板将对话拼接为 "<|im_start|>user 你好 <|im_end|><|im_start|>assistant 你也好 <|im_end|>" 这样的结构化文本。
3. 计算损失函数时候，Pretrain 是对每一个 token 计算损失，而 SFT 仅对 Assistant 部分计算损失

所以说 SFT 我们只需要对 Dataset 进行改进，训练方式还是之前的 Teacher-Forcing。

## 改造 Dataset

先来看看我们 SFT 的数据集是啥样的：

```json
{
	"conversations": [
		{"role": "user", "content": "hello!"},
		{"role": "assistant", "content": "hi!"}	
	]
}
```

在 MiniMind 实现里面，除了把 conversations 里面的对话组成 message，还加入了一个 system 的 prompt：

```python
    def prepare_message(
        self, conversations: list, random_threshold: float = 0.2
    ) -> list:
        assert len(conversations) >= 2

        SYSTEM_PROMPTS = [
            "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
            "你是minimind，一个小巧但有用的语言模型。",
            "你是一个专业的AI助手，请提供有价值的回答。",
            "你是minimind，请尽力帮助用户解决问题。",
            "你是一个可靠的AI，请给出准确的回答。",
            "You are a helpful AI assistant.",
            "You are minimind, a lightweight intelligent assistant.",
            "You are a friendly chatbot. Please answer the user's questions carefully.",
            "You are a knowledgeable AI. Try your best to provide accurate information.",
            "You are minimind, a small but useful language model.",
        ]

        message = []
        for turn in conversations:
            message.append({"role": turn["role"], "content": turn["content"]})

        if message[0]["role"] != "system" and random.random() < random_threshold:
            message = [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + message

        return message
```

根据 Qwen 和 Grok 所说，MiniMind 参数量较小，自身无法稳定维持"助手"的角色认知，如果没有 system prompt 模型容易角色混淆、生成无意义接龙、对模糊指令无法正确响应。稍后我会对加入和未加入 system prompt 的数据集的数据集进行对比测试。

之后我们就可以用 tokenizer 的 `apply_chat_template` 生成结构化的文本了:

```python
message = self.prepare_message(conversations)
input = self.tokenizer.apply_chat_template(
   message, tokenize=False, add_generation_prompt=False
)
input_ids = self.tokenizer(input).input_ids[: self.max_length]
input_ids += self.tokenizer.pad_token_id * (self.max_length - len(input_ids))
```

在模型的 forward 里面我们规定的训练时候会让 logits 和 labels 进行偏移，所以在数据集就不用处理了。如上面所说： 计算损失函数时候，SFT 仅对 Assistant 部分计算损失。所以我们需要把 labels 的其余部分置为 -100，这样求交叉熵损失时候设置的 `ignore_index=-100` 就会起作用了。

```python
def pad_labels(self, input_ids: list[int]):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i : i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

## 修改 SFT 脚本

SFT 的训练代码和 Pretrain 的代码几乎一致，我们 copy 之后修改一下超参数就好了。

| 超参数               | Pretrain    | SFT         | 主要原因                                  |
| ----------------- | ----------- | ----------- | ------------------------------------- |
| **Learning Rate** | 1e-4 ~ 1e-3 | 1e-6 ~ 5e-5 | SFT 要微调，避免破坏预训练权重；Pretrain 需要快速学习基础表示 |
| **Epochs**        | 1~3 epoch   | 3~20 epoch  | SFT 数据少，需要多次学习高质量样本                   |
| **Batch Size**    | 128         | 16          | Pretrain 数据多、序列短；SFT 数据少、序列长          |
| **Weight Decay**  | 0 ~ 0.01    | 0.01 ~ 0.1  | SFT 需要更强正则化防止过拟合小数据集                  |
| **Dropout**       | 0~0.1       | 0 或更低       | SFT 数据高质量，不需强 dropout                 |

## 训练

## 评估