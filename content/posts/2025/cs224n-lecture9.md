---
title: "CS224N Lecture 9: Pretraining"
date: '2025-11-27T11:24:11+08:00'
authors: [Xilyfe]
series: ["CS224N"]
tags: ["深度学习"]
lastmod: 2026-01-21T12:36:40+08:00
--- 


## Subword Modeling

在过去的实践中，我们处理文本会先进行分词，然后把一个个 token 转换为其对应的 idx 索引，在用它去进行词嵌入。但这样的做法有一个问题，在测试集中我们遇到没见过的词语就把他用 \<UNK> 来替换，这样会丢失大量的信息。

**现代 NLP 模型不会直接把完整单词当作基本单位，而是把单词拆成若干 subwords（子词）来表示。**

- 出现非常频繁的词（如 _hat、learn_）会被当作 **完整的词** 放进词表，少见词、奇怪词、新造词会被**拆成多个子词**。
- 如果是特别奇怪的词、根本没见过、字母组合很怪，模型可能把每个字符都拆开。

---

- **常用词**：例如 hat, learn 等词语，Embedding 会完整学习不拆分。
- **变体**：例如 taaaaast，会把词语拆为常见的 Subword 和常用词的组合 taa##aaa##sty。
- **拼写错误**：laern，模型没见过于是拆分为 la##ern，模型没见过这个错词，所以分成多个子词。
- **新词 / 自创词**：Transformerify 拆为 Transformer## ify。

这些拆分后的 Subword 会得到多个 Embedding，我们可以取最后一个 Embedding，对 Embedding 进行平均，或者用 RNN 等模型对其进行学习。

## Pretraining

### 三种预训练架构

> Pretraining 干的事情就是给模型安排一个非常困难的任务，迫使它在数据中尽可能的学习。

- Encoder Only：几乎只能用 MLM / 替换 token 检测等，因为它没有生成能力。
- Decoder Only：几乎只能用 next-token prediction（因为它看不到未来），或者说 TeachForcing。
- Encoder-Decoder：传统的带对齐的翻译（supervised），Next-token prediction（纯自回归），Span corruption（T5 式）

---

Bert 就是 Encoder Only Pretaining 的典型代表，它的训练方式包括：

- 随机遮掉 15% 的 **Subword**
- 随机替换 15% 的 **Subword**
- 随机选择 15% 的 **Subword** 不替换，但还是让它预测
- 判断两个句子是否相邻

**对于不同的任务会选择不同的预训练方法**：例如 Bert 更适合为文档选择一个 Tag，它不适合进行文本生成的任务。

---

**一个 Misunderstanding**

经典的 Transformer（Vaswani et al., 2017 那篇 “Attention is All You Need”）本身不是一种，而是一个“积木盒子”。 它同时提供了三种可拼装的部件：

1. Transformer Encoder 块（双向注意力，允许看未来）
2. Transformer Decoder 块（带 masked self-attention，只能看过去）
3. Cross-attention（Decoder 看 Encoder 的输出）

根据你怎么拼这三个积木，就自然得到了刚才 PPT 里的三种预训练架构：

|你拿 Transformer 的哪些部分|拼出来的是哪种架构|经典代表模型|预训练方式|
|---|---|---|---|
|只用 Encoder 块堆 6/12 层|→ 纯 Encoder|BERT、RoBERTa、DeBERTa|MLM（遮词填空）|
|Encoder + Decoder 都用|→ Encoder-Decoder|T5、BART、Flan-T5、UL2|Span Corruption 等|
|只用 Decoder 块堆很多层|→ 纯 Decoder（GPT式）|GPT-1/2/3/4、LLaMA、Grok、PaLM、Gemma、Qwen、Mistral、DeepSeek|Next-token prediction（下一词预测）|

## Prefix-Tunning

**Prefix Tuning** 是一种参数高效微调 (PEFT) 技术。它保持预训练语言模型（如 GPT, BERT）的参数完全**冻结**，只在Transformer的**每一层**输入前添加一小段可训练的连续向量（即 Prefix）。

以 Decoder Only 的模型进行参数微调为例，我们会初始化一部分可训练参数 prefix_k 和 prefix_v。假如 Decoder 每一层的 Q, K, V 形状为 \[batch_size, seq_len, d_model]，那么 prefix_k 和 prefix_v 的形状就是 \[batch_size, prefix_len, d_model]。我们将 prefix_k 和 K 矩阵在第 2 个维度进行拼接，得到新的 K 和 V，形状为 \[batch_size, prefix_len+seq_len, d_model]。

```python
# 在 modeling_llama.py 里，你会看到类似的代码（伪代码）
def forward(...):
    # 正常计算
    key = self.k_proj(hidden_states)      # W_K * x
    value = self.v_proj(hidden_states)    # W_V * x

    # ↓↓↓ 关键：Prefix-Tuning 的注入点 ↓↓↓
    if self.config.peft_type == "PREFIX_TUNING":
        past_key_value = self.get_prompt(batch_size)   # 就是 P_K^l, P_V^l
        key = torch.cat([past_key_value[0], key], dim=2)   # cat 在 seq_len 维度
        value = torch.cat([past_key_value[1], value], dim=2)

    # 后面的 attention 计算完全不变
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) / sqrt(...)
```

**为什么输入形状不变，K 和 V 形状变了不影响呢？**

$$
\begin{align}
shape(softmax(\frac{QK^T}{\sqrt{d_k}})V) &= [batch\_size, prefix\_len, d\_model] \times  [batch\_size, d\_model, prefix\_len+seq\_len] \times  [batch\_size, prefix\_len+seq\_len, d\_model]\\
&= [batch\_size, prefix\_len, d\_model]
\end{align}
$$

### Span Corruption

Span Corruption 的方法是随机把原文里连续的几段文字（span）整段抠掉，换成一个特殊的 mask token，然后让模型把这些被抠掉的原文原封不动地生成回来。

举个例子，我们有一段文本：我星期天要去参加比赛，然后会对他随机扣掉一段 span：

- Span Corruption 后输入：我 \<extra_id_0> 要去参加 \<extra_id_1>
- 目标输出：\<extra_id_0> 星期天 \<extra_id_1> 比赛

之后损失函数就是一个标准的自回归训练：在 Encoder 输入 Span Corruption 后文本，在 Decoder 中每一歩都把真实的上一个 token 喂给模型，将预测的 logtis 和目标输出求交叉熵损失。

我们用一个极小的词表：

```
0: <pad>
1: 我
2: 星期天      (把“星期天”当 1 token)
3: 要
4: 去
5: 参加
6: 比赛
7: 。
8: <extra_id_0>
9: <extra_id_1>
10: <extra_id_2>
11: <s>   (decoder start)
12: </s>  (decoder end)
```

于是就有：

```python
raw_input_ids = [1, 2, 3, 4, 5, 6, 7]
encoder_input_ids = [1, 8, 3, 4, 5, 9, 7]
# 对应： 我 <extra_id_0> 要 去 参加 <extra_id_1> 。
decoder_input_ids = [11, 8, 2, 9, 6, 10]   
# [<s>, <extra_id_0>, 星期天, <extra_id_1>, 比赛, <extra_id_2>]
decoder_target_ids = [8, 2, 9, 6, 10, 12]  
# 预测目标（要与 logits 对齐）
```

它的好处与 MLM 和 Next-token Prediction 相比在于：

- 它既逼模型理解上下文（像 BERT）
- 又逼模型学会生成长序列（像 GPT）
