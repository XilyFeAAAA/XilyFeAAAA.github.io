---
title: "Lecture 7: Attention, Final Project and LLM intro"
date: '2025-11-23T11:24:11+08:00'
authors: [Xilyfe]
series: ["CS224N"]
tags: ["深度学习"]
--- 

## BLEU 评估指标

BLEU 是机器翻译中最经典的自动评价指标之一，用来衡量模型生成的译文（candidate）和 人工参考译文（reference）之间的相似度。

BLEU 的核心思想: **N-gram（词序列）有多少和参考译文匹配。匹配越多，得分越高。**

$$
BLEU = BP \times \exp(\sum_{n=1..N}{w_n * \log{p_n}})
$$

| 符号  | 含义  |
| --- | --- |
| $p_n$ | n-gram 精确率 precision |
| $w_n$ | 权重，一般 BLEU-4 时为 1/4  |
| BP| 长度惩罚 |

---

**计算过程**

```
candidate: the cat the cat on the mat
reference: the cat is on the mat
```

**1. Step 1: unigram precision**


|word|cand|ref|clipped|
|-|-|-|-|
|the|4|2|2|
|cat|2|1|1|
|on|1|1|1|
|mat|1|1|1|

- total clipped = 5
- total candidate unigram = 8
- unigram precision = 5/8

**Step 2：bigram precision**

|word|cand|ref|clipped|
|-|-|-|-|
|the cat|2|1|1|
|cat the|1|0|0|
|cat on|1|0|0|
|on the|1|1|1|
|the mat|1|1|1

- total clipped = 3
- total candidate bigram = 6
- bigram precision = 3/6

同理计算 p3 和 p4。

**Step 3：长度惩罚**

len(candidate) > len(reference) 所以没有惩罚，BP=1

**Step 4：综合 BLEU**

$$
BLEU = 1 \times \exp((\log{p_1} + \log{p_2} + \log{p_3} + \log{p_4})/4)
$$

> 在这个例子中，假如翻译的句子 candidate=the，那么它在 1-gram 中就能得到很高的分数，避免预测过短的句子就会采用惩罚机制：当预测长度>参考长度则不惩罚，BP=1；当预测长度<参考长度，BP=exp(1 - r/c)

---

**机器翻译很怕高阶 n-gram 全不匹配**

举个例子：

```
candidate: the the the the the the
reference: the cat is on the table
```

对于 2-gram 有:

```
candidate bigram: "the the", "the the", ...
reference bigram: "the cat", "cat is", ...
```

总 clipped=0, 总 can bigram=5, 得到 p(2)=0。  
计算 BLEU 分数时：$geo\_mean = exp(\frac{1}{4}( \log(p_1) + \log(p_2) + \log(p_3) + \log(p_4)))$  

由于 log(0) 等于负无穷，所以取对数之后 BLEU 分数为 0。

## 注意力机制

### 经典点积注意力

![](http://img.xilyfe.top/img/20260119120610905.png)

- $S^T$ 是 Decoder 在当前时间步的隐藏状态 dec_hidden
- $h_i$ 是 Encoder 的第 i 个 token 对应的隐藏状态 enc_hidden_i
- $s^Th_i$ 就能得到 Decoder 当前 token 和 Encoder 第 i 个 token 的注意力分数，对 Encoder 的每一个注意力分数进行 softmax 就能得到注意力权重

这个就是最早提出的 Bahdanau 注意力，但是这个算法的问题在于：$h_i$ 这个向量包含了完整信息，也就是有太多不需要的信息了。

### 乘法注意力

![](http://img.xilyfe.top/img/20260119120627811.png)

乘法注意力的解决方法就是：在两个向量之间乘一个矩阵，这个矩阵可以学习隐藏状态哪一部分是有用的

### 低秩乘法注意力

![](http://img.xilyfe.top/img/20260119120641801.png)

乘法注意力的问题在于，当隐藏状态长度很大时，中间矩阵的参数量就会非常大。解决办法是把大方阵拆为两个低秩的矩阵，它们有一样的效果。
