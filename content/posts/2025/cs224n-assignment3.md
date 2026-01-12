---
title: "CS224N Assignment 3"
date: '2025-11-20T11:24:11+08:00'
authors: [Xilyfe]
series: ["CS224N"]
tags: ["深度学习"]
--- 


## 前置知识

### 一、Bahdanau 注意力机制

Bahdanau 注意力机制的思想就是在 Decoder 的每一个时刻，用当前 token 和 Encoder 的每一个 token 计算注意力分数，判断 Encoder 的哪一个 token 和当前 token 关系最大。

### 二、Teacher-forcing

在早期的 seq2seq 结构中只包含最简单的 Encoder 和 Decoder 两部分。训练时如果让 Decoder 完全靠自己前一步的预测输出当下一步输入，容易出错 → 错误不断累积，训练非常难收敛。为了解决这个问题，Teacher-forcing 的解决办法是直接把答案告诉模型。

假设正在翻译 "I love you" -> "我爱你":
- t=0，输入句首 token，将预测的第一个 token 和答案求损失
- t=1，输入 "我"，将预测的第二个 token 求损失
- t=2，输入 “我爱”, 将预测的第三个 token 求损失

这个方法的好处在于可以避免一步错步步错的现象，但是坏处就是这个方法在训练集有很好的表现，但是在测试集和验证集上仍然可能看到之前错误的预测。

## NMT 分析

<div align="center">
    <img src="../../../../resource/ai/llm/NMT.png" width="80%"/>
</div>

### 一、模型核心框架总览

NMT 模型做的事情就是实现 seq2seq，它有一个 Encoder （双向 LSTM）+一个 Decoder （单向 LSTM 和注意力机制）实现。待翻译的文本首先会通过 CNN 卷积层，有助于提取上下文 Token 之间的信息。然后进入双向 LSTM，输出每时刻的隐藏状态，Last_hidden 和 Last_cell （后面会成为 Decoder 的初始向量）。Decoder 阶段的输入是真实的翻译文本也就是答案）, Decoder 根据答案+隐藏状态+注意力机制预测下一个 token 是什么，最后损失则为每一损失的平均值。

### 二、Encode

编码器的作用是通过双向 LSTM 解析原文本，根据原文本的语义信息生成向量提供给解码器生成目标文本。
#### 1. 填充文本
- 输入到 NMT 模型的文本形状为 [B, L]，这里的 L 指的是训练集每条评论不同的长度。
- 经过处理，这些张量会统一到相同的长度得到原文本张量 [SL, B]，这里的 SL指的是原文本最大长度。
> Q:为什么输入张量的形状是 [SL, B]，通常不都是 [B, SL] 吗？
> A:因为 PyTorch 中，LSTM 模型默认 `batch_first=False`，也就是要求输入格式为 [L, B, embedding_dim]，而训练数据通常是 [B, L]，所以必须转成 [L, B]。假如设置了 `batch_first=True`那么就可以沿用之前的形状 [B, L]。

#### 2. 词嵌入
- NMT 内部维护着两个可训练的 Embedding，分别对应原文本和目标文本。
- 对输入张量进行词嵌入之后，形状从 [SL, B] 变成 [SL, B, E]
- 由于要对张量进行卷积操作，需要先变形为 [B, E, SL] 再变回 [SL, B, E]
- 为了避免带 padding 的输入影响模型训练效果，通过 `pack_padded_sequence` 得到处理过的张量放进 LSTM 模型，他可以只处理有效数据，节省计算，并且让 RNN 层只看真实部分，自动忽略 padding。
- 双向 LSTM 模型会返回三个张量`enc_hiddens, (last_hidden, last_cell)` 
	- enc_hiddens 是双向 LSTM 每个时刻两个方向的隐藏状态，需要通过 `pad_packed_sequence` 转为 Tensor，形状为 [SL, B, 2*H]，最后转为 [B, SL, 2*H]
	- last_hidden 和 last_cell 是双向 LSTM 最后输出的两个状态张量，形状为 [2, B, H]
- 由于 Encoder 是双向 LSTM，forward LSTM 和 backward LSTM 会分别返回一个隐藏状态向量和细胞向量，所以形状是 [2, B, H]。但是 Decoder 是一个单向 LSTM，所以它的隐藏状态向量和细胞向量形状为 [B, H]。解决办法就是把2个向量在第二个维度拼接为 [B, 2*H]。
>Q:为什么要进行卷积？
>A:因为词向量是逐词的，每个 token 对应的 embedding 只包含它自己的信息，不同 token 组合起来有不同的含义。再词嵌入和 LSTM 之间加入卷积层，可以让模型更好的学习输入文本的语义信息。

### 三、Decode

解码器也采用了 LSTM 模型，但目的不像 Encode 部分一样是为了得到隐藏状态向量，所以没有用 nn.LSTM 而用了 nn.LSTMCell，通过手动 for 循环一步步计算注意力分数。

- 由于 enc_hiddens 的形状是 [B, SL, 2*H]，所以需要对它做一次线性变换变成 [B, SL, H]
- 对目标文本进行词嵌入之后得到 Y，形状也是 [TL, B, E]。
- 沿着 Y 的第一维遍历 Y，能得到每个时刻向量 y_t，形状为[B, E]。
- o_prev 是 Encoder 上一个时间步的隐藏状态，或者可以说是综合考量注意力分数和正确答案得到的隐藏状态，形状也是 [B, H]
- 每一个时间步 Decode 做的事情包括：
	- 将 y_t 和 o_prev 裁剪为 [B, E+H] 输入到 Decode 的单向 LSTM，得到 dec_hidden, dec_cell, 形状为 [B, H]
	- 通过矩阵乘法，将 enc_hiddens_proj [B, SL, H] 和 dec_hidden.unsequeeze(2) [B, H, 1] 相乘，得到注意力分数 [B, SL, 1] 再变形为 [B, SL]，其中第二维每一个元素就代表每个 token 和当前的匹配度。
	- 由于通过 padding 将文本长度补齐到 SL，所以需要把 padding 部分的 score 降到负无穷，不会影响上下文。
	- Softmax 得到注意力权重 alpha_t，表示在生成当前目标词时，源句 token 对结果的重要性。
	- 矩阵乘法，将 alpha_t.unsqueeze(1) [B, 1, SL] 和 enc_hiddens [B, SL, 2H] 相乘，得到注意力分数 [B, 1, 2H] 再变形为 [B, 2H]。
	- 最后将 decoder 的隐藏状态张量和 上下文向量 a_t 拼接得到 U_t [B, 3H]，经过线性变换最后得到 [B, H] 的张量，也就是下一个时间步的 o_prev

> 上一步得到注意力分数只是告诉模型“在源句的每个位置上，我该关注多少？”。它只是**权重**，不能直接用于生成词或更新 decoder。而 a_t 它是一个**真正的向量表示**，包含你“关注的位置合成出来的语义与上下文信息”。decoder 需要 **这个语义向量** 来决定下一步要输出什么词。

### 四、最后处理

最后 decoder 会返回一个 [SL, B, H] 的张量，翻译或者说单词预测也是一个多分类问题，所以还需要通过线性变换将 [SL, B, H] 变为 [SL, B, E] 再做 softmax，其中 [a, b, c] 代表第 b 个句子中第 a 个 token 是 vocal[c] 的可能性。


## 训练过程

### 一、损失计算

#### 1. 负对数似然

机器翻译的本质给定源句子“我爱你”，让模型学会输出目标句子“I love you”。或者说我们希望模型学会条件概率分布：$p(y | x) = p(I love you | 我爱你)$，概率越大越好。假设我们有 N 个平行句对训练样本，那么希望找到一组超参数 θ，让所有训练样本出现的联合概率最大：

$$
\theta^{*} = argmax_{\theta}\prod_{i=1}^{N}p(y^{i}|x^{i};\theta)
$$

为了方便计算对其取对数：

$$
\theta^{*} = argmax_{\theta}\sum_{i=1}^{N}\log{p(y^{i}|x^{i};\theta)}
$$

因为求最小值比求最大值容易，所以取负数：

$$
\theta^{*} = argmin_{\theta}\sum_{i=1}^{N}-\log{p(y^{i}|x^{i};\theta)}
$$

---

**对于单个句子而言是如何计算概率的呢？**

根据概率公式：

$$
p(y|x)=p(y_1|<s>,x) \times p(y_2|<s>,y_1,x) \times \dots = \sum_{t=1}^{T}\log{p(y_t|y_{<t},x)}
$$

NMT 是一个多分类问题，采用 softmax 得到概率分布：

$$
p(y_t=k|y_{<k},x)=\frac{exp(logit_k)}{\sum_{k'}{exp(logiy_{k'})}}
$$

#### 2. 困惑度 Perpelxity

困惑度就是对平均负对数损失做指数：

$$
ppl=e^{\frac{loss}{len(token)}}=e^{-\frac{1}{N}\sum_{i=1}^{N}\log{P(y_i)}}
$$

### 二、早停机制

早停机制可以有效的避免模型进行无效训练或者陷入过拟合：

1. 如果当前困惑度最低，那么 patience 清零并且保存模型
2. 否则 patience++
3. 如果 patience 达到上限则触发 trial，学习率下降并且从上一个 checkpoint 重新加载模型和优化器，当trial 到达上限则早停退出。

### 三、BLEU 分数

BLEU 是常用的“自动评翻译好不好”的指标，步骤是：  

1. 算“n-gram 精度”（比如 1-gram 是单个词的匹配度，2-gram 是两个词连起来的匹配度，比如“adequate resources”在参考翻译里有没有）；  
2. 算“ brevity penalty（简短惩罚）”：如果模型翻译太短（比如参考句 10 个词，模型只翻 5 个），会扣分项；  
3. 把精度、惩罚结合起来，算最终 BLEU 分（0-1 之间，越高越好）。  


### 四、分析 beam search

beam search 是“让模型生成更好翻译”的方法，训练时会记录不同迭代次数的翻译结果（比如第 200 次、第 3000 次迭代），要做两件事：  

1. 看翻译质量有没有随迭代提升（比如第 200 次翻得不通顺，第 3000 次接近参考翻译）；  
2. 分析同一迭代下，beam search 生成的多个候选翻译（比如 10 个候选）有什么差别（比如有的候选多了个“the”，有的少了个“and”）。