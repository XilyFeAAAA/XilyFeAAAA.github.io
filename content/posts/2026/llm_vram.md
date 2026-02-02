---
title: 估算模型需要的显存
date: 2025-08-22T16:56:43+08:00
featuredImage: http://img.xilyfe.top/img/20260202165752948.png
authors:
  - Xilyfe
series:
  - LLM
tags: []
lastmod: 2026-02-02T07:45:07+08:00
---
到底需要多少算力才能部署一个模型，这是一个非常常见的问题。我们就从训练和推理两个场景，分析一下如何估计模型所需要的显存。

## 训练

训练显存大致分为以下四部分：
1. 模型权重：取决于存储的精度，常见的 BF16 和 FP16 占用大小为 2B
2. 梯度：反向传播计算的梯度，和权重一样常见情况下占用 2B
3. 优化器状态：常见的 Adam 会为每个参数都保存它的 Momentum、Variance 和 Master weights，精度为 FP32 所以总计 12B
4. 中间激活值：简单来说就是为了计算反向传播的梯度，需要把前向计算的中间值存储起来，具体计算见下文。

因此使用 AdamW 优化器 + 混合精度训练的经验公式为：

$$
\text{VRAM}_{train} \approx 20 \times N (Bytes)
$$

>以 70B 的模型为例，训练显存需要 $\approx 70 \times 10^9 \times 20 \text{Bytes} = 1400 \text{GB}$ ，80GB 的 A100 需要至少 18 张并行训练。

## 推理

推理只需要存模型的权重（不考虑 KVCache 的情况）：

$$
\text{VRAM}_{infer} \approx 2 \times N (Bytes)
$$

如果是 4-bit 量化：

$$
\text{VRAM}_{infer} \approx 0.5 \times N (Bytes)
$$

考虑 KVCache 加速推理的情况下，如果精度为 FP16，那么额外需要：

$$
\text{VRAM}_{KVCache} \approx b \times s \times l \times h \times 2 (Bytes)
$$

>在长序列推理中， KVCache 占据显存非常大。示例：LLaMA-7B (hidden_size=4096, layers=32)，batch=1, seq_len=32k → KV Cache ≈ 32k × 32 × 2 × 4096 × 2 ≈ **8 GB**。 seq_len=128k 的情况下仅仅 KV Cache 会到 **32 GB+**。

## 中间激活值分析

### 什么是激活值

这里的激活值和激活函数没有啥关系，以一个四个 Linear 的模型结构为例进行说明。其前向传播和损失函数的公式如下所示：

$$
\begin{split}
x_1 &= W_1 x + b_1 \\
x_2 &= W_2 x_1 + b_2 \\
x_3 &= W_3 x_2 + b_3 \\
x_4 &= W_4 x_3 + b_4 \\
l &= (y - x_4)^2
\end{split}
$$
在该公式中：$x$ 和 $y$ 为数据的特征和标签；$W_1$、$b_1$、$W_2$、$b_2$、$W_3$、$b_3$、$W_4$、$b_4$ 为四个 Linear 层的权重和偏置；$x_1$、x$_2$、$x_3$、$x_4$ 都是计算过程中的中间状态。反向传播过程中要对权重进行更新，也就是求损失相对于 $W_1$、$W_2$、$W_3$、$W_4$ 的偏导，按照链式求导法则得到公式如下：

$$
\begin{split}
\frac{\partial l}{\partial W_4} &= \frac{\partial l}{\partial x_4} \cdot \frac{\partial x_4}{\partial W_4} = \Bigg[ -2(y-x_4)\Bigg] \cdot x_3 \\
\frac{\partial l}{\partial W_3} &= \frac{\partial l}{\partial x_4} \cdot \frac{\partial x_4}{\partial x_3} \cdot \frac{\partial x_3}{\partial W_3} =  \Bigg[ [-2(y-x_4)] \cdot W_4 \Bigg] \cdot x_2 \\
\frac{\partial l}{\partial W_2} &= \frac{\partial l}{\partial x_4} \cdot \frac{\partial x_4}{\partial x_3} \cdot \frac{\partial x_3}{\partial x_2} \cdot \frac{\partial x_2}{\partial W_2} =  \Bigg[ [-2(y-x_4)] \cdot W_4 \cdot W_3 \Bigg] \cdot x_1 \\
\frac{\partial l}{\partial W_1} &= \frac{\partial l}{\partial x_4} \cdot \frac{\partial x_4}{\partial x_3} \cdot \frac{\partial x_3}{\partial x_2} \cdot \frac{\partial x_2}{\partial x_1} \cdot \frac{\partial x_1}{\partial W_1} =  \Bigg[ [-2(y-x_4)] \cdot W_4 \cdot W_3 \cdot W_2 \Bigg] \cdot x \\
\end{split}
$$

对上面这四个权重矩阵的链式求导公式找一下规律，可以发现对于权重矩阵 $W_i$ 的梯度在计算时主要有两项：
- 第一项是上述公式中使用特别大的中括号扩起来的部分，这部分是第 i+1 层反传回来的值，我们使用符号 $i+1$ 来表示这一项；
- 另一项则是第 $i−1$ 层计算出来的中间值，使用符号 $x_{i−1}$ 来表示；

那么对于 $W_i$ 的梯度计算公式就变为了 $\frac{\partial l}{\partial W_i} = l_{i+1} \cdot x_{i-1}$，这里的 $l_{i+1}$ 是第 $i+1$ 层反传过来的，所以计算第 $i$ 层的梯度时只需要做一次矩阵乘法即可。这里的 $x_{i−1}$ 正是在前向传播时计算出来的中间状态，比较官方的术语为 **中间激活值**。

### 显存分析

#### transformer 结构

这里把 transformer 层分为两部分，一部分是 MHA 层，一部分是 FFN 层。下面分别写一下这两部分的公式。一般的资料中关于 transformer 的公式仅写主要的部分，像dropout、normalize、激活函数都会被省略，但是这里由于需要分析中间激活值的显存，所以会把整个 transformer 的所有操作都体现到公式中，如下。

MHA 层的公式如下：

$$
\begin{equation}\begin{split}
Q &= x \cdot W_Q, \quad K = x \cdot W_k, \quad V = x \cdot W_v \\
x_{\text{self}} &= \text{Dropout}\Big[ \text{softmax}\big(\frac{Q \cdot K^T}{\sqrt{d}} \big) \Big] \cdot V \\
x_{\text{attn}} &= \text{LN}\Big[ \text{Dropout}\big(x_{\text{self}} \cdot w_o \big) + x \Big]
\end{split}\end{equation}
$$

FFN 层的公式如下：

$$
\begin{equation}\begin{split}
x_{\text{ffn}} &= \text{GeLU}(x_{\text{attn}} \cdot W_{\text{ff1}}) \cdot W_{\text{ff2}} \\ 
x_o &= \text{LN}\Big[\text{Dropout}\big(x_{\text{ffn}} \big) + x_{\text{attn}} \Big]
\end{split}\end{equation}
$$

总的来说，MHA 层的输入为 $x$，输出为 $x_{attn}$；FFN 层的输入为 $x_{attn}$，输出为 $x_o$；

#### tranformer 的中间激活值分析

首先定义几个符号：
- b：表示batch_size；
- s：表示seq_length，为文本长度；
- h：表示hidden_dim，为隐藏层的维度；
- a：表示多头注意力中有多个头；
- ha：表示hidden_dim_per_head，为多头注意力中每个头的隐藏层维度；

另外，在实际使用时一般都有 ha∗a=h 成立。
MHA 层需要保存的激活值，以及每个激活值的大小：

$$
\begin{alignat}{10}
Q = x \cdot W_Q \quad &: \quad \text{维度为 } [b, a, s, h_a] = [b, s, h], &\text{大小为 } 2bsh \text{ 字节} \\
K = x \cdot W_k \quad &: \quad \text{维度为 } [b, a, s, h_a] = [b, s, h], &\text{大小为 } 2bsh \text{ 字节} \\
V = x \cdot W_v \quad &: \quad \text{维度为 } [b, a, s, h_a] = [b, s, h], &\text{大小为 } 2bsh \text{ 字节} \\
Q \cdot K^T \quad &: \quad \text{维度为 } [b, a, s, s], &\text{大小为 } 2bas^2 \text{ 字节} \\
\text{softmax}(\frac{Q^T K}{\sqrt{d}}) \quad &: \quad \text{维度为 } [b, a, s, s], &\text{大小为 } 2bas^2 \text{ 字节} \\
\text{Dropout}\Big[ \text{softmax}\big(\frac{Q \cdot K^T}{\sqrt{d}} \big) \Big] \quad &: \quad \text{维度为 } [b, a, s, s], &\text{Dropout 层大小为 } bas^2 \text{ 字节} \\
x_{\text{self}} = \text{Dropout}\Big[ \text{softmax}\big(\frac{Q \cdot K^T}{\sqrt{d}} \big) \Big] \cdot V \quad &: \quad \text{维度为 } [b, a, s, h_a] = [b, s, h], &\text{大小为 } 2bsh \text{ 字节} \\
x_{\text{self}} \cdot W_o \quad &: \quad \text{维度为 } [b, s, h], &\text{大小为 } 2bsh \text{ 字节} \\
\text{Dropout}\big(x_{\text{self} \cdot w_o} \big) \quad &: \quad \text{维度为 } [b, s, h], &\text{Dropout 层大小为 } bsh \text{ 字节} \\
x_{\text{attn}} = \text{LN}\Big[ \text{Dropout}\big(x_{\text{self}} \cdot w_o \big) + x \Big] \quad &: \quad \text{维度为 } [b, s, h], &\text{大小为 } 2bsh \text{ 字节}
\end{alignat}
$$

FFN 层需要保存的激活值，以及每个激活值的大小：

$$
\begin{alignat}{2}
x_{\text{attn}} \cdot W_{\text{ff1}} \quad &: \quad \text{维度为 } [b, s, 4h], &\text{大小为 } 8bsh \text{ 字节} \\
\text{GeLU} (x_{\text{attn}} \cdot W_{\text{ff1}}) \quad &: \quad \text{维度为 } [b, s, 4h], &\text{大小为 } 8bsh \text{ 字节} \\
x_{\text{ffn}} = \text{GeLU}(x_{\text{attn}} \cdot W_{\text{ff1}}) \cdot W_{\text{ff2}} \quad &: \quad \text{维度为 } [b, s, h], &\text{大小为 } 2bsh \text{ 字节} \\
\text{Dropout}\big(x_{\text{ffn}} \big) \quad &: \quad \text{维度为 } [b, s, h], \qquad &\text{Dropout 层大小为 } bsh \text{ 字节} \\
\text{LN}\Big[\text{Dropout}\big(x_{\text{ffn}} \big) + x_{\text{attn}} \Big] \quad &: \quad \text{维度为 } [b, s, h], &\text{大小为 } 2bsh \text{ 字节} \\
\end{alignat}
$$

将 MHA 和 FFN 层全部加起来得到：

$$
\begin{split}
& 2bsh+2bsh+2bsh+2bas^2+2bas^2+bas^2+2bsh+2bsh+bsh+2bsh+ \\
& 8bsh+8bsh+2bsh+bsh+2bsh=34bsh+5bas^2
\end{split}
$$

如果有 $l$ 层 transformer，那么这 $l$ 层 transformer 总的中间激活值占用的显存为：$l∗(34bsh+5bas^2)$

#### embedding 层和解码层

上面仅分析了多个 transformer 对应的中间激活值消耗的显存的大小。模型中还会有 embedding 层和解码层。其中解码层没有对应的中间激活值，只需要分析一下 embedding 层即可。

embedding 层的功能是将输入的 token ID 转为向量，其输出的矩阵维度为 \[batch_size, seq_length, hidden_size]，即 \[b, s, h]，该中间激活值占用的显存为 2bsh。

综上所述，整个模型所有的中间激活值的大小为$l∗(34bsh+5bas^2)+2bsh$。随着模型越来越大，$l$ 是比较大的，所以有时会忽略 $2bsh$ 这一项，直接使用 $l∗(34bsh+5bas^2)$ 来估计模型的中间激活值的大小。