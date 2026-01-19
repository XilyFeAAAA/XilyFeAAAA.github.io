---
title: "Lecture 3: Architectures & Hyperparameters"
date: '2025-12-19T16:00:11+08:00'
authors: [Xilyfe]
series: ["CS336"]
tags: ["大模型"]
--- 

## Architectures

- Normorlization
	- Pre - Post
	- Layer Norm - RMS Norm
- Activations
	- ReLU, GeLU, GLU
- HyperParameters
	- $d_{ff}$,$d_{model}$
	- num_heads
	- vocabulary
	- dropout & regularization
- Stability Tricks
- Other MHA

## Norm

### PreNorm

![](http://img.xilyfe.top/img/prenorm.png)

现代的 Transformer 架构中，Transformer Block 都采用 PreNorm 而不是 PostNorm，具体来说就是把 Norm 放在注意力机制和 FFN 前馈网络层前面，而不是进行残差连接之后再 Norm。优点在于==训练更稳定，可以采用更大的学习率==。

### RMSNorm

$$
y=\frac{x}{\sqrt{||x||_2^2+\epsilon}}*\gamma
$$

和 LayerNorm 对比性能相当，并且更快（不需要减均值，计算量小）。

### Drop Bias

FFN 前馈网络层去掉 Bias 优化更稳定，并且内存占用更少。

## Activations

### GLU

GLU 全程是 Gated Linear Unit，是由 **"门控"**（gating）机制和 **线性激活**（linear activation）结合而成的，其工作原理受到了 **LSTM**（长短期记忆网络）中门控结构的启发。

$$
GLU(x)=σ(xW+b)⊗(xV+c)
$$

- $\sigma$ 是 sigmoid 函数
- $\odot$ 是逐元素点乘

思路就是对 x 做两次计算，一次线性计算丰富 x 的信息，一次 sigmoid 将其映射到 0-1 区间，==决定那些信息保存，哪一些去除==。

GLU 的变体包括：

- ReGLU：$ReGLU(x) = max(0, xW) \odot xV$
- GeGLU：$ReGLU(x) = GeLU(xW) \odot xV$
- SwiGLU：$ReGLU(x) = Swish(xW) \odot xV$
- ...

!!! Attention 
	相较于普通激活函数的 FNN， $FFN(x) = h(xW_1)*W_2$，GLU 的参数量为其 3/2，因为多了一个门控的矩阵。所以为了保证参数量一致，带 GLU 的 FFN 一般 $d_{ff}$ 都设为普通 FFN 的 2/3。


## HyperParameters

- FFN：$d_{ff}=4d_{model}$ 如果是 GLU 则是 $d_{ff}=2.66d_{model}$
- Head Dim：$num\_heads * d_{k} = d_{model}$ 
- Aspect Ratio：模型的宽高比，增加 num_layers 会导致无法并行，增加 d_model 可以拆分矩阵并行计算
- Vocabulary Size：单语言 35-50K，多语言 100-250K
- Dropout & Weight Decay：在预训练时候 Weight Decay 作用不是防止过拟合，而是能在后期训练更快更平滑。

## Stability Tricks

### Z-loss

$$
softmax=\frac{e^i}{\sum_j{e^j}}=\frac{e^i}{Z(x)}
$$

Transformer 中有两个 Softmax,分别在 attention 中和最后输出的地方。Softmax的不稳定性主要来自于分母的计算，容易 overflow 导致整个计算结果无效，使梯度计算崩溃。Z-loss 目的引入一个辅助损失项，使得 Z(x) 趋近于 1。z-loss就像一个正则化项，专门作用于 logits 的大小。

### QK Norm

把每个 query 向量和 key 向量都单独做 L2 归一化，然后再算 attention score,送入 softmax。这能固定score的范围，使 softmax 更平滑，训练更稳定。

## Other MHA