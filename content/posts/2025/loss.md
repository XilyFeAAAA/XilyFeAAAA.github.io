---
title: "Loss Function"
date: '2025-12-27T11:01:11+08:00'
featuredImage: "http://img.xilyfe.top/img/20260113120251867.jpg"
authors: [Xilyfe]
series: ["DeepLearning"]
tags: ["深度学习", "Loss"]
--- 

## MSELoss

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N}(\hat y_i - y_i)^2
$$
均方差损失任务适合用于==回归任务==。

## CrossEntropyLoss

$$
loss = -\sum_i^Cy_i\log{p_i}
$$

- $C$：类别数
- $y_i$：对于标签的 one-hot 编码，真实标签为 1
- $p_i$：概率分布

> 注意 $p_i$ 是概率分布而不是 logits。

==交叉熵损失内部做的是 log_softmax + nll==

模型通常输出的是没有归一化的 logits，所以需要进行 softmax 将其变成概率，然后求负对数似然损失，NLLLoss 本质就是拿 log 概率，取出真实类别那一项，取负数：

$$
nll=−\log(p_{\text{真实类别}}​)
$$

因为对于错误项 $y_i=0$，所以公式可以直接化简为：

$$
loss=-\log{p_i}
$$

因此 CrossEntropyLoss 函数也只需要传入正确标签，不需要传入 one-hot 编码。

```python
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# 假设：
# logits: [N, C]
# targets: [N]
N, C = 8, 10
logits = torch.randn(N, C)      # shape (N, C)
targets = torch.randint(0, C, (N,))  # shape (N,)

# loss:
criterion = CrossEntropyLoss(reduction='mean')
loss = criterion(logits, targets)

# shapes:
# - CrossEntropyLoss expects logits: (N, C)
# - targets: (N,)
# - If reduction='none', output shape would be (N,) (per-sample loss)
print(logits.shape, targets.shape, loss.shape)  # (8,10) (8,) torch.Size([])
```

==注意！==

CrossEntropyLoss 对 logits 和 labels 的形状有要求：

- logits: \[N, C]
- labels: \[N]

所以在 NLP 问题中通常输出的 logits 是 \[batch_size, seq_len, vocal_size] 需要变形到 \[batch_size * seq_len, vocal_size]，并且 labels 需要变形为 \[batch_size * seq_len]


### 对数概率

首先我们经过模型输出得到的是 logtis，但是他是一个分布我们需要通过 softmax 把它变成一个概率：

$$
softmax(x_i)=\frac{e^{x_i}}{\sum_j^{dim}{e^{x_j}}}
$$

我们可以通过减去最大值等方法让它变得更稳定。除此之外，由于计算指数的开销很大，我们可以通过对数化 prob 来减少计算的次数：

$$
log\_softmax = x_i - \log({\sum_j^{dim}{e^{x_j}}})
$$

### 负对数似然

**1. 概率（Probability）：已知模型，预测结果**

假设硬币正面朝上的概率是 $\theta$（模型参数），抛 10 次硬币，出现 “正正反正” 这样的结果（数据 D）的概率是：\($P(D|\theta) = \theta \times \theta \times (1-\theta) \times \theta = \theta^3(1-\theta)^1$)

概率回答的是：已知参数 θ，结果 D 发生的可能性有多大？

**2. 似然（Likelihood）：已知结果，反推模型**

现在反过来：我们已经抛了 10 次硬币，得到了 “正正反正” 这个结果（数据 D），想反推参数 θ（正面概率）应该取多少，才能让这个结果 “最合理”。

似然函数就是把概率的自变量反过来：$L(\theta|D) = P(D|\theta)$

似然回答的是：已知结果 D，参数 θ 取某个值时，能让这个结果发生的可能性有多大？我们的目标是找到让 ($L(\theta|D)$) 最大的 θ（最大似然估计）。

为了简化计算，所以类似 softmax 在似然函数前面加上 log：$logL(θ∣D)=logP(D∣θ)=∑logP(单个结果∣θ)$

---





## BCEWithLogitsLoss

二元交叉熵损失用于二分类问题：

```python
# logits: [N] or [N, 1]
# targets: [N]  (0/1 floats or ints)

import torch
from torch.nn import BCEWithLogitsLoss

N = 12
logits = torch.randn(N)             # shape (N,)
targets = torch.randint(0, 2, (N,)).float()  # shape (N,)

criterion = BCEWithLogitsLoss(reduction='mean')
loss = criterion(logits, targets)  # scalar

# shapes:
# - if logits shape is (N,), targets must be (N,)
# - you can also use logits (N,1) and targets (N,1)
print(logits.shape, targets.shape, loss.shape)
```

> 注意 BCEWithLogitsLoss 要求 logits 和 targets 的形状相同

二分类问题除了用二元交叉熵还可以通过交叉熵损失来计算，将 logits 投影到 \[N, 2] 就可以了。

---

二元交叉熵损失也可以用于多标签问题

```python
# B 指的是 batch_size
# C 指的是标签个数 label_size

import torch
from torch.nn import BCEWithLogitsLoss

B, C = 4, 6
logits = torch.randn(B, C)                 # (B, C)
targets = torch.randint(0, 2, (B, C)).float()  # (B, C)

criterion = BCEWithLogitsLoss(reduction='mean')
loss = criterion(logits, targets)  # scalar

# shapes:
# - logits: (B, C)
# - targets: (B, C)
# - if reduction='none', output shape would be (B, C) -- per-label loss
print(logits.shape, targets.shape, loss.shape)
```




