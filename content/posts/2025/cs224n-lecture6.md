---
title: "CS224N Lecture 6: Sequence to Sequence Models"
date: 2025-11-22T11:24:11+08:00
authors:
  - Xilyfe
series:
  - CS224N
tags:
  - 深度学习
lastmod: 2026-01-30T02:46:59+08:00
featuredImage: http://img.xilyfe.top/img/20260130144509787.png
---


## 梯度消失&梯度爆炸

RNN 在进行反向传播，计算第 j 步的损失对前面的一步梯度的时候，需要运用链式法则：

$$
\begin{aligned}
\frac{\partial J^j}{\partial h^i}
&= \frac{\partial J^j}{\partial h^j} \cdot \frac{\partial h^j}{\partial h^i} \\
&= \frac{\partial J^j}{\partial h^j} \cdot \frac{\partial h^j}{\partial h^{j-1}} \cdot \ldots \cdot \frac{\partial h^{i+1}}{\partial h^i} \\
&= \frac{\partial J^j}{\partial h^j} \cdot \prod_{i<t\le j} \frac{\partial h^t}{\partial h^{t-1}}
\end{aligned}
$$

根据 RNN 的公式$h_t=tanh(Wx \cdot x_t + W_h \cdot h_{t-1} + b_h)$，将激活函数忽略可以得到，$h_t$对$h_{t-1}$的偏导就是 W，所以我们可以近似得到：

$$
\frac{\partial{J^j}}{\partial{h^i}} = \frac{\partial{J^j}}{\partial{h^j}} \cdot W^{j-i}
$$

可以看出，当 W 很小或者很大，同时 i 和 j 相差很远的时候，由于公式里有一个指数运算，这个梯度就会出现异常，变得超大或者超小，也就是所谓的“梯度消失/梯度爆炸”问题。

梯度爆炸的解决办法很暴力很简单，就是当梯度超过一个阈值时候，将它裁剪成阈值大小：

![](http://img.xilyfe.top/img/20260119120536522.png)

## LSTM

LSTM 在 RNN 的基础上很好的解决了长距离详细传递的问题，它引入了 Cell State 和三个门 Forget Gate, Input Gate 和 Output Gate 来传输记忆和决定哪些记忆是需要的，哪些不需要。

![](http://img.xilyfe.top/img/20260119120554596.png)

- 遗忘门：根据$h^{t-1}$和$x^t$判断 Cell State 哪一些需要遗忘
- 输入门：根据$h^{t-1}$和$x^t$判断需要向 Cell State 传入哪些当前信息
- 输出门：根据$h^{t-1}$和$x^t$判断需要从 Cell State 中输出哪些信息

以 Forget Gate 举例：

$$
f^t=\sigma(w_fh^{t-1}+U_fx^t+b_f)
$$

sigmoid 激活函数会将计算结果隐射到 0-1 的区间，然后与 $c^{t-1}$相乘。值越接近于 1，历史记忆就保留；相反值趋于 0，历史记忆就遗忘。

**为什么 LSTM 相对于 RNN 能够记忆更长的记忆？**

我们回顾一下 RNN 的公式：

$$
h_t=tanh(Wx \cdot x_t + W_h \cdot h_{t-1} + b_h)
$$

由于参数矩阵是固定的，所以进行反向传播时候，梯度要么会非常大要么会非常小。

但是对于 LSTM，它的三个门控机制可以选择每次保留 or 遗忘记忆，使得历史记忆可以长期保存。举一个极端的例子，遗忘门总是为 1，输入门总是为 0，那么历史记忆就能一直在 Cell State 上流通。

实际上，LSTM 不光是解决了长距离依赖的问题，它的各种门，使得模型的学习潜力大大提升，各种门的开闭的组合，让模型可以学习出自然语言中各种复杂的关系。比如遗忘门的使用，可以让模型学习出什么时候该把历史的信息给忘掉，这样就可以让模型在特点的时候排除干扰。