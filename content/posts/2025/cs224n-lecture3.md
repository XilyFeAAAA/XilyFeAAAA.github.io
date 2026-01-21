---
title: "CS224N Lecture 3: Backpropagation, Neural Network"
date: '2025-11-19T11:22:11+08:00'
authors: [Xilyfe]
series: ["CS224N"]
tags: ["深度学习"]
lastmod: 2026-01-21T12:35:41+08:00
--- 


## 矩阵微积分

$$
\frac{\partial}{\partial{\textbf{x}}}(\textbf{Wx+b})=\textbf{W}
$$

$$
\frac{\partial}{\partial{\textbf{b}}}(\textbf{Wx+b})=\textbf{I}
$$

$$
\frac{\partial}{\partial{\textbf{u}}}(\textbf{u}^Th)=\textbf{h}^T
$$

## 雅克比行列式

$$
\textbf{h}=f(\textbf{z}) \\
h_i=f(z_i) 且 \textbf{h} ,\textbf{z} \in R^n
$$

可以得到:

$$
(\frac{\partial{h}}{\partial{z}})_{ij}=\frac{\partial{h_i}}{\partial{z_j}}=\frac{\partial{f(z_i)}}{\partial{z_j}}=
\begin{cases}
f'(z_i),\;if\;i=j\\
0,\;if\;otherwise
\end{cases} 
\\
\frac{\partial h}{\partial z} =
\begin{pmatrix}
f'(z_1) &        & 0 \\
        & \ddots &   \\
0       &        & f'(z_n)
\end{pmatrix}
= \operatorname{diag}(f'(z))
$$

## 反向传播

<div align="center">
    <img src="../../../../resource/ai/llm/flow.png" width="70%"/>
</div>

- 前向传播就是单纯的计算
- 反向传播是根据梯度进行学习

---
<div align="center">
    <img src="../../../../resource/ai/llm/siso.png" width="70%"/>
</div>

单输入单输出的情况下,下游梯度就是局部梯度与上游梯度的乘积。  
$$
\frac{\partial{s}}{\partial{z}}=\frac{\partial{h}}{\partial{z}} \times \frac{\partial{s}}{\partial{h}}
$$
<div align="center">
    <img src="../../../../resource/ai/llm/miso.png" width="70%"/>
</div>

多输入的情况下仍然遵循链式法则。

---

<div align="center">
    <img src="../../../../resource/ai/llm/error_back.png" width="78%"/>
</div>

如图是一种错误的计算反向传播的方式，如果依次计算$\frac{\partial{s}}{\partial{W}}$和$\frac{\partial{s}}{\partial{b}}$，那么会有一部分计算是重复的，就导致反向传播的效率下降。正确的方式应该是先计算公共部分，然后再计算单独的部分，这可以通过拓扑排序来实现。

## demo实现

```python
class multiplyGate():
    def forward(x, y):
        self.x = x
        self.y = y
        return x*y
    def backward(dz):
        dx = self.y * dz  # dz/dx * dL/dz
        dy = self.x * dz
        return [dx, dy]
```