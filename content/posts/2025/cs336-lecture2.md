---
title: "Lecture 2: Computing"
date: '2025-12-19T11:48:11+08:00'
authors: [Xilyfe]
series: ["CS336"]
tags: ["大模型"]
--- 

!!! abstract 
    本节主要讲解内存计算问题，首先介绍了float32, float 16 等数据类型。随后介绍 PyTorch 中的 tensor 这一重要的数据类型。最后举例介绍了在模型训练中各个部分所需要的计算量，并介绍了浮点运算利用率这一指标以衡量硬件计算效率。重点如下： 
    1. 在PyTorch 中 tensor 是对已分配内存的指针，很多操作无需新占用内存 
    2. 大型矩阵乘法在深度学习所需计算量最大 
    3. 浮点运算利用率 $MFU=\frac{actualFLOP/s}{promised FLOP/s}$  
    4. 前向传播所需计算量：2×(#tokens)×(#parameters) 
    5. 反向传播所需计算量：4×(#tokens)×(#parameters)

## Memory Accounting

介绍了 FP32、FP16 以及 BP16 和 FP8 四种精度，在 CS224 介绍过，就不多说了。

## Compute Accounting

在 Ptorch 中，张量存储了指向数据的指针，初次之外还有一些 Metadata，例如 strides。

> stride 指的是：沿着这个方向到达下一行需要经过几个元素

<div style="text-align: center">
	<img src="https://raw.githubusercontent.com/XilyFeAAAA/ImgRepository/main/img/pytorch_tensor.png?token=A3JNYA5O3GQ7FWC2Y2NVA23JII6NI" width="70%" /> 
</div>

在索引 Tensor 中元素(x,y)时，只需要在如图所示的长数组中找到第 N 个元素即可，索引计算方式为：

$$
N = x \times \text{stride}[1] + y \times \text{stride}[0]
$$

在PyTorch 中，很多对于 tensor 的操作实际上只是在创建新的视图(view)，而无需重新分配内存。例如对于 Slice 操作来说，假如我有变量 `x = torch.arrange(10)`，就会将 \[0, 1, 2, 3, 4, ..., 9] 存在内存中。当我对他进行切片操作 `y = x[2:9:2]` 时，张量 y 的 Pointer 还是指向 x 的同一块区域，但是 y 的 Metadata 就该改变了：

- `assert y.data_prt() == x.data_prt()`
- `assert y.stride() == 2`
- `assert y.storage_offset() == 2`


Pytorch 中大部分对 tensor 的操作默认要求其是 contiguous，连续的定义是==逻辑上相邻的元素在内存中也相邻==，具体来说：

1. stride\[n-1] == 1
2. stride\[i] == stride\[i+1] * size\[i+1]

进行切片操作 \[::2] 或者转置(strides\[-1]和strides\[-2]互换)时，tensor 就是 non-contiguous 的了。

> `.contiguous()` 会返回一个在内存布局上是 contiguous 的 tensor，并且在数值上与原 tensor 等价。


## Compute Cost

一次浮点数运算是指一次基本计算，包括加法(”+”)和 乘法(”×”)。注意区分如下两个简写：

1. FLOPs：浮点计算次数，用以衡量计算量
2. FLOP/s：每秒浮点计算数量，也叫 FLOPS，用以衡量硬件计算速度

**对于 Linear 计算**：

```python
x = torch.ones(B, D)
w = torch.randn(D, K)
y = x @ w
```

总浮点计算量为：$flops=2*B*D*K$。乘以 2 的原因是，$y[i][k]=\sum_{j=0}^{D-1}{x[i][j] * w[j][k]}$，所以需要乘法+加法两次（加法和乘法看做是等价的），我们可以把它进一步分解为：

```python
for i in range(B):
	for k in range(K):
		for j in range(D):
			y[i][k] += x[i][j] + w[j][k]
```

**对于反向传播**：

```python
h1 = x[B,D] * w1[D,D]
h2 = h1[B,D] * w2[D,K]
loss = loss(h2[B,K])
```

- $\frac{\partial{loss}}{\partial{h_1}}=\frac{\partial{loss}}{\partial{h_2}}[B,K] @ w_2^T[K,D]$ ：计算量为 $2*B*D*K$
- $\frac{\partial{loss}}{\partial{w_1}}=\frac{\partial{loss}}{\partial{h_1}}[B,D] @ x^T[D,B]$ ：计算量为 $2*B*D*D$

同时 $\frac{\partial{loss}}{\partial{x}}$ 和 $\frac{\partial{loss}}{\partial{h_2}}$ 一样