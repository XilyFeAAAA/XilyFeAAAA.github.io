---
title: Flash Attention
date: 2026-03-14T10:59:39+08:00
featuredImage: http://img.xilyfe.top/img/20260314174134259.png
authors:
  - Xilyfe
series: []
tags:
  - GPU
lastmod: 2026-03-14T11:44:21+08:00
---
## 前情提要

![image.png](http://img.xilyfe.top/img/20260314111856134.png)

GPU 存储分为芯片内和芯片外，芯片内的 SRAM 用于储存需要计算的临时数据，显存 HBM 在芯片外：
- HBM：位于 GPU 芯片外，就是我们所说的显存，类似于 CPU 的 DRAM，储存模型训练和推理时的参数，容量大，例如 A100 一般为 40G 或 80G。
- SRAM：位于 GPU 芯片上，仅用于存储 CUDA Kernel 计算时所需的临时数据，容量极限一般在 20MB
- CUDA Kernel：GPU 上执行并行的计算函数，是实现并行计算任务的基本单元

![image.png](http://img.xilyfe.top/img/20260314112212450.png)

原始的 Attention 计算是 $S = QK^T,\quad P=\text{softmax}(S),\quad O = PV$，GPU 需要以下步骤，总计 6 次通信：
1. 从 HBM 中加载 $Q$、$K$ 到 SRAM
2. Kernel 计算出 $S=QK^T$
3. 将 $S$ 写会 HBM
4. 将 $S$ 加载到 SRAM
5. 计算 $P=\text{softmax}(S)$
6. 将 $P$ 写回 HBM
7. 将 $P$、$V$ 加载到 SRAM
8. 计算 $O=PV$
9. 将 $O$ 写回 HBM

但我这里就很奇怪了，既然 Kernel 能计算出 $S=QK^T$，那不就代表 SRAM 能存下整个 $S$ 矩阵了吗？那何必在 SRAM 和 HBM 里面来回移动呢？实际上，在传统 kernel 里 GPU 并不是把 **整个 Q、K 都读进 SRAM** 再算出 $S$ 的，而是 **分 tile读入并计算**。我们的矩阵会被分为一个个 tile 如下图：

```
+----+----+----+----+
|t11 |t12 |t13 |... |
+----+----+----+----+
|t21 |t22 |t23 |... |
+----+----+----+----+
|t31 |t32 |t33 |... |
+----+----+----+----+
```

然后 kernel 按照下面的方式，一次次计算一个 tile 并返回，最后得到完整的计算结果，实际上就是分块矩阵的思想：

```python
for Qi in Q_tiles:
    load Qi from HBM

    for Kj in K_tiles:
        load Kj from HBM
        Sij = Qi @ Kj^T
        store Sij to HBM
```

所以实际上 SRAM 和 HBM 的通信次数是 $n^2 \, / \, \text{tile\_size} \times 6$，我们说的 6 次是在 Matrix Level。


## 优化思路

![image.png](http://img.xilyfe.top/img/20260314115324360.png)

假设不考虑 softmax 的过程，我们计算 $O=QK^TV$，这样我们只需要两次 HBM 和 SRAM 的通信了，把 $Q$、$K$、$V$ 分块从 HBM 读入 SRAM，计算之后再把这一小块 $O$ 从 SRAM 写会 HBM，最后就能得到计算结果了。但是问题就出在 softmax 身上，由于 softmax 每次需要一整行数据，但是分块后 $S=QK^T$ 只有一小块，并不是一整行，如下图。

![image.png](http://img.xilyfe.top/img/20260314115905015.png)

## 前向传播

{{< admonition type=info title="补充知识">}} 

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}}
$$
float16 支持的范围是 ±65504，意味着当 x>11时候，$e^x$ 将超过有效范围出现溢出，这就引出了 safe softmax 的概念（PS：在 CS336 手写 softmax 时候我们就实现过）。

$$
\text{safe softmax}(x_i) = \frac{e^{x_i} / e^{max(x)}}{\sum_{j=1}^N e^{x_j}  / e^{max(x)}} = \frac{e^{x_i-max(x)}}{\sum_{j=1}^N e^{x_j-max(x)}}
$$

每个数字减去最大值再求 softmax 不会改变最终结果，所以在实际使用时都用 safe softmax。
{{< /admonition >}}

假设 $S=QK^T$ 的第一行为 $[1,2,3,4]$，我们把矩阵 $Q$ 和 $V$ 分块，从 HBM 读入数据到 SRAM 分别得到了第一行的一部分 $[1,2]$ 和 $[3,4]$，此时需要计算第一行的 softmax 值 $P=\text{softmax}(S)$：
1. 计算每一块的最大值：$m_1=max(1,2)=2$ 和 $m_2=max(3,4)=4$
2. 计算每一块的分子：$f_1=[e^{-1},e^0]$ 和 $f_2=[e^{-1},e^0]$
3. 计算每一块的分母：$l_1=e^{-1}+e^0$ 和 $l_2=e^{-1}+e^0$
4. 合并最大值：$m = \max(m_1,m_2)=4$
5. 计算全局分母：$l=e^{m_1-m}\times l_1+e^{m_2-m}\times l_2$ 
6. 计算最终 softmax 结果：$o_1 = \frac{e^{m_1-m} \times f_1}{l}$ 和 $o_2 = \frac{e^{m_2-m} \times f_2}{l}$

但是我们计算 $[1,2]$ 的时候如何知道整个序列的 $m$ 和 $l$ 呢？我们进一步看一下论文是怎么写的：

![image.png](http://img.xilyfe.top/img/20260314174429904.png)

我认为 Flash Attention 里面很巧妙的一点在于，它忽略了 $S$ 和 $P$ 直接计算 $O$。刚刚我们认为存在问题是因为站在了 $P$ 矩阵或者 $S$ 矩阵的视角上，它的形状是 $N \times N$，我们总想着怎么把它在列方向拆分。而 $O=PV$ 的形状是 $N \times d$，这样我们就可以让分块矩阵 $Q$、$K$、$V$ 在 $O$ 的每一行上<mark>原地更新</mark>。

这里我举一个例子就能完全理解：将 $P$ 求平均值然后和 $V$ 做乘法。假设我们有行向量：

$$
P=\begin{bmatrix}  1 & 2 & 3   \end{bmatrix}
$$

和列向量：

$$
V =  \begin{bmatrix} 4 \\  5 \\ 6  \end{bmatrix}
$$

我们的目标是计算：

$$
O=\frac{P_1V_1+P_2V_2+P_3V_3}{P_1+P_2+P_3}
$$
由于空间不足不能一次性读入 $P$ 矩阵，所以我们只能一个一个获取 $P_i$ 和 $V_i$：
1. 先看 $P_1 = 1$，此时分母为 $0+1=1$，我们计算 $O \leftarrow \frac{O*1+P_1V_1}{1} = \frac{0 + 1 \cdot 4}{1} = 4$ 
2. 加上 $P_2 = 2$，新分母为 $1+2=3$，把旧 $O$ 乘回旧分母，加上新项，再除新分母 $O \leftarrow \frac{O*1+P_2V_2}{3} = \frac{4 \times 1 + 2 \cdot 5}{3} = \frac{14}{3}$
3. 加上 $P_3=3$，新分母为 $1+2+3=6$，我们同样操作 $O \leftarrow \frac{O*3+P_3V_3}{6} = \frac{\frac{14}{3} \times 3 + 3 \cdot 6}{6} = \frac{16}{3}$

这就是 <mark>在线归一化</mark> 的思想。在 Flash Attention 中，我们除了需要重新计算全局分母，还需要在最大值更新时更新分子，这就是 Flash Attention 的全部思想。

{{< admonition type=question title="为什么先遍历 KV 矩阵再遍历 Q 矩阵？">}} 

```python
for KV_block in KV:          # 外层
    load K_block, V_block -> shared memory
    
    for Q_block in Q:        # 内层
        load Q_block
        compute Q_block @ K_block^T
        online softmax update
        accumulate with V_block
```

假如我们有 n 个 Q block 和 m 个 KV block（通常情况下 n 都是大等于 m 的，比如 Inference 的情况）。先加载 KV block，它就可以和所有的 Q block 比较，总共需要 load 一次全部 KV block 和 m 次全部 Q block。假如我们先加载 Q block，那么总共需要 load 一次全部 Q block 和 n 次 KV block，明显这种 HBM 访问次数更多。
{{< /admonition >}}

## 反向传播

![image.png](http://img.xilyfe.top/img/20260314203557920.png)

反向传播的梯度计算太复杂了，这里就不具体推到了。它的核心在于，虽然前向计算中省略了 $S$ 和 $P$ 矩阵的计算，缺少了激活值，但是我们在 HBM 里面也存了 $l$ 和 $m$ 可以帮助我们在反向传播中很快的 recompute，性能不会差非常多。
