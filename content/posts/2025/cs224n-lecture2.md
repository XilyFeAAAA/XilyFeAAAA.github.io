---
title: "CS224N Lecture 2: Word Vectors and Language Models"
date: 2025-11-19T17:13:11+08:00
authors:
  - Xilyfe
series:
  - CS224N
tags:
  - 深度学习
lastmod: 2026-01-30T02:50:12+08:00
featuredImage: http://img.xilyfe.top/img/20260130144509787.png
---
## 梯度下降

$$
\theta^{new}_j := \theta^{old}_j - \eta \, \nabla_{\theta} J(\theta)
$$

传统梯度下降算法的缺点在于：
1. 计算成本大：$J(\theta) = \frac{1}{N} \sum_{i=1}^{N} L(x_i, y_i; \theta)$，这意味着每一次计算梯度都需要全部样本参与计算，开销很大
2. 容易陷入局部最小值


## 随机梯度下降

仅仅在全部数据集中选取一个很小的子集，例如16或32个数据，用这些数据充当完整的数据集来计算损失函数和优化梯度

> 在深度学习中出现噪声经常会有更好的效果

## 跳字负采样

Skip-Gram用的是Softmax概率:
$$
P(w_o \mid w_i) = \frac{\exp(v_{w_o}'^T v_{w_i})}{\sum_{w=1}^{V} \exp(v_w'^T v_{w_i})}
$$
也就是说会计算整个Corpus的分母，计算量非常大。

---
负采样的思想是，不再计算整个词表的概率，而是单单判断正样本（上下文）和负样本（随机采样的非上下文词）

对于一个中心词$w_i$和一个上下文词$w_o$以及负样本集K，目标函数是：
$$
J(w_i, w_o) = - \log \sigma(v_{w_o}' \cdot v_{w_i}) - \sum_{k=1}^{K} \log \sigma(-v_{w_k}' \cdot v_{w_i})
$$


## 共现矩阵

回顾之前提到的构造词向量的方法：one-hot编码和word2vec思想，还有一种更简单的表示思路。让相邻的词的向量表示相似，直接统计哪些词是经常一起出现的。下面图片就是CS224N课程中举的例子，根据三个句子的语料库构造的共现矩阵。

![](http://img.xilyfe.top/img/20260130144952699.png)

这样的表示明显优于one-hot表示，因为它的每一维都有含义——共现次数，因此这样的向量表示可以求词语之间的相似度。  
但是这样表示还有有一些问题：
1. 维度=词汇量大小，还是太大了
2. 还是太过于稀疏，在做下游任务的时候依然不够方便。
---

对于第一个问题可以采用SVD奇异值分解的方法，它可以将任意矩阵分解为三个矩阵的乘积。为了减少尺度同时尽量保存有效信息，保留对角矩阵的最大的k个值，并将矩阵U，V 的相应的行列保留。

![](http://img.xilyfe.top/img/20260130145007040.png)

## GloVe

GloVe=Global+Vector，它的核心理念是：如果两个词在语料库中经常出现在相似的上下文中，它们的词向量就应该相似。

但不同于 Word2Vec 的「预测式」学习，GloVe 是一个「统计式」模型，直接利用共现矩阵的全局统计信息。

### 共现概率

人为的定义共现概率：
$$
P(j|i)=P_{ij}=\frac{X_{ij}}{X_i}
$$

- $X_{ij}$是词j出现在词i上下文的总次数
- $X_i=\sum_kX_{ik}$，也就是词i的上下文总数

举个例子，Corpus中有下面两个句子，并且我们需要计算P(I,like)：
```
I like A
I like B
```

那我们可以得到共现矩阵:
|        |   I   |  like  |   A   |   B   |
|  ----  | ----  |  ----  | ----  |  ---- |
|    I   |   0   |    2   |   0   |   0   |
|  like  |   2   |    0   |   1   |   1   |
|    A   |   0   |    1   |   0   |   0   |
|    B   |   0   |    1   |   0   |   0   |

根据公式可以得到：
$$
P(I,like)=\frac{X_{I,like}}{X_I}
$$
从共现矩阵可以看到，$X_{I,like}=2$。假设窗口大小为1，那么$X_I=2$，$P(I,like)=1$；假设窗口大小为2，那么$P(I,like)=\frac{1}{2}$

---

GloVe 的一个非常直观的出发点是：
> 某些词之间的意义差异可以通过共现概率比值体现。

$$
\frac{P(apple|red)}{P(apple|green)} = \frac{P_{red,apple}}{P_{green,apple}} \ge 1
$$
那么就认为red比green相对apple更有意义.

于是GloVe希望学习到一个函数能够近似概率比值:
$$
F(wi, wj, \tilde{w_k})=\frac{P_{ik}}{P_{jk}}
$$

经过一系列数学推导(下文会计算)，词向量的点乘可以近似于:
$$
w_i^Tw_j+b_i+b_j \approx \log{X_{ij}}
$$

此时损失函数就变成了拟合这个关系，我们希望他们尽可能相等，也就是
$$
w_i^Tw_j+b_i+b_j - \log{X_{ij}}
$$
尽可能等于0
$$
J=\sum_{i,j}(​{w_i^T​w_j​+b_i​+b_j​−\log{X_{ij}}​})^2
$$

得到目标函数/损失函数的表达式后的操作就很简单了，对式子求偏导进行梯度下降，优化随机生成的词向量$w_i$和$w_j$

---

某些词（比如 “the”）出现太多，会让损失被它们主导。所以 GloVe 加了一个权重函数$f(X_{ij})$。$f(x)$ 越大，代表词的重要性越高。
$$
J=\sum_{i,j}f(X_{ij})(​{w_i^T​w_j​+b_i​+b_j​−\log{X_{ij}}​})^2
$$


> 与Word2Vec的目标函数相比，GloVe的损失函数更像回归问题

---

Glove 的作者认为，单词词向量空间是一个线性结构，例如 “man” - “women” 的差与 “king” - “queen” 的差很相近.

作者假设存在一个函数：
$$
F(w_i-w_j,w_k)=\frac{P_{ik}}{P_{jk}}
$$
并且假设F是一个点乘的计算即：
$$
F(w_i-w_j,w_k)=w_k^T(w_i-w_j)=\log{\frac{P_{ik}}{P_{jk}}}=\log{P_{ik}}-\log{P_{jk}}
$$
这个式子可以化简为:
$$
w_i^T w_k - w_j^T w_k = \log(P_{ik}) - \log(P_{jk})
$$
从中可以看到存在这样的关系：
$$
w_i^T w_k \approx \log(P_{ik}) = \log{\frac{X_{ik}}{X_i}}=\log{X_{ik}-\log(X_i)}
$$
即：
$$
w_i^T w_k + \log(X_i) \approx \log{X_{ik}}
$$
我们可以抽象出一般的形式,$b_i$和$\tilde{b_k}$是偏置，吸收了$\log(X_i)$和其他常数：
$$
w_i^Tw_j + b_i + \tilde{b_j} \approx \log{X_{ij}}
$$

值得注意的是，在 GloVe 中，每个词都有两个向量：

词向量 $w_i$ — 表示这个词本身

上下文向量 $\tilde{w_j}$ — 表示当这个词出现在别的词的上下文中时的“角色”

并不是每次出现的上下文词直接拿句子里的其他词的向量，而是每个词都有一个固定的上下文向量，用来参与 共现对的预测。

最后训练结束词向量一般直接用$w_i$或者取$w_i$和$\tilde{w_i}$的平均值

## 评估