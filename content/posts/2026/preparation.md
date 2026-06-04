---
title: 八股 & 手撕
date: 2026-06-02T17:55:39+08:00
featuredImage: http://img.xilyfe.top/img/20260604102855590.png
authors:
  - Xilyfe
series: []
tags: []
lastmod: 2026-06-03T09:15:02+08:00
---
>准备 2026 暑假 LLM 算法实习ing


## 1. 八股

### 1.1 Transformer

#### 架构

Transformer 的核心是用 **Self-Attention + FFN** 替代 RNN/CNN，实现对序列中任意位置的全局建模。

经典 Transformer 分为：

- **Encoder-only**：如 BERT，适合理解类任务。
- **Decoder-only**：如 GPT / LLaMA / Qwen，适合自回归生成。
- **Encoder-Decoder**：如原始 Transformer / T5，适合 Seq2Seq 任务。

LLM 中最常见的是 **Decoder-only Transformer**，单层结构通常为：

```
Input Tokens
   ↓
Token Embedding + Position Encoding
   ↓
[ Transformer Block ] × N
   ↓
Final Norm
   ↓
LM Head
   ↓
Next Token Distribution
```

一个 Decoder Transformer Block 通常包含：

```
x
│
├── RMSNorm / LayerNorm
│
├── Masked Multi-Head Self-Attention
│
├── Residual Add
│
├── RMSNorm / LayerNorm
│
├── FFN / MLP / SwiGLU
│
└── Residual Add
```

| 结构         | 特点                                                 |
| ---------- | -------------------------------------------------- |
| Post-Norm  | `x + Sublayer(LN(x))` 之前的早期 Transformer 常用，深层训练不稳定 |
| Pre-Norm   | `x + Sublayer(Norm(x))`，LLM 常用，训练更稳定               |
| RMSNorm    | 只归一化均方根，不减均值，计算更省，LLaMA 系常用                        |
| SwiGLU FFN | 比 ReLU/GELU FFN 表达能力更强，现代 LLM 常用                   |
| RoPE       | 旋转位置编码，支持相对位置信息和一定长度外推                             |
| GQA/MQA    | 减少 KV Cache 和推理显存                                  |

#### Q/K/V 解决了什么问题

Self-Attention 的核心公式：

$$ \text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

其中：

- **Q / Query**：当前位置想要查找什么信息。
- **K / Key**：每个位置提供什么索引特征。
- **V / Value**：真正被聚合的信息内容。

假设输入为：

$$ X \in \mathbb{R}^{B \times L \times d_{\text{model}}} $$

通过三个线性变换得到：

$$ Q=XW_Q,\quad K=XW_K,\quad V=XW_V $$

Q/K/V 的作用：

1. **解耦匹配与内容**
    - `QK^T` 负责计算 token 之间的相关性。
    - `V` 负责提供最终被加权汇聚的信息。
    - 如果不用 Q/K/V，而直接对原始 embedding 做相似度，会限制模型表达能力。
2. **允许不同子空间学习不同关系**
    - 多头注意力中，每个 head 有自己的 Q/K/V。
    - 不同 head 可以关注语法、实体、指代、位置、格式等不同模式。
3. **支持动态上下文建模**
    - Attention 权重由当前输入动态决定。
    - 相比固定卷积核，Self-Attention 能根据上下文自适应选择信息来源。
4. **支持并行计算**
    - RNN 依赖时间步递推。
    - Transformer 可以一次性计算整个序列的 Q/K/V，训练并行度高。

#### 多层 transformer 叠加出现的问题

#### 残差连接作用？会带来什么问题？

#### 各种 norm 的差异&作用

#### RoPE

#### 有哪些减少 attention 计算量的方法


### 1.2 LoRA 微调

#### 原理

#### 参数选择

#### LoRA 加在哪些模块上

#### 多轮对话微调


### 1.3 Supervised Finetune

#### 原理

#### sft 到什么程度可以 rl

#### 怎么评估 sft 效果

#### sft 数据怎么评估好坏

#### 样本长度差异大怎么办

### 1.4 RLHF

#### rl 比 sft 好在哪

#### 重要性采样

#### KL divergence

#### Policy Gradient

#### 蒙特卡洛估计

#### 广义优势估计

#### TD Error

#### PPO

#### DPO

#### GRPO

#### DAPO

#### GSPO

#### agentic rl 的 credit assignment

#### RLHF 训练的指标

>如何判断 early stop

#### 熵崩塌

#### RL数据和SFT数据需要有重合吗？

### 1.4 分布式训练

#### Data Parallel

#### Tensor Parallel

#### Deepspeed Zero


### 1.5 训练

#### 参数量计算

#### 显存计算

#### 训练出现 NaN 的原因

>lr,除0，log，clamp



## 2. 手撕

### 2.1 LLM

#### Softmax

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}
$$

>数值稳定的 Softmax，减去最大值防止溢出。

```python
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - x_max)
    return e / e.sum(dim=dim, keepdim=True)
```

$$
\log\text{softmax}(x_i) = \log{\frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}} = x_i - m - \log\left(\sum_{j=1}^{n} e^{x_j - m}\right)
$$

```python
def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - x_max)
    return x - x_max - e.sum(dim=dim, keepdim=True)
```


#### CrossEntropyLoss

假设真实分布为 $p(x)$，模型预测分布为 $q(x)$，那么交叉熵即为：

$$
H(p,q) = -\sum_x p(x)\log q(x)
$$

在 LLM 的语境下：
- $q(x)$ 表示模型预测得到的 token 概率分布；
- $p(x)$ 表示真实 next token 的概率分布。

由于训练数据中的真实 token 通常使用 one-hot 分布表示，因此于是交叉熵可以化简为：

$$
H(p,q)=-\log q(x_t)=-\log P(x_t \mid x_{<t})
$$

对于整条序列而言，语言模型的交叉熵损失即所有 token 交叉熵的平均值：

$$
L = -\frac{1}{T}\sum_{t=1}^{T}\log P(x_t \mid x_{<t}) = -\frac{1}{T}\sum_{t=1}^{T}\log { \frac{e^{x_t - m}}{\sum_{j=1}^{n} e^{x_j - m}}}
$$

```python
def cross_entropy_loss(logits, targets):
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1) # [bs, len, vocab]
    logprobs_flat = logprobs.view(-1, logprobs.size(-1))
    targets_flat = targets.view(-1)
	return -logprobs_flat[range(len(targets_flat)), targets_flat].mean()
```



#### PPL

$$
\text{PPL}(x_1, x_2, \ldots, x_N) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log q(x_i \mid x_1, \ldots, x_{i-1})\right)
$$

```python
import torch
import torch.nn.functional as F
import math

def compute_perplexity(logits, targets, ignore_index=-100):
    bs, seq_len, vocab_size = logits.size()
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction="mean"
    )
    return torch.exp(loss)
```

#### MHA

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model can not be divided by num_heads without reminders"
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        seq_len = 0
        self.mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), diagonal=1)


    def forward(self, Q, K, V):
        bs, seq_len, _ = Q.size()
        q = self.q_proj(Q)
        k = self.k_proj(K)
        v = self.v_proj(V)

        q = q.view(*q.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(*v.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(*v.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if self.mask is not None:
            scores[..., -seq_len:] += self.mask
        
        attn = torch.softmax(scores, dim=-1)
        output = (attn @ v).transpose(1, 2).reshape(bs, seq_len, self.d_model)
        return self.o_proj(output)

```

#### MHA + KVCache

```python
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # 在此实现
        assert d_model % num_heads == 0, "d_model can not be divided by num_heads without reminders"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None, mask=None):
        # 在此实现
        B, T, _ = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        if cache is not None:
            k = torch.concat([cache["k"], k], dim=2)  # seq_len 维度 concat
            v = torch.concat([cache["v"], v], dim=2)
        
        new_cache = {
            "k": k,
            "v": v
        }

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        output = (attn @ v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.o_proj(output), new_cache
```

#### GQA

#### FFN

#### MSE

#### KL divergence
#### LayerNorm / RMSNorm

#### LoRA

#### ReLU / SwiGLU

#### Importance Sampling

#### Loss

#### Greedy Search
#### Beam Search
#### Top-K
#### Top-p


### 2.1 算法与数据结构
#### 最长上升子序列
#### 手撕解析括号
#### 已知 rand5 求 rand3 和 rand7
#### 英文句子分部分反转
#### 找链表第一个公共节点
#### 大数乘法
#### On 数组第 k 大
#### 编辑距离
#### 链表题（具体忘了）
#### 岛屿数量
#### 最长上升子序列
#### 三数之和