---
title: Mask On Transformer
date: 2026-01-24T12:22:15+08:00
featuredImage: http://img.xilyfe.top/img/20260124122403258.png
authors:
  - Xilyfe
series:
  - Transformer相关
tags:
  - 掩码
  - Transformer
lastmod: 2026-01-26T09:17:09+08:00
---
在 Transformer 中同时运用了两种掩码技术：
1. 用于**处理非定长序列**的 padding mask
2. 用于**防止标签泄露**的 causal mask

## Padding Mask

NLP 任务中，输入的长度往往不是统一的，训练的数据集里面样本长度各有不同。但是我们在实际训练中，往往需要把多个数据合成一个大的 batch 一同训练，这样可以充分利用显卡的性能。那么问题就来了，不同长度的文本如何合成一个大 batch 呢。NLP 的解决思路是：把所有输入的文本统一成一个固定长度，多余的位置用特殊字符 \<PAD> 来填充。

生成 Padding Mask 的思路非常简单，我们只需要把输入的 token_id 和 \<PAD> 的 special_id 进行对比，得到一个布尔矩阵就好了，它表示了哪些 token 是填充的。迭代中一个 batch 里 x 形状是 \[batch_size, seq_len]，而应用 mask 时候 attn 形状为 \[batch_size, n_heads, seq_len, seq_len]，所以为了向量能够广播，我们需要把维度对齐。

```python
import tokenizer

for x, y in dataloader:
	padding_mask = (x != tokenizer.pad_idx).unsqueeze(1).unsqueeze(-1)
	logits = model(x, padding_mask)
```

我们使用 Padding Mask 的目的是让每个 token 的注意力不浪费在那些无意义的填充字符上，所以我们需要在 softmax 之前对注意力分数进行处理。我们把注意力分数里那些不希望关注的部分，置为一个非常大的负数，这样 softmax 之后它们的注意力权重就会接近于 0。

```python
class Attention(nn.Module):

	def forward(self, x, padding_mask):
		# ...
		scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(self.head_dim)
		if padding_mask is not None:
	        scores = scores.masked_fill(~padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = nn.Softmax(scores, dim=-1)
```

`masked_fill` 函数会把 Padding Mask 中值为 True 的位置置为 -INF。

>这里需要注意 `masked_fill` 和 `masked_fill_` 的区别，后者是一个原地操作。

## Causal Mask

Causal Mask 主要用于限定模型的可视范围，防止模型看到未来的数据。

我们知道 Transformer 是一个自回归模型，它的预训练方式称为 Teacher-Forcing。对于一个数据 "I love eating lunch"，它会不断用 "I" 预测 "love"，用 "I love" 预测 "eating"，用 "I love eating" 预测 "lunch"。但是我们同样知道注意力机制它的优势在于可以 **观察上下文**，也就是它会通过下文来帮助理解 token，这就与 Teacher-Forcing 冲突了。所以我们需要用 Causal Mask 因果注意力把下文掩码掉。

![](http://img.xilyfe.top/img/20260126202902147.png)

在具体应用中，Causal Mask 将所有未来的 token 的注意力分数设为负无穷，这样注意力权重就会接近于 0，从注意力机制中屏蔽掉这些令牌，使得模型在进行预测时只能关注过去和当前的 token，并确保模型仅基于每个时间步骤可用的信息进行预测。

```python
class Attention(nn.Module):
	def forward(self, x):
		# ...
		scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(self.head_dim)
		causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        scores[..., -seq_len:] += causal_mask
```

**这里需要注意：scores 的形状是 \[bsize, n_heads, seq_q, seq_k] 而不是 \[bsize, n_heads, seq_len, seq_len]。**
为什么 query 的长度和 key 的长度不一样？我们需要回忆一下 KVCache 的知识点。

![](http://img.xilyfe.top/img/20260119121117689.png)

对于第 i 次循环我们要生成 $token_i$​，它只需要 $QK^T$ 这个下三角矩阵的最后一行和 $V$ 矩阵。再拆细一点，我们只需要 $Q_i$​ 和 $K$ 矩阵相乘得到下三角矩阵最后一行还有 $V$，所以我们只需要缓存 $K$ 和 $V$ 矩阵。

```python
# KVCache
if past_key_value is not None:
	key = torch.cat([past_key_value[0], key], dim=1)
	value = torch.cat([past_key_value[1], value], dim=1)
past_key_value = (key, value) if use_cache else None
```

可以看到，自回归训练利用 KVCache 之后每次 Query 的长度都是 1，而 Key 和 Value 的长度是 past_len + q_len，我们只需要对 **“当前新增的 key 部分”** 施加 causal 结构，`-seq_len:` 正好选中最后新增的那一小段 key。