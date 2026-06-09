---
title: Interview Q&A
date: 2026-06-02T17:55:39+08:00
featuredImage: http://img.xilyfe.top/img/20260609104852216.png
authors:
  - Xilyfe
series:
  - 面经
tags: []
lastmod: 2026-06-09T11:32:27+08:00
---
>准备 2026 暑假 LLM 算法实习ing

## Transformer

### Architecture

Transformer 的核心是用 **Self-Attention + FFN** 替代 RNN/CNN，实现对序列中任意位置的全局建模。结构通常分为：

1. Encoder-only 用的是**双向自注意力**，能看到两侧的 token，所以能捕捉整体信息。
2. Decoder-only 用的**因果掩码注意力**，符合自回归生成，适合 NLP 问题。
3. Encoder-Decoder 是**交叉注意力**，Encoder 理解 Decoder 生成，所以适合 seq2seq。


![image.png](http://img.xilyfe.top/img/20260608112300509.png)


>让 Image-2 根据我的手稿生成了一个手绘风格示意图，感觉还可以。

### Attention

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V 
$$

{{< qa q="softmax 数值不稳定" >}}
$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}
$$
数值稳定的 Softmax，减去最大值防止溢出。
{{< /qa >}}

{{< qa q="softmax 里面缩放点积的作用" >}}
softmax 的数值越大，最大值与其他值差距越悬殊，softmax 越接近 one-hot。对 $QK^T$ 缩放点积使得输入始终保持在 softmax 的**线性敏感区间**，梯度不消失。
{{< /qa >}}

{{< qa q="自注意力的时间复杂度" >}}
1. 计算 $QK^T$ 是 $(n \times d_k) \cdot (d_k \times n)$ 的矩阵乘法，计算 $QK^T$ 的每个元素 $A_{ij}$ 需要 $A_{ij}=Q_i\cdot K_j$ 进行点积，所以 $n^2$ 个元素每个 $d_k$ 次计算，一共 $O(n^2d_k)$
2. softmax 操作 $O(n^2)$
3. 计算 $AV$ 是 $(n \times n)(n \times d_v)$ 的矩阵乘法，一共 $O(n^2d_v)$
4. 合计 $O(n^2 d)$
{{< /qa >}}

{{< qa q="如何实现并行计算的" >}}
attention 把 Q/K/V 矩阵的 $d_{model}$ 拆为 $\text{num\_heads} * d_k$ 维度，通过矩阵乘法就可以并行的计算不同注意力头，最后再拼接到一起。
{{< /qa >}}

{{< qa q="有哪些减少 attention 计算量的方法" >}}
1. FlashAttention：注意力计算的时间复杂度依然是 $O(n^2d)$，但是它通过分块读取和 online softmax 降低了显存访问开销，把显存访问复杂度从 $O(n^2)$ 降到了 $O(nd)$。
2. KVCache：在生成下一个 token 时，只需要计算当前 token 的 query、key、value。之前的 key/value 被缓存，直接拼接使用，避免重复计算整个历史序列的 attention，从而提高**推理阶段**的速度。
3. MQA/GQA：减少 K 和 V 矩阵注意力头，减小计算量。
{{< /qa >}}

### KVCache

![image.png](http://img.xilyfe.top/img/20260608145701282.png)
从图中可以注意到：对于第 $i$ 次循环我们要生成 token $i$，它只需要 $QK^T$ 这个下三角矩阵的最后一行和 $V$ 矩阵。再拆细一点，我们只需要 $Q_i$ 和 $K$ 矩阵相乘得到下三角矩阵最后一行还有 $V$，所以我们只需要缓存 $K$ 和 $V$ 矩阵。


```python
class KVCacheAttention(nn.Module):
    def forward(self, x, cache=None, mask=None):
        # ...
        if cache is not None:
            k = torch.concat([cache["k"], k], dim=2)  # seq_len 维度 concat
            v = torch.concat([cache["v"], v], dim=2)
        
        new_cache = {
            "k": k,
            "v": v
        }

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask[:t_q, :t_k]==0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        output = (attn @ v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.o_proj(output), new_cache
```

>加入 KVCache 之后 `seq_k` 和 `seq_q` 的长度就不一样了，所以需要进行截长。

### Position Embedding

1. Learnable Position Embedding

像 token embedding 一样，用长度为 `dim` 的向量表示 token 的位置信息，无需多言。问题在于长度是固定的，假如训练时 `max_seq=2048` 推理时候 `max_seq=4096`，就会出现位置不存在，所以 Learnable Position Embedding 的问题就是**不能外推**。

2. Sinusoidal Position Embedding

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right),\quad PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

正余弦位置编码的思想也是**给每个位置生成唯一编码**，从公式可以看出在低维度数值变化快频率高，在高度为数值变化低频率低：

```text
低维：sin(1)      sin(2)      sin(3)      sin(5)
高维：sin(0.0001) sin(0.0002) sin(0.0003) sin(0.0004) 
```

通过这种方法就可以给每一个 position 一个独一无二的向量，因为多个频率组合起来，几乎不可能出现两个位置完全一样。它的问题在于**不好体现位置的相对关系**，把 sinusoidal pe 加入 token embedding 之后模型得自己找到 $x_i=e_i+PE_i$ 的相对关系。

3. RoPE

假设我们有 `pos=m` 的词向量 $q_m$ 和 `pos=n` 的词向量 $k_n$，我们分别用旋转矩阵 $R$ 对两个向量进行旋转，然后计算它们的内积：

$$
\begin{align}
<q_m^{rope}, k_n^{rope}>&=<R(m\theta)q_m, R(n\theta)k_n>\\
&=q_m^TR^T(m\theta)R(n\theta)k_n\\
&=q_m^TR(-m\theta)R(n\theta)k_n\tag{证明1.1}\\
&=q_m^TR((n-m)\theta)k_n\tag{证明1.2}
\end{align}
$$

可以发现内积的结果只和两个词向量的相对位置有关， 最终结果 $q_m^TR((n-m)\theta)k_n$ 与 $m$ 和 $n$ 的绝对值无关，只与相对位置 $(m−n)$ 有关。扩展到高维之后可以看下图：

![image.png](http://img.xilyfe.top/img/20260608154720669.png)

### Normalization

早期 BatchNorm 论文认为它通过减少 Internal Covariate Shift 来提升训练效果，但后续研究发现这并不是主要原因。现代观点认为 Normalization 的核心作用是控制激活值和梯度的尺度，使网络各层输入保持在稳定范围内，从而改善优化问题、平滑 Loss Surface、提高训练稳定性，并允许使用更大的学习率。

1. Batch/Layer Normalization

$$
y=\gamma \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta
$$

BatchNorm 和 LayerNorm 的都是**标准的归一化公式**，区别在于 BatchNorm 的均值和方差计算都在 batch 维度，而 LayerNorm 在 hidden dim 上计算均值和方差。

2. RMS Normalization

$$
\text{RMSNorm} \left(\right. \mathbf{x} \left.\right) = \frac{\mathbf{x}}{\sqrt{\frac{1}{H} \sum_{i = 1}^{H} x_{i}^{2} + \epsilon}} \bigodot \gamma + \beta
$$

一方面 RMSNorm 发现去掉均值 $\mu$ 之后影响不大，减少这一步计算可以大幅度节省时间。另一方面是去掉均值可以减少信息损失，让训练更稳定。

{{< qa q="为什么 LLM 不适合用 BatchNorm">}}
- 每条数据的长度不相同，在 batch 维度计算均值和方差不稳定。
- batch 大小不够大，显存有限情况下 `micro_batch` 可能为 1。
- 训练和推理的 `batch_size` 可能不同。
{{< /qa>}}

{{< qa q="PreNorm 和 PostNorm 区别" >}}
对 Post-Norm 求关于 $x_l$​ 的梯度，由链式法则：
$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+1}} \cdot \frac{\partial \text{LN}(x_l + F(x_l))}{\partial x_l}
$$
梯度回传时候必须经过 LayerNorm 的缩放，对于深层的注意力网络，梯度容易消失或者爆炸。
而对 Pre-Norm 求梯度则有：
$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \left(1 + \frac{\partial \sum F}{\partial x_l}\right)
$$
可以看到第一项是**直连梯度**，不经过任何 LayerNorm，梯度**不会消失**。
{{< /qa >}}

{{< qa q="残差连接作用" >}}
假设没有残差连接，深层神经网络里参数的梯度是各层梯度的连乘 $\frac{\partial L}{\partial x_1}=\prod_{i=1}^{100}\frac{\partial F_i}{\partial x_i}$，容易出现梯度消失或者梯度爆炸。而对残差神经网络 $y=x+F(x)$ 求导得到 $\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}$，这个 $I$ 是恒等映射。
{{< /qa >}}


{{< qa q="梯度裁剪" >}}
LLM 里面的梯度裁剪一般是按照 **L2 范数** 进行裁剪，从而**解决梯度爆炸的问题**。
假设梯度向量为 $g=(g_1,g_2,\dots,g_n)$，我们计算得到 L2 范数：$\|g\|=\sqrt{\sum_i g_i^2}$。假设 L2 范数超过了设定的阈值 $\|g\| > c$ 那么就对他进行裁剪。

$$
\begin{array}{c} g'=
\begin{cases}
g, & \|g\|\le c\\
g\frac{c}{\|g\|}, & \|g\|>c
\end{cases} \end{array}
$$

```python
loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()
```
{{< /qa >}}


{{< qa q="学习率调整" >}}
学习率调整的核心是**前后期稳定，中期探索**。
1. 初期模型的参数随机，太大学习率可能导致梯度爆炸，所以一般用 warmup 让学习率逐步提高。
2. 后期模型已经趋于稳定，所以不希望太大学习率导致先前学习的内容被覆盖，可以采用余弦退火算法。
3. 一般 sft 预训练时候的学习率相对较大，设置在 2e-5 这样可以快速收敛。在后训练或者对某个方向微调时候，学习率一般比较小在 1e-6 或者更小，避免灾难性遗忘。
{{< /qa >}}

{{< qa q="为什么 LLM 里很少用 dropout" >}}
1. LLM 本身参数量就很大，基本不会出现过拟合的问题。
2. dropout 会破坏数据的一致性，影响模型训练的稳定性。
{{< /qa >}}

{{< qa q="激活函数的选择" >}}
1. ReLU 是硬阈值，如果某个神经元长时间 $x<0$ 没有得到梯度，那么它基本废掉了。
2. GeLU 是平滑的 ReLU，训练更稳定一些，但是计算了比 ReLU 大。
3. SwiGLU 是门控版的 GeLU，它可以控制哪些特征保留，哪些特征抑制。SwiGLU 的非线性表达能力最强，同时可以提高表达能力。
{{< /qa >}}


## LoRA

$$
W_{update} = W + \Delta W
$$

LoRA 微调的思路是不在原矩阵 $W$ 上进行参数更新，而是用矩阵加法训练一个新的参数矩阵 $\Delta W$。LoRA 牛逼之处在于 $\Delta{W}$ 不需要和 $W$ 一样的参数量，他可以被拆为两个低秩矩阵的乘积:

$$
\Delta W = A \cdot B
$$

假设原矩阵 $W \in \mathbb{R}^{d \times d}$，低秩矩阵为 $A \in \mathbb{R}^{d \times r}$ 和 $B \in \mathbb{R}^{r \times d}$，那么训练参数量由 $d\times d$ 变成了 $2 \times d \times r$。

```python
class LoRA(nn.Module):

	def __init__(self, in_feature: int, out_feature: int, rank: int, alpha: int):
		super(LoRA, self).__init__()
		self.rank = rank
		self.alpha = alpha
		
		self.A = nn.Linear(in_feature, rank, bias=False)
		self.B = nn.Linear(rank, out_feature, bias=False)

	def forward(self, x):
		return self.B(self.A(x)) * self.alpha / self.rank
```

{{< qa q="LoRA 加在哪些模块上" >}}
最初的 LoRA 论文主要作用于 attention 的 $W_q$ 和 $W_k$。但后来的实践表明，将 LoRA 应用于所有的线性层，包括 MLP，通常能获得更好的效果，尽管这会增加一些参数量。
{{< /qa >}}

{{< qa q="rank 和 alpha 怎么选" >}}
1. $r$ 越大，增量权重的表示能力越强，可以拟合更多特定任务。
2. $\alpha$ 是 **LoRA scaling**，控制增量权重在 forward 中的幅度。
3. 一般小参数模型的 rank 在 8\~16，然后控制缩放系数 $\alpha / r$ 在 1\~4 的范围，视任务而定。
{{< /qa >}}

{{< qa q="低秩矩阵矩阵初始化" >}}
一般是对 $A$ 矩阵应用 kaiming 初始化，对 $B$ 矩阵置为 0：
1. 首先 $A$ 和 $B$ 矩阵需要至少有一个为零矩阵，这样 LoRA 一开始更新时 $\Delta W = B A$ 接近于零矩阵，就不会破坏预训练权重。
2. 其次 $A$ 和 $B$ 矩阵不能全部为零矩阵，要不然计算出来它们的梯度都是 0，没法更新参数
3. 而在前向传播中，低秩更新实际走的路径是：x → A → (scale) → B，也就是说反向传播时是从矩阵 $B$ 到矩阵 $A$。假如矩阵 $A$ 为零矩阵，那么矩阵 $B$ 的梯度为 0，训练就会先更新矩阵 $A$，$A$ 更新的数值尺度就会收到 $B$ 的初始化分布影响，容易放大早期更新的尺度。如果初始化矩阵 $B$ 为零矩阵，那么会先更新矩阵 $B$，把 $B$ 从零矩阵拉开，再更新 $A$，训练更稳定。
{{< /qa >}}

{{< qa q="为什么不用 SVD 进行矩阵分解" >}}
SVD 分解是对确定的矩阵进行分解。而在微调开始时，我们并不知道目标更新矩阵 $\Delta{W}$ 是什么。LoRA 的做法是直接定义两个低秩矩阵 $A$ 和 $B$ 作为可学习参数，通过梯度下降让模型自己去寻找最优的低秩空间，而不是对已知矩阵做数学分解。
{{< /qa >}}

{{< qa q="LoRA在推理阶段会增加延迟吗" >}}
在推理前可以通过重参数化，将训练好的低秩矩阵 合并到原权重中 $W_{merged}=W+\Delta{W}$。推理时只需要使用 $W_{merged}$ 进行计算，网络结构和参数量与原模型完全一致。
{{< /qa >}}

{{< qa q="LoRA 的局限性" >}}
LoRA 没法同时满足多任务并发和零推理延迟。如果希望零推理延迟，那么就必须把地址矩阵 $\Delta{W}$ 的权重合并到原权重上。如果希望多任务并发，就得把 $W$ 和 $A$、$B$ 权重分离，动态的选择加载哪个 LoRA。
{{< /qa >}}

## SFT

### how

sft 就是让模型在收集的 trajectory 上做 off-policy 的 teacher-forcing 训练。我们把 `labels[:, :t-1]` 喂给模型让它进行 next token prediction，然后用预测的 token 和 `labels[:, t]` 计算交叉熵损失以此优化模型。

>具体来说，我们把 `labels[:, :-1]` 喂给模型，由于因果掩码的存在，模型预测的 `logits`（形状为 \[B, T-1]） 里面每个 $\text{token}_t$ 都是基于前 $t-1$ 个 token，也就实现了 teacher-forcing，然后我们计算 `logits` 和 `labels[:, 1:]` 的交叉熵损失就好了。

```python
def sft_loss(logits, labels, ignore_index=-100):
	assert logits.dim() == 3
	assert labels.dim() == 2
	
	loss = F.cross_entropy(
		logits.view(-1, logits.size(-1)),
		labels.view(),
		ignore_index=ignore_index,
		reduction="mean"
	)
	return loss
```

{{< admonition type=warning title="">}} 
这里的 `logits` 和 `labels` 应该是都已经移位过的，`logits[:, :-1]` 对应 `labels[:, 1:]`。
{{< /admonition >}}

写到这里突然想起来一个很重要的问题，sft 和 pretrain 的一个很重要的差异就是：pretrain 是在大量数据上学习**整个文本**，而 sft 是在学习**该如何回复 prompt**，所以训练时候需要：
1. sft 需要给 prompt+response 应用 chat_template，而 pretrain 只需要把 text encode 就好了。
2. sft 需要把非 assistant 的部分 mask 掉，避免模型学 prompt 本身的分布。

```python
def only_assistant(input_ids, max_prompt_len, ignore_index=-100):
	labels = input_ids[:, 1:]
	labels[:, :max_prompt_len] = ignore_index
	return labels
```

### tricks

[LLM训练-sft](https://zhuanlan.zhihu.com/p/809229182) 这篇文章里面提到的一个很重要的技巧就是：**对不同 task_type 和 special token 分别观察 channel_loss**。

```python
def compute_loss(logits, labels, task_ids):
	per_token_loss = F.cross_entropy(
	    logits.view(-1, vocab_size),
	    labels.view(-1),
	    ignore_index=-100,
	    reduction='none'
	).view(B, T)  # (B, T)
	
	# 按 task_type 分组
	for task in unique_tasks:
	    task_mask = (task_ids == task).unsqueeze(1)  # (B, 1) broadcast to (B, T)
	    valid_mask = (labels != -100) & task_mask    # (B, T)
	    
	    task_loss = per_token_loss[valid_mask].mean()
	    log(f"loss/{task}", task_loss)
	
	# 训练用的总 loss（所有 task 合并）
	total_loss = per_token_loss[labels != -100].mean()
```

我们通过 batch 内数据的 task_type 生成掩码，就可以计算特定 task_type 数据的 loss。然后根据 loss 我们就知道不同 task_type 数据的拟合情况，如果过拟合了我们就需要删减这个 task_type 的数据，或者增加其他类型数据，如果欠拟合就需要扩大这类数据量。

special_token 同理，我们可以通过类似方法观察 special_token 的 loss 变化，正常来说在 sft 初期 special_token 的 loss 是比较高的，因为这些 token 在 pretrain 时候没有见过。

```python
SPECIAL_TOKEN_IDS = {tokenizer.convert_tokens_to_ids(t) 
                     for t in ["<think>", "</think>"]}

# special token 位置 mask
special_mask = torch.zeros_like(labels, dtype=torch.bool)
for tok_id in SPECIAL_TOKEN_IDS:
    special_mask |= (input_ids == tok_id)

# 只在 response 范围内且是 special token
special_valid = special_mask & (labels != -100)
normal_valid = ~special_mask & (labels != -100)

special_token_loss = per_token_loss[special_valid].mean()
normal_loss = per_token_loss[normal_valid].mean()

log("loss/special_tokens", special_token_loss)
log("loss/normal_tokens", normal_loss)
```

---

{{< qa q="各种各样的 mask" >}}
1. attention mask 负责的只是 PAD token，在计算注意力的时候不注意到 PAD token。
2. causal mask 负责的是因果掩码，让每个 token 不注意到之后的 token，应用时候需要和 attention mask 取 & 然后 apply。
3. loss mask 控制的是哪些 token 应该参与计算损失，比如 PAD token，sft 中 prompt 部分的 token，或者 agentic rollout 时候检索的部分。
{{< /qa >}}

{{< qa q="left or right padding" >}}
首先结论是：训练阶段倾向 right padding，推理阶段倾向 left padding。
1. 
{{< /qa >}}

{{< qa q="多轮对话微调" >}}
- 只训练最后一轮 assistant 会导致**上下文利用变差：模型更依赖最后一轮信息**。
- 对每一轮 assistant 都训练会导致**回答更顺但更爱跑题，还会把旧回答当输出习惯**。

解决方案有两种：
1. 把多轮对话拆为多个样本进行训练，然后每个样本只训练最后一轮 assistant，并且要把多个样本分散在不同 batch。
2. 对每一轮 assistant 都训练，但是加权。
{{< /qa >}}

{{< qa q="同一个 batch 里面长度差异很大怎么办" >}}
解决办法是**按长度分桶**，先 sort by length 然后把 shuffle batch 再训练，可以减少无效 padding。
{{< /qa >}}

{{< qa q="sft loss 低但是对话效果差" >}}
很可能是 chat template 或者 loss mask 的问题，模型没有学习到该回复什么内容。
{{< /qa >}}

{{< qa q="packing or padding" >}}
在去年的一篇文章里我详细介绍了 packing 和 padding 的区别，简单来说 packing 就是把多条文本拼到一个 sequence 里，padding 就是一条数据一个 sequence，很明显 packing 可以减少 pad token 带来的无意义计算量，那该怎么选择的？

结论是不管是 pt 还是 sft 都可以用 packing 策略来对齐，需要注意的是：
1. 不同数据放在同一个 sequence 里面，如果用传统的 causal_mask，seq_2 可以注意到 seq_1 的 token。**需要用分块的 causal mask 和用 eos token 分割不同句子**。
2. 如果 packing 导致一个 sample 被截断，那么在下一个 block 的后半部分 sample 计算 attention 时候就看不到上文信息了。
{{< /qa >}}

{{< qa q="sft loss 持续上升可能是什么原因" >}}
next-token prediction 本质是"背书"——即使数据是乱码，loss 也应持平或缓慢下降，持续上升说明存在以下问题：
1. 训练代码 bug：梯度反传、优化器、学习率调度等逻辑错误
2. 数据格式错误：labels shift 错位、chat template 有问题、eos 错位、**prompt 没有 mask**、packing mask 错误
3. 学习率过大：loss 震荡或发散，需降低 lr 或检查 warmup
{{< /qa >}}

{{< qa q="初始 loss 的范围、loss 过高或过低" >}}
- 小模型先验知识较少，初始的 loss 相对更大，在 1.8~2.5。大模型一般在 1~1.8。
- loss 过高说明数据太难没有相关知识，ntp 预测的很随机。loss 过低说明没有新学习的知识，和 pretrain 的分布接近。
- 对于不同领域数据 sft 的 loss 范围也不同，开放性问题相较于检索性问题 loss 就更高。
{{< /qa >}}

{{< qa q="sft 到什么程度可以 rl" >}}
最简单的途径，你 sft 模型测一下 pass@k 的指标，取 k 条最大值，如果能比 pass@1 的指标高很多，就值得做RL。pass@k 明显大于 pass@1 说明模型能回答出来，但是需要多次尝试有概率，做 rl 可以优化模型的参数分布。
{{< /qa >}}

{{< qa q="怎么评估 sft 效果" >}}
sft 的评估是需要看经典的 3H 原则的：**Helpfulness、Honesty、Harmlessness**。当然，实际工作的评估中，倒也不必完全是按照这三个原则进行评估，可以按需求制定自己模型的指标：是否 follow 指令，是否 system 穿透，是否内容准确，是否产生幻觉，是否安全……等等等等。
{{< /qa >}}

{{< qa q="special token 的 loss 行为" >}}
`<|im_start|>` 等 special token 的 loss 应该先高后快速下降。这些 special token 是 sft 后 chat template 带来的，一开始模型并不熟悉所以预测这些 token 的概率非常低，但随着训练模型就知道了 `<|im_start|>user` 后面跟的是 prompt， `<|im_start|>assistant` 后面跟的是 response。 

假如 special token 的 loss 下降很慢可能是：
1. special token 设置有问题，tokenizer 没有把 special token 当成单个 token
2. chat template 的问题
{{< /qa >}}

{{< qa q="sft 训练策略" >}}
1. **多任务学习**：直接混合不同数据源，简单有效，但无法针对专业任务精细控制
2. **顺序训练**：依次在每个数据集上 SFT，灵活，但容易发生灾难性遗忘
3. **混合序列训练**：先在专业数据集上多任务学习，再在通用数据集上 SFT，兼顾专业与通用
4. **双阶段混合微调 DMT**：第一阶段专业数据 SFT，第二阶段混合少量专业数据防止遗忘，综合效果最佳
5. **渐进式混合**：初期 100% 专业 → 中期 50/50 → 后期 10% 专业+90% 通用，像课程学习逐步过渡
{{< /qa >}}

{{< qa q="混合加训和 cpt 怎么选择" >}}
1. 混合加训优点是**保留原始能力+均衡的吸收新知识**，缺点是混合了通用和专业数据集，**训练数据量大**。
2. cpt 优点是**训练速度快，显著偏向新数据**，缺点是**容易出现灾难性遗忘**。

其实都选混合加训。
{{< /qa >}}

{{< qa q="sft 过拟合怎么办" >}}
sft 的过拟合并不像传统深度学习一样，通过调整训练 epoch、学习率、dropout、weight_decay 来解决。因为大概率模型只是某项能力局部过拟合了，大部分能力都是正常的，盲目调整超参数反倒会让模型整体上欠拟合。具体地，在确定模型并没有全局过拟合之后（如果是全局过拟合，模型整体的效果应该都很差劲，那就通过炼丹来解决，这里不赘述了），我们主要的解决方案是**通过优化训来数据来缓解过拟合，主要措施是删减对应 task_type 的数据，或是扩充该 task_type 的数据多样性**。过拟合的难点是让模型暴露出来它到底对什么过拟合了，好让我们去 grep 对应的训练数据来做修改。通常，我们观察到模型过拟合是因为它回答错了某个知识，而且是非常蠢的错误：比如日本的首都是北京。
{{< /qa >}}

## RLHF

### rl 比 sft 好在哪

### 重要性采样

### KL divergence

### Policy Gradient

### 蒙特卡洛估计

### 广义优势估计

### TD Error

### PPO

### DPO

### GRPO

### DAPO

### GSPO

### agentic rl 的 credit assignment

### RLHF 训练的指标

>如何判断 early stop

### 熵崩塌

### RL数据和SFT数据需要有重合吗？

## 分布式训练

### Data Parallel

### Tensor Parallel

### Deepspeed Zero


## 训练

### 参数量计算

### 显存计算

### 训练出现 NaN 的原因

>lr,除0，log，clamp

