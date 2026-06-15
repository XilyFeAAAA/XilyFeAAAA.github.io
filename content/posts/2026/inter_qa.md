---
title: Interview Q&A
date: 2026-06-02T17:55:39+08:00
featuredImage: http://img.xilyfe.top/img/20260609104852216.png
authors:
  - Xilyfe
series:
  - 面经
tags: []
lastmod: 2026-06-15T12:06:19+08:00
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
首先结论是：训练阶段无所谓，推理阶段倾向 left padding。在 inference 的时候 next token prediction 会取 `logits` 的最后一个 token，也就是 `next_token_logits = outputs.logits[:, -1, :]`。假如我们进行 right padding，那么模型 generate 的第一个 token 就是取 PAD token 对应的 `logits` 向量。问题在于：PAD token 的 embedding 是随机初始化的，模型从来没有学过"PAD 位置之后应该生成什么"，所以这时候生成的 next token 是随机无意义的，就会导致接下去生成的 token 都出现问题。
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

{{< qa q="rl 和 sft 区别" >}}
{{< /qa >}}

{{< qa q="credit assignment" >}}
{{< /qa >}}

{{< qa q="rl 数据和 sft 数据需要有重合吗" >}}
{{< /qa >}}

### Monte Carlo

### TD Error

### GAE

### KL Divergence

f-散度的提出是为了解决 **两个分布到底有多么不同** 这样一个问题，由于分布是曲线没法相减，所以需要一种把"两条曲线的差异"压缩成一个数的方法，这就是**散度**。f-散度的核心思路：在每个点 x 处，看 P 和 Q 的密度之比 $r(x)=\frac{p(x)}{q(x)}$。如果 $r(x) = 1$ 处处成立，两个分布完全一样散度应该是 0。如果 $r$ 偏离 1，说明有差异应该被惩罚。用一个**凸函数** $f(r)$ 来做这个惩罚，然后对全空间积分（以 Q 为权重）。选不同的 $f$，就得到不同的散度：

$$
D_f(P \| Q) = \int q(x) f\left( \frac{p(x)}{q(x)} \right) dx
$$

我们在大模型训练中常见的 KL 散度就是 $f(r)=r\log r$ 形状的 f-散度：

$$
D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)}  dx
$$

KL 散度理论上是用概率分布的积分定义的，但现实中我们只有**有限的样本数据**，没法精确计算，所以需要用样本来**近似估算**——这个近似方法就叫"估计器"。

>一个优秀的估计器通常需要考量两个核心指标：
>1. **偏置**：估计器的数学期望是否等于真实值？如果相等，就是**无偏估计**；如果不等，就是**有偏估计**。  
>2. **方差**：不同批次的样本算出来的估计值，上下波动大不大？方差太大会导致强化学习的梯度震荡，训练崩盘。

#### 三种估计器

{{< admonition type=info title="">}} 
在推导之前证明一个大前提：对于任何从 $q(x)$ 中采样的比率 $r = \frac{p(x)}{q(x)}$，它的数学期望永远为 1。

首先，我们要明确**数学期望的定义**。对于任何从分布 $q(x)$ 中采样出来的随机变量 $f(x)$，它的数学期望就是把所有可能的 $x$ 对应的函数值 $f(x)$，乘以它在 $q(x)$ 中的概率密度，然后全空间积分：

$$\mathbb{E}_{x \sim q}[f(x)] = \int q(x) \cdot f(x) \, dx$$

现在，我们把 $f(x) = r = \frac{p(x)}{q(x)}$ 代入这个定义公式中：

$$\mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} \right] = \int q(x) \cdot \frac{p(x)}{q(x)} \, dx$$

注意到积分符号内部的 $q(x)$ 了吗？分子和分母上各有一个 $q(x)$，它们可以**直接约掉（消去）**：

$$= \int p(x) \, dx$$

根据概率论的基本公理，**任何一个合法的概率密度函数，它在全空间的积分（总概率）必须严格等于 1**。因为 $p(x)$ 是一个合法的概率分布，所以：

$$\int p(x) \, dx = 1$$

{{< /admonition >}}

k1 估计器是最直接的推导，既然我们要计算 $\mathbb{E}_{x \sim q} [-\log r]$，那么直接脱掉期望符号，用单样本的函数值作为估计：

$$
k_1 = \log \frac{q(x)}{p(x)} = -\log \frac{p(x)}{q(x)} = -\log r
$$

虽然 $\mathbb{E}[k_1]$ 严格等于真实 KL（**绝对无偏**），但是它的**方差极大**。因为当某个样本下 $q(x) < p(x)$ 时，$k_1$ 会变成负数。尽管理论上整体 KL 散度永远 $\ge 0$，但 $k_1$ 单个样本却频繁在正负之间剧烈摆动，这在代码里做强化学习策略裁剪（Clip）或加惩罚项时，会带来巨大的不稳定因素。

---

在强化学习中，由于我们往往会限制新旧策略不能离得太远，因此可以假设 $p \approx q$，这意味着比率 $r \approx 1$。 我们可以利用泰勒展开，在 $r = 1$ 处对函数进行逼近：

$$
-\log r \approx f(1) + f'(1)(r-1) + \frac{1}{2}f''(1)(r-1)^2 = -(r-1) + \frac{1}{2}(r-1)^2
$$

接着，我们在期望意义下看待这个公式。因为上面提到了 $\mathbb{E}_{x \sim q}[r - 1] = 0$，所以线性项 $-(r-1)$ 在求期望时直接归零了。因此：

$$
\mathbb{E}[-\log r] \approx \mathbb{E}\left[ \frac{1}{2}(r-1)^2 \right]
$$

同时，我们知道当 $r \approx 1$ 时，由一阶展开可知 $\log r \approx r - 1$。我们将这个关系代入上式，用 $(\log r)^2$ 替换掉 $(r-1)^2$，就得到了 $k_2$：

$$
k_2 = \frac{1}{2}(\log r)^2 = \frac{1}{2}\left(\log \frac{p(x)}{q(x)}\right)^2
$$

因为带有平方，**$k_2 \ge 0$ 恒成立**，完美避开了 $k_1$ 产生负数导致的剧烈摆动，方差极小。但它是截断泰勒展开的产物，所以是**有偏估计**。

---

k3 估计器的思路是设计一个估计器，**既像 $k_1$ 一样严格无偏，又像 $k_2$ 一样恒大于 0 且方差极小？** 为此，他引入了统计学中大名鼎鼎的**控制变量技术**：在无偏估计器 $k_1$ 上，加上一个**期望值严格为 0 的项**，利用它们之间的负相关性来抵消波动。前面我们已经证明了 $\mathbb{E}_{x \sim q}[r - 1] = 0$。那么我们直接把这一项无条件加到 $k_1$ 里面去：

$$
k_3 = k_1 + (r - 1) = -\log r + r - 1 = r - 1 - \log r
$$

![image.png](http://img.xilyfe.top/img/20260614194511569.png)

k3 的方差问题根源在于 $r-1$ 这一项：当 $r=\pi_{\theta}/\pi_{ref}$ 很大时（即训练策略对某个 token 分配远高于参考模型的概率)，$r一1$ 按 $r 线性增长，而 $k1=-\log r$ 只是对数增长。结果就是：KL 大时，k3 的方差比 k1 高出几个数量级。

#### 前向/反向 KL 散度

KL 散度是不对称，$D_{K L} \left(\right. P \parallel Q \left.\right) \neq D_{K L} \left(\right. Q \parallel P \left.\right)$，所以"哪个在前哪个在后"非常重要。$D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)}  dx$ 从 KL 散度公式可以观察到，传统 KL 散度也就是 Forward KL 是从分布 $p(x)$ 里面采样，而 $D_{KL}(Q \| P)$  称作 Reverse KL 反向 KL 散度，他从分布 $q(x)$ 采样。

1. 从分布 $p(x)$ 中采样意味着：我们需要能获得 $p(x)$ 的数据。所以我们在做 OPD 时候必须反向 KL 散度，OPD 规定了 trajectory 必须从学生模型 $q(x)$ 采样。
2. FKL 偏向于把概率质量"摊开"，覆盖两个峰之间的低概率谷地，生成的内容是所有 teacher 模式的**模糊平均**。
3. RKL 中 student 只需要找到 teacher 的**某一个高概率模式**，集中概率质量进去。

#### 惩罚系数

DAPO、VAPO、MiniMax CISPO 主张完全去掉 KL 散度项，原因是对于这些关注 reasoning RL 的工作：
- 奖励目标本身就是远离 SFT 分布，模型要学会"反思"、"aha moment"，分布必然大幅漂移，KL会阻碍学习。
- RL有可验证 reward(比如 rule-based、math/code verifier)，Reward Hacking 风险小，没有必要用 KL 散度。
- 资源上节约 reference model 显存和 forward 计算，训练效率提升明显。

DeepSeek GRPO/Kimi/GLM 保留 KL 散度项，原因是对于基座模型来说，统一 RL stage 里混了 alignment 和 general task 多种任务，很多都是经典 RLHF 里 reward hacking 的高触发场景，KL 能必要的防护。但是这些工作都在 KL 上进行了精细化：
- DeepSeek V3.2为例子，进行了以下几个调整：
	- 不同领域适用不同 KL 系数（per-domain）：数学场景（Reasoning主导）系数接近 0，通用对齐保留系数。
	- 修正KL估计器。
- Kimi K1.5/K2 也使用了 KL 强度动态调整。

### Importance Sampling

重要性采样 IS 的核心思想是用一个分布 $q \left(\right. x \left.\right)$ 采样的数据去估计另一个分布 $p \left(\right. x \left.\right)$ 下的期望，只需要乘以一个修正比率：

$$
\begin{align*}
\mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_{x \sim q}\left[ f(x) \cdot \frac{p(x)}{q(x)} \right]
\end{align*}
$$

- $p \left(\right. x \left.\right)$ 是真正想估计期望的分布（目标策略）
- $q \left(\right. x \left.\right)$ 是实际用来采样的分布（行为策略）
- $\frac{p \left(\right. r \left.\right)}{q \left(\right. x \left.\right)}$ 就是重要性比率



### PPO

![image.png](http://img.xilyfe.top/img/20260611223048584.png)

{{< qa q="PPO 公式是怎么得到的" >}}
我们在强化学习里的终极目标，是让动作带来的**期望回报最高**，也就是说我们希望最大化：

$$
J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t \gamma^t r_t\right]
$$

**策略梯度定理**告诉我们，这个目标函数的梯度可以写成:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{(s,a)\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a|s)\cdot Q^{\pi_\theta}(s,a)\right]
$$

为了减小估计的方差，在实践中我们通常用优势函数 $A^{\pi_\theta}(s,a)$ 代替状态动作价值函数 $Q(s,a)$，这不会改变梯度的期望值。于是梯度写为

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s,a) \right]
$$

注意上式的期望是在**当前策略 $\pi_\theta$** 下取的，advantage 也是 $\pi_\theta$​ 下的 $A^{\pi_\theta}$​。但我们实际拥有的 rollout 数据是用  $\pi_{old}$​ 采样、用 $\pi_{old}$​ 算出来的 $A^{\pi_{old}}$​，所以需要做重要性采样把分布换到 $\pi_{old}$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{a\sim\pi_{old}}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}\nabla_\theta \log\pi_\theta(a|s)\cdot A(s,a)\right]
$$

**这里是一个近似而非严格等式**——只有在 $\pi_\theta$ 与 $\pi_{old}$​ 足够接近时，用 $\pi_{old}$ 下采样的轨迹和 $A^{\pi_{old}}$​ 去估计 $\pi_\theta$ 下的真实梯度才是可靠的。这一点很重要,它正是 PPO 后续引入 clip 机制的根本原因:clip 通过限制 $\pi_\theta/\pi_{old}$ 的比值范围,把更新约束在这个近似成立的"信任区域"内,防止单次更新让 $\pi_\theta$​ 跑得离 $\pi_{old}$ 太远导致上式近似失效。

那么我们找到一个函数 $L(\theta)$，只要使得 $\nabla_\theta L(\theta)$ 恰好等于上面这个式子，对 $L(\theta)$ 进行梯度上升（或者说对 $-L(\theta)$ 进行梯度下降）就等价于对 $J(\theta)$ 进行梯度上升，让 $J(\theta)$ 变大，也就是我们强化学习的优化目标：

$$
L(\theta) = \mathbb{E}_{a\sim\pi_{old}}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A(s,a)\right]
$$

这就是 PPO 的 surrogate loss，再加上 clip 操作来约束信任区间，就构成了完整的 PPO 目标函数。
{{< /qa >}}

{{< qa q="公式里面 min 和 clip 的组合" >}}
min+clip 的组合是起到了一个**熔断机制**，我们已经可以通过 clip 限制单次更新的幅度了，但是万一策略的更新幅度还是太大了，我们需要停止策略的参数更新。观察公式，假如优势 advantage 大于 0，若 $r_t>1+\epsilon$，那么最小值函数会取右边被 clip 的部分，此时 loss 中就只剩常量了不产生任何梯度则停止参数更新，同理优势小于 0 且 $r_t<1-\epsilon$ 也是。那为什么我们不用管 Adv 大于 0 且 r 小于 0.8 的情况？或者 Adv 小于 0 且 r 大于 1.2 的情况？Adv 大于 0 的情况说明当前策略是好的，如果 r 小于 0.8 说明：这个策略是好的，旧模型偏向这个策略，但是新模型不怎么偏向这个策略了，那我们肯定希望能尽可能朝现在这个方向来更新参数，强化新策略做出这个选择的概率。
{{< /qa >}}

{{< qa q="PPO 和 DPO 对 reward 的要求有什么不同" >}}
- PPO 的 reward 要能对每个 token 或每个 response 给出相对精确的分值，用于计算 advantage。对 reward 的绝对值和方差都比较敏感。
- DPO 的 reward 只需要能区分好坏（排序能力），是 point-wise 打分后做比较，不需要特别精确的绝对值，容忍度更高。
{{< /qa >}}

{{< qa q="critic model 重要吗" >}}
- 在 PPO 训练中，critic model 用来估计 baseline，计算advantage = reward - value，减少policy gradient的方差，训练更稳定。没有好的critic，PPO的训练信号噪声很大。
- GRPO 等方法用组内 reward 均值做 baseline，避免了单独训练critic的成本，同时在reasoning任务上效果接近甚至更好。
{{< /qa >}}

{{< qa q="PPO 训练的指标" >}}
1. KL 散度：太大可能存在 reward hacking 风险，太小可能没有充分更新
2. policy loss/critic loss
3. entropy：出现 collapse 多样性消失，reward hacking
{{< /qa >}}

{{< qa q="Reward Model 训练的指标" >}}
1. auc：chosen > rejected 的 排序准确率
2. chosen 和 rejected 的 reward margin：margin 大说明很自信
3. ood：RM 在没见过的分布上的表现
4. reward distribution：reward 方差是不是太大或者太小，太大容易训练不稳定，太小缺乏梯度信号。
{{< /qa >}}

{{< qa q="Reward model 训练时候碰到的问题" >}}
1. 标注数据不足与偏差：如果偏好数据主要来自单一群体或话题，模型在其他领域的表现会较差，甚至带有该群体的主观偏见。奖励模型可能过度偏好训练集中常见的回答风格（如过度详细或倾向某种语气）。例如医疗助手训练后，reward model 对长回复都评为高分，导致 PPO 之后容易长篇大论。解决方案是：**扩充多样化数据**、**在训练时加入字数正则项**。
2. 过拟合泛化能力差：过拟合往往由数据匮乏和模型容量过高共同导致。大模型微调出的奖励模型有能力记忆训练集中偏好对比的细节，当标注数据有限或包含噪声时，模型可能学习到伪相关特征（如特定词频、长度等）作为判断依据，削弱了真正偏好信号的泛化。解决方案是：**正则化**、**根据验证集曲线早停**、**缩小模型参数**。
3. reward hacking：解决方案是 **设计对抗样本加入训练**、**KL 散度限制模型差异**，**设计针对性的正则项例如长度正则**。
4. 正负样本 margin 小：缺乏 hard negative sample。
5. 分布偏移：RLHF 模型在奖励模型的打分中表现极佳，但人工质检觉得输出空洞或跑偏，未真正提升体验。解决方案是：**加大 KL 损失权重**、**定期迭代映入新样本修正偏差**、**拆分多个目标奖励**。例如客服模型很礼貌但是专业性很差，把回复正确率和礼物拆为两个 reward model 加权组合。
{{< /qa >}}

{{< qa q="PPO 是 off-policy 还是 on-policy，有什么区别" >}}
PPO 理论上是 on-policy，每次训练的轨迹是从模型自身采样的。但是在实践工程上为了提高数据利用率（`ppo_epochs`）或者受限于显存需要梯度累计，就会导致变成 off-policy，数据是从前几个版本的模型上 rollout 的，这就需要 PPO 公式里面的 **重要性采样** 来修正。
{{< /qa >}}

{{< qa q="为什么用 actor-critic 而不是纯 critic" >}}
Actor-Critic 的核心原因是：Critic 只能评估状态或动作的好坏（V/Q），但无法直接生成可学习的策略更新方向；而Actor负责输出可微的策略分布 π(a|s)，将 Critic 提供的优势信号 $A(s,a)=Q−V$ 转化为参数更新的梯度 $\nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s,a)$，从而实现“评价→改进”的闭环。仅有Critic 在高维或连续动作空间中会面临 argmax 困难、不可微以及无法高效表示策略分布的问题，因此需要 Actor 来承载策略表示，使 Critic 的评分能够转化为稳定可优化的参数更新方向。
{{< /qa >}}

### DPO

![image.png](http://img.xilyfe.top/img/20260611223633880.png)

{{< qa q="DPO 的 chosen 和 reject 的 loss 同时下降是因为什么" >}}
{{< /qa >}}

{{< qa q="DPO 训练为什么会导致输出变长" >}}
{{< /qa >}}

### GRPO

{{< qa q="不同 RL 场景怎么设计 reward" >}}
{{< /qa >}}

{{< qa q="GRPO 的优势为什么要减 baseline，一定要除 std 吗" >}}
{{< /qa >}}

{{< qa q="GRPO 为什么加上 KL 散度，KL 散度怎么计算，为什么 DAPO、GSPO 又去掉了KL散度？" >}}
{{< /qa >}}



### 熵崩塌


## 分布式训练

### Data Parallel

### Tensor Parallel

### Deepspeed Zero


## 训练

### 参数量计算

### 显存计算

### 训练出现 NaN 的原因

>lr,除0，log，clamp

