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

### Q&As


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

{{< qa q="在计算 attention 时有什么节省显存资源的策略吗？" >}}
1. 混合精度 BF16/FP16
2. Sparse Attention：限制每个 token 只关注部分 token
3. Flash Attention：前向传播时候没有保存中间激活值，在反向传播时候重新计算
{{< /qa >}}

{{< qa q="有哪些减少 attention 计算量的方法" >}}
1. FlashAttention：注意力计算的时间复杂度依然是 $O(n^2d)$，但是它通过分块读取和 online softmax 降低了显存访问开销，把显存访问复杂度从 $O(n^2)$ 降到了 $O(nd)$。
2. KVCache：在生成下一个 token 时，只需要计算当前 token 的 query、key、value。之前的 key/value 被缓存，直接拼接使用，避免重复计算整个历史序列的 attention，从而提高**推理阶段**的速度。
3. MQA/GQA：减少 K 和 V 矩阵注意力头，减小计算量。
{{< /qa >}}


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

{{< qa q="梯度检查点" >}}

![image.png](http://img.xilyfe.top/img/20260629194509255.png)

如图可以发现，**反向传播必须用到前向传播中产生的中间激活值**。梯度检查点的核心思路是 **以计算换显存**：前向传播时只保留少量"检查点"激活，其余激活用完即丢。反向传播需要某个激活时，从最近的检查点**重新跑一次局部前向**来复原它。
{{< /qa >}}


## LoRA

### 原理

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

### Q&As

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

### 原理

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


### Q&As

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
### Monte Carlo

>LLM 的 RLHF 中我们需要 critic model 来进行状态价值 $V(S_t)$ 的预测，但是无法知道真实值。Monte Carlo、TD Error 和 GAE 采用不同方法估计一个策略的长期收益也就是 $V_t$，它们各自在**偏差和方差做了不同的权衡**。

Monte Carlo 想要估计的是状态价值 $V(S_t)$，它用 $G_t$ 来直接作为状态价值 $V(S_t)$ 的目标值：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

状态价值 $V(S_t)$ 是 $t$ 时刻的状态价值是后续状态奖励的折扣求和。

- 它的优点就是无偏差，因为 $G_t$ 是完全使用真实发生的奖励算出来的，没有包含任何主观的猜想，所以它的期望值完全等于真正的价值。
- 缺点是它的方差很大，每一步的随机性会随着时间步累乘，这个在强化学习训练中很忌讳。其次就是必须等整个序列采样结束才能计算。

### TD Error

TD Error 的思路是：当你从 $S_t$ 走到 $S_{t+1}$ 时，对未来收益的预估相对于上一时刻更近了一步，因为你知道了 $S_{t+1}$ 时刻的实际奖励。TD Error 用 TD Target 当做状态价值函数 $V(S_t)$ 的真实值：

$$
TD_t = R_{t+1} + \gamma V(S_{t+1})
$$

然后 TD Error 也就是 TD 误差即为真实值与预测值的差距：

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

- 优点是低方差， 每一小步就更新一次，只包含了一步的随机性，后面的长远未来被 $V(S_{t+1})$ 这个平滑的期望值代替了，因此波动极小。
- 缺点是有偏差，神经网络 $V$ 在训练初期通常是瞎猜的。用一个不准的预判去更新另一个预判，会引入不可避免的偏差，甚至可能导致训练不稳定。

### GAE

在现代策略梯度算法（如 PPO、TRPO）中，我们通常不直接使用价值 $V$，而是使用**优势函数** $A(S, A) = Q(S, A) - V(S)$，用来衡量某个动作比平均表现好多少。通过贝尔曼方程，我们可以用状态价值 $V$ 表示动作价值 $Q$：

$$
Q^{\pi} \left(\right. s , a \left.\right) = r \left(\right. s , a \left.\right) + \gamma V^{\pi} \left(\right. s^{'} \left.\right)
$$

得到优势函数：

$$
A^{\pi} \left(\right. s_{t} , a_{t} \left.\right) = R_{t} + \gamma V \left(\right. S_{t + 1} \left.\right) - V \left(\right. S_{t} \left.\right) \approx \delta_{t}
$$

会发现它就是 TD Error，但是 TD Error 偏差太大，容易陷入局部最优的问题。于是我们用 $V \left(\right. S_{t} \left.\right) = R_{t} + \gamma V \left(\right. S_{t + 1} \left.\right)$ 不断展开从而增加精度，就能得到多步 TD Error：

$$
\begin{array}{c} \delta_{t} & = R_{t} + \gamma V \left(\right. S_{t + 1} \left.\right) - V \left(\right. S_{t} \left.\right) \\ \delta_{t + 1} & = R_{t + 1} + \gamma V \left(\right. S_{t + 2} \left.\right) - V \left(\right. S_{t + 1} \left.\right) \\ \delta_{t + 2} & = R_{t + 2} + \gamma V \left(\right. S_{t + 3} \left.\right) - V \left(\right. S_{t + 2} \left.\right) \end{array}
$$

进而可以用多步 TD Error 来表示优势 $\hat{A}_{t}^{\left(\right. k \left.\right)} = \sum_{l = 0}^{k - 1} \gamma^{l} \delta_{t + l}$。多步估计确实减小了误差提高了进度，但是**随机变量越多，叠加在一起，整体的波动就越大**，导致随着 $k$ 增加方差越来越大。GAE 的思路是：**对所有步数的估计取加权平均，步数越多权重越小**。引入参数 $\lambda \in \left[0 , 1 \right]$，权重是 $\left(\right. 1 - \lambda \left.\right) \lambda^{k - 1}$：

$$
\hat{A}^{\text{GAE}}_t = (1-\lambda)\left[\hat{A}^{(1)}_t + \lambda\hat{A}^{(2)}_t + \lambda^2\hat{A}^{(3)}_t + \ldots\right]
$$

这个式子可以进一步化简得到：

$$
\begin{align}
\hat{A}^{\text{GAE}}_t &= \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l} \\
	                   &= \delta_t + \gamma\lambda\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \ldots \\
	                   &= \delta_t + \gamma\lambda \hat{A}^{\text{GAE}}_{t+1}
\end{align}
$$

### Bradly-Terry

BT 模型假设每个对象有一个隐含的分数，通常用 $r$ 表示。当比较两个对象 $i$ 和 $j$ 时，$i$ 优于 $j$ 的概率计算公式为：

$$
P(i > j) = \frac{\exp{r_i}}{\exp{r_i} + \exp{r_j}} = \frac{1}{1 + \exp{(r_j - r_i)}}=\sigma(r_i - r_j)
$$

>BT 模型要求分数都为正数，所以对 $r$ 套一个指数。

为了让我们的模型预测的分值 $r$ 尽可能符合现实，我们需要最大化观测到这些结果的总概率。假设每对比较都是独立的，我们可以写出**似然函数**：

$$L = \prod_{(i, j) \in \mathcal{D}} P(i \succ j) = \prod_{(i, j) \in \mathcal{D}} \sigma(r_i - r_j)$$

我们的目标是找到一组参数，使得 $L$ 最大。在深度学习和最优化中，我们更习惯**最小化一个损失函数**，而不是最大化一个连乘的概率（连乘容易导致浮点数下溢，且求导困难）。因此，我们对似然函数 $L$ 取**负对数**，把连乘变成连加，就得到了最终的损失函数：

$$\mathcal{L} = -\ln L = -\sum_{(i, j) \in \mathcal{D}} \ln \sigma(r_i - r_j)$$

这就是经典的 **Bradley-Terry 损失函数**：

$$\mathcal{L}_{RM} = -E_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \ln \sigma(r(x, y_w) - r(x, y_l)) \right]$$

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

$$\mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} \right] = \int q(x) \cdot \frac{p(x)}{q(x)} \, dx= \int p(x) \, dx$$


根据概率论的基本公理，**任何一个合法的概率密度函数，它在全空间的积分必须严格等于 1**。因为 $p(x)$ 是一个合法的概率分布，所以：

$$\mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} \right]=\int p(x) \, dx = 1$$

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


### DPO

![image.png](http://img.xilyfe.top/img/20260611223633880.png)


### GRPO

$$
\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G}  \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right)  - \beta D_{\text{KL}} \right]
$$

GRPO 是对 PPO 算法的变形：
1. 把 PPO 的 token-mean loss 变成了 seq-mean-token-mean loss
2. PPO 的优势是通过 TD Error 和 GAE 算的，GRPO 省去了 critic model，用组内的平均值和方差计算相对优势
3. PPO 把 KL 加在 reward 里面，而 GRPO 把 KL 加在 loss 里面当正则项。
### Q&As

{{< qa q="KL 散度和交叉熵、MLE的关系" >}}
假设我们有一组真实世界的数据集 $D = \{x_1, x_2, \dots, x_N\}$，那么根据经验分布可以得到：

$$
P(x) = \frac{1}{N} \sum_{i=1}^N \delta(x - x_i)
$$

然后我们进一步拆解 KL 散度，可以得到：

$$
KL(P\|Q) = \sum P(x)\log\frac{P(x)}{Q(x)} = \underbrace{-\sum P(x)\log Q(x)}_{\text{交叉熵} H(P,Q)} - \underbrace{\left(-\sum P(x)\log P(x)\right)}_{\text{熵} H(P)}
$$
因为后半部分信息熵 $\sum_{x} P(x) \log P(x)$ 完全由已知的数据集决定，里面没有模型的参数 $\theta$。所以在对 $\theta$ 求极小化的时候，这一项可以直接当作常数无视掉。因此，我们得到了第一个等价关系：**最小化 KL 散度在优化过程中等价于最小化交叉熵**

$$
\arg\min_{\theta} D_{KL}(P \parallel P_{\theta}) \iff \arg\min_{\theta} \left( - \sum_{x} P(x) \log P_{\theta}(x) \right)
$$

然后我们把经验分布代入：

$$
\begin{align}
- \sum_{x} P_{data}(x) \log P_{\theta}(x) &= - \sum_{x} \left( \frac{1}{N} \sum_{i=1}^N \mathbb{I}(x = x_i) \right) \log P_{\theta}(x) \\
&= - \frac{1}{N} \sum_{i=1}^N \left( \sum_{x} \mathbb{I}(x = x_i) \log P_{\theta}(x) \right) \\
&= - \frac{1}{N} \sum_{i=1}^N \log P_{\theta}(x_i)
\end{align}
$$

因此可以发现：**最大似然估计本质上等价于最小化数据的经验分布与模型分布之间的 KL 散度**

$$
\min_{\theta} D_{KL}(P \parallel P_{\theta}) \iff \min_{\theta} \left( - \frac{1}{N} \sum_{i=1}^N \log P_{\theta}(x_i) \right)
$$

{{< /qa >}}

{{< qa q="RL 能不能拓宽模型的能力边界？" >}}
	首先我认为模型的能力边界取决于 **模型的知识覆盖**，所以这部分能力一般只能由 pretraining 来提高。而 RL 做的事情是 **优化搜索策略**，以 passk 测试举例。RL 训练后 pass@1 提高但 pass@k 而不如 base model，说明 RL 更像是把 base model 已经"内含"的能力提取/放大出来，让高概率路径更容易被采样到，而不是让模型学会全新的推理模式——本质是**分布 sharpening**，而非能力扩展。
{{< /qa >}}

{{< qa q="如何 scale RL 训练边界？" >}}
- **数据/任务规模化**：扩大 prompt 多样性和难度覆盖，类似预训练的 scaling law 思路，但 RL 数据构造成本更高,需要 verifier/reward model 配合
- **Rollout 规模化**“：增大每个 prompt 的采样数，更多样本才能估计出更准的 advantage，同时增加发现稀有正确路径的概率
- **计算基础设施 scaling**：多机多卡分布式训练，async rollout 生成和训练解耦，避免同步等待浪费算力
- **课程学习**：从简单任务逐步过渡到难任务，避免 reward 过于稀疏导致训练早期完全学不到东西
- **Reward 设计的可扩展性**：从单一 ORM 扩展到 PRM，或结合 rule-based verifier + reward model 混合，让 reward 信号能覆盖更复杂任务
{{< /qa >}}

{{< qa q="LLM 推理能力是在哪一个训练阶段产生的？" >}}
主流观点认为：**推理能力的"种子"在预训练阶段就已经存在**，后续阶段是激发/强化，而非从零创造：
- **预训练阶段**：海量文本中天然包含大量推理模式（数学证明、代码逻辑、因果论证等），模型通过 ntp 隐式学到了这些模式的统计规律，这是推理能力的**知识基础**
- **SFT 阶段**：通过高质量 CoT 数据，让模型学会"以推理链的形式组织输出"这个**行为模式**——即让模型知道"应该这样表达推理过程"，更多是格式/习惯层面的教学
- **RL 阶段(如 GRPO/PPO)**：通过 reward 信号强化那些**能导向正确答案**的推理路径,抑制错误路径的概率
{{< /qa >}}

{{< qa q="Rejection Sampling 是什么" >}}
拒绝采样和重要性采样一样，都是"用一个分布的样本去处理另一个分布"的技术，核心动机都是**目标分布难以直接采样/优化**。在 LLM RL 里都用于处理 **off-policy** 问题，即用旧策略生成的数据，来服务新策略。

- Importance Sampling 是给每个样本一个**连续权重** $w=\frac{p(x)}{q(x)}$，样本全部保留只是加权，一般用于**梯度估计**的无偏修正
- Rejection Sampling 根据一定概率接受或者拒绝样本，拒绝的样本直接丢弃，不参与后续计算，拒绝的样本白白生成了，浪费算力。一般用于**数据构造/筛选**,产出的是干净数据集，再喂给标准 SFT/RL。

假设你想让一个 LLM 更擅长解数学题，可以拿一批带标准答案的数学题 {$\{(q_i, a_i^*)\}$​，$a_i^*$ 是已知的正确答案。对每道题 $q_i$​,用当前 LLM以较高 temperature采样 $N$ 个候选解答，然后用 rm 或者规则验证器等方式 **接受/拒绝**，最后用过滤后的干净数据做标准 SFT。

>rejection sampling 的本质是通过 SFT loss 在筛选出的正确样本上做梯度下降，直接提高 $\pi_\theta$ 对这些正确解答序列的似然，重塑模型的输出分布，让它更集中在正确答案(以及正确的推理路径)上。

{{< /qa >}}


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

{{< qa q="DPO 的 chosen 和 reject 的 loss 同时下降是因为什么" >}}
DPO 的损失函数为：

$$
\mathcal{L}_{\text{DPO}}= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Bigg[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\rm ref}(y_l|x)}\right)\Bigg]
$$

实际上，从上式可以看出，要让总的 loss 下降，优化偏好数据对的概率有多种情况，比如偏好-非偏好答案概率都下降/都上升，偏好答案概率上升-非偏好答案概率下降等，所以在 loss 下降的情况下，不一定是偏好答案概率上升-非偏好答案概率下降导致的，还可能是二者的概率都下降的情况。其次 Bradly-Terry 模型本身就存在优化不确定的问题，它只关心两个对象谁好谁坏。所以一般情况下 DPO 训练都更关心 **chosen reward 和 rejected reward 的 margin**。

单纯的关心 reward margin 也存在一个问题，**概率是守恒的，margin 只盯着两个点，没管剩下的概率质量去哪了**。$\pi_\theta(\cdot|x)$ 是整个输出空间上的一个分布，总和恒为 1。如果 $\pi_\theta(y_w|x)$ 被大幅压低，这部分概率质量必然要流向分布里的某个地方——DPO 的 loss 完全没有约束这部分质量流向哪里，它只比较了 $y_w$ 和 $y_l$​ 这两个特定的点。Princeton 那篇研究 "likelihood displacement" 的论文把这个现象的后果讲得很直白：这种 displacement 可能是灾难性的，会把概率质量从 preferred response 转移到含义完全相反的 response 上去——比如训练模型偏好"No"而不是"Never"，结果反而大幅推高了"Yes"的概率。

DPO-P 的做法是在 loss 里面加一个 chosen response 的 sft 损失项。给 chosen 的绝对似然加一个锚，不让它相对reference model掉太多，这样不管 margin 怎么变化，chosen 本身的概率底线是被保护的，displacement 没有空间发生。
{{< /qa >}}

{{< qa q="DPO 训练为什么会导致输出变长" >}}
原因：

1. 从隐式奖励来看，DPO 的隐式奖励为 $r=\beta\log{\frac{\pi}{\pi_{ref}}}=\beta \sum_{t=1}^{|y|}[\log{\pi_{ref}(y_t|x,y_{<t})} - \log{\pi_\theta(y_t|x,y_{<t})}]$。可以看到这个奖励是逐 token 对 log-ratio 求和，也就是说当 $|y_w| > |y_l|$ 时，即使每个 token 的 log-ratio 差异很小，累计效应也会让 $r_\theta(x,y_w)$ 系统性大于 $r_\theta(x,y_l)$。
2. 相比于上面来自 DPO 算法建模的固有偏差，训练数据中存在长度偏置也是造成 DPO 长度偏移的又一个原因。这种偏置源于 rm 固有的长度偏好，导致大多数偏好回复显著长于不偏好的回复。在统计数据中，不管是人工标注还是 GPT-4 标注都偏爱长回复。

解决方案有两种：

第一个是 SimPO。它去掉了 ref model，并且用长度归一化的平均 log-likelihood 作为隐式奖励：$r(x,y)=\frac{\beta}{|y|}\log{\pi_\theta(y,x)=\frac{\beta}{|y|}\sum_{t=1}^{|y|}\log{\pi_\theta(y_t|x, y_{<t})}}$。代入得到 SimPO 的损失函数为：

$$
\mathcal{L}_{SimPO}=-\mathbb{E}\left[\log{\sigma(\frac{\beta}{|y_w|}\log{\pi_\theta(y_w,x)} - \frac{\beta}{|y_l|}\log{\pi_\theta(y_l,x)} - \gamma)}\right]
$$

1. 除以 $|y|$ 对 token 的 log-ratio 进行归一化，消除了 DPO 的长度偏置。
2. 减去了 reward margin $\gamma$，强制要求 chosen 和 rejected 有一定区分度。

---

第二个方案是类似 R-DPO 的方法，加入长度差惩罚：

$$
L_{\mathrm{R\text{-}DPO}}=-\mathbb E\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}+\alpha |y_w|-\alpha |y_l|\right)\right]
$$

R-DPO 通过 logit 偏置改变样本的梯度权重，降低长 chosen 样本梯度/提高短 chosen 样本梯度。

>长度差惩罚加在 loss 里面求导之后不会之间消去吗，为什么可以对梯度有影响？
>因为 length margin penalty 在 $\log\sigma(\cdot)$ 里面，所以对 $L=-\log\sigma(z)$ 求导之后 $\nabla_\theta L=(\sigma(z)-1)\nabla_\theta z$  会受到影响。

---

第三个方案就是偏好数据建模的时候对长度去偏（De-Bias），其中可能涉及到多模型投票、Prompt Engineering等多种方法等混合。类似的方法：
- LIFT-DPO 将长度约束指令融入通用指令数据集，并拒绝超出指定长度限制的 chosen 回复。
- SamPO 从 chosen 和 rejected 的 token 中按相同数量随机下采样，确保参与梯度计算的 token 数相等。

{{< /qa >}}


{{< qa q="不同 RL 场景怎么设计 reward" >}}
1. 数学/代码/SQL/tool call：这些可验证结果的场景可以用 rule-based reward
2. 通用对话：reward model 打分
3. 长 CoT：ORM + PRM
4. 安全对其：多 reward 加权
{{< /qa >}}

{{< qa q="GRPO 的优势为什么要减 baseline" >}}
1. 可以降低梯度估计的方差
	- 如果直接用原始奖励值 $r_i$ 代替 $A_i$，因为 $r_i$ 通常永远是正数，这会导致**策略梯度的方差极大**，模型训练极不稳定，甚至不收敛。
	- 减去一个与当前 Action 无关的 Baseline，在数学上完全不改变梯度的期望值，但能**显著降低方差**。
2. 从**绝对好坏变成相对优势**，和 PPO 的 advantage 减去 critic model 预估的 value 一样。

>策略梯度的更新公式为 $g = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log \pi_\theta(a_i|s) \cdot r_i$，如果所有 $r_i$ 都是很大的正数，梯度更新会尝试把这组采样里的**所有动作的概率都往上推**。  但是，概率的总和永远只能是 $1$，这就会导致模型的参数空间发生剧烈的改变：所有的梯度向量都指向相同的正方向，彼此严重抵消，只有微弱的相对分量在起作用。

{{< /qa >}}

{{< qa q="GRPO 的优势为什么要除 std" >}}
1. 强化学习不同任务的 reward 定义不同，**奖励的尺度也不相同**。有的 $[-3,3]$ 有的 $[0,1]$，如果不除以 std，advantage 的绝对大小会完全受制于 reward 的尺度。
2. 大模型在训练的不同阶段，**组内得分的分布是动态变化的**。训练后期当模型收敛，生成的回答都非常好，分差极小，如果不除 std 的话 advantage 只有 $\pm 0.001$，梯度接近于 0，模型就会停止进化。
{{< /qa >}}


{{< qa q="为什么 GRPO 的 KL 是 loss 正则项，而 PPO 是加在 reward 里" >}}
1. GRPO 用组内相对优势替代了 PPO critic model 的细粒度 reward，如果直接把 KL 加在 GRPO 的 advantage 上会导致语义出现变化，例如某个 response 的优势的 $0.2$ 加上 KL 之后优势变成 $-0.3$ 了。而 PPO 的 critic model 要同时学会预测"未来还能拿到多少任务reward"和"未来还要扣多少KL惩罚"，它预测的 value 已经是考虑 KL 以后的数值了，不存在这个问题。
2. 在 PPO 中 KL 加在 token level 的 reward 上。因为有 critic 网络通过贝尔曼方程在时间步上进行前向递推，模型能够通过 GAE 明确知道是哪一个特定的 Token 导致了过大的 KL 漂移。而 GRPO 如果粗暴地将全序列 kl 累加到 seq-level reward 中，无法引导模型去学习序列内部的 token level 的演变关系。
{{< /qa >}}

{{< qa q="GRPO 为什么加上 KL 散度，KL 散度怎么计算，为什么 DAPO、GSPO 又去掉了KL散度？" >}}
GRPO 在 loss 中加入了 KL 当做正则项，目的是：
- 防止 policy 偏离 reference model 太远，避免 reward hacking
- 保持语言流畅性（ref model是SFT后的模型，有基本语言能力）
- 正则化作用，稳定训练

但是 DAPO 和 GSPO 认为：
- **DAPO**：认为 token-level KL 惩罚会抑制模型生成长链 CoT 的能力，导致模型倾向于生成短 response 以减小 KL，去掉后模型能更自由地探索长推理链，用 clip+entropy bonus 代替 KL 约束。
- **GSPO**：从理论上证明group-level的约束比token-level KL更合理。
{{< /qa >}}

{{< qa q="RL training 和 test-time scaling 各自是如何 explore 的" >}}
- RL  Training
	1. 提高 temperature 增加采样的多样性
	2. topP/topK 增加采样多样性
	3. Entropy bonus：在 loss 项里面加 token-mean 的熵当做正则项 $\beta H(\pi)$
- Test-time Scaling
	1. Best-of-N：采样N个response，用verifier选最优
	2. Beam Search：维护多条候选链，逐步扩展
	3. Sequential revision：让模型自我反思和修正（Reflection）
{{< /qa >}}

{{< qa q="熵崩塌的解决方案" >}}
1. DAPO 的解决方案 **clip-higher**。熵崩塌主要源于：在 $r_{t} \left(\right. \theta \left.\right) > 1 + \epsilon$ 的情况下，旧策略的概率本来就不高，还限制了更新幅度的上限，抑制了低概率 token 的增长。Clip-Higher 采用非对称裁剪机制，解耦上下裁剪的范围：上裁剪阈值 $\epsilon_{h i g h} = 0.28$：放宽低概率Token的探索限制。下裁剪阈值 $\epsilon_{l o w} = 0.2$：抑制高概率Token的过度利用。
2. 适当降低 KL 的系数 $\beta$，允许模型单次更新变化更大一些。
3. Entropy bonus：在 loss 项里面加 token-mean 的熵当做正则项 $\beta H(\pi)$
4. 动态 rollout 温度：随训练收敛提高采样温度，防止采样分布过于尖锐。
{{< /qa >}}

## On-Policy Distillation

### Q&As


{{< qa q="OPD 相比传统 RL/SFT 的改进及应用" >}}
**相比 SFT**：
- SFT 用 forward KL，是 mode-covering，容易学到"平均化"的行为，且完全依赖静态数据集，存在 exposure bias，训练时用 teacher forcing，推理时自回归生成，分布不一致。
- OPD 是 on-policy 采样（用学生模型自己的 rollout），用 teacher 模型的 logprobs 作为监督信号，直接优化 reverse KL 或类似目标，让学生在**自己会犯错的地方**接受矫正，而不是在教师的静态轨迹上模仿。

**相比传统 RL**：
- 传统 RL 依赖 reward model 或 rule-based verifier 提供**稀疏**信号
- OPD 用 teacher 模型提供**每个 token 位置**的密集监督信号，不需要设计复杂 reward，梯度信号更稠密，训练更稳定，样本效率更高
{{< /qa >}}

{{< qa q="student 模型和 teacher 模型的 tokenizer 不一致" >}}
student 和 teacher 模型的 eos token 不同，例如 student 的 eos token 是 <|endoftext|>，teacher的 eos token 是 <|im_end|>。这会导致计算 loss 的时候，student 生成 <|endoftext|> 的概率被大幅压低，但由于 teacher 模型的 <|im_end|> 不在 student 的 top-k 支撑集，student 模型学不到 teacher 的停止方法，就导致无限重复了。
{{< /qa >}}

## 分布式训练

### Data Parallel

>DP 和 DDP 的应用场景一般是 **单卡能够装下模型，并且要提高 `batch_size`**。

Data Parallel 的思路是 **每个 GPU 上都放有完整的模型参数**，把数据按 `batch_size` 维度进行切分，然后传输到不同 GPU 上进行前向传播、反向传播，然后计算梯度传回 GPU-0。GPU-0 会负责对所有梯度进行平均，然后在 GPU-0 上进行更新模型。之后 GPU-0 会把更新后的新参数传给其他 GPU 进行更新。

它的缺点在于数据的传输量太大了，并且都集中在 GPU-0 上压力太大了。假设参数量为 $\Psi$ 节点数量为 $N$，那么 GPU-0 需要传入梯度 $\left(\right. N - 1 \left.\right) \Psi$，传出参数量为 $\left(\right. N - 1 \left.\right) \Psi$。其他 GPU 传出梯度量为 $\Psi$ 传入参数为 $\Psi$。

### Distributed Data Parallel

![](https://img.xilyfe.top/img/20260202233932894.png)

>DDP 采用了 Ring-AllReduce 这种集群通信方式，来解决 DP 通讯量大的问题。

首先 PyTorch 会把模型内的参数按照倒序排列（因为是反向传播求梯度，顺序和代码是相反的），然后将参数依次放在桶里。每个参数都会挂一个监听器，当参数求得梯度之后监听器被触发，此时检查桶内参数是不是全都计算好梯度了。假如每个 GPU 的同一个桶都装满了，也就是说对应的梯度就计算好了，就会对桶内参数的梯度用 Ring-AllReduce 进行同步。当全部桶都同步完整，各个 GPU 的模型就应该同步了，此时就可以调用优化器对参数进行更新。

假设参数量为 $\Psi$ 节点数量为 $N$，那么对于每个 GPU 有：

- Scatter-Reduce 阶段传入/传出：$\left(\right. N - 1 \left.\right) \frac{\Psi}{N} \approx \Psi$
- AllGather 阶段传入/传出：$\left(\right. N - 1 \left.\right) \frac{\Psi}{N} \approx \Psi$

可以看到每个 GPU 的通讯量和节点数量是无关的，相比 DP 节省了大量通讯资源和时间。

### Tensor Parallel

 >DP 和 DDP 它们的思路是 **用显存冗余换吞吐量**。每张 GPU**都有完整模型**，但是只**处理不同的数据**，它的本质是**复制模型 → 并行处理数据 → 最后通过 AllReduce 同步梯度**，代价是模型被复制 $N$ 份，占用 **$N$ 倍显存**。张量并行 Tensor Parallel 的思路正好相反是 **用通信换显存**。现在的大模型参数量巨大一张卡很可能放不下，所以把模型拆到多卡，每张 GPU **只有部分模型**，但是**处理完整的数据**，最后进行合并。

Tensor Parallel 分为列拆分和行拆分两种，顾名思义就是把参数矩阵按列拆分开和按行拆分开，放在不同的 GPU 上。

![image.png](http://img.xilyfe.top/img/20260623141456509.png)

对矩阵进行列切分，得到的输出为 $Y = \left[ Y_{1} \mid Y_{2} \mid . . . \mid Y_{p} \right]$，输出被切分了，每卡只有一部分。它的优点就是各个 GPU 之间不需要通信，计算完全独立。比如我们把线性层之后要接一个激活函数，各个 GPU 计算得到中间值的一部分之后，可以直接计算激活函数的值。但如果下一层需要完整的 $Y$ 那么仍然需要 AllGather 通信。

---

![](https://img.xilyfe.top/img/20260317125901490.png)

把矩阵按列切分，$X$ 也需要切分，每个 GPU 计算的都是 **部分贡献**，最终需要对他们进行求和才能得到完整的 $Y = \sum Y_{i}$，所以必须通过 AllReduce 进行通信。但优点是它的输出是完整的，下一层可以直接使用。

---

一般情况下，列切分和行切分都是同时使用的。例如再一个 MLP 层中，我们需要进行 $y=W_2(\sigma(W_1x))$ 的变化：
- 假如我们两层都采用行拆分，那么每一层都需要一次 AllGather 开销太大了
- 假如我们两层都采用列拆分，我们按照 `X → A → Y → B → Z` 的流程。第一层我们把 $A$ 矩阵切分为 $A_{1}$ 和 $A_{2}$，得到 GPU1 上有 $Y_{1} = X \cdot A_{1}$，GPU2 上有 $Y_{2} = X \cdot A_{2}$，目前还是正常的。但是第二层就有问题了，此时 GPU1 上有 $Y_{1}$，GPU2 上有 $Y_{2}$，然后我们把 $B$ 矩阵按照列切分，GPU1 上有 $Z_{1} = Y_{1} \cdot B_{1}$，GPU2 上有 $Z_{2} = Y_{2} \cdot B_{2}$，他们各自少了 $Y_{2}$ 和 $Y_{1}$，每个 GPU 只算了一半的贡献。也就是说，用列拆分还是需要两次 AllGather。
- 如果我们采用先列拆分再行拆分的方式，那么第一层计算后，两个 GPU 在第二层都会正好得到需要的列切分过的输入 $Y_1$ 和 $Y_2$，最终只需要一次 AllGather。
### Deepspeed Zero

不管是 DP 还是 DDP，每个 GPU 都保存了完整的模型参数，中间激活值以及优化器状态，这里面优化器状态占用的显存最大。我们拿 AdamW 举例，一共需要：
1. FP16 的参数、梯度（模型参数）
2. FP32 的梯度、一阶动量、二阶动量、Master Weight（优化器状态）

而 DeepSpeed ZeRO 的 ZeRO 含义是 Zero Redundancy Optimizer，其核心思想是 **消除冗余存储的优化器状态**。大模型在传统数据并行下需要每个GPU 存完整参数、梯度、优化器状态。ZeRO 的解决方法是把**参数、梯度、优化器状态**分成 $N$ 份，每个 GPU 只存 $\frac{1}{N}$。为了用这些分片数据计算，**ZeRO 需要付出通信的代价**，ZeRO 需要通过 AIl-Gather 和 Reduce-Scatter 在 GPU 间频繁传递数据。

#### AllGather
![image.png](http://img.xilyfe.top/img/20260629181824533.png)

#### ReduceScatter

- **AllGather**：每个 GPU 都有对应的 shard，节点按逻辑环依次发送和接收数据块，例如节点 1 发送块 1 →节点2，同时接收节点 4 的块 4。经过 N-1 轮，所有节点通过拼接累积数据块完成全局同步。

![image.png](http://img.xilyfe.top/img/20260629183133126.png)

- **Reduce-Scatter**：先把所有人的数据**归约**（求和/平均），再把结果**切碎分发**，每人只留一片。

>两个操作的**总通信量完全相同**，都是 `(N-1)/N × 数据总量 × GPU数`，只是流向不同：Reduce-Scatter 是"多合一再分发"，AllGather 是"各出一份再广播"。

#### ZeRO-1

![](https://img.xilyfe.top/img/20260204160715456.png)

>- 深蓝色代表优化器状态
>- 浅蓝色代表参数和梯度

ZeRO-1 的运行流程是这样的：
1. 每个 GPU 都存储了完整的模型参数，所以可以分别独立的进行前向传播，计算得到 loss
2. 反向传播时，每个 GPU 都从后向前计算出每一层参数的梯度
3. 这时候 GPU-1 和 GPU-2 把计算出来的 **前三层梯度** ReduceScatter 传给 GPU-0。这时候 GPU-0 就可以计算平均梯度，并且它有前三层的优化器状态，就可以对前三层进行更新。然后再用 AllGather 把前三层更新后的参数广播给 GPU-1 和 GPU-2，中三层和后三层的参数也是如此更新。

假设参数量为 $\Psi$ 节点数量为 $N$，那么对于每个 GPU 有：
- 梯度收集阶段传入/传出：$(N-1)\frac{\Psi}{N}\approx\Psi$ 
- 参数收集阶段传入/传出：$(N-1)\frac{\Psi}{N}\approx\Psi$ 

所以 ZeRO-1 最终总传入/传出参数量为 $2\Psi$ 和 DDP 通讯量相同，但是每一个 GPU 上占用的显存量大幅度减少了。

#### ZeRO-2

![](https://img.xilyfe.top/img/20260204170258315.png)

DeepSpeed ZeRO-2 相对于 ZeRO- 1 的核心优化在于进一步分区了梯度从而显著降低显存占用，想法很简单：每个 GPU 只负责更新对应的参数，那么只需要保存这部分参数的梯度就好了。训练过程如下：
1. 由于每个 GPU 都存储了完整的模型参数，所以可以分别独立的进行前向传播，计算得到 loss
2. 反向传播时，每个 GPU 都从后向前计算出每一层参数的梯度
3. 然后 GPU-0 和 GPU-1 计算出最后一层参数的梯度，它们会把这些梯度放到一个 bucket 里面，再传给 GPU-2。当 GPU-2 计算得到最后一层的平均梯度，GPU-0 和 GPU-1 就把这些梯度删除，因为不是自己需要的，以此减少了显存占用。倒数第二、三层也是如何，计算得到梯度再传给 GPU-2 计算平均梯度，然后自己再把不需要的这部分梯度删除。而 GPU-2 得到了后三层平均梯度，就可以更新自己对应的优化器状态，再更新参数。
4. 最后三个 GPU 再 AllGather 分别传递自己更新好的参数，使得每个 GPU 上的模型保存一致。

#### ZeRO-3

![](https://img.xilyfe.top/img/20260204173843936.png)

DeepSpeed ZeRO-3 又进一步分区了模型的参数，在前向传播时候通过其他 GPU 来广播自己所缺的那一部分参数。

假设参数量为 $\Psi$ 节点数量为 $N$，那么对于每个 GPU 有：

- 参数广播阶段传入/传出：$\left(\right. N - 1 \left.\right) \frac{\Psi}{N} \approx \Psi$
- 梯度广播阶段传入/传出：$\left(\right. N - 1 \left.\right) \frac{\Psi}{N} \approx \Psi$
- 参数收集阶段传入/传出：$\left(\right. N - 1 \left.\right) \frac{\Psi}{N} \approx \Psi$

>为什么 ZeRO-3 的通信量不是 $4\Psi$？前向传播 AllGather 一次参数，反向传播计算梯度时候还要 AllGather一次参数，然后计算完梯度需要 Reduce Scatter，然后把更新完的参数再 AllGather。
>**ZeRO-3 的参数在 optimizer step 之后依然保持分片状态**，不需要重新广播给所有卡。每张卡只更新自己持有的那段参数分片（fp32 master weight → 转回 fp16），更新完就结束了。下一个 iteration 的 forward 需要用参数时，再按层做 AllGather——这个通信已经算在「下一步的 forward Ψ」里，而不是当前 step 的额外开销。

### Q&As

{{< qa q="在 LLM 训练时，如果不小心多 AllReduce 了几次 loss，会发生什么" >}}
正常流程是 forward → backward → AllReduce梯度 → optimizer step。每次 AllReduce是对梯度做 **sum/mean聚合**，多做一次相当于梯度被**重复累加**，效果等价于**梯度被放大N倍**，后果就等效学习率增加，会导致训练不稳定，梯度爆炸，参数更新幅度异常大，严重时直接 NaN。
{{< /qa >}}

{{< qa q="ZeRO 在进行并行计算时，这个 GPU 怎么拿到别的 GPU 上的参数？" >}}
以 ZeRO-3 举例，在前向传播时候：
- 当前 layer 的参数被切分在 $N$ 块卡上，每块卡只持有 $\frac{1}{N}$​ 的参数
- 执行 **AllGather**：每块卡把自己持有的参数 shard 广播给所有其他卡
- 所有卡拼到完整参数后执行前向计算
- 计算完成后**立刻丢弃**刚才 gather 来的参数

然后反向传播时候：
- 执行 **AllGather**：每块卡把自己持有的参数 shard 广播给所有其他卡
- 所有卡分别进行反向传播，计算对应参数 shard 的梯度
- 所有卡 Reduce Scatter，这样每个 GPU 就能得到对应 shard 的完整梯度
- 每个 GPU 更新自己对应的那部分参数 shard
{{< /qa >}}

{{< qa q="有 4 块卡，数据被切成了 4 份，每个卡上有一份数据，设一次通信量是 $x$，如果要实现一次 AllReduce 操作，需要多少通信量？" >}}
**Ring-AllReduce分两个阶段：**

| 阶段     | 操作             | 每块卡发送量                     |
| ------ | -------------- | -------------------------- |
| 第一阶段   | Reduce-Scatter | $\frac{N-1}{N} \cdot x$    |
| 第二阶段   | AllGather      | $\frac{N-1}{N} \cdot x$    |
| **合计** |                | $\frac{2(N-1)}{N} \cdot x$ |

N=4时，每块卡总发送量 = $\frac{3}{2}x$，**4块卡总通信量 = $4 \times \frac{3}{2}x = 6x$**。
{{< /qa >}}

## Infra

### Q&As

{{< qa q="RL rollout 中的长尾问题是什么，有哪些解决方案" >}}
RL rollout 中的长尾问题指的是，推理时候不同样本的 response 长度可能差异很大，这会导致有些样本很早结束但由于需要等待长序列，一直输出 eos_token 空转浪费 GPU 资源。

- **Async rollout**：rollout 和 training 解耦成独立的 worker，生成快的样本不用等慢的，持续填充训练队列
- **长度惩罚/截断策略**：设置最大长度上限,超长序列强制截断或给予长度惩罚
- **动态 batch 调度**：按预估长度把 sample 放到不同 bucket 里，把长度相近的样本分到同一 batch，减少组内方差
- **Force-end 机制**：对多轮 agentic rollout 设置最大轮数强制结束，避免个别样本无限循环拖慢整体
{{< /qa >}}

{{< qa q="continuous batching 在 RL 训练时会有什么问题" >}}
1. **off-policy 问题**：如果生成阶段用的策略和训练阶段实际更新的参数出现版本差异（生成时用的是几步之前的旧参数），就引入了 off-policy 偏差，需要 importance sampling ratio 做矫正。
2. **同步开销**：例如 GRPO 需要在组内都 rollout 结束才能算 advantage，而 continuous batching 恰恰是希望不同请求独立异步完成。这两者存在天然张力，需要额外的 group-level barrier 同步。
3. **显存管理复杂度**：continuous batching 依赖 KV cache 的动态分配，训练阶段还需要额外保存 log-prob、activation 用于反向传播，显存压力比纯推理场景更大。
{{< /qa >}}

## 训练


{{< qa q="参数量计算" >}}
1. Embedding 层：token embedding 的参数量是 `vocab_size * hidden_size`，position embedding 现在一般都是 RoPE 或者变体，非可学习位置编码参数量都是零。
2. Transformer Block：单层 transformer block 包含一个 MHA（或者 MQA 或者 GQA），和一个 FFN 还有一个 LayerNorm。
	- 注意力机制包括四个注意力矩阵，参数量是 `4*hidden_size^2`，如果采用 MQA 或者 GQA 它的参数量是 `2*hidden_size*^2 + 2*hidden_size * num_kv_heads*head_dim`。
	- 前馈神经网络 SwigLU FFN 包括 gate/up/down 三个矩阵，参数量是 `3*hidden_size*ffn_dim`。中间维度一般取 `ffn_dim=8*hidden_size/3`，所以参数量约等于 `8*hidden_size^2`
	- RMSNorm 的参数量可以忽略不计
3. LM_Head：就是反向 token embedding，参数量也是 `vocab_size * hidden_size`
4. 总参数量约为 `2*hidden_size*vocab_size + 12*hidden_size^2`
{{< /qa >}}

{{< qa q="推理需要的显存" >}}
推理需要的显存包含两部分，模型权重和 KVCache 两部分：
1. 模型权重占用的显存取决于参数量和数据类型，以 FP16 为例就是 `参数量 * 2B`
2. KVCache 的显存量为 `2*batch_size*num_layers*seq_len*(num_kv_heads*head_dim)*2B`
{{< /qa >}}

{{< qa q="全参训练需要的显存" >}}
训练显存大致分为以下四部分：
1. 模型权重：取决于存储的精度，常见的 BF16 和 FP16 占用大小为 2B
2. 梯度：反向传播计算的梯度，和权重一样常见情况下占用 2B
3. 优化器状态：常见的 Adam 会为每个参数都保存它的 Momentum、Variance 和 Master weights，精度为 FP32 所以总计 12B
4. 中间激活值：简单来说就是为了计算反向传播的梯度，需要把前向计算的中间值存储起来，具体计算见下文。

合计起来，显存占用的经验公式为：$\text{VRAM}_{t r a i n} \approx 20 \times N \left(\right. B y t e s \left.\right)$
{{< /qa >}}

{{< qa q="lora训练需要的显存" >}}
LoRA 和全参训练相比，它冻结原模型权重，只训练低秩矩阵A、B。
- 模型权重：前向传播要用不能节省，还是 `总参数量 * 2B`
- 梯度：`LoRA 参数量 * 2B`
- 优化器状态：`LoRA 参数量 * 12B`
- 激活值：和全参一样
{{< /qa >}}

{{< qa q="sft/ppo/dpo/grpo 需要的显存" >}}
**SFT**：就一个 policy model，等同于普通全参/LoRA训练

**PPO**：
- actor model 需要权重、梯度、激活值、优化器状态
- critic model 需要权重、梯度、激活值、优化器状态
- reference model 仅仅推理得到 logprobs，所以只需要权重
- reward model 只需要权重

**DPO**：
- actor model 需要权重、梯度、激活值、优化器状态
- reference model 仅仅推理得到 logprobs，所以只需要权重

**GRPO**：
- actor model — 训练
- reward model — 推理
- reference model — 推理
{{< /qa >}}

{{< qa q="deepspeed zero/fsdp 需要的显存" >}}
- **ZeRO-1**：把优化器状态拆分，所以单个 GPU 显存占用是 `2+2+(12/N)B * 参数量`
- **ZeRO-2**：优化器状态和梯度拆分，显存为 `2+(2+12)/N B * 参数量`
- **ZeRO-3**：优化器状态和梯度还有参数都拆分，显存为 `(2+2+12)/N B * 参数量`
- **FSDP**：与 ZeRO 类似

>这里不考虑激活值。

{{< /qa >}}


## 项目

### MedicalGPT

### Search-R1

#### dataset

{{< qa q="各阶段数据是什么格式" >}}
第一阶段是 **在 Llama-Factory 上进行 SFT 冷启动**。Llama-Factory 支持的数据集格式为 sharegpt 或者 alpaca：

```text
# alpaca
[
    {
        "instruction": "任务指令",
        "input": "可选的输入上下文",
        "output": "期望的输出响应"
    }
]
# sharegpt
[
    {
        "conversations": [
            {
                "from": "human",
                "value": "用户说的话"
            },
            {
                "from": "gpt",
                "value": "助手的回复"
            },
            {
                "from": "human",
                "value": "用户下一句话"
            }
        ],
        "system": "可选的系统提示词"
    }
]
```

然后使用这些自定义数据集需要我们重写 `dataset_info.json` 文件。

第二阶段是 **在 verl 上进行 GRPO 训练**。verl 需要的数据是包含下面字段的 parquet 类型数据：

```json
{
    "data_source": data_source,
    "prompt": [{
        "role": "user",
        "content": question,
    }],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "target": example["golden_answers"]
        }
    },
    "extra_info": {
        "split": split,
        "index": idx,
    }
}
```

我们下载 [FLASHRAG](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/) 数据集，然后用 Search-R1 提供的转换脚本 `qa_search_train_merge.py` 就好了，它会自动填充字段并且转换为 parquet 格式。
{{< /qa >}}

{{< qa q="各阶段数据是怎么构造的" >}}
Search-R1 用的数据集是 Huggingface 上的 FlashRAG，这个数据集提供了 question 和 golden_answer，正好适用于 Search-R1 这种 ORM 的 RL 训练。FlashRAG 包含多种类型的 QA 数据：

| 数据集      | 说明                                               |
| -------- | ------------------------------------------------ |
| hotpotqa | 多步跨文档推理数据集，需要多轮检索、多信总融合推理，适合验证模型多轮搜索决策能力。        |
| triviaqa | 海量开放域事实问答数据集，侧重单步精准事实检索，用验证模型 query 生成质量与精准搜索能力。 |
{{< /qa >}}

#### retriever

{{< qa q="检索这块用的是什么方案？" >}}
E5-base 向量模型对 Wikipedia 语料做 dense embedding，建 FAISS 索引，本地部署成检索服务，
封装成 VERL Tool 注册进 Tool Agent Loop，模型在 rollout 过程中自主发起 query、
拿到 top-k 段落后拼回上下文继续推理。纯 dense retrieval，没有做 BM25 混合。
{{< /qa >}}

{{< qa q="为什么不用现成的搜索引擎 API？" >}}
1) 复现对齐：Search-R1 原论文用的是固定的 Wikipedia dump（wiki-18），
   用本地语料+索引能保证检索内容可控、可复现，不会随外部搜索引擎结果变化而漂移，
   便于做消融和对比原论文指标。
2) 训练稳定性：RL rollout 阶段检索调用量大且并发高，外部 API 有限流、延迟抖动、
   可用性问题，会直接拖慢/打断整个 rollout pipeline；本地服务延迟低且可控，
   可以配合 async rollout 提升吞吐。
3) 成本：训练阶段调用次数是数量级的，外部 API 计费不现实。
{{< /qa >}}

{{< qa q="retrieval 服务挂了怎么办" >}}
1. rollout 阶段进行检索服务时，如果遇到 retrieval 失败会进行多次重试。
2. 在 Search-R1 基础上引入异常轨迹监控与过滤机制，检索服务不可用/超时会被监控模块捕获，对应 trajectory 直接做 loss mask，不参与 GRPO 更新，避免因基础设施问题产生的低质量/异常样本污染训练信号。
3. 监控机制如果发现 retrieval 连续失败超过指定阈值，会直接终止训练。
{{< /qa >}}

#### training

{{< qa q="项目里怎么处理模型输出格式不对的情况？" >}}
1. SFT 冷启动阶段用教师模型蒸馏出的含 CoT + Tool Call 的轨迹数据训练， 从源头上让模型学会规范的输出格式
2. RL 阶段用格式约束作为辅助奖励， 配合精确匹配主奖励一起引导
3. 如果输出格式解析失败， 归为异常轨迹类型之一，做 loss mask 处理，不让这类样本产生有效梯度
{{< /qa >}}


{{< qa q="为什么不用 HuggingFace 直接推理而用 vLLM？" >}}
1. 吞吐：vLLM 的 PagedAttention + continuous batching 相比 HF 原生推理 在 rollout 阶段吞吐高出一个量级，RL 训练里 rollout 生成是主要耗时瓶颈 
2. 框架适配：VERL 的 HybridEngine 依赖 vLLM 做 rollout 引擎、FSDP 做训练引擎， 两者之间做权重 reshard/同步，这是新版 VERL 架构默认的 rollout backend
3. async rollout：要提升异步 rollout 机制的吞吐收益，需要 vLLM 支持的 高并发生成能力配合，HF 原生推理无法支撑这个并发规模
{{< /qa >}}

{{< qa q=" 训练时怎么平衡 exploration 和 exploitation？" >}}
1. 采样温度/top-p 控制单条轨迹的多样性
2. group size 控制组内探索广度，group 内做相对优势计算
3. 对比 GRPO 加 KL 正则和不加 KL 正则的效果
{{< /qa >}}

{{< qa q="评价指标" >}}
{{< /qa >}}