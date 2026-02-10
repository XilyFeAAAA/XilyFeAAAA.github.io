---
title: MiniMind 学习指北(一)：Model
date: 2026-01-22T13:47:19+08:00
featuredImage: http://img.xilyfe.top/img/20260122134824760.png
authors:
  - Xilyfe
series:
  - minimind
tags:
  - 大模型
  - 深度学习
lastmod: 2026-02-10T02:22:22+08:00
---

这是 Minimind 学习指北系列的第一节，整个系列我们大致分为：
1. 模型实现
2. Tokenizer
3. 预训练
4. 评估
5. 监督微调
6. LoRA
7. 强化学习
8. 蒸馏
9. 推理

![LLM-structure.png](http://img.xilyfe.top/img/20260122135047734.png)

## RMSNorm

### 前身

RMSNorm 也是归一化 Normalization 的一种，它的提出是为了解决这样一个问题：当数据分布非常不一致的时候，模型无法很好的学习到数据里面的信息。从数学层面来看，我们举个例子：假设我们有一个函数 
$$
y=wx
$$
在反向传播时候我们需要计算损失函数对参数的偏导 $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \times x$，可以看到梯度大小和 x 有关系，x 过大过小都容易导致梯度爆炸或消失。而归一化把数据变成变成均值为 0，标准差为 1 的分布，就可以缓解这个问题。

在 Transformer 原论文中归一化采用的是 LayerNorm：
$$
\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 + \epsilon}} \odot \gamma
$$
而 RMSNorm 在 LayerNorm 的基础上把均值 $\mu$ 这一项去掉了，在大幅减少计算量的同时性能相差不大：
$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{H} \sum_{i=1}^{H} x_i^2 + \epsilon}} \odot \gamma
$$

### 代码

```python
class RMSNorm(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        dtype: torch.dtype,
        device: torch.device,
        eps: float = 1e-8
    ) -> None:
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model), dtype=dtype, device=device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x.square(), dim=-1, keepdim=True)
        norm = x / torch.sqrt(mean + self.eps)
        return norm * self.weight 
```

>[!Note]
>值得注意，公式中 x 需要和均方根相除，所以求均值时候 `keepdim` 需要设为 True。否则 x 的维度是 \[bs, len, d_model]，mean 的维度是 \[bs, len]，没办法广播相除。

## RoPE

RoPE 的基础知识可以查看文章：[RoPE]({{< ref "posts/2025/rope" >}})


![](http://img.xilyfe.top/img/20260119121142785.png)
这里我引用 RoPE 文章里面的示例图，一步步拆解代码：
1. 根据公式 $\theta_i=10000^{-2i/d}$，计算 $\theta_0$ 到 $\theta_{d/2-1}$ 向量
2. 将它与 position 向量（也就是公式里面的 m） 求外积
3. 然后对整个矩阵求余弦值，同理正弦矩阵的步骤也相同

由于每次对 Q 和 K 矩阵进行 RoPE 时，它们的位置信息都是固定的，所以我们可以把求正余弦矩阵的步骤拆解出来。

```python
def precompute_freqs(dim: int, end: int, base: float = 1e6):
    position = torch.arange(end, dtype=torch.float)
    freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    sinusoid = torch.outer(position, freq)
    return torch.cos(sinusoid), torch.sin(sinusoid)


def apply_rotary_pos_emb(qk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    even, odd = qk[..., 0::2], qk[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = odd * cos + even * sin
    
    rotated_qk = torch.stack([rotated_even, rotated_odd], dim=-1)
    return rotated_qk.reshape_as(qk)
    
```

## YaRN

### 前言

在研究 YaRN 之前，我们需要回到 RoPE 的公式，看看它存在什么问题。
我们已经知道，RoPE 本质上是在每个 attention head 的偶数/奇数维度上做一个二维旋转，旋转角度是 $\theta = m \cdot \omega$ 。由于 $\omega_i = 10000^{-2(i-1)/D}$ ，所以：
- 在低维度（也就是i 比较小的维度） $\omega_i$ 比较大更接近于 1，它的旋转角度是 $m \cdot 1$。m 稍微变化一点，它的角度变化就非常明显。这种高频信号更适合捕捉短距离、精细的位置差异。
- 而在高纬度的时候 $\omega_i$ 比较小，频率就低，适合捕捉长距离、粗粒度的位置关系。

外推时会发生什么？
- 现在给模型一个长度为 4096 的句子。对于第 3000 个词会发生什么？
- 它的旋转角度 $\theta = m \cdot \omega$。由于 sin 和 cos 函数是周期性的，当 m 变得很大时，$m \cdot \omega$ 可能会绕回来。
- 当位置 m 超出预训练范围 L 时，比如 m=L+k，模型计算出的旋转角度可能和一个在预训练范围内的位置 m'(m'<L)的角度一模一样。
- 结果就是：模型会彻底混淆位置。它看到第 3000 个词，可能觉得它的位置和第 500 个词差不多，导致相对位置信息完全错乱，注意力计算也就随之崩溃，最终模型性能急剧下降。或者说：在高频维度 $\omega_d$​ 和 $j-i$ 大导致 $\Delta\theta_d$ 很大疯狂绕圈，cos/sin 落在训练中从没见过的组合。


### 解决方案

YaRN 的解决方法**并不是拉短高频**，而是拉长低频，让**低频在远距离还能保持可区分的相位**。
由于高频区域负责短距离信息，缩放会导致细节丢失，所以不缩放；我们可以通过使低频维度旋转变慢，从而覆盖更长的位置范围：
$$
h(\theta_d) = (1 - \gamma(r(d))) \cdot \frac{\theta_d}{s} + \gamma(r(d)) \cdot \theta_d
$$
以及：
$$
\begin{array}{c} \gamma(r) = 
\begin{cases} 
0 & \text{if } r < \alpha \\
1 & \text{if } r > \beta \\
\frac{r - \alpha}{\beta - \alpha} & \text{otherwise}
\end{cases} \end{array}
$$
YaRN 的官方写法采用波长来判断频率高低，由于正余弦函数周期为 2π，所以波长为 $\lambda(d) = \frac{2\pi}{\theta_d}$，而 $r(d)=\frac{L}{\lambda(d)}$ 代表在最大上下文长度 L 内，这个维度会转多少圈。假如 $r \ll 1$ 那么说明一圈都转不完属于低频。
- 在高频维度（小 $\theta_d$，小 $\lambda_d$，大 $r(d)$）：几乎不缩放（保持原 $\theta_d$​）
- 在低频维度（大 $\theta_d$，大 $\lambda_d$，小 $r(d)$）：完全缩放 $\theta_d / s$
- 中间维度：线性过渡

```python
def precompute_freqs(dim: int, end: int, base: float = 1e6, rope_scaling: Optional[dict]=None):
    position = torch.arange(end, dtype=torch.float)
    freq, attn_factor = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim), 1.0
    
    if rope_scaling is not None:
        beta_fast, beta_slow, factor, ori_max, attn_factor = (
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("factor", 16),
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("attention_factor", 1.0)
        )
        if end > ori_max:
            r = ori_max * freq / 2 * torch.pi
            gamma = torch.clamp((r - beta_slow) / (beta_fast - beta_slow), 0, 1)
            freq = (1 - gamma) * freq / factor + gamma * freq
    
    sinusoid = torch.outer(position, freq)
    cos = torch.cos(sinusoid) * attn_factor
    sin = torch.sin(sinusoid) * attn_factor
    return cos, sin

```

>注意这里用固定值 factor 替换了YaRN 里面的 s 这个温度参数

## GQA


![image.png](http://img.xilyfe.top/img/20260126212710505.png)

传统 MHA 的做法是把 Query，Key 和 Value 都分为多个注意力头分别计算，但是在实际应用中通常采用 KVCache 空间换时间的策略，Key 和 Value 占用的显存非常大。GQA 的思路非常直接，让 Key 和 Value 不要分那么多个注意力头了，一组 Query 共用一个 Key 和 Value。虽然注意力头减少了，但是实际应用中性能差距不大，显存占用显著减小。


在 `__init__` 阶段，我们需要保证 Query 的注意力头数可以整除 KV 的注意力头数，这样组数才是一个整数。

```python
class GQA(nn.Module):
    def __init__(self, args: PretrainedConfig):
        super(GQA, self).__init__()
        assert args.hidden_size % args.num_attention_heads == 0
        assert args.num_attention_heads & args.num_key_value_heads == 0

        self.flash_attn = args.flash_attn
        self.n_rep = args.num_attention_heads // args.num_key_value_heads
        self.n_query_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.hidden_size = args.hidden_size
        self.head_dim = self.hidden_size // self.n_query_heads
        self.q_proj = nn.Linear(self.hidden_size, self.n_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
```

为了保证 Query 和 KV 的维度相同可以进行矩阵乘法计算，我们需要把 KV 的张量进行扩展：利用 `repeat_interleave` 这个内置函数就可以在张量的第 k 维进行重复了。

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return torch.repeat_interleave(x, dim=1, repeats=n_rep)
```

然后在位置编码之后对 KV 应用 `repeat_kv`进行扩张：

```python
# RoPE & YaRN
cos, sin = position_embeddings
key = apply_rotary_pos_emb(key, cos, sin)
query = apply_rotary_pos_emb(key, cos, sin)

key = repeat_kv(key, self.n_rep)
value = repeat_kv(value, self.n_rep)

# KVCache
if past_key_value is not None:
    key = torch.cat([past_key_value[0], key], dim=2)
    value = torch.cat([past_key_value[1], value], dim=2)
past_key_value = (key, value) if use_cache else None

scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
# Casual Mask
causal_mask = torch.triu(
    torch.full((seq_len, seq_len), float("-inf")), diagonal=1
).to(scores.device)
scores[..., -seq_len:] = scores[..., -seq_len:] + causal_m
# Padding Mask
if attention_mask is not None:
    assert attention_mask.dim() == 2  # [b, k_len]
    scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf")

attn = torch.softmax(scores, dim=-1)
attn = self.attn_dropout(attn)
attn = torch.matmul(attn, value)
```

>[!Note]
>假设在预训练阶段（即 KVCache 没有开启），scores 的形状为 \[batch_size, num_heads, seq_len, seq_len]，attention_mask 的形状是 \[batch_size, seq_len]，那`attention_mask.unsqueeze(1).unsqueeze(2)` 而不是 `attention_mask.unsqueeze(1).unsqueeze(-1)` 都可以吗？
>并不是。PyTorch 的广播规则是从右向左对齐维度，假如我们用第二种方法：
>- scores: \[b, h, s, s ]
>- mask: \[b, 1, s, 1 ]
>
>那么 mask 的 s 会被对其到 scores 的 query 维度，也就是 dim=-2 的行维度。这意味着如果某个 query 位置 i 是 padded 的，那么整个第 i 行都会被填成 -inf。如果我们采用第一种方法，那么 s 会被对其到 scores 的 dim=-1 列维度，假如 query 位置 i 是 padded 那么 scores 的第 i 列都会被填充，这才是正确的。


## FFN

FFN 全称是 FeedforwardNet，前馈神经网络。他主要是对输入进行升维，以及加入非线性变化来丰富神经网络的细节。

升维是一个数学的概念，达到三维之后已经超过我们的想像范围了毕竟难理解，我们可以用一个例子来类比一下。我们评价一个人可以从三个方面进行打分：人品、外貌、能力，但是对于全世界这么多的人，都只从三个维度进行评价， 就会太笼统了，可能有很多人评分相同。所以我们可以把评分指标升维，例如外貌就可以进一步细化为：身高、体重、长相、年龄等等。所以前馈神经网络对输入数据进行升维，可以理解为把数据里面的信息更细节化，加深神经网络的理解。

![image.png](http://img.xilyfe.top/img/20260127134545216.png)

在神经网络加入非线性是为了增加网络的表达能力。如果单一的只使用线性变换，再复杂的组合 $y=f_1(f_2(f_3(f_4(x)+b_4)+b_3)+b_2)+b_1$ 都可以被简化为 $y=f(x)+b$ 这个一次函数。在网络中加入非线性因素就可以**解决线性不可分问题**，通过分段函数来增加它的拟合能力，提高表达能力。

```python
class FeedForwardNet(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super(FeedForwardNet, self).__init__()

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.act_fn = ACT2FN[args.hidden_act]
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x))))
```

>ACT2FN 是 Transformer 库自带的一个激活函数库，就不用我们自己手写了。
>![image.png](http://img.xilyfe.top/img/20260127134821291.png)

## Model

简单来说 MiniMindModel 整合了每个模块，并且预计算了正余弦值，这样就减少了重复计算的开销。

```python
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig) -> None:
        super(MiniMindModel, self).__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.vocab_size)
        self.norm = RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(config) for _ in range(config.num_hidden_layers)])

        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        _, seq_len = input_ids.shape
        hidden_state = self.embedding(input_ids)
        hidden_state = self.dropout(hidden_state)
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        sta_pos = 0 if past_key_values[0] is None else past_key_values[0][0].shape[2]
        position_embeddings = (
            self.freqs_cos[sta_pos : sta_pos + seq_len, :],
            self.freqs_sin[sta_pos : sta_pos + seq_len, :],
        )

        cur_key_values = []
        for idx, layer in enumerate(self.layers):
            hidden_state, cur_key_value = layer(
                hidden_state,
                position_embeddings,
                past_key_values[idx],
                use_cache,
                padding_mask,
            )
            cur_key_values.append(cur_key_value)

        hidden_state = self.norm(hidden_state)
        logits = self.proj(hidden_state)

        return logits, cur_key_values
```

>由于预计算的正余弦值形状是 \[max_seq_len, dim // 2]，而使用 KVCache 之后推理阶段每次 hidden_states 的长度为 1，所以我们只需要取对应位置的 cos 和 sin 就好了，而不是传入整个 freqs_cos 和 freqs_sin。

在 MiniMindModel 继承上我们又封装了一层 MiniMindForCausalLM，它继承了 PreTrainedModel 和 GenerationMixin 两个基类。
- PreTrainedModel 是 Huggingface 用来规范预训练模型的，规定了创建和定义预训练模型所需的核心功能和属性
- GenerationMixin 是 Huggingface 是用来规范生成类模型的基类，提供统一的生成功能和接口

>具体细节可见文章 [transformer库的基类](hg_transformer)。

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig) -> None:
        self.config = config
        super().__init__(self.config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embedding.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
```

当处于训练模式的时候，由于因果语言模型的训练逻辑是 **预测下一个词（Next Token Prediction）**，所以我们需要把 logits 和labels 错位，将“模型的预测结果”和“真实的标准答案”对齐。

```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
```

当处于推理模式时候，由于 token 需要一个一个生成，我们每次只需要最后一个 token 的隐藏状态，所以可以指定 `logits_to_keep`，一般设为 1。

```python
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

