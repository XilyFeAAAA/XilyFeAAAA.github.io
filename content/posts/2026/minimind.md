---
title: MiniMind 学习指北
date: 2026-01-22T13:47:19+08:00
featuredImage: http://img.xilyfe.top/img/20260122134824760.png
authors:
  - Xilyfe
series: []
tags:
  - 大模型
  - 深度学习
lastmod: 2026-02-02T03:38:54+08:00
---
0#图

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
    key = torch.cat([past_key_value[0], key], dim=1)
    value = torch.cat([past_key_value[1], value], dim=1)
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
        return x + self.dropout(self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x))))
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


## 数据集

`__getitem__` 方法有几个需要注意的地方：
1. 在 input_ids 前后加上 bos 和 eos 两个 special token 可以帮助模型理解，句子该从哪里开始，在什么时候结束，不会在长文本胡言乱语。
2. 由于我们在模型的 `forward` 里面规定了：训练模式下将 logits 和 labels 进行偏移，所以在 Dataset 里面返回的 x 和 y 就不用额外的进行 shift 了。
3. padding 补充的 token 不参与 loss 计算，所以在 labels 里面把这部分 token 的 id 设为 -100，和 `F.cross_entropy(...,ignore_index=-100)` 一致。

```python
class PreTrainDataset(Dataset):
    def __init__(
        self, tokenizer, datapath: Union[str, PathLike[str]], max_length: int = 512
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(datapath)

    def load_data(self, datapath: Union[str, PathLike[str]]) -> list[str]:
        samples = []
        with open(datapath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(data)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> list[int]:
        sample = self.samples[index]
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length - 2,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = torch.tensor(
            [self.tokenizer.bos_token_id]
            + encoding["input_ids"]
            + [self.tokenizer.eos_token.id],
            dtype=torch.long,
        )
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels

```


## Tokenizer

>在 CS336 的笔记中我已经完整介绍了一个 Tokenizer 是如何训练并且读取的，详情可见 [[cs336_assignment1]]。

简单来说，训练一个 tokenizer 经过以下步骤：
1. 通过正则分词，获得文本中全部 token，将其和 special_tokens 一起记录。
2. 不断把文本中出现频率最高的 token_pair 合并得到新 token，然后用新 token 替换文本中原先的 pair。
3. 重复上一步直到 vocab 达到指定规模。

上面的代码我们已经在 CS336 里实现过了，这一次我们通过 Huggingface 的 tokenizers 库直接生成。为了方便阅读，我先从如何得到一个 tokenizer 讲起。

首先 `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` 会根据参数自动加载对应的 tokenizer，假如我们传入的是模型名称，例如 "bert-base-uncased"、"gpt2"，它会先请求 config.json，从中判断 tokenizer 类型（BertTokenizer、GPT2Tokenizer、LlamaTokenizer 等），然后下载 tokenizer 必需的文件到本地。假如我们传入的是一个路径，它就会直接从文件夹中读取 tokenizer 的核心文件，它包括：

- tokenizer_config.json：里面包含 special_tokens、是否自动在文本开头添加 `bos_token` 等等 tokenizer 的配置信息。
- 词汇表文件（二选一）
	- vocab.json + merges.txt
	- tokenizer.json：实际上就是把 vocab.json + merges.txt 放到一起了

那我们训练 Tokenizer 就是要得到 tokenizer.json 和 tokenizer_config.json 两个文件。

```python
VOCAB_SIZE = 6400
SAVE_PATH = "out"
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def data_iterator(
    datapath: Union[str, PathLike[str]], max_sample: Optional[int] = None
):
    with open(datapath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_sample is not None and i == max_sample:
                break
            data = json.loads(line)
            yield data["text"]


def train_tokenizer(datapath: Union[str, PathLike[str]]):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    iter = data_iterator(datapath)
    tokenizer.train_from_iterator(iter, trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(path.join(SAVE_PATH, "tokenizer.json"))
```

代码的核心就是 `tokenizer.train_from_iterator` 这个函数，我们提供一个数据集的迭代器还有一个训练器（例如这里我们用的是 BPE），就可以用 huggingface 提供的函数进行训练了。记得前面我们还说到，AutoTokenizer 除了 tokenizer.json 里面的词汇表和合并规则，还需要 tokenizer_config.json 的 tokenizer 配置信息，我们需要手动保存：

```python
config = {
    "add_bos_token": False,
    "add_eos_token": False,
    "add_prefix_space": False,
    "added_tokens_decoder": {
        "0": {
            "content": "<|endoftext|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        "1": {
            "content": "<|im_start|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        "2": {
            "content": "<|im_end|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
    },
    "additional_special_tokens": [],
    "bos_token": "<|im_start|>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|im_end|>",
    "legacy": True,
    "model_max_length": 32768,
    "pad_token": "<|endoftext|>",
    "sp_model_kwargs": {},
    "spaces_between_special_tokens": False,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "<|endoftext|>",
    "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}",
}

with open(os.path.join(SAVE_PATH, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(config, config_file, ensure_ascii=False, indent=4)
```

| 字段名                             | 解释                                                                        |
| ------------------------------- | ------------------------------------------------------------------------- |
| `add_bos_token`                 | 是否自动在文本开头添加 `bos_token`。                                                  |
| `add_eos_token`                 | 是否自动在文本末尾添加 `eos_token`。                                                  |
| `add_prefix_space`              | Byte-level 分词时是否在文本前加空格。通常英文中启用（True）更好，中文中设为 False。                      |
| `added_tokens_decoder`          | 特殊 token 的详细配置。包括 token 内容、是否为特殊 token、是否仅限单词等，key 是内部 token ID。          |
| `additional_special_tokens`     | 除了 `bos/eos/pad/unk` 外，额外声明的特殊 token 列表。当前为空。                             |
| `bos_token`                     | 起始 token，通常用于语言模型的开头控制符。                                                  |
| `clean_up_tokenization_spaces`  | 解码时是否清理 token 化带来的空格冗余。False 表示不清理。                                       |
| `eos_token`                     | 结束 token，通常用于语言模型输出结束的标记。                                                 |
| `legacy`                        | 设置为 `True` 兼容旧版本 `tokenizer` 行为。推荐保持默认。                                   |
| `model_max_length`              | 模型支持的最大 token 长度。超过将触发截断或报错。这里为 32768。                                    |
| `pad_token`                     | 用于对齐 padding 的特殊 token。                                                   |
| `sp_model_kwargs`               | SentencePiece 模型的额外配置参数（当前为 BPE，未使用，故为空）。                                 |
| `spaces_between_special_tokens` | 是否在特殊 token 之间自动添加空格，设置为 False。                                           |
| `tokenizer_class`               | 指定 tokenizer 类型。Hugging Face 使用 `"PreTrainedTokenizerFast"` 支持 Rust 实现加速。 |
| `unk_token`                     | 用于标记未知词（out-of-vocabulary）的 token。                                        |
| `chat_template`                 | Jinja2 模板字符串，用于格式化对话数据为模型输入格式。适用于 Chat 模型（如 LLaMA2-Chat、ChatGPT）。         |

## 预训练 - 单机

预训练的代码中包含了很多 tricks，这里先放上源码：

```python

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=340)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=762)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--from_resume", type=int, default=0)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--from_weight", type=str, default="none")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    return parser.parse_args()


def train_epoch(
    epoch: int, loader: DataLoader, iters: int, sta_step: int = 0, wandb=None
):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=sta_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = input_ids != tokenizer.pad_token_id
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with context:
            res = model(input_ids, attention_mask, labels)
            loss = res.loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            print(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        if step % args.save_interval == 0 or step == iters - 1:
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            state_dict = model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
            )
            model.train()
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    args = get_args()

    # 随机种子
    seed_everything(args.seed)

    # 配置目录、模型参数
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume
        else None
    )

    # 混合精度
    device = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    context = (
        nullcontext() if device == "cpu" else torch.amp.autocast(device, dtype=dtype)
    )

    # wandb
    wandb = None
    if args.use_wandb:
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # model & tokenzier
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        print("torch.compile enabled")
    train_ds = PreTrainDataset(tokenizer, args.data_path, args.max_length)
    scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # recovery
    sta_epoch, sta_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        sta_epoch = ckp_data["epoch"]
        sta_step = ckp_data["step"]

    # 开始训练
    for epoch in range(sta_epoch, args.epochs):
        loader = DataLoader(
            train_ds,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        it = iter(loader)

        if epoch == sta_epoch and sta_epoch > 0:
            for _ in range(sta_step):
                next(it)

            print(f"跳过前 {sta_step} 个 step。")
            train_epoch(
                epoch, it, total_steps=len(loader), sta_step=sta_step, wandb=wandb
            )
        else:
            train_epoch(epoch, it, total_steps=len(loader), sta_step=0, wandb=wandb)

```

### ckp & resume

大型模型的训练耗时非常久，难免会出现间断的情况，所以我们需要定时将模型当前的状态进行保存，并且可以恢复继续训练，类似“断点续传”。

- checkpoint：通常只包含模型的权重 `model.state_dict()`，用于推理或者作为预训练权重被别人加载
- resume：通常包含训练的全部信息，包含模型权重、optimizer 权重、epoch、step 等等，用来恢复间断的训练

```python
state_dict = model.state_dict()
state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
ckp_tmp = ckp_path + ".tmp"
torch.save(state_dict, ckp_tmp)
os.replace(ckp_tmp, ckp_path)
```

这里用到工程化的设计，先把 ckp 的数据写到临时文件里，然后再把临时文件转存到目标路径，可以避免意外情况下留下半个有问题的 checkpoint。

```python
resume_data = {
    "model": state_dict,
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "step": step,
    "wandb_id": wandb_id,
}
for key, value in kwargs.items():
    if value is not None:
        if hasattr(value, "state_dict"):
	        resume_data[key] = value.state_dict()
        else:
            resume_data[key] = value
```

前面说到 resume 里面需要保存完整的训练环境，除了前面 checkpoint 里面保存的模型权重，还需要优化器权重等等。

---

等到训练的时候就需要根据 args 判断是不是需要读取 resume 继续训练：

```python
ckp_data = (
    lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
    if args.from_resume
    else None
)
sta_epoch, sta_step = 0, 0
if ckp_data:
    model.load_state_dict(ckp_data["model"])
    optimizer.load_state_dict(ckp_data["optimizer"])
    scaler.load_state_dict(ckp_data["scaler"])
    sta_epoch = ckp_data["epoch"]
    sta_step = ckp_data["step"]
```

每隔 `args.save_interval` 或者每个 batch 结束，就需要保存当前环境：

```python
if step % args.save_interval == 0 or step == iters - 1:
    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        step=step,
        wandb=wandb,
        save_dir="../checkpoints",
	)
```

### 混合精度训练

>在 cs224n 的帖子中，我已经详细记录了混合精度训练的知识点，具体可见 [[cs224n-lecture12]]。

简单来说，大模型的参数如果都用 FP32 来提高计算精度，可能导致显存不够出现内存溢出。如果浮点数精度用 FP16，性能不会大幅度下降，但是存在参数超出 FP16 表示范围的问题。解决方式是：一方面分情况使用 FP32 或者 FP16（在容易导致溢出的操作使用 FP16）；一方面在反向传播时候把 loss 扩大，避免参数小于 FP16 的表示范围，然后在更新参数之前再把梯度缩小。

具体训练中我们一般遵循下面框架：
```python
scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16"))

with torch.amp.autocast(device, dtype=dtype):
	loss = model(input_ids, attention_mask, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

首先说说 `torch.amp.autocast` 这个上下文管理器，它在前向传播也就是计算 loss 的过程中，判断哪些地方需要 FP16 哪些地方需要 FP32。例如 matmul / conv / attention 这些不容易导致参数计算溢出的地方，就使用 FP16，在 softmax / norm / reduction 这些地方就采用 FP32。

并且我们需要注意，`torch.amp.autocast` 控制的是“算的时候用什么”，不是“存的时候用什么”。也就是说模型存储的参数，其类型就是我们代码里面预先规定的。而前向计算时候碰到 softmax 这样的操作，它们会被临时 cast 成 FP16，然后用 FP16 的 kernel 跑；如果碰到矩阵乘法，参与的 tensor 就会被 cast 为 FP32，然后用 FP32 的 matmul kernel 处理。

>既然没有改变全部参数存储的类型，比如规定 FP32 存储还是 FP32，那如何解决占用显存大的问题？在绝大多数现代模型里：参数 + optimizer state ≈ 30–40%，而激活值（activations）+ 中间结果 ≈ 50–70%，所以 AMP 能大幅度减小峰值显存占用。

在反向传播的时候 backward kernel 被调用，它只能用 forward 保存下来的 dtype，对于 softmax 这些操作 backward 自然也是 FP16。

`torch.amp.GradScaler` 是为了解决反向传播更容易梯度消失的问题 - 求偏导更容易导致参数或者梯度超过 FP16 的精度。`scaler.scale(loss).backward()` 可以在反向传播之前把 loss 乘上一定倍数，这样放大的 loss 求偏导就不容易梯度消失，最后 `scaler.step(optimizer)` 会先 unscale 梯度，然后检查 inf / nan，如果安全则 `optimizer.step()` 更新参数。

### 余弦衰退学习率

$$
\frac{\text{lr}}{10} + 0.5 \times \text{lr} \times (1+\cos(\pi \times \frac{\text{cnt\_step}}{\text{total\_step}}))
$$
根据余弦函数的性质，我们可以看出 lr 呈逐渐减小的趋势。前期提高模型学习率，后期避免在最优点附近震荡所有减小学习率。

### 梯度累积

我们都知道，深度学习里面相对大的 batch_size 比小 batch_size 训练，loss 曲线更平滑噪声更小。但是对于参数量如此庞大的大模型，采用大 batch_size 可能会爆显存。而梯度累积则是一种 quick fix，它在时间维度上模拟更大的 batch，多次前向 + 反向传播只更新一次参数。

具体的做法如下：

```python
for step, (x, y) in enumerate(dataloader):
	logits = model(x)
	loss = loss_fn(y, logits) / step_accumulations
	loss.backward()
	
	if (step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

>注意：当我们执行 backward()，会将小 batch 的平均梯度添加到参数的 .grad 上。假如我们没有让 `loss /= step_accumulations`，那么经过 k 轮我们会得到小 batch 平均梯度的 k倍，而不是大 batch 的平均梯度。

### 梯度裁剪

梯度裁剪是为了解决：在反向传播时梯度是链式相乘的，深层模型里可能出现 loss 或者梯度达到天文数字超过范围的情况。它可以在 `optimizer.step()` 之前，将梯度缩小。

在混合精度中，需要用 scaler 将梯度先缩放回去在计算梯度的范数：
```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```