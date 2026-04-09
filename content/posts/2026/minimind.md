---
title: MiniMind 学习指北
date: 2026-02-13T14:08:48+08:00
featuredImage: http://img.xilyfe.top/img/20260122134824760.png
authors:
  - Xilyfe
series:
  - 项目笔记
tags:
  - 大模型
  - 深度学习
lastmod: 2026-02-19T12:25:36+08:00
---


## 1. Tokenizer

### Tokenizer

>在 CS336 的笔记中我已经完整介绍了一个 Tokenizer 是如何训练并且读取的，详情可见 [cs336_assignment1](../cs336_assignment1)。

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

测试 Tokenizer 很简单，我们对 prompt 进行编码之后再解码，将它和原始 prompt 对比，一致则说明成功了：

```python
def eval_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "你是一个优秀的聊天机器人，总是给我正确的回应！",
            },
            {"role": "user", "content": "你来自哪里？"},
            {"role": "assistant", "content": "我来自地球"},
        ],
        tokenize=False,
    )
    print(f"输入文本: \n{prompt}")
    print(f"词表长度 {len(tokenizer)}")
    input_ids = tokenizer(prompt)["input_ids"]
    print(input_ids)
    print(f"encoded prompt length = {len(input_ids)}")
    decoded_prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"decoded prompt == raw prompt: {decoded_prompt == prompt}")
```

>Chat 模型并不是我们发送什么他就输入什么，它会把输入内容改写为一种带特殊标记的结构化格式（就是 tokenizer_config.json 里面 Jinja2 字符串模板）。所以真正的 input 应该写成 `tokenizer.apply_chat_template(prompt)`。

## 2. Model

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

![](http://img.xilyfe.top/img/20260122135047734.png)

### RMSNorm

#### 前身

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

#### 代码

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

### RoPE

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

### YaRN

#### 前言

在研究 YaRN 之前，我们需要回到 RoPE 的公式，看看它存在什么问题。
我们已经知道，RoPE 本质上是在每个 attention head 的偶数/奇数维度上做一个二维旋转，旋转角度是 $\theta = m \cdot \omega$ 。由于 $\omega_i = 10000^{-2(i-1)/D}$ ，所以：
- 在低维度（也就是i 比较小的维度） $\omega_i$ 比较大更接近于 1，它的旋转角度是 $m \cdot 1$。m 稍微变化一点，它的角度变化就非常明显。这种高频信号更适合捕捉短距离、精细的位置差异。
- 而在高纬度的时候 $\omega_i$ 比较小，频率就低，适合捕捉长距离、粗粒度的位置关系。

外推时会发生什么？
- 现在给模型一个长度为 4096 的句子。对于第 3000 个词会发生什么？
- 它的旋转角度 $\theta = m \cdot \omega$。由于 sin 和 cos 函数是周期性的，当 m 变得很大时，$m \cdot \omega$ 可能会绕回来。
- 当位置 m 超出预训练范围 L 时，比如 m=L+k，模型计算出的旋转角度可能和一个在预训练范围内的位置 m'(m'<L)的角度一模一样。
- 结果就是：模型会彻底混淆位置。它看到第 3000 个词，可能觉得它的位置和第 500 个词差不多，导致相对位置信息完全错乱，注意力计算也就随之崩溃，最终模型性能急剧下降。或者说：在高频维度 $\omega_d$​ 和 $j-i$ 大导致 $\Delta\theta_d$ 很大疯狂绕圈，cos/sin 落在训练中从没见过的组合。


#### 解决方案

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

### GQA


![](http://img.xilyfe.top/img/20260126212710505.png)

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


### FFN

FFN 全称是 FeedforwardNet，前馈神经网络。他主要是对输入进行升维，以及加入非线性变化来丰富神经网络的细节。

升维是一个数学的概念，达到三维之后已经超过我们的想像范围了毕竟难理解，我们可以用一个例子来类比一下。我们评价一个人可以从三个方面进行打分：人品、外貌、能力，但是对于全世界这么多的人，都只从三个维度进行评价， 就会太笼统了，可能有很多人评分相同。所以我们可以把评分指标升维，例如外貌就可以进一步细化为：身高、体重、长相、年龄等等。所以前馈神经网络对输入数据进行升维，可以理解为把数据里面的信息更细节化，加深神经网络的理解。

![](http://img.xilyfe.top/img/20260127134545216.png)

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
>![](http://img.xilyfe.top/img/20260127134821291.png)

### Model

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

>具体细节可见文章 [transformer库的基类](../hg_transformer)。

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

当处于训练模式的时候，由于因果语言模型的训练逻辑是 **预测下一个词**，所以我们需要把 logits 和labels 错位，将“模型的预测结果”和“真实的标准答案”对齐。

```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
```

当处于推理模式时候，由于 token 需要一个一个生成，我们每次只需要最后一个 token 的隐藏状态，所以可以指定 `logits_to_keep`，一般设为 1。

```python
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

{{< admonition type=info title="权重共享">}} 
由于输入 `embedding` 和输出 `lm_head` 是同一个空间，输入 embedding 是一个编码过程，把 token 变成向量。输出是一个解码过程，把向量变成 token 概率。如果不共享就会导致两个空间不对齐，训练很困难。

理论上 embedding 的权重矩阵为：

$$
W_{emb} \in \mathbb{R}^{V \times H}
$$

lm_head 的权重矩阵为：

$$
W_{out} \in \mathbb{R}^{H \times V}
$$

并且希望：

$$
W_{out} = W_{emb}^T
$$

但是由于 PyTorch 里面 Linear 是反着存的，`nn.Linear(in_features, out_features)` 存储的 shape 为 \[out_features,in_features]，所以代码里直接 `self.model.embedding.weight = self.lm_head.weight` 就好了。
{{< /admonition >}}

## 3. Pretrain

### 数据集

>预训练我们采用的是 Teacher-Forcing，所以需要的数据格式应该是偏移的 `input_ids` 和 `labels`。

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


预训练的代码中包含了很多 tricks，这里先放上源码：

```python
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=340)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=16)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_weight", type=str, default="pretrain")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--from_resume", type=int, default=0)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_test.jsonl")
    parser.add_argument("--from_weight", type=str, default="none")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser.parse_args()


def train_epoch(
    model: MiniMindForCausalLM,
    tokenizer: Tokenizer,
    epoch: int,
    loader: DataLoader,
    iters: int,
    sta_step: int = 0,
    wandb=None,
) -> None:
    start_time = time.time()
    for step, (input_ids, labels) in tqdm(enumerate(loader, start=sta_step), total=iters, desc=f"Epoch {epoch+1}"):
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
            tqdm.write(
                f"Epoch:[{epoch+1}/{args.epochs}]({step+1}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.3f}min"
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
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.lr}"
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

    assert tokenizer.pad_token_id is not None

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
        seed_everything(args.seed + epoch)
        loader = DataLoader(
            train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        
        it = iter(loader)
        if epoch == sta_epoch and sta_step > 0:
            for _ in range(sta_step):
                next(it)

            print(f"跳过前 {sta_step} 个 step。")
            train_epoch(
                model,
                tokenizer,
                epoch,
                it,
                iters=len(loader),
                sta_step=sta_step,
                wandb=wandb,
            )
        else:
            train_epoch(
                model, tokenizer, epoch, it, iters=len(loader), sta_step=0, wandb=wandb
            )
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


### 训练

继续我们模型、数据和预训练的脚步都准备好了，那肯定得跑一轮看看我们的模型是啥样了，于是我在 AutoDL 租了一张 RTX 5090 跑了两个小时。

![](http://img.xilyfe.top/img/20260210143025198.png)

这里我按照 MiniMind 推荐的参数 `hidden_size=768, num_hidden_layers=16`：104M 的模型，跑了两个 epoch 最终 loss 稳定在 1.6 上下我就不浪费马内了。

{{< admonition type=question title="为什么 LLM 预训练通常是一到两个 epoch？">}} 
1. 现代大模型不同于以前的深度学习，训练使用万亿级 token 的大规模语料，所以如果和以前 DL 一样用几十上百个 epoch 进行训练一定会过拟合。
2. 模型参数巨大，训练的成本很高
{{< /admonition >}}

## 4. Sft

前面我们已经进行了预训练，得到了一个只会续写的模型。这是因为我们预训练数据集的文本都是简单的一句话，然后通过加工我们得到了类似 "<|im_start|> 秦始皇的功绩包括统一货币、文字" 的文本，通过 Teacher-Forcing 它只能做到预测下一个 token，或者说只会机械接龙，不会对话。

SFT 全称是 Supervised-Finetune，也就是监督微调。我们通过 **对话文本** 的数据集在预训练的模型基础上进行训练，就能让模型学会对话。简单来说 SFT 和 Pretrain 者有以下区别：

1. Pretrain 的数据都是纯文本如 "今天天气很好..."，而 SFT 的数据集是对话如 {"user":"你好", "assistant": "你也好"}
2. Pretrain 会直接 tokenize 整个文本，而 SFT 会用 template 模板将对话拼接为 "<|im_start|>user 你好 <|im_end|><|im_start|>assistant 你也好 <|im_end|>" 这样的结构化文本。
3. 计算损失函数时候，Pretrain 是对每一个 token 计算损失，而 SFT 仅对 Assistant 部分计算损失

所以说 SFT 我们只需要对 Dataset 进行改进，训练方式还是之前的 Teacher-Forcing。

### 改造 Dataset

先来看看我们 SFT 的数据集是啥样的：

```json
{
	"conversations": [
		{"role": "user", "content": "hello!"},
		{"role": "assistant", "content": "hi!"}	
	]
}
```

在 MiniMind 实现里面，除了把 conversations 里面的对话组成 message，还加入了一个 system 的 prompt：

```python
    def prepare_message(
        self, conversations: list, random_threshold: float = 0.2
    ) -> list:
        assert len(conversations) >= 2

        SYSTEM_PROMPTS = [
            "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
            "你是minimind，一个小巧但有用的语言模型。",
            "你是一个专业的AI助手，请提供有价值的回答。",
            "你是minimind，请尽力帮助用户解决问题。",
            "你是一个可靠的AI，请给出准确的回答。",
            "You are a helpful AI assistant.",
            "You are minimind, a lightweight intelligent assistant.",
            "You are a friendly chatbot. Please answer the user's questions carefully.",
            "You are a knowledgeable AI. Try your best to provide accurate information.",
            "You are minimind, a small but useful language model.",
        ]

        message = []
        for turn in conversations:
            message.append({"role": turn["role"], "content": turn["content"]})

        if message[0]["role"] != "system" and random.random() < random_threshold:
            message = [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + message

        return message
```

根据 Qwen 和 Grok 所说，MiniMind 参数量较小，自身无法稳定维持"助手"的角色认知，如果没有 system prompt 模型容易角色混淆、生成无意义接龙、对模糊指令无法正确响应。稍后我会对加入和未加入 system prompt 的数据集的数据集进行对比测试。

之后我们就可以用 tokenizer 的 `apply_chat_template` 生成结构化的文本了:

```python
message = self.prepare_message(conversations)
input = self.tokenizer.apply_chat_template(
   message, tokenize=False, add_generation_prompt=False
)
input_ids = self.tokenizer(input).input_ids[: self.max_length]
input_ids += self.tokenizer.pad_token_id * (self.max_length - len(input_ids))
```

在模型的 forward 里面我们规定的训练时候会让 logits 和 labels 进行偏移，所以在数据集就不用处理了。如上面所说： 计算损失函数时候，SFT 仅对 Assistant 部分计算损失。所以我们需要把 labels 的其余部分置为 -100，这样求交叉熵损失时候设置的 `ignore_index=-100` 就会起作用了。

```python
def pad_labels(self, input_ids: list[int]):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i : i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

### 修改 SFT 脚本

SFT 的训练代码和 Pretrain 的代码几乎一致，我们 copy 之后修改一下超参数就好了。

| 超参数               | Pretrain    | SFT         | 主要原因                                  |
| ----------------- | ----------- | ----------- | ------------------------------------- |
| **Learning Rate** | 1e-4 ~ 1e-3 | 1e-6 ~ 5e-5 | SFT 要微调，避免破坏预训练权重；Pretrain 需要快速学习基础表示 |
| **Epochs**        | 1~3 epoch   | 3~20 epoch  | SFT 数据少，需要多次学习高质量样本                   |
| **Batch Size**    | 128         | 16          | Pretrain 数据多、序列短；SFT 数据少、序列长          |
| **Weight Decay**  | 0 ~ 0.01    | 0.01 ~ 0.1  | SFT 需要更强正则化防止过拟合小数据集                  |
| **Dropout**       | 0~0.1       | 0 或更低       | SFT 数据高质量，不需强 dropout                 |

### 评估

![image.png](http://img.xilyfe.top/img/20260213134345985.png)

MiniMind 提供了两种训练方式，由于我用的是 104M 的模型所以最终选择用 `sft_mini_512.jsonl` 加上 `sft_1024.jsonl` 进行两轮 SFT 训练，每轮进行 1.5 个 epoch，最终 RTX 5090 耗时 7 个小时，loss 稳定在 1.65。实验结果如下：

```text
💬: 你有什么特长？
🤖: 我是人工智能模型，没有个人特点或兴趣。但我可以提供信息、解答问题、生成文本、进行对话等服务。如果你有任何具体的问题或需要帮助，请随时告
诉我！

[Speed]: 26.26 tokens/s


💬: 为什么天空是蓝色的
🤖: 天空之所以呈现出蓝色，主要有以下几个原因：
1. 云层散射：太阳光中的各种颜色的光线在大气中的散射程度较大，使得我们看到的是蓝色。当阳光照射到地面时，蓝色光的波长比较短，容易被空气分子 
散射。而这些波长较短的光则更容易穿透大气层到达我们的视线。这就是为什么天空看起来非常蓝的颜色。
2. 天空的散射作用：由于空气分子会把光线分散到各个方向，导致不同颜色的光线混合在一起。当太阳光照耀在空气中时，空气分子将其中的水蒸气吸收并
将其分散得更快，从而让我们看到天空。
3. 日落或日出的时刻：随着太阳处于地球同一平面上升起，然后逐渐变暗。如果白天没有太阳能辐射出来，那么光线也不会完全失去可见度，最终呈现出红
、橙等暖黄色的颜色。因此，从明亮到深蓝等角度来看，天空呈现出了美丽而略带沉重的一面。
总的来说，天空呈现出一种蓝色的视觉效果，这是因为它的大气层散射效应使得光线在穿过大气时发生了散射，而不是被物体反射或吸收，因此它在天空中表
现为一个蓝色的色彩。

[Speed]: 33.96 tokens/s


💬: 解释一下"光合作用"的基本过程
🤖: 光合作用是植物通过吸收阳光、水和二氧化碳（CO2）等有机物并释放出氧气的过程。在大气中，叶绿体中的叶绿素能够将阳光转化为化学能，为生态系
统提供能量和营养物质。光合作用发生在太阳能的照射下进行的阶段，这个过程中产生了大量的光能，这些光子被储存起来来利用，用于构建生物链，促进着
森林覆盖率的提高以及对气候变化的影响。同时，光合作用也涉及到其他重要的生理和化学反应，如代谢和储存等。

[Speed]: 34.13 tokens/s


💬: 如果明天下雨，我应该如何出门
🤖: 如果您要在明天进行户外活动，建议选择一个有良好风景的地点，避免暴晒。同时，准备一把备用的伞或雨衣，以防万一导致天气突变。
如果你想在室内享受大自然的美丽，可以考虑搭建帐篷、喝杯热茶或者阅读书籍等。此外，您也可以选择携带一些简单的工具，比如太阳镜、毛巾、手电筒、
吸管等等。如果您想更多地融入大自然，可以选择种植一些盆栽植物和绿植来提高环境质量。
总之，选择一个具有美丽自然环境的地方非常重要，这样您的户外活动就能让你更加舒适愉悦地度过今天的阳光时光。

[Speed]: 35.02 tokens/s


💬: 比较一下猫和狗作为宠物的优缺点
🤖: 猫（猫和狗）与狗各有不同的优缺点。以下是一些常见的宠物：

1. **驯化性猫**：
   - 驯化性猫，对体型、毛发类型和行为都有敏感性，但它们需要人类的照顾和定期锻炼。虽然它们的天生灵活性使其成为家庭宠物，但饲养猫可能较小，
且在某些情况下可能会表现出不适或过热的情况。此外，许多人选择与猫相处，这取决于他们的日常活动水平。
   - 在一些文化中，猫被视为社交动物，而在其他文化中则被看作是照顾者。它们往往更容易照顾家中的成员，因为它们通常更易于训练。

2. **温顺而独立**：
   - 猫通常非常柔软且温顺，不会主动打盹或者表达自己的感情。相比之下，狗更加警惕，更愿意独自行动。猫也经常在夜间活动，尤其是在城市中心。
   - 有些人喜欢用猫抓板来捕食或抓住昆虫，这有助于减少宠物的压力，同时也能增强他们的心理健康。

3. **独立性猫**：
   - 猫通常比较独立，对于那些不那么孤单的狗来说，没有太多的互动时间。这种特质使得它们能够更好地适应各种生活环境。
   - 在很多文化中，猫通常被用来安慰人们，帮助减轻焦虑和压力，提供一种安全感和归属感。

4. **低维护成本**：
   - 如果宠物拥有较少的工作空间或经济实惠的环境，猫会更容易维护。这是因为狗通常更适合在相对稳定的工作地点生活，而不是家庭生活。

5. **长寿伴侣**：
   - 猫需要长时间的活动，因此长寿伴侣可以提供一个忠诚和陪伴的平台，让宠物感到舒适和幸福。同时，猫也会表现出耐心、好奇心和社交性，这些都增
加了它们在同龄人的生活中所需的时间。

总之，尽管猫和狗都是受欢迎的选择，但饲养猫和它们之间还有很多需要改进的地方，例如猫的数量、饮食习惯以及是否满足个性化的需求等。通过综合考虑
以上因素，我们可以得出结论，无论何时，一只猫和一只狗都可以在某种程度上满足他们的需求，并且在长期的陪伴下，它们的身影可能会有所不同。

[Speed]: 35.09 tokens/s


💬: 解释什么是机器学习
🤖: 机器学习是一种人工智能技术，它通过让计算机从数据中自动学习并改进任务的性能。这意味着系统可以利用大量的数据进行训练和优化，从而提高其性
能、准确性和可靠性。这种方法通常用于识别模式、分类、预测等场景中的各种情况和异常情况。常见的机器学习算法包括监督学习（无标签数据）、非监督
学习和强化学习（如深度学习和强化学习）。

[Speed]: 34.94 tokens/s


💬: 推荐一些中国的美食
🤖: 1. 北京烤鸭：北京是中国的传统美食，以皮脆肉嫩、味道独特而闻名。你可以去附近的糖葫芦小火锅店尝试一下。
2. 广东粤菜：广东地区拥有许多美食，例如糖鑰鱼火锅和广州炸酱面等。在这些餐厅里你可以品尝到正宗的火锅口味。
3. 福建火锅：福建是一个非常有特色的国际大都市，这里有各种不同口味的点心供你选择。你可以找到很多地方用餐并享受美味佳肴。
4. 山西铁板烧：山西铁板烧是一家位于南京市的古老火锅连锁店，提供多种口味的烧烤食物，同时还有许多精美的小吃，如“烤鸡卷”、“蒸饺子”和“火锅海鲜
”。

[Speed]: 35.35 tokens/s
```

可以看得出来效果有提升了，不过还是会出现幻觉，训练的数据集还是太小了。

## 5. Lora

### LoRA 是什么

PEFT 大致包含三类：Prompt-Tuning、Adapter-Tuning 以及 LoRA，而 MiniMind 里面采用的就是 LoRA 进行指令微调。
在 CS224N 的课程中已经学习了 LoRA 的原理，简单来说我们在经过 Pretrain 和 SFT 的模型基础上，对参数 $y=Wx$ 加上一个增量矩阵 $\Delta{W}$ 来微调模型，并且这个 $\Delta{W}$ 是通过 **低秩近似** 得到的，所以实际参数量远小于 $W$，计算开销小。具体可以看之前的笔记：

{{< link_ref "cs224n-lecture12" >}}

{{< link_ref "lora&qlora" >}}

### 实现细节

#### LoRA 模块

前面我们数学公式是 $y=Wx+\Delta Wx = Wx+BAx$，但是在 PyTorch里面如果我们用 `nn.Parameter()` 手动实现得写成：

```python
def __init__(self, in_features, out_features, r)
	self.A = nn.Parameter(torch.zeros(r, in_features))
	self.B = nn.Parameter(torch.zeros(out_features, r))

def forward(self, x):
	return x @ (self.B @ self.A).T
```

PyTorch 默认把特征维放在最后：输入形状是 `(batch, ..., in_features)`，这样所有前导维都当作批维自动广播，所以在 PyTorch 里面都是 x 右乘一个矩阵而不是像线性代数里面都是 $W \times x$ 这样左乘一个矩阵。

```python
class LoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: int) -> None:
        super(LoRA, self).__init__()

        self.scaling = alpha / rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x)) * self.scaling
```

如果直接用 `nn.Linear` 那么只需要先应用 A 再应用 B 就好了。

#### 应用 LoRA

```python
def apply_lora(model: nn.Module, rank: int, alpha: int) -> None:
    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # collect linear
    lora_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lora" not in name and "lm_head" not in name:
            lora_modules.append((name, module))

    for name, module in lora_modules:
        lora_module = LoRA(in_features=module.weight.shape[1], out_features=module.weight.shape[0], rank=rank, alpha=alpha).to(module.weight.device).to(module.weight.dtype)

        setattr(module, "lora", lora_module)
        ori_forward = module.forward

        def forward_with_lora(x):
	        return ori_forward(x) + lora_module(x)

        module.forward = forward_with_lora

```

为了避免将 LoRA 应用到 LoRA 层自身，我们对每一个 `nn.Module` 检测他的权重矩阵形状是否为 rank。其次需要注意我们初始化 LoRA 模块的时候，`in_features=module.weight.shape[1]` 。这是因为 `nn.Linear` 层内部初始化的权重矩阵是 `W=[out_features, in_features]`，然后计算 $y=xW^T$，所以 `in_features` 应该是权重矩阵的第二维。

这个代码看似没啥问题，但是我调试时候 debug 了半个多小时，最后还是 Gemini 帮我解决了。这是一个非常经典的 **Python 闭包** 导致的错误。闭包函数内的 `lora` 和 `ori_forward` 是从外部作用域“引用”的变量，它们指向的是循环结束时的“最后一个值”，而不是当前循环的值。所以当我们前向传播计算线性层的时候，它调用的 forward 方法其实都是最后一个线性层的 forward + lora。关于 Python 的闭包问题可以见下面这个文章，这里就讲一下怎么解决：

{{< link_ref "python-closure" >}}

解决方案有两种：
1. 我们用 **默认参数** 将当前的 `lora` 和 `ori_forward` 绑定到函数内部。

```python
def forward_with_lora(x, lora=lora, ori_forward=ori_forward):
    return ori_forward(x) + lora(x)
```

2. 使用 **工厂函数** 创建，每次调用都会生成新的闭包环境

```python
def _create_lora_forward(lora_module, original_forwarda):
    def forward(x):
        return original_forward(x) + lora_module(x)
    return forward

def apply_lora(model: nn.Module, rank: int) -> None:
    for _, module in model.named_modules():
        module.forward = _create_lora_forward(lora, module.forward, rank, rank*2)
```

#### 保存 LoRA

既然我们训练了 LoRA 模块，那就需要把里面的权重保存下来。我们之前用 `setattr(module, "lora", lora)` 把 LoRA 模块插入了 model 里面，所以 `lm_checkpoint` 方法通过 `model.state_dict()` 可以获得 LoRA 的权重。但是我们需要的是 LoRA 的 **可插拔** 的特性，所以只需要把 LoRA 的权重留下来即可，需要的时候把这部分权重挂载上去，所以我们需要再写一个方法来实现：

```python
def save_lora(model: nn.Module, path: str):
	state_dict = {}
	for name, module in model.named_modules():
		if hasattr(module, "lora"):
			tmp_state = {f"{name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
			state_dict.update(tmp_state)
	torch.save(state_dict, path)
```

#### 应用 LoRA

```python
def load_lora(model: nn.Module, path: str):
	state_dict = torch.load(path, map_location=model.device)
	for name, module in model.named_modules():
		if hasattr(module, "lora"):
			lora_state = {"A.weight": state_dict[f"{name}.lora.A.weight"], "B.weight": state_dict[f"{name}.lora.B.weight"]}
            module.lora.load_state_dict(lora_state)
```

### 实验结果

#### MiniMind 微调

```
💬: 你有什么特长？
🤖: 我是能够像人类一样思考和感知环境的智能机器。

[Speed]: 15.22 tokens/s


💬: 为什么天空是蓝色的 
🤖: 因为星星和太阳光会反射不同波长的光线，导致我们看到的是蓝色。

[Speed]: 21.41 tokens/s                                                                                          t Outlook 或者 Yaho.ai Gam
                                                                                                                  游的普通民众，也可以选择 

💬: 解释一下"光合作用"的基本过程
🤖: 蛋白质是植物和一些细菌利用阳光、水或其他生物能进行细胞呼吸，产生氧气并释放出能量。这个过程涉及到蛋白质的结构
、分子间相互作用以及酶催化等多种机制。在植物体内，氨基酸通过转录成RNA，然后是碳水化合物（O2）捕获二氧化碳，生成有
机物质；同时，还可以通过代谢途径将有机物质转换为营养物质来帮助植物生长和传递养分。当植物受到外部环境压力时，其代
谢产物会被激活，供应相应的水分和营养成分；这一过程中还有助于调节人体免疫系统，防止感染和疾病。总之，生物体内发生DNA聚合酶的分解和转运，是至关重要的。

[Speed]: 22.23 tokens/s


💬: 如果明天下雨，我应该如何出门
🤖: 确保你的衣服干净、干燥。你可以穿外套去学校或图书馆。如果你想带伞，但不会下雨的话就带上雨伞。

[Speed]: 20.53 tokens/s


💬: 比较一下猫和狗作为宠物的优缺点
🤖: 猫更喜欢自己独处，而狗则相对独立。但是它们的性格不同，有时候甚至会被驯服，但它们总是可以变得非常亲近人类，喜
欢在主人身边打扰自己的安宁睡觉。

[Speed]: 21.54 tokens/s


💬: 解释什么是机器学习
🤖: 机器学习（Machine Learning）是一种基于数据训练的算法，它能够通过分析和模式识别数据中的模式、趋势或关联性，从
而实现自动化决策或者预测未来结果的能力。这种技术被广泛应用于各种领域，如自然语言处理、计算机视觉、医疗诊断等，是
现代人工智能的重要组成部分。

[Speed]: 22.86 tokens/s


💬: 推荐一些中国的美食
🤖: 我推荐一下：宫保鸡丁、麻辣香锅、回锅肉、饺子皮等。这些都是非常受人们欢迎的美味哦，你可以尝试一下就可以了！

[Speed]: 22.49 tokens/s
```

训练了 20 个 epoch，可以看到模型已经有了自己的说话风格了。

#### Qwen3-0.6B 微调

但是，由于我的基模太拉跨了，所以我下载了 Qwen3-0.6B 模型进行 LoRA 微调。下面代码是手动通过 PyTorch 进行 LoRA 微调，调用 Transformers 库进行 LoRA 微调的方法可以见博文：

{{< link_ref "lora&qlora" >}}

具体代码如下：

```python
import argparse

import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.lora_dataset import LoRADataset
from model.lora import apply_lora, save_lora


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--batch_size", type=int, default="8")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--max_length", type=int, default=340)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--lora_path", type=str, default="lora")
    parser.add_argument("--lora_name", type=str, default="xiaoxue")
    return parser.parse_args()


def train(args: argparse.Namespace):
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    apply_lora(model, args.rank, args.alpha)

    train_ds = LoRADataset(tokenizer, args.dataset_path, args.max_length)
    dataloader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params} / {all_params} ({trainable_params / all_params:.2%})")

    step = 0
    model.train()
    for epoch in range(args.epochs):
        for input_ids, labels in tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            step += 1
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            mask = (input_ids != tokenizer.pad_token_id).bool()
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=mask)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_interval == 0:
                tqdm.write(f"Epoch:[{epoch + 1}/{args.epochs}], loss: {loss.item():.4f}")
            if step & args.save_interval == 0:
                save_lora(model, f"{args.lora_path}/{args.lora_name}_{args.rank}.pth")


if __name__ == "__main__":
    args = get_args()
    train(args)
```

模型输出如下：

```
💬: 你有什么特长？
🤖: 嗯...我有写故事的能力哦，能写出好多个有趣的故事呢！不过这可是我的天赋嘛（恼）

[Speed]: 10.99 tokens/s


💬: 为什么天空是蓝色的
🤖: 因为水分子在阳光下会散射，所以天空呈现蓝色！（答）

[Speed]: 13.29 tokens/s


💬: 解释一下"光合作用"的基本过程
🤖: 光合作用是一种植物、藻类和某些细菌等生物通过叶绿体中的色素吸收太阳光，将水分解为氧气，并释放出二氧化碳的过程。在这一过程中，植物利用阳光能量合成糖分并储存这些物质作为储藏库供以后使用。这一步是生命活动的基础之一！（注意：这个回
答可能有些误导性哦）

[Speed]: 13.17 tokens/s


💬: 如果明天下雨，我应该如何出门
🤖: 先看天气预报！再决定是否带伞。如果是晴天那就直接出发吧

[Speed]: 13.34 tokens/s


💬: 比较一下猫和狗作为宠物的优缺点
🤖: 优点：猫比狗更聪明，能感知到主人的情绪；而狗则有更强的责任感。但需要注意的是，在这种情况下，它们可能缺乏足够的互动时间来培养彼此的感情。因此在选择时需要权衡这些因素，并考虑是否能够为对方提供足够的情感支持。如果你们愿意投入更多
的时间与精力去建立情感联系的话，那么可以尝试将两者放在同一个家庭中，共同成长吧！

[Speed]: 12.99 tokens/s


💬: 解释什么是机器学习
🤖: 简单来说，就是让计算机通过经验来发现规律和模式，从而提高对新数据的理解与预测能力。例如，在分类问题中，训练模型时可以将大量已知的数据集分成训练集、测试集等部分，并用这些数据集上的信息去优化模型参数（即过拟合），最后就能在新的测
试集中做出更准确的判断了！

[Speed]: 12.85 tokens/s


💬: 推荐一些中国的美食
🤖: 嗯...中国有很多好吃的东西呢，比如饺子、包子、汤圆等等。不过这些应该都是比较普通的了喵！

[Speed]: 12.58 tokens/s
```

可以看出来数据集的语言风格已经可以模仿个 8-9 成了。

---

记录微调 Qwen3 时候出现的一个 BUG：`loss=nan`。

一开始看到 `loss=nan` 我的反应是加上梯度裁剪、修改学习率，然后还是爆 nan 了。之后我怀疑是不是精度的问题，把 float16 改成 bfloat16，然后加上混合精度训练还是爆 nan 了。由于模型是预训练的肯定没有问题，我的 LoRA 训练脚本之前也是 ok 的，所以我怀疑是不是数据有问题，于是在 debugger 里面对 LoRADataSet 进行步入。

我在 `__getitem__()` 方法里面断点时候，怀疑是不是 Qwen 的 tokenizer `apply_chat_template` 加入的模板和我 MiniMind 不同，导致对非 assistant 进行pad 时候出错。后面发现确实是 `pad_labels` 方法出错了，但问题不是模板不同，而是 Qwen3 的 tokenizer 没有设置 bos_token。我把代码改为：

```python
self.bos_id = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
# self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
```

训练就成功了，这次经验告诉我 `loss=nan` 可能是 **数据集/标签问题**。

## 6. Eval

>这一章我们需要设计一个脚本来验证大模型的对话能力

### 评估脚本

我们预训练是让模型学会说话的能力，或者说词语接龙的能力，给他一个 prompt 它可以接着说下去。因此我们在处理 prompt 时候需要稍加处理：

```python
inputs = tokenizer.bos_tok_id + prompt
```

这样我们就能得到一个类似 "<im_start> 今天天气" 这样的输入文本了。 下一步我们用 tokenizer 把输入文本变长 token 序列：

```python
input_ids = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
```

在 [MiniMind 学习指北(一)：模型](../minimind-model) 这一节中，我们的模型继承了 Transformers 库的 `GenerationMixin` 这个基类，所以我们可以很方便的调用它的 `generate` 这个方法来生成文本。与此同时为了让模型输出更流程，我们可以调用它的 TextStream 这个类来实现流式输出：

```python
model, tokenizer = init_model()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generated_ids = model.generate(
	inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"].bool(),
    max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
)
```

okey，现在我们就可以使用之前预训练得到的模型权重进行评估，不出所料又爆 BUG 了。。。

### 修改模型

![](http://img.xilyfe.top/img/20260210151653349.png)

我们先来看看这个 BUG 出在哪，Traceback 里面提示我们的保存 KVcache 的数组 `past_key_values` 是一个 **DynamicCache** 对象，不能用索引来取第一个元素。经过研究问题原来是 `GenerationMixin` 基类的 `generate` 方法会调用我们写好的 `forward` 方法，根据 `use_cache` 这个参数帮我们生成 `past_key_values`。然而新版本的 Transformers 库引入了 `DynamicCache` 作为 KV Cache 的标准实现，用于在自回归生成中高效管理注意力键值缓存，彻底替代旧版 `list[tuple]` 格式。

先了解一下 `DynamicCache` 里需要用到的方法：
- `get_seq_length(layer_idx: int) -> int`：获取第 i 层的 KV 长度
- `update(key: Tensor, value: Tensor, layer_idx: int) -> tuple`：类似之前的 `torch.concat` 帮我们更新 key 和 value，并且返回完整 key、value。

```python
class MiniMindModel(nn.Module):
	def forward(self, ...):
		# if past_key_values is None:
        #     past_key_values = [None] * len(self.layers)

        # sta_pos = 0 if past_key_values[0] is None else past_key_values[0][0].shape[2]
        if past_key_values is not None and use_cache:
            sta_pos = past_key_values.get_seq_length()
        else:
            sta_pos = 0
            if use_cache:
                past_key_values = DynamicCache()
	    #...
	    for idx, layer in enumerate(self.layers):
            hidden_states, past_key_values = layer(
                hidden_states,
                idx,
                position_embeddings,
                past_key_values,
                use_cache,
                attention_mask,
            )

class MiniMindBlock(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, past_key_values = self.attention(
            self.attn_norm(hidden_states),
            layer_idx,
            position_embeddings,
            past_key_values,
            use_cache,
            attention_mask,
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.ffn_norm(hidden_states))

        return hidden_states, past_key_values

class GQA(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        # if past_key_value is not None:
        #     key = torch.cat([past_key_value[0], key], dim=2)
        #     value = torch.cat([past_key_value[1], value], dim=2)
        # past_key_value = (key, value) if use_cache else None
        
        if use_cache:
            key, value = past_key_values.update(key, value, layer_idx)
```

### 评估结果

```
💬: 为什么天空是蓝色的
🤖: ？天空之所以呈现蓝色，是因为大气层中的氮气、氧气、水分子和其他气体分子会散射光线，使得天空看起来是蓝色的。

[Speed]: 38.49 tokens/s


💬: 请用Python写一个计算斐波那契数列的函数
🤖: 。函数可以使用任何一个函数来写，但是它需要一个大型的函数来存储，例如while()函数。因此，要编写程序时，要考虑到函数的复杂度，它需要大规模的函数来存储和处理数据。

[Speed]: 38.62 tokens/s


💬: 解释一下"光合作用"的基本过程
🤖: 。光合作用是指光能转化为化学能的过程。在光合作用中，植物、动物和微生物将太阳能转化为有机物，并释放出氧气。这个过程发生在光合作用的过程中，将二氧化碳和水转化为葡萄糖和氧气，同时释放出氧气。
光合作用的另一个基本过程是光合色素的合成。当光合色素通过光能被转化为化学能时，光合色素吸收光能并将其转化为化学能，从而生成葡萄糖和氧气。这个过程在光合作用中起到非常重要的作用，因为光合作用可以产生出植物和动物等光合生物所需的能量。

[Speed]: 36.80 tokens/s



💬: 解释什么是机器学习
🤖: ，并提供一个实际的应用。
机器学习是一种人工智能领域的算法和模型，通过训练计算机程序，使其能够从数据中学习，自动改进和适应，从而实现自动化决策和预测。      汽车，因为它们能够使汽车在道路上行驶，并提高行驶安全性。
机器学习是一种人工智能领域的算法和模型，通过训练计算机程序，使其能够从数据中学习，自动改进和适应，从而实现自动化决策和预测。      
一个实际应用的例子是图像识别。例如，利用深度学习技术，例如卷积神经网络（CNN）可以识别图像中的物体，并将其分类为正面或负面。       
一个实际的应用是自动驾驶汽车。自动驾驶汽车通过使用传感器和算法来实现自主驾驶，如激光雷达、激光雷达和摄像头。这些技术可用于自动驾驶
一个实际的应用是自动驾驶汽车。自动驾驶汽车通过使用传感器和算法来实现自主驾驶，如激光雷达、激光雷达和摄像头。这些技术可用于自动驾驶
汽车，因为它们能够使汽车在道路上行驶，并提高行驶安全性。

[Speed]: 36.99 tokens/s
```

可以看到模型在接龙方面还过得去，下一步我们需要进行监督微调 SFT 进一步提高模型对话能力。

## 7. Save

>在之前的训练中，我们都是通过 `torch.save()` 来保存模型的权重，我们是可以通过训练脚本来加载并且评估，但是在推理框架上这种权重就无法直接使用，所以这篇记录一下如何保存和使用 Huggingface 格式的模型。

在深度学习中，模型通常会以不同的格式保存，不同框架或工具链会使用不同的标准。最基础的是 **PyTorch 原生格式**。这种方式通常通过 `torch.save` 保存模型权重或整个模型对象。例如我们之前就是通过 `model.state_dict()` 获得模型的权重，然后调用 `torch.save()` 保存。这种格式的特点是非常灵活，可以保存任何 Python 对象，例如模型权重、优化器状态、训练轮数等，例如：

```python
 resume_data = {
    "model": state_dict,
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "step": step,
    "wandb_id": wandb_id
}
torch.save(resume_data, "checkpoint.pth")
```

但缺点是缺乏统一标准，不同项目之间很难直接复用。

>保存的后缀名是 `.pt` 或者 `.pth` 没有影响，每个人习惯不同。


另一类是 **Hugging Face Transformers 标准格式**。这是目前大模型生态最常见的一种格式，由 Hugging Face 在其库 Transformers 中定义。模型通过 `save_pretrained` 保存，通过 `from_pretrained` 加载。这种格式的优点是统一规范，很多推理框架都可以直接使用，例如 vLLM。HuggingFace 格式的模型结构如下：

```
my_model/
 ├── config.json
 ├── model.safetensors
 ├── tokenizer.json
 ├── tokenizer_config.json
 ├── special_tokens_map.json
 └── generation_config.json
```

- **config.json**：这是模型的结构配置文件，包含模型架构信息，例如层数、隐藏层维度、attention head 数量等。加载模型时，Transformers 会先读取这个文件，再构建模型结构。
- **model.safetensors**：这是 Hugging Face 推出的新型权重格式，主要解决安全性和加载速度问题。很多模型仓库已经从 `pytorch_model.bin` 迁移到 `model.safetensors`。
- **tokenizer.json**：Tokenizer 的相关信息，涵盖了 `merges.json` 和 `vocab.json`

HuggingFace 中 `save_pretrained` 保存的模型包含了完整信息（结构+权重+tokenizer），各种推理框架可以直接加载。

```python
model.save_pretrained(PATH, safe_serialization=True)
config.save_pretrained(PATH)
tokenizer.save_pretrained(PATH)
```

{{< admonition type=info title="注意点">}} 
如果要让模型变成 Huggingface Transformers 兼容模型需要进行一定修改。

1. config 类需要继承 `transformers.PretrainedConfig`，model 需要继承 `transformers.PreTrainedModel`
2. model 内部必须定义类成员 `config_class`

同时我们需要注意，例如 vLLM、sgLang 的推理框架内部都事先存储了有名的开源模型结构，比如：Llama、Mistral 或者 Qwen 等等。如果 architecture 是我们自己定义的，我们需要给 vLLM 写 custom model loader。
{{< /admonition >}}

---
