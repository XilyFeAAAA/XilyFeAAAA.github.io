---
title: 分布式训练技术 - 张量并行
date: 2026-02-05T19:50:43+08:00
featuredImage: http://img.xilyfe.top/img/20260315231138396.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 分布式
lastmod: 2026-03-19T02:17:32+08:00
---
>上一篇文章我们学的是 DP 和 DDP，它们的思路是 **用显存冗余换吞吐量**。每张 GPU**都有完整模型**，但是只**处理不同的数据**，它的本质是**复制模型 → 并行处理数据 → 最后通过 AllReduce 同步梯度**，代价是模型被复制 N 份，占用 **N 倍显存**。张量并行 Tensor Parallel 的思路正好相反是 **用通信换显存**。现在的大模型参数量巨大一张卡很可能放不下，所以把模型拆到多卡，每张 GPU **只有部分模型**，但是**处理完整的数据**，最后进行合并。

## 引言

上一篇提到的 Data Parallel 的核心机制是：
- 把一个 **batch** 里的样本拆分到多个 DP worker 上。
- 每个 worker 拿到 batch 里的一个 micro-batch，独立做 forward + backward。
- 最后 All-Reduce 同步梯度。

但这存在一个关键前提：必须存在一个 batch dimension 才能把样本切开。它要求输入张量的形状为 \[bs, seq_len, dim], DP 切分的是第 0 维，每个 worker 拿到 \[bs/dp_size, seq_len, dim]。

而我们在处理我们在处理 LLM 训练任务，通常是 SFT，样本序列是变长的。我们通常采用两种办法：
1. Batching + Padding 模式：把一个 batch 内样本的序列都 padding 到最长，这样LLM的输入是 \[bs, max_seq_len, hidden_size]，然后通过 attention mask 对 PAD TOKEN 进行掩码
2. Packing 模式：在 vLLM 中为了提高计算效率，我们去掉了 batch 维度，把所有 batch 的 sequence 都压成了一个长序列，通过一个张量记录每个 sequence 的结束位置，这样就不用对变长序列插入 PAD 浪费算力

对于 Packing 模型，我们去掉了 batch 维度就无法采用 Data Parallel 了，这一章就研究一下 Tensor Parallel 是如何实现的。

## 核心思想

张量并行的核心思想是**将单层内的权重矩阵切分到多张 GPU 上，协同完成矩阵运算**，切分的方式有两种：按行切分权重和按列切分权重。

### 按列切分权重

![image.png](http://img.xilyfe.top/img/20260317124726586.png)

以一个线性层 $Y = XW$ 为例，如果 $W \in \mathbb{R}^{d \times d}$，可以将 $W$ 按列切分为两半：$W = [W_1, W_2]$，分别放在两张 GPU 上。每张 GPU 计算 $Y_i = XW_i$，得到输出的一半。最后拼接结果：$Y = [Y_1, Y_2]$。

![image.png](http://img.xilyfe.top/img/20260317125249472.png)

### 按行切分权重

按行切分权重矩阵，我们需要把输入张量 $X$ 也按列切开：

![image.png](http://img.xilyfe.top/img/20260317125055021.png)

对权重的梯度我们有：

$$
\frac{\partial L}{\partial W_i}= X_i^T \cdot \frac{\partial L}{\partial Y_i}
$$
对输入的梯度我们有：

$$
\frac{\partial L}{\partial X_i}
= \frac{\partial L}{\partial Y_i} \cdot W_i^T
= \frac{\partial L}{\partial Y} \cdot W_i^T
$$

![image.png](http://img.xilyfe.top/img/20260317125901490.png)

这里的 $X$ 代表的不是输入的 input_ids，而是上一层 Decode Layer 传来的中间值，所以我们需要对图中的 $X$ 求偏导。

{{< admonition type=question title="行切分和列切分有什么区别？">}} 
1. 对矩阵进行列切分，得到的输出为 $Y = [Y_1 \mid Y_2 \mid ... \mid Y_p]$，输出被切分了，每卡只有一部分。它的优点就是各个 GPU 之间不需要通信，计算完全独立。比如我们把线性层之后要接一个激活函数，各个 GPU 计算得到中间值的一部分之后，可以直接计算激活函数的值。但如果下一层需要完整的 Y 那么仍然需要通信。
2. 把矩阵按列切分，$X$ 也需要切分，每个 GPU 计算的都是 **部分贡献**，最终需要对他们进行求和才能得到完整的 $Y=\sum Y_i$，所以必须通过 AllReduce 进行通信。但优点是它的输出是完整的，下一层可以直接使用。

![image.png](http://img.xilyfe.top/img/20260319120446657.png)


{{< /admonition >}}

## Embedding 层

![image.png](http://img.xilyfe.top/img/20260317122603823.png)

Embeddings 的难点在于 weight 较大，需要拆分到多个设备上，并实现正确的lookup，下面以4张卡简述其实现步骤：
1. 将 wte 较均等分布到多张卡上
2. 将 input_ids 复制到所有卡上
3. 在每一张卡上input_ids分别lookup 卡上的子wte
4. 将所有卡上的值 all-reduce

## MLP 层

![image.png](http://img.xilyfe.top/img/20260317192106769.png)

在 MLP 里面我们采样对 $A$ 进行列切分，对 $B$ 进行行切分，为什么呢？
1. 假设我们全部采用行切分，那么 $A$ 和 $B$ 需要两个 AllReduce，通信量很大。
2. 假设都采用列切分，我们按照 `X → A → Y → B → Z` 的流程。第一层我们把 $A$ 矩阵切分为 $A_1$ 和 $A_2$，得到 GPU1 上有 $Y_1=X\cdot A_1$，GPU2 上有 $Y_2=X\cdot A_2$，目前还是正常的。但是第二层就有问题了，此时 GPU1 上有 $Y_1$，GPU2 上有 $Y_2$，然后我们把 $B$ 矩阵按照列切分，GPU1 上有 $Z_1=Y_1\cdot B_1$，GPU2 上有 $Z_2=Y_2\cdot B_2$，他们各自少了 $Y_2$ 和 $Y_1$，每个 GPU 只算了一半的贡献。

假如我们先把 $A$ 列切分，把 $B$ 行切分，那么就有：经过第一次按列切分 GPU1 有 $A_1$ 计算得到 $Y_1$， GPU2 有 $A_2$ 计算得到 $Y_2$。然后第二次按行切分，GPU1 有 $B_1$ 和 $Y_1$ 计算得到 $Z$ 的部分贡献，GPU2 有 $B_2$ 和 $Y_2$ 计算得到 $Z$ 的部分贡献，通过一次 AllReduce 将两者加在一起得到了完整的 $Z$。

## Attention 层

Self-Attention 的张量并行更简单，因为self-attention天然的是多头注意力机制，可以将每个头的计算分配到不同的 GPU 上。由于有多个头，可以考虑使用 head 的某个因子数(n)作为设备数，每张卡跑 $head//n$ 个头，那么问题就变成了如何拆分 weight 以及同步最终结果。

假设我们有 4 张 GPU，$W_q$、$W_k$、$W_v$ 矩阵可以被拆分为 4 块如下：

![image.png](http://img.xilyfe.top/img/20260317123731021.png)

然后我们把输入 $x$ 传到每一个 GPU 和拆分的权重矩阵进行矩阵乘法得到部分的输出，最后拼接起来。

## 通讯量

- MLP、Attention：forward 和 backward 各一次 AllReduce，AllReduce 分为 Reduce-Scatter 和 All-Gather 两个阶段，总通讯量为 $4\Psi$。
- Embedding：forward 部分每个 GPU 只负责 vocab 的一部分，lookup 不需要通信。而 backward 需要对 embedding weight 做梯度聚合是一次 AllReduce，所以通信量为 $2\Psi$。

## 具体实现

### Embedding

在介绍 Embedding 层之前说明一下 TP 里面出现的参数：
- `tp_size`：总 GPU 数量
- `tp_rank`：当前所在 GPU 编号

```python
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y
```

`__init__()` 中获取了当前所在的 GPU 编号，并且对词表进行了划分。然后在前向计算过程中：
1. 先计算掩码，把不在划分范围内的 token id 记为 False。
2. 然后对 token id 进行一个 shift 操作，把 $[a_1,a_2,\ldots,a_n]$ 移到 $[0, 1, \ldots, n]$，再应用 mask。
3. 得到 $y$ 之后还需要应用一下 mask，因为 mask 掉 token id 变成 0，0 对应的 embedding tensor 也要置为全零。
4. 最后通过 AllReduce 传出去。

### Lm_Head

>Lm_Head 就是 Embedding 的一个逆过程

```python
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        # x: [tokens, dim]
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

这里需要注意一下，在推理框架中我们通过 Continuous Batching 把多条序列被**直接拼接**成一个扁平的 tensor：

```
seq1: [t1, t2, t3]          →
seq2: [t4, t5]              →   [t1, t2, t3, t4, t5, t6, t7, t8]  shape: [8, dim]
seq3: [t6, t7, t8]          →
```

`cu_seqlens_q` 记录的就是边界，例如：`[0, 3, 5, 8]`。对于每条序列，我们只需要**最后一个位置**的 logits 来预测 next token，所以通过 `cu_seqlens_q[1:] - 1` 获得每个序列的最后一个 token 的位置。

最后把各个 GPU 计算汇合：
- `dist.gather` 将各卡的分片 logits 汇聚到 rank 0
- `torch.cat(..., -1)` 在最后一维（词表维）拼接，还原完整 `[batch, vocab_size]`
- 只有 rank 0 持有完整 logits，其余 rank 返回 `None`，上层调用方需注意判空

### Linear

```python
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
```

Linear 基类定义了一些初始参数，初始化了权重矩阵，然后 `tp_dim` 记录了这个 Linear 是列切分还是行切分。

```python
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
```

Linear 按照列切分之后它的形状完整，但是数值不完整，所以不需要进行 AllReduce。

```python
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

行切分之后需要进行一次 AllReduce，所有节点的 y 值加起来得到最终结果。

```python
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)
```

QKV 的计算方法就是把 `vocab_size` 投影到 `3*hidden_size`，然后在 dim 维度进行拆分得到 Q/K/V。

### MLP

```python
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
```

MLP 的第一个 Linear 是列切分，计算后每个 GPU 会得到一部分的结果：

$$
X * [W_{a1},W_{a2}]= [Y_1,Y_2]
$$

GPU0 得到 $Y_1$，GPU2 得到 $Y_2$。接着经过激活函数，会再次进行 Linear 的线性变换，这次是行切分。我们之前说过行切分需要对 $X$ 进行列切分，正好上一步计算每个 GPU 得到一半的 $Y$, GPU0 拿着 $Y_1$ 和 $W_{b1}$ 计算得到 $Y_1W_{b1}=Z_1$， GPU1 拿着 $Y_2$ 和 $W_{b2}$ 计算得到 $Y_2W_{b2}=Z_2$，最后 AllReduce 之后得到 $Z=Z_1+Z_2$。

{{< admonition type=question title="初始化一个模型为什么可以分布到多个 GPU 呢？">}} 
由于我们采用了 `import torch.distributed as dist`，在程序运行之后会启动 `tp_size` 个 Python 进程。每个进程绑定一张 GPU，从 `train.py` 第一行开始执行。假设我们的模型只有 MLP，那么每个进程都会执行 `model = Qwen3MLP(...)`。每个 GPU 都会输入完整的 x，然后用切分过得权重矩阵对他进行计算，最终得到不完整的 y，然后通过 AllReduce 所有 GPU 都得到了完整的 y，它们就可以继续下去了。
{{< /admonition >}}


## AllReduce 模拟

1. 定义行切分 Linear

```python
class RowParallelLinear(nn.Module):
    """
    权重 W [H_out, H_in] 按输入特征维（行）切分：
      rank k 持有 W[:, col_start:col_end]，形状 [H_out, shard_size]
    输入 X [B, H_in] 同样取对应特征列 X[:, col_start:col_end]
    各 rank 计算 partial sum，最终 All-Reduce 求和
    """
    def __init__(self, in_features: int, out_features: int,
                 tp_rank: int, tp_size: int, bias: bool = False):
        super().__init__()
        assert in_features % tp_size == 0, \
            f"in_features={in_features} 必须能被 tp_size={tp_size} 整除"
 
        self.in_features  = in_features
        self.out_features = out_features
        self.tp_rank      = tp_rank
        self.tp_size      = tp_size
        self.shard_size   = in_features // tp_size  # 每个 rank 负责的输入特征列数
 
        self.col_start = tp_rank * self.shard_size
        self.col_end   = self.col_start + self.shard_size
 
        # 本 rank 只持有权重的一个列分片 [H_out, shard_size]
        self.weight = nn.Parameter(torch.empty(out_features, self.shard_size))
        nn.init.xavier_uniform_(self.weight)
 
        # bias 只有 rank 0 持有，All-Reduce 后加，避免被重复累加 tp_size 次
        self.bias = nn.Parameter(torch.zeros(out_features)) \
            if (bias and tp_rank == 0) else None
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H_in] 完整输入，取本 rank 负责的特征列
        x_shard   = x[:, self.col_start:self.col_end]   # [B, shard_size]
        y_partial = x_shard @ self.weight.T              # [B, H_out]
        return y_partial  # 调用方负责 All-Reduce
```

2. 随机初始化输出和权重矩阵

```python
torch.manual_seed(42)
 
IN_FEATURES  = 6   # 输入特征维度（必须能被 TP_SIZE 整除）
OUT_FEATURES = 4
TP_SIZE      = 2
BATCH_SIZE   = 3

X = torch.arange(BATCH_SIZE * IN_FEATURES, dtype=torch.float32).reshape(BATCH_SIZE, IN_FEATURES)
 
print("=" * 60)
print(f"模拟开始: TP_SIZE={TP_SIZE}")
print(f"输入 X shape: {list(X.shape)}  [B={BATCH_SIZE}, H_in={IN_FEATURES}]")
print(f"输出期望 shape: [B={BATCH_SIZE}, H_out={OUT_FEATURES}]")
print(f"每个 Rank 负责的输入特征列数: {IN_FEATURES // TP_SIZE}")
print("=" * 60)
 

W_full = torch.arange(OUT_FEATURES * IN_FEATURES, dtype=torch.float32).reshape(OUT_FEATURES, IN_FEATURES) * 0.1
```

输出为：

```
============================================================
模拟开始: TP_SIZE=2
输入 X shape: [3, 6]  [B=3, H_in=6]
输出期望 shape: [B=3, H_out=4]
每个 Rank 负责的输入特征列数: 3
============================================================
```

3. 模拟 Tensor Parallel 过程

```python
partial_outputs = []
 
for tp_rank in range(TP_SIZE):
    print(f"\n{'#' * 20} 模拟 Rank {tp_rank} 的计算过程 {'#' * 20}")
 
    layer = RowParallelLinear(IN_FEATURES, OUT_FEATURES, tp_rank=tp_rank, tp_size=TP_SIZE)
 
    col_start = layer.col_start
    col_end   = layer.col_end
 
    # 从完整权重里取本 rank 的列分片，写入层权重
    W_shard = W_full[:, col_start:col_end]   # [H_out, shard_size]
    with torch.no_grad():
        layer.weight.copy_(W_shard)
 
    print(f"\nRank {tp_rank} 负责输入特征列范围: [{col_start}, {col_end})")
    print(f"权重分片 W_shard shape: {list(W_shard.shape)}")
    print(f"权重分片 W_shard:\n{W_shard}")
 
    # 取对应输入列
    X_shard = X[:, col_start:col_end]
    print(f"\nX_shard (X[:, {col_start}:{col_end}]):\n{X_shard}")
 
    # 局部矩阵乘法 → partial sum
    Y_partial = layer(X)   # 内部自动切 X_shard
    print(f"\nY_partial = X_shard @ W_shard.T:\n{Y_partial}")
 
    partial_outputs.append(Y_partial)
```

>这里注意一下，Linear 里面权重矩阵的形状是相反的，所以我们对 `W_shard` 进行列切分，计算时候转置就是行切分了。

输出为：

```
#################### 模拟 Rank 0 的计算过程 ####################

Rank 0 负责输入特征列范围: [0, 3)
权重分片 W_shard shape: [4, 3]
权重分片 W_shard:
tensor([[0.0000, 0.1000, 0.2000],
        [0.6000, 0.7000, 0.8000],
        [1.2000, 1.3000, 1.4000],
        [1.8000, 1.9000, 2.0000]])

X_shard (X[:, 0:3]):
tensor([[ 0.,  1.,  2.],
        [ 6.,  7.,  8.],
        [12., 13., 14.]])

Y_partial = X_shard @ W_shard.T:
tensor([[ 0.5000,  2.3000,  4.1000,  5.9000],
        [ 2.3000, 14.9000, 27.5000, 40.1000],
        [ 4.1000, 27.5000, 50.9000, 74.3000]], grad_fn=<MmBackward0>)

#################### 模拟 Rank 1 的计算过程 ####################

Rank 1 负责输入特征列范围: [3, 6)
权重分片 W_shard shape: [4, 3]
权重分片 W_shard:
tensor([[0.3000, 0.4000, 0.5000],
        [0.9000, 1.0000, 1.1000],
        [1.5000, 1.6000, 1.7000],
        [2.1000, 2.2000, 2.3000]])

X_shard (X[:, 3:6]):
tensor([[ 3.,  4.,  5.],
        [ 9., 10., 11.],
        [15., 16., 17.]])

Y_partial = X_shard @ W_shard.T:
tensor([[  5.0000,  12.2000,  19.4000,  26.6000],
        [ 12.2000,  30.2000,  48.2000,  66.2000],
        [ 19.4000,  48.2000,  77.0000, 105.8000]], grad_fn=<MmBackward0>)

```

4. AllReduce 聚合

```python
print(f"\n\n{'=' * 25} 模拟 All-Reduce 聚合 {'=' * 25}")
for i, p_out in enumerate(partial_outputs):
    print(f"\n来自 Rank {i} 的 partial sum:\n{p_out}")
 
final_output = torch.stack(partial_outputs).sum(dim=0)
print(f"\n聚合后的最终结果 (sum of all partial outputs):\n{final_output}")
```

输出为：

```
========================= 模拟 All-Reduce 聚合 =========================

来自 Rank 0 的 partial sum:
tensor([[ 0.5000,  2.3000,  4.1000,  5.9000],
        [ 2.3000, 14.9000, 27.5000, 40.1000],
        [ 4.1000, 27.5000, 50.9000, 74.3000]], grad_fn=<MmBackward0>)

来自 Rank 1 的 partial sum:
tensor([[  5.0000,  12.2000,  19.4000,  26.6000],
        [ 12.2000,  30.2000,  48.2000,  66.2000],
        [ 19.4000,  48.2000,  77.0000, 105.8000]], grad_fn=<MmBackward0>)

聚合后的最终结果 (sum of all partial outputs):
tensor([[  5.5000,  14.5000,  23.5000,  32.5000],
        [ 14.5000,  45.1000,  75.7000, 106.3000],
        [ 23.5000,  75.7000, 127.9000, 180.1000]], grad_fn=<SumBackward1>)
```

5. 验证结果

```python
print(f"\n\n{'=' * 28} 验证结果 {'=' * 28}")
 
print(f"\n完整权重矩阵 W_full:\n{W_full}")
 
ref = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
with torch.no_grad():
    ref.weight.copy_(W_full)
 
ref_output = ref(X)
print(f"\n标准 nn.Linear 计算结果:\n{ref_output}")
 
# 逐样本对比
print(f"\n--- 逐样本对比 ---")
for b in range(BATCH_SIZE):
    tp_vec  = final_output[b]
    ref_vec = ref_output[b]
    match   = torch.allclose(tp_vec, ref_vec, atol=1e-5)
    print(f"样本 {b}: TP={tp_vec.tolist()}  REF={ref_vec.tolist()}  {'✓' if match else '✗'}")
 
are_equal = torch.allclose(final_output, ref_output, atol=1e-5)
print(f"\n并行计算结果与标准 nn.Linear 结果是否一致: {are_equal}")
```

输出为：

```
============================ 验证结果 ============================

完整权重矩阵 W_full:
tensor([[0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000],
        [0.6000, 0.7000, 0.8000, 0.9000, 1.0000, 1.1000],
        [1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000],
        [1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000]])

标准 nn.Linear 计算结果:
tensor([[  5.5000,  14.5000,  23.5000,  32.5000],
        [ 14.5000,  45.1000,  75.7000, 106.3000],
        [ 23.5000,  75.7000, 127.9000, 180.1000]], grad_fn=<MmBackward0>)

--- 逐样本对比 ---
样本 0: TP=[5.5, 14.5, 23.5, 32.5]  REF=[5.5, 14.5, 23.5, 32.5]  ✓
样本 1: TP=[14.5, 45.099998474121094, 75.69999694824219, 106.29999542236328]  REF=[14.5, 45.099998474121094, 75.69999694824219, 106.30000305175781]  ✓
样本 2: TP=[23.5, 75.69999694824219, 127.9000015258789, 180.10000610351562]  REF=[23.5, 75.69999694824219, 127.9000015258789, 180.10000610351562]  ✓

并行计算结果与标准 nn.Linear 结果是否一致: True
```