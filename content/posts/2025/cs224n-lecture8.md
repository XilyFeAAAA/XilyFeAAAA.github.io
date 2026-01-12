
---
title: "Lecture 8: Self-Attention and Transformers"
date: '2025-11-24T11:24:11+08:00'
authors: [Xilyfe]
series: ["CS224N"]
tags: ["深度学习"]
--- 

## Self-Attention

### Key, Query, Value

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/self-attention.png" width="70%" />
</div>

从单条文本来看，矩阵 $x=Ew_t$ 形状为 [SeqLen, EmbeddingSize]。

**1. 用权重矩阵 Q,K,V 转换词向量**

$$
\begin{align}
q_i &= Qx_i \\
k_i &= Kx_i \\
v_i &= Vx_i
\end{align}
$$
> 为了提高计算效率，可以把 Q,K,V 合并为一个大矩阵和 x 相乘，然后再把各个部分取出来，类似 BiLSTM 中四个门控计算合并到一个 [4*H, B] 的大矩阵，然后再通过 `.chunk` 分开。

**2. 用输入 x 去查询其他 token 的 Key**

$$
\begin{align}
e_{ij} &= q_i^Tk_j \\
\alpha_{ij} &= \frac{\exp(e_{ij})}{\sum_{j'}{\exp(e_{ij'})}}
\end{align}
$$

用词向量的 Query 去查询其他词向量的 Key 得到它们的相似度，具体数学运算就是向量的点积，最后对每个 token 的相似度做z softmax 就能得到相似权重。

我们把 $e_{ij}$ 拆开来看:

$$
e_{ij} = q_i^Tk_j = x_i^TQ^TKx_i
$$

---

Q：为什么不把 $Q^TK$ 直接用一个矩阵表示呢，还有乘两次这么麻烦？  
A：因为提前把它们合并成一个矩阵，那么这个矩阵的大小也会变成 [SeqLen, SeqLen] 意味着矩阵大小依赖于输入序列的长度，参数矩阵可能会变得非常大。

---

Q：为什么采用点积计算两个向量的相似度？不采用余弦相似度其他的？

---

Q：这种拆解矩阵的低秩表示会不会对最终结果有影响？
A：实验证明模型并不需要高秩的注意力矩阵，低秩反而是归纳偏置，有助于泛化。

**3. 计算 x 的输出**

将每个 token 的权重乘上它对应的 Value 得到加权和就是最后 x 的输出，这个输出值代表 x 在句子中的上下文含义：

$$
o_i = \sum_i{\alpha_{ij}v_j}
$$

### 自注意力机制的问题和解决方法

#### 1. 不知道文本顺序

自注意力机制和 RNN 或者 LSTM 不同，RNN 是从左到右或者右到左来更新隐藏状态矩阵的，而自注意力机制各个 token 之间不相互依赖，可以进行并行操作，但带来的缺点就是 模型不知道 token 之间的先后关系，例如: I love you 和 You love me 在它看来是一样的。

解决方案是：

**正余弦位置编码**

$$
\begin{align}
PE(pos, 2i) &= \sin(\frac{pos}{10000^{2i/d_{model}}}) \\
PE(pos, 2i+1) &= \cos(\frac{pos}{10000^{2i/d_{model}}})
\end{align}
$$

- 优点: 不需要学习，可以泛化到更长的序列,例如模型只训练到 512 长度，推理时给 4096 → 仍可用
- 缺点：表达能力有限

**可学习位置编码**

定义一个和 x 大小相同的可训练矩阵，类似:

```python
self.position_embedding = nn.Embedding(seqlen, embedding_size)
```

直接把 x 和 position_embedding 叠加，让模型训练学习位置信息。缺点在于无法extrapolation，训练 max_len=512，推理时来个 2048 没 embedding 只能报错。

#### 2. 没有非线性因素

可以在 self-attention 层计算各个 token 注意力分数之后通过一个前馈网络 MLP:

$$
\begin{align}
m_i &= MLP(output_i) \\
    &= W_2 * ReLU(W_1 output_i + b_1) + b_2
\end{align}
$$

#### 3. 不能窥视后续文本

以 "I love you" 生成 "我爱你" 为例， 在 Encoder 上生成 "爱" 的时候，我们只能看到之前的 "\<sta>"；在生成 "爱" 的时候只能看到之前的 "\<sta>我"，之后的 token 对当前 token 来说应该是 invisible 的。但是之前的自注意力机制中计算注意力分数，我们直接整个矩阵相乘了，也就是说任意 token 都能窥视之后的信息，解决办法就是利用掩码：

$$
e_{ij}= 
\begin{cases}
q_i^Tk_j, & j <= i \\
-\infty, & j > i
\end{cases}
$$

这样经过 softmax 之后，掩码位置都会变成 0，和 Value 相乘就不会影响模型。


## Transformer

### 多头注意力

多头注意力的出现是为了解决单头注意力机制的表达能力瓶颈。

人处理语言时会同时从多个角度去看的：

- 一个角度看语法结构（主语-谓语关系）
- 一个角度看语义相似度
- 一个角度看指代消解
- 一个角度看世界常识

单头必须把这所有线索压缩到一个 $d_k$ 维的空间里去学太勉强了，容易学偏或者学模糊。

多头注意力就是把 $d_k$（比如 512）维分成 h=8 份，每份 64 维，让 8 个空间各自专心学一种关注模式，例如：

- Head 1：专门学语法结构（比如主语更关注谓语）
- Head 2：专门学短距离依赖（相邻词关注多一些）
- Head 3：专门学长距离依赖（句首和句尾关注）
- Head 4：专门学指代关系
- Head 5：专门学情感极性

每个 head 都在一个低维子空间（64维）里独立学习一种“关注策略”，互不干扰。最后再把 8 个 head 的结果拼接起来，经过一个线性层融合，就相当于模型同时从 8 个不同角度理解了这个句子。

### 缩放点积注意力

注意力机制存在一个问题就是：**当模型规模变大之后，向量的点积也会随之变得很大**

---

**数学证明**：

首先我们要明确 Softmax 是一个 **n → n 的函数**，所以它的梯度不是一个数而是一个雅可比矩阵：

$$
J_{ij}=\frac{\partial{y_i}}{\partial{x_j}}
$$

根据链式法则有:

$$
\frac{\partial{L}}{\partial{x_j}}=\sum_{i=1}^n{\frac{\partial{L}}{\partial{y_i}}\frac{\partial{y_i}}{\partial{x_j}}}
$$

给定输入向量:

$$
\mathbf{x} = (x_1,x_2,x_3,\dots)
$$

softmax 之后得到输出的第 i 个分量为:

$$
\begin{align}
\mathbf{y_i}&=softmax({\mathbf{x}}_i)=\frac{e^{x_i}}{\sum_{k=1}^n{e^{x_k}}} \\
			&=\frac{e^{x_i}}{S}
\end{align}
$$

求偏导得到:

$$
\begin{align}
\frac{\partial{\mathbf{y_i}}}{\partial{\mathbf{x_i}}}&=\frac{e^{x_i}S-e^{x_i}e^{x_i}}{S^2}\\
&=\frac{e^{x_i}}{S}(1-\frac{e^{x_i}}{S})\\
&=y_i(1-y_i) \\
\frac{\partial{\mathbf{y_i}}}{\partial{\mathbf{x_j}}}&=-\frac{e^{x_i}e^{x_j}}{S^2} \\
&=-y_iy_j
\end{align}
$$

当 softmax 的输入某个 $x_i$ 特别大的时候，经过 softmax 就会变得接近 one-hot 编码，这时候最大项的梯度 $p_i(1-p_i)=0$ ，其他项梯度 $p_ip_j=0$，最后累加起来就是 0 了。

---

解决办法：**缩放点积注意力**

$$
Attention(\mathbf{Q},\mathbf{K},\mathbf{V})=softmax(\frac{\mathbf{QK}^T}{\sqrt{d_k}})V
$$

把 $QK^T$ _除以 $\sqrt{d_k}$_，能把它的方差重新拉回到一个合理的范围（大约接近 1）。

### 残差连接

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/residual.png" width="70%" />
</div>

梯度回传时候，链式计算可能会乘非常多的偏导。如果之间的偏导都是小于 1 的数，就有可能导致最后梯度接近于 0；反之梯度有可能非常大。梯度爆炸比较好解决，超过一定值的时候就把它裁剪掉。ResNet 的残差连接和 $1*1$ 卷积，可以解决梯度消失这个问题。

$$
Out(x)=f(x)+x
$$

残差连接解决的是**退化问题**:

从数学上看，假设一个浅层网络（比如 10 层）已经能很好地拟合一个函数 H(x)，我们希望再把网络加深 40 层得到更好的性能。假如我们让加深 40 层后的网络性能保持不变，那么这最少 40 词的非线性嵌套做的事就是学会一个恒等映射，这样准确率才不会下降。但是作为一个深层结构参数间高度耦合的网络，它很难学会恒等映射

$$
H(x) = x
$$

因为你要让几十层参数全部精确地抵消成“什么都不干”太难了，优化器几乎做不到。所以我们退而求其次，希望能学会这样一个函数:

$$
F(x) = H(x) - x
$$

这样让模型学会把输出变成全 0，比学会把输出精确等于输入 x 容易很多。

### 层归一化

先明确一点，Lay Normalization 是**在最后一个维度进行归一化**。换句话说，在 Transformer 中 Lay Norm 是对每个 token 对应的隐藏状态向量 $h_i$ 进行归一化，它是不依赖于 Batch Size 或者 Sequence Size 的。

$$
\begin{align}
\mu_i &= \frac{x_{i1} + x_{i2}}{2} \\
\sigma_i^2 &= \frac{(x_{i1}-\mu_i)^2 + (x_{i2}-\mu_i)^2}{2} \\
\hat{x}_{ij} &= \frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2 + \epsilon}}
\end{align}
$$


> Attention 机制在归一化阶段希望“先让每个 token 自己内部整洁”，再去交流。

---

Q：为什么需要对 feature 维度进行归一化
A：feature 数值差太大 → 在线性层中被放大或压缩

假设你有输入特征向量：$x = [1000,\; 0.1]$，经过一个线性层：$y = Wx$。其中

$$
\begin{array}{c} W = \begin{bmatrix} w_1 & w_2 \\ w_3 & w_4\end{bmatrix} \end{array}
$$
计算：

$$
y_1 = w_1\cdot1000 + w_2\cdot0.1
$$

不管权重 $w_2$ 多努力，最终贡献都会被 $1000 * w_1$ 盖掉。

### 源码解析

#### 一、Transformer

```python
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

```


整体来说 Transformer 包含 Encoder、Decoder 和投影三部分，Encoder 负责学习输入，Deocder 负责生成输出，最后把输出向量投影到期望的空间。

**注意！！！**

因为 PyTorch 的交叉熵损失要求输入形状如下：

```math
input:    [N, C]
target:   [N]
```

其中：

- N = 样本数量
- C = 类别数量 = vocab_size

而 Transformer 解码器输出的 logits 是：\[B, T, V]，所以需要展平到二维。

#### 二、Encoder

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/encoder.png" width="60%" />
</div>

```python
class Encoder(nn.Module):
    """
    - Word Embedding
    - Position Embedding
    - MultiEncoderLayer
    """    
    def __init__(self, vocal_size: int,  d_model: int,  layer_nums: int,  max_len: int,  hidden_size: int, n_heads: int = 8, p: float = 0.1):
        super(Encoder, self).__init__()
        self.word_embedding = nn.Embedding(vocal_size, d_model)
        self.pos_embedding = PositionEmbedding(max_len, d_model, p)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, hidden_size, p) for _ in range(n_layers)])
        
        
    def forward(self, x):
        attns = []
        enc_masks = padding_mask(x, x)
        enc_embed = self.word_embedding(x)
        enc_embed = self.pos_embedding(enc_embed)
        
        enc_output = enc_embed
        for layer in self.layers:
            enc_output, enc_attn = layer(enc_output, enc_masks)
            attns.append(enc_attn)

        return enc_output, attns
```

Encoder 包含 Word Embedding、Position Embedding 和多头注意力层，矩阵形状变化如下:

1. 输入为 batch_size 行文本，文本长度固定为 seq_len：\[batch_size, seq_len]
2. 通过词嵌入之后得到: \[batch_size, seq_len, d_model]
3. Position Embedding 大小和输入相同直接叠加: \[batch_size, seq_len, d_model]
4. 多头注意力机制不改变矩阵形状，最后还是 \[batch_size, seq_len, d_model]

> 严格来说 Encoder 和 Decoder 返回注意力权重 `attn` 没有什么用，不过它可以让我们查看模型在翻译时关注了源句子的哪些词。

##### 1. Word Embedding

词嵌入就是简单的使用了 torch.nn.Embedding 这个自学习矩阵，它的形状是 \[src_vocal_size, d_modal]，根据输入文本 token 的索引就能从中取出对应的词向量。

##### 2. Position Embedding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```

正余弦编码的公式是：

$$
\begin{align}
PE(pos, 2i) &= \sin(\frac{pos}{10000^{2i/d_{model}}}) \\
PE(pos, 2i+1) &= \cos(\frac{pos}{10000^{2i/d_{model}}})
\end{align}
$$

通过数学性质：

$$
\exp(a) = e^a，\text{而}a^b = \exp(b \ln a)
$$

所以有：

$$
\frac{pos}{10000^{2i/d_{model}}}=pos \times \exp{-\frac{2i}{d_{model}}\ln{10000}}
$$

---

**为什么要把 position 从 \[max_len] 扩展到 \[max_len, 1]?**

从公式来看，我们要做的事就是把每条句子的奇数和偶数位置的值替换掉，通过 pytorch 的切片操作 `tensor[:, 0::2]` 就可以用一个 \[seq_len, d_model/2] 形状的矩阵替换掉偶数位置的值。同时利用 pytorch 的广播机制，把 position 扩展到 \[seq_len, 1] 和 div_term \[d_model/2] 相乘就能得到 \[seq_len, d_model/2] 的新矩阵。

> PS: 我感觉这个计算方法真的很诡异，div_term 的维度是 \[d_model/2, ]，我们可以看做是一个长为 d_model/2 的行向量。然后和 position 主元素相乘之后，广播到 \[seq_len, d_model/2] ，这时候可以看做 d_model/2 的行向量进行了一个转置变长 d_model/2 的列向量，然后水平方向扩展为 \[seq_len, d_model/2] 的矩阵。

**用一个实际数字例子看广播计算**

假设 position（5×1）:

```
[[0],
 [1],
 [2],
 [3],
 [4]]
```

div_term（视为 1×3）：

```
[ a  b  c ]
```

计算 position * div_term：

```
[[0*a, 0*b, 0*c],
 [1*a, 1*b, 1*c],
 [2*a, 2*b, 2*c],
 [3*a, 3*b, 3*c],
 [4*a, 4*b, 4*c]]
```

---

最后前向计算中也是利用到了广播机制，x 的形状是 \[seq_len, batch_size, d_model], pe 的形状是 \[seq_len, 1, d_model]， 保持 x 的形状不变。

> Pytorch 的广播机制为：如果两个张量维度不一致，则在较短张量前面补 1。如果维度一致，则从右往左对比，如果两个维度一致或者有一个为 1 则可以广播。例如 \[1, 2, 3] 和 \[3, 2, 1]，从右往左：有一个 1，两个都是 2，有一个 1，最后广播为 \[3, 2, 3], 每个维度是两个 shape 各自的最大值。

##### 3. EncoderLayer

```python
class EncoderLayer(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        hidden_size: int,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForwardNet(d_model, hidden_size, dropout)
        
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        self.pre_norm = pre_norm
    
    def forward(self, x, attn_masks=None):
        
        if self.pre_norm:
            # Pre-LN: norm -> sublayer -> dropout -> residual
            norm_x = self.attn_norm(x)
            context, attn = self.mha(norm_x, norm_x, norm_x, attn_masks)
            x = self.dropout(context) + x
            
            norm_x = self.ff_norm(x)
            ff_out = self.ff(norm_x)
            x = self.dropout(ff_out) + x
            
            return x, attn
        else:
            # Post-LN: sublayer -> dropout -> residual -> norm
            context, attn = self.mha(x, x, x, attn_mask)
            x = self.attn_norm(x + self.dropout(context))

            ff_out = self.ff(x)
            x = self.ff_norm(x + self.dropout(ff_out))
            return x, attn
```

- pre_norm 模型一般比 post_norm 更稳定，易于训练。这在深层Transformer中尤为明显。
- post_norm 在较浅层时效果尚可，但模型越深，梯度消失或爆炸问题可能更严重。

##### 4. MultiHeadAttention

```python
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, p: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        

        self.W_Q = nn.Linear(d_model, n_heads * d_k)
        self.W_K = nn.Linear(d_model, n_heads * d_k)
        self.W_V = nn.Linear(d_model, n_heads * d_k)
        self.fc = nn.Linear(n_heads * d_k, d_model)
        
        # self.norm = nn.LayerNorm() Laynorm 统一放在 EncoderLayer
        self.attn_dropout = nn.Dropout(p)
        self.proj_dropout = nn.Dropout(p)
        
    def forward(self, q, k, v, masks = None):
        batch_size, seq_len, _ = q.size()
        Q = self.W_Q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        
        if mask:
            scores = scores.masked_fill(mask, float('-inf'))
        
        
        attn = nn.Softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        
        context = torch.matmul(attn, V).transpose(1, 2)
        context = context.reshape(batch_size, seq_len, self.n_heads * self.d_k)
        context = self.fc(context)
        context = self.proj_dropout(context)
        
        # return self.norm(context + q, -1), attn
        return context, attn
```

**前馈计算中的形状变化**

1. 输入的 q,k,v 形状都是三维矩阵 \[batch_size, seq_len, d_model]
2. 经过自学习矩阵 W_Q,W_K,W_V 变化之后得到 Q,K,V \[batch_size, seq_len, n_heads*d_k]
> 其实形状没变，因为 Transformer 论文中规定 $d_{model}=n\_heads \times d_k$
3. 将 n_heads 从矩阵第三维拉出来 \[batch_size, seq_len, n_heads, d_k]
4. 我们拉出 n_heads 是为了利用矩阵的形状同时计算多个注意力头，所以把 "头" 的维度拉到前面 \[batch_size, n_heads, seq_len, d_k]
5. 根据公式 $softmax({\frac{QK^T}{\sqrt{d_k}}})V$ 计算得到 \[batch_size, n_heads, seq_len, d_k]
6. 最后将形状变回去 \[batch_size, seq_len, n_heads, d_k] -> \[batch_size, seq_len, n_heads * d_k] -> \[batch_size, seq_len, d_model]

对形状变化还是不理解的可以考虑一下朴素版本通过 for 循环实现多头注意力，每个注意力头得到 \[batch_size, seq_len, d_k] 然后 n_heads 个头的结果拼接在一起得到 \[batch_size, seq_len, n_heads * d_k]

---

**view 和 shape 都是对矩阵形状做变化，有什么区别？**

- `view` 只能对连续存储的张量进行形状变化，例如 `self.W_Q(input_Q)` 得到的张量通常是连续的。
- `reshape` 可以对任意张量进行形状变化，但是性能不如 `view`。Q,K,V 通过 `transpose` 转置后不能保证在内存中连续，所以用 `reshape` 安全。

---

**为啥注意力机制计算结束还要加一个全连接？**

多头注意力机制最早出现在 Transformer 模型中（论文《Attention is All You Need》），它的定义明确包含了最后的线性变换层：

```python
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) * W_O
```

它的好处是前面的拼接只是机械地将每个头的输出放在一起，维度虽然匹配，信息却没有经过进一步的处理，并且拼接是无参数的，而 Linear 层引入了可学习的权重 `W_o`，让模型能够根据任务动态调整每个头的贡献。

---

**为什么要把张量拆成四维并调整维度顺序呢？**

在单头自注意力中，输入形状为 $[B, L, H]$。  
我们可以直接计算注意力得分：

$$
Q K^\top :\ [B, L, H] \times [B, H, L] = [B, L, L]
$$

对于多头注意力，我们首先将输入从 $[B, L, H]$ 映射为 $[B, L, n\_\text{heads} \times d_k]$。这表示第三个维度已经包含了所有头的投影结果。

接着，我们把这一维拆分成两维：

$$
[B, L, n\_\text{heads}, d_k]
$$

再通过 permute 调整维度顺序：

$$
[B, n\_\text{heads}, L, d_k]
$$

此时可以将其理解为：每个 batch 里有 $n\_\text{heads}$ 个独立的注意力头，每个头对应一个 $[L, d_k]$ 的投影空间，相当于单头时的$[L, H]$ 但维度较小（$d_k$​）。

---

**四维矩阵如何进行乘法的？**

在本科的线性代数课上，我们学过了二维矩阵的乘法 $[a, b] \times [b, c] = [a, c]$，那四维矩阵呢？

对于超过二维的张量，`torch.matmul` 会把张量的前几个维度（称为“批次维度”）看作独立的批次，对每个批次单独执行二维矩阵乘法。最后两维被视为矩阵的行和列，进行标准的矩阵乘法。

- 假设两个张量 A 和 B：
	- A 的形状：\[batch_dim1, batch_dim2, m, n]。
	- B 的形状：\[batch_dim1, batch_dim2, n, p]。
- 前面的批次维度 \[batch_dim1, batch_dim2] 必须匹配。
- 最后两维 \[m, n] 和 \[n, p]按二维矩阵乘法规则计算。
- 结果形状：\[batch_dim1, batch_dim2, m, p]。

##### 5. FeedforwardNet

```python
class FeedForwardNet(nn.Module):
    
    def __init__(self, d_model: int, hidden_size: int, p: float = 0.1):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.relu = nn.ReLU()
        # self.norm = nn.LayerNorm()
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.dropout(output)
        # return self.norm(x + output, -1)
        return output
```
#### 三、Decoder

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/decoder.jpg" width="60%" />
</div>

Transformer Decoder 和 Encoder 稍有不同，它也包含 Word Embedding 和 Position Embedding，不过 DecoderLayer 每一层中是 MaskedMultiHeadAttention 和 CrossMultiHeadAttention。

```python
class Decoder(nn.Module):
    
    def __init__(
        self,
        vocal_size: int,
        d_model: int,
        max_len: int,
        layer_nums: int,
        hidden_size: int,
        p: float = 0.1
    ):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(vocal_size, d_model)
        self.position_embedding = PositionEmbedding(max_len, d_model, p)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, hidden_size, p) for _ in range(layer_nums)])
        self.attn_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_x, enc_context):
        self_attns, cross_attns = [], []
        batch_size, seq_len = x.size()
        dec_embed = self.word_embedding(x)
        dec_embed = self.position_embedding(dec_embed)
        
        dec_padding_mask = padding_mask(x, x)  # [batch_size, seq_len, seq_len]
        cross_mask = padding_mask(enc_x, x)
        
        # 对 dec_masks 应用 Future Mask
        future_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        future_mask = future_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
        self_mask = dec_padding_mask.bool() | future_mask
        
        
        dec_output = dec_embed
        for layer in self.layers:
            context, dec_self_attn, dec_cross_attn = layer(dec_output, enc_context, self_mask, cross_mask)
            self_attns.append(dec_self_attn)
            cross_attns.append(dec_cross_attn)
        
        return context, self_attns, cross_attns
```

##### 1. Mask

Mask 掩码分为两种，一种是使 Sequence 固定长度用于掩盖空位的掩码，另一种是在 Transformer Decoder 中用于掩盖未来 token 的掩码。

**Padding Mask**

在 Transformer Encoder 的多头注意力机制中，我们用每个 token 的 Query 去查询其他 token 的 Key，得到了每个 token 在 源输入中的注意力权重 \[seq_len, seq_len]，其中第一个维度代表有多少 token，第二个维度代表每个 token 对其他 token 的注意力权重，所以我们应该应用 Mask 在第二个维度。在 Decoder 的交叉多头注意力机制中，我们用 Decoder 的每一个 token 去查询 Encoder 的 token，也就是说这时候产生的注意力权重形状是 \[enc_len, dec_len]。

```python
def padding_mask(enc_seq: torch.Tensor, dec_seq: torch.Tensor, pad_idx: int = 0):
    batch_size, enc_len = enc_seq.size()
    batch_size, dec_len = dec_seq.size()
    
    mask = dec_seq.detach().eq(pad_idx).unsqueeze(1)  # [batch_size, 1, dec_len]
    return mask.expand(batch_size, enc_len, dec_len)
```

**Future Mask**

Future Mask 的原理前文介绍过了，主要就是为了使 Decoder 不能窥视后续文本。我们仍然是在注意力权重矩阵中应用 Mask，它的形状是 \[batch_size, seq_len, seq_len]，对这个矩阵的上三角部分掩盖即可。

```python
    future_masks = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    future_masks = future_masks.unsqueeze(0).expand(batch_size, seq_len, seq_len)
    dec_masks = dec_masks | future_masks
```

我们生成一个上三角全是 True, 主对角和下三角为 False 的矩阵，然后和 Padding Mask 相与。

##### 2. DecoderLayer

```python
class DecoderLayer(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForwardNet(d_model, hidden_size, dropout)
        
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
    
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, enc_output, self_mask, cross_mask):
        norm_x = self.self_norm(x)
        self_context, dec_self_attn = self.self_mha(norm_x, norm_x, norm_x, self_mask)
        x = self.dropout(self_context) + x
        
        norm_x = self.cross_norm(x)
        cross_context, dec_cross_attn = self.cross_mha(norm_x, enc_output, enc_output, cross_mask)
        x = self.dropout(cross_context) + x       
        
        norm_x = self.ff_norm(x)
        ff_output = self.ff(norm_x)
        
        return self.dropout(ff_output) + x, dec_self_attn, dec_cross_attn
```

DecoderLayer 依次应用掩码多头注意力，交叉多头注意力和前馈神经网络，为了让 CrossMultiHeadAttention 可以服用多头注意力的代码，我们只需要把输入 x 区分为 input_q, input_k, input_v。用掩码多头注意力得到的 Query 去查询 Encoder 输出的 Key就行了。