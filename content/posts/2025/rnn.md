---
title: "RNN"
date: '2025-11-19T18:06:11+08:00'
authors: [Xilyfe]
series: ["DeepLearning"]
tags: ["Module", "深度学习"]
--- 

> 在 CS224N 的课程中学习了 RNN 的基本知识，为了深入了解背后的机制和代码实现，我让 GPT 设计了一个 RNN 相关的深度学习任务，通过 PyTorch 手搓一个 RNN 网络。

## 实验题目

使用 RNN 实现英文字母序列预测任务（Character-Level Sequence Prediction）

## 实验目标

1. 理解 RNN 的结构和前向传播机制。
2. 掌握用 PyTorch 构建和训练循环神经网络的过程。
3. 学会将序列数据转化为模型可以处理的张量形式。
4. 观察 RNN 在学习序列模式（如字母顺序）时的表现。

## 实验内容

让模型学习英文字母序列（如 “abcde…z”），然后预测下一个字母。

例如输入 a, b, c, d, e，目标输出 b, c, d, e, f。
训练完成后，输入 'a'，模型应该能输出 'b'；输入 'm'，输出 'n'。

## 实验过程

**1. 整体思路**

- 将输入序列转为 0-25 的整型数组传入模型
- 模型内部通过 embedding-lookup 查找到对应的 one-hot 向量，例如 a 对应 [0,0,0,...,0,1]
- 通过 RNN 模型进行训练

**2. nn.Module**

模型具体设计遵循 RNN 的变化公式：

$$
h_t=tanh(Wx \cdot x_t + W_h \cdot h_{t-1} + b_h)
$$

模型的参数如下，矩阵形状需要注意：在 PyTorch 中计算时候是 $x \cdot W$ 而不是 $W \cdot x$

```python
class RNN(torch.nn.Module):
    
    def __init__(
        self,
        embeddings,
        embed_size,
        hidden_size,
        output_size
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.hidden_size = hidden_size
        
        # xt = [batch_size, seq_len * embed_size]
        # Wx = [seq_len * embed_size, hidden_size]
        # Wh = [batch_size, hidden_size]
        self.embed_to_hidden_weight = torch.nn.Parameter(torch.empty(embed_size, hidden_size))
        torch.nn.init.xavier_uniform_(self.embed_to_hidden_weight)
        
        self.hidden_to_logits_weight = torch.nn.Parameter(torch.empty(hidden_size, output_size))
        torch.nn.init.xavier_uniform_(self.hidden_to_logits_weight)
        
        self.last_to_new_weight = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        torch.nn.init.xavier_uniform_(self.last_to_new_weight)
        
        embed_to_hidden_bias_tensor = torch.empty(hidden_size)
        torch.nn.init.uniform_(embed_to_hidden_bias_tensor)
        self.embed_to_hidden_bias = torch.nn.Parameter(embed_to_hidden_bias_tensor)

        hidden_to_logits_bias_tensor = torch.empty(output_size)
        torch.nn.init.uniform_(hidden_to_logits_bias_tensor)
        self.hidden_to_logits_bias = torch.nn.Parameter(hidden_to_logits_bias_tensor)
```

embedding_lookup 的设计我参考的 cs224n 的词向量嵌入，由于 RNN 模型中循环的特性，每次需要取出当前 batch 的一个词向量，所以我没有把它展平成二维。

```python
    def embedding_lookup(self, w):
        # [batch_size, seq_len] -> [batch_size, seq_len, embed_size]
        return self.embeddings[w]

```

前馈计算的代码其实就是实现公式了，这里用 torch 中的张量索引 [:, i: ,] 可以很容易得到 batch_size 大小的第 i 个输入的 one-hot 矩阵

```python
    def forward(self, w):
        x = self.embedding_lookup(w)
        batch_size, seq_len, _ = x.shape
        ht = torch.zeros(batch_size, self.hidden_size)
        for i in range(seq_len):
            inp = x[:, i, :]
            ht = torch.tanh(torch.matmul(inp, self.embed_to_hidden_weight) + torch.matmul(ht, self.last_to_new_weight) + self.embed_to_hidden_bias)
        
        return torch.matmul(ht, self.hidden_to_logits_weight) + self.hidden_to_logits_bias
```

**3. 实验结果**

在 1000 轮训练中可以看到 loss 已经降到非常小的数据了：
```
第 988/1001 轮训练，loss=2.3454136680811644e-05
第 989/1001 轮训练，loss=2.3394533855025657e-05
第 990/1001 轮训练，loss=2.3357280952041037e-05
第 991/1001 轮训练，loss=2.3320028049056418e-05
第 992/1001 轮训练，loss=2.330512688786257e-05
第 993/1001 轮训练，loss=2.3282778784050606e-05
第 994/1001 轮训练，loss=2.3260427042259835e-05
```

但是在测试集上效果非常差：

```
正确：tensor([14, 12,  3, 22, 19,  9, 22])
预测：tensor([16, 18, 15, 17, 25,  4,  5])
loss=7.221210956573486
```

大概率是数据集问题，不过实验题目限制数据集就这么大==，所以代码应该没什么问题。