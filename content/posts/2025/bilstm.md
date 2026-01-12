---
title: "Bilateral LSTM"
date: '2025-11-27T11:49:11+08:00'
authors: [Xilyfe]
series: ["DeepLearning"]
tags: ["Module", "深度学习"]
--- 

## 双向 LSTM

有些时候预测可能需要由前面若干输入和后面若干输入共同决定，这样会更加准确。因此提出了双向循环神经网络，网络结构如下图。可以看到 Forward 层和 Backward 层共同连接着输出层，其中包含了 6 个共享权值 w1-w6。

<div style="text-align: center">
    <img src="../../../resource/ai/llm/bilstm.png"/>
</div>

### 代码实现

在 GPT 过程中发现了一种计算效率更高的方法，代码如下:

```python
class LSTM(nn.Module):

    def __init__(self):

        self.Wih = nn.Parameter(torch.empty(4*H, B))
        self.Whh = nn.Parameter(torch.empty(4*H, B))
        self.Bih = nn.Parameter(torch.empty(4*H))
        self.Bhh = nn.Parameter(torch.empty(4*H))
        self.setup_parameter()


    def forward(self, x):
        h_prev = x.new_zeros(2, B, H)
        c_prev = x.new_zeros(2, B, H)
        h, c = [], []

        seq_len, batch_size, embed_size = x.shape
        for i in range(seq_len):
            x_t = x[i]  # x[i] = x[i, :, :]
            gates = F.linear(x_t, self.Wih, self.Bih) + F.linear(h_prev, self.Whh, self.Bhh)
            f, i, o, g = gates.chunk(4, dim=1)
            f, i, o, g = torch.sigmoid(f), torch.sigmoid(i), torch.tanh(g), torch.sigmoid(o)
            c_prev = f * c_prev + i * g
            h_prev = o * torch.tanh(c_prev)
            h.append(h_prev)
            c.append(c_prev)
        
        h = torch.stack(h, dim=0)
```

在初始化参数时，将原先的 2*4 个 Weight 直接合并到一个矩阵中，也就是形状从 [Batch_size, Hidden_size] 变成 [Batch_size, Hidden_size * 4]，这样就可以使用一次矩阵乘法代替之前四次矩阵乘法。之后 chunk(4, dim=0) 把这条 [B, 4H] 的张量沿着维度 1 均分成 4 份，每份形状都是 [B, H]，并按序赋值给 i, f, g, o。

> F.linear(x, w, b) 等价于 y = x * W^T + b，所以在定义参数时候需要定义为 [H, B] 的形状。

这种写法的好处在于：我之前的做法每个门各自一套权重（Whf/Wxf、Whi/Wxi、Who/Wxo、Whc/Wxc），每步要做 8 次 matmul（4 个门 × 2 条支路：输入与隐状态）再逐门相加；偏置也分门单独加。新的方法把四个门拼成一个大矩阵，用两次 F.linear 就拿到全部门值，再用 chunk(4) 拆成 i,f,g,o。这样每步只需 2 次 GEMM，显著更高效，也更接近 nn.LSTM 的实现。


```python
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMManual(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        H, D = hidden_size, input_size

        # Forward direction parameters
        self.W_ih_f = nn.Parameter(torch.empty(4 * H, D))
        self.W_hh_f = nn.Parameter(torch.empty(4 * H, H))
        self.b_ih_f = nn.Parameter(torch.empty(4 * H))
        self.b_hh_f = nn.Parameter(torch.empty(4 * H))

        # Backward direction parameters
        self.W_ih_b = nn.Parameter(torch.empty(4 * H, D))
        self.W_hh_b = nn.Parameter(torch.empty(4 * H, H))
        self.b_ih_b = nn.Parameter(torch.empty(4 * H))
        self.b_hh_b = nn.Parameter(torch.empty(4 * H))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier for input weights, orthogonal for recurrent; forget gate bias to 1
        for W in (self.W_ih_f, self.W_ih_b):
            nn.init.xavier_uniform_(W)
        for W in (self.W_hh_f, self.W_hh_b):
            nn.init.orthogonal_(W)
        for b in (self.b_ih_f, self.b_hh_f, self.b_ih_b, self.b_hh_b):
            nn.init.zeros_(b)
        # Set forget gate bias (second quarter) to 1
        for b in (self.b_ih_f, self.b_hh_f, self.b_ih_b, self.b_hh_b):
            H4 = b.numel() // 4
            b.data[H4 : 2 * H4].fill_(1.0)

    @staticmethod
    def _step(x_t, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh):
        """One LSTM step.
        x_t: [B, D]; h_prev, c_prev: [B, H]
        returns h_t, c_t
        """
        gates = F.linear(x_t, W_ih, b_ih) + F.linear(h_prev, W_hh, b_hh)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.batch_first:
            # [B, T, D] -> [T, B, D]
            x = x.transpose(0, 1)
        T, B, D = x.shape
        H = self.hidden_size

        if hx is None:
            h0 = x.new_zeros(2, B, H)
            c0 = x.new_zeros(2, B, H)
        else:
            h0, c0 = hx
            assert h0.shape == (2, B, H) and c0.shape == (2, B, H)

        # Forward pass (left -> right)
        h_f = []
        h_prev_f = h0[0]
        c_prev_f = c0[0]
        for t in range(T):
            h_prev_f, c_prev_f = self._step(
                x[t], h_prev_f, c_prev_f, self.W_ih_f, self.W_hh_f, self.b_ih_f, self.b_hh_f
            )
            h_f.append(h_prev_f)
        h_f = torch.stack(h_f, dim=0)  # [T, B, H]

        # Backward pass (right -> left)
        h_b_rev = []
        h_prev_b = h0[1]
        c_prev_b = c0[1]
        for t in reversed(range(T)):
            h_prev_b, c_prev_b = self._step(
                x[t], h_prev_b, c_prev_b, self.W_ih_b, self.W_hh_b, self.b_ih_b, self.b_hh_b
            )
            h_b_rev.append(h_prev_b)
        h_b_rev = torch.stack(h_b_rev, dim=0)  # [T, B, H] in reversed time
        h_b = torch.flip(h_b_rev, dims=[0])    # align to original time

        # Concatenate features from both directions
        y = torch.cat([h_f, h_b], dim=2)  # [T, B, 2H]

        # Final hidden/cell states per direction
        h_n = torch.stack([h_f[-1], h_b[0]], dim=0)  # [2, B, H]
        c_n = torch.stack([c_prev_f, c_prev_b], dim=0)  # [2, B, H]

        if self.batch_first:
            y = y.transpose(0, 1)  # [B, T, 2H]
        return y, (h_n, c_n)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, H = 3, 5, 8, 16
    x = torch.randn(B, T, D)
    bilstm = BiLSTMManual(input_size=D, hidden_size=H, batch_first=True)
    y, (h_n, c_n) = bilstm(x)
    print(y.shape, h_n.shape, c_n.shape)  # [3, 5, 32] [2, 3, 16] [2, 3, 16]

```

- 前向隐藏序列：h_f 是 每个时间步 的前向隐藏态，形状 [T, B, H]
- 反向隐藏序列：h_b 也是 每个时间步 的反向隐藏态，形状 [T, B, H]（因为是反向从右往左算的，我们先得到 h_b_rev 按“反序时间”，然后用torch.flip(h_b_rev, dims=[0]) 把它对齐到“原时间顺序”，得到 h_b）
- 拼接后的序列输出：y = torch.cat([h_f, h_b], dim=2) → [T, B, 2H] 这是每个时间步的双向表示（把前向与反向在特征维拼起来）
- h_n、c_n 要按照 PyTorch/LSTM 的约定形状给出：h_n.shape == (num_layers * num_directions, B, H) 单层双向就是 [2, B, H]。