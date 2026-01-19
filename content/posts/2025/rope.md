---
title: "RoPE"
date: '2025-11-28T16:23:11+08:00'
authors: [Xilyfe]
series: ["LLM"]
tags: ["大模型", "Transformer"]
--- 
 

## 作用

RoPE 相对于正余弦位置编码和可学习位置编码，更能够表达**相对位置信息**，便于模型捕捉序列中元素之间的关系，还便于模型泛化到更长的序列，支持超长文本推理。

RoPE 的思路：**对 Q,K 矩阵进行旋转，使计算得到的注意力权重天然带有两个 token 之间的相对距离**。

## 公式推导

对输入序列 $\{x_0, x_1, \dots\}$，假设我们取两个 token：第 m 个和第 n 个

$$
x_m,\ x_n \in \mathbb{R}^{d_{\text{model}}}
$$

方便推导，先假设 $d_{\text{model}} = 2$。令它们的 Query、Key 为：

$$
q_m = W_q x_m,\quad k_n = W_k x_n
$$

注意力的核心是内积：

$$
\langle q_m,\ k_n\rangle = q_m^\top k_n
$$

RoPE 的想法是用“旋转”后的 Q/K 来计算内积：

$$
\langle q_m^{rope},\ k_n^{rope}\rangle
$$

其中

$$
q_m^{rope} = q_m e^{i m\theta},\qquad k_n^{rope} = k_n e^{i n\theta}
$$


---

这里补充一下向量旋转的知识点：

复数的形式是：

$$
z = a + bi
$$

其中 a 是实部，b 是虚部，i 是虚数单位。如果你把复平面画出来，会发现它和普通的 2D 坐标系完全一样：

- 横轴 = 实轴（real axis）
- 纵轴 = 虚轴（imag axis）    
- 一个点 $(x, y)$ 就是复数 $x + yi$

所以二维向量 $(x, y)$ 可以等价地写成复数 $z = x + yi$，对于上文的二维向量 $x_m$ 有：

$$
q_m = 
\begin{bmatrix}
q_m^1 \\
q_m^2
\end{bmatrix} = q_m^1 + q_m^2i
$$

因为复数的乘法，天然包含了“旋转 + 缩放”的操作，对一个复数 $z = x + yi$，乘上一个单位模的复数：
$$
e^{i\theta} = \cos\theta + i\sin\theta
$$


就会让它绕原点旋转角度 $\theta$，因此 $q_m^{rope} = q_m e^{i m\theta}$ 表示把 $q_m$​ 旋转 $m\theta$。

对任意二维向量应用二维旋转矩阵也可以逆时针旋转：

$$
\begin{array}{c} R(\theta) = 
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix} \end{array}
$$


旋转矩阵的效果与 $e^{i m\theta}$ 相同， 证明如下：

$$
\begin{align}
q_m^{rope}&=q_me^{im\theta}\\
&=\begin{bmatrix}q_m^1\\q_m^2\end{bmatrix}e^{im\theta}\\
&=(q_m^i+q_m^2i)(\cos(m\theta) + i\sin(m\theta))\\
&=(q_m^1\cos(m\theta)-q_m^2\sin(m\theta)) + i(q_m^2\cos(m\theta) + q_m^1\sin(m\theta))\\
&=\begin{bmatrix}q_m^1\cos(m\theta)-q_m^2\sin(m\theta)\\q_m^2\cos(m\theta) + q_m^1\sin(m\theta)\end{bmatrix}\\
&=\begin{bmatrix}\cos(m\theta) & -\sin(m\theta) \\\sin(m\theta) & \cos(m\theta)\end{bmatrix}\begin{bmatrix}q_m^1\\q_m^2\end{bmatrix} \\
&=R(m\theta)\begin{bmatrix}q_m^1\\q_m^2\end{bmatrix}
\end{align}
$$

---

基于前面的向量旋转的知识，我们进而得到：

$$
\begin{align}
<q_m^{rope}, k_n^{rope}>&=<R(m\theta)q_m, R(n\theta)k_n>\\
&=q_m^TR^T(m\theta)R(n\theta)k_n\\
&=q_m^TR(-m\theta)R(n\theta)k_n\tag{证明1.1}\\
&=q_m^TR((n-m)\theta)k_n\tag{证明1.2}
\end{align}
$$

至此就可以看出，应用 RoPE 对向量进行旋转后，注意力权重就与两个 token 之间距离相关了。两个 token 距离越远，n-m 越大，旋转角度越大，注意力权重越小。

证明-1.1：

$$
\begin{array}{c} R^T(\theta) = 
\begin{bmatrix}
\cos\theta & \sin\theta \\
-\sin\theta & \cos\theta
\end{bmatrix} =\begin{bmatrix}
\cos(-\theta) & -\sin(-\theta) \\
\sin(-\theta) & \cos(-\theta)
\end{bmatrix} = R(-\theta)
\end{array}
$$

证明-1.2：

$$
R(-m\theta)R(n\theta)\ \longleftrightarrow\ e^{-im\theta}e^{in\theta}=e^{i(n-m)\theta}=R((n-m)\theta)
$$

---

扩展到高维就有：

![](http://img.xilyfe.top/img/20260119121133066.png)

可以看到矩阵计算时候有非常多的 0，增大了计算量，简便方法就是：

![](http://img.xilyfe.top/img/20260119121142785.png)

并且有：

$$
\theta_i=10000^{-2i/d}
$$

## 代码


### 线性组合

```python
def linear_RoPE(qk: torch.Tensor):
    # x = [bs, len, dmodel]
    _, seq_len, d_model = qk.size()
    assert d_model % 2 == 0
        
    position = torch.arange(seq_len, dtype=torch.float) # [max_len]
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    sinusoid = torch.outer(position, freq)
    
    cos, sin = torch.cos(sinusoid), torch.sin(sinusoid)
    even, odd = qk[..., 0::2], qk[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = odd * cos + even * sin

    return torch.stack([rotated_even, rotated_odd], dim=-1).reshape_as(qk)
```

代码的朴素实现就是一比一参考上图。

- 首先计算频率，也就是上图的 $m\theta$，需要注意图片里面是一维向量，但实际上应该是三维的 \[batch_size, seq_len, d_model//2]。我们可以构造出 0-seq_len-1 的向量和 $\theta_0$-$\theta_{d/2-1}$ 向量，然后求外积(ps: `torch.outer` 等同于 `unsqueeze(1)` 之后逐点相乘)就能得到 \[seq_len, d_model//2]。
- 然后将输入矩阵的 d_model 为按照奇偶分开。
- 最后偶数列就是 "偶数列\*cos-奇数列\*sin"，奇数列就是 "奇数列\*cos+偶数列\*sin"
- 通过 `torch.stack` 叠加在第四维，然后再 `reshape` 交错拼接在第三维。

### LLaMA

```python
def llama_RoPE(qk: torch.Tensor):
    _, _, seq_len, dim = qk.shape
    
    assert dim % 2 == 0, "dim must be even"
    
    qk_complex = qk.view(*qk.shape[:-1], dim//2, 2) # [bsize, nheads, seq_len, dim//2, 2]
    qk_complex = torch.view_as_complex(qk_complex)  # [bsize, nheads, seq_len, dim//2]
    
    position = torch.arange(seq_len, dtype=torch.float) # [max_len]
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    sinusoid = torch.outer(position, freq)
    rot = torch.exp(1j * sinusoid)  # [seq_len, dim//2]
    # rot = torch.polar(torch.ones_like(sinusoid), sinusoid)
    rotated_qk_complex = qk_complex * rot
    rotated_qk = torch.view_as_real(rotated_qk_complex)  # [bsize, nheads, seq_len, dim//2, 2]
    rotated_qk = rotated_qk.view_as(qk)
    
    return rotated_qk
    
```

LLaMA 的实现方式更接近 RoPE 最朴素的想法：**对 Q/K 进行旋转**，它等价于 **对 Q/K 的每对维度进行一个二维旋转**：

$$
\begin{pmatrix}
x'_{2i}\\
x'_{2i+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{pmatrix}
\begin{pmatrix}
x_{2i}\\
x_{2i+1}
\end{pmatrix}
$$

LLaMA 的想法是 **把二维向量看成复数**，二维旋转矩阵实际上等价于乘上复数：$e^{j\theta} = \cos\theta + j\sin\theta$ 。`qk_complex` 就是将 qk 最后一个维度两两拆开组成复数，然后和单位模复数相乘将其旋转，最后再还原为二维向量。


