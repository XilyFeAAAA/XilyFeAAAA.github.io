---
title: Optimizer
date: 2025-12-01T09:41:11+08:00
authors:
  - Xilyfe
series:
  - DeepLearning
tags:
  - 深度学习
featuredImage: http://img.xilyfe.top/img/20260203144751043.png
lastmod: 2026-02-03T11:34:35+08:00
---
深度学习一般的过程为：前向传播 → 计算 loss → 反向传播 → 计算梯度 → 更新参数，optimizer 的作用就是利用梯度来更新参数。梯度下降是一种流行的优化算法，算法计算出损失函数相对于神经网络中每个参数的梯度，然后在负梯度方向上更新参数，这样就能减少损失函数。

## SGD

第一代的优化算法就是 SGD，随机梯度下降：

$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$

在理想情况下我们应该用整个数据集训练得到的全部梯度进行一轮更新，也就是：

$$
g_{\text{ideal}} = \frac{1}{N}\sum_{\text{All Data}}\text{Gradients}
$$

但实际上受限于显存大小，我们采用批次训练 batch_size 个数据进行一轮训练：

$$
g_{\text{realistic}}=\frac{1}{B}\sum_{\text{Sample}}\text{Gradients}
$$

但是这 batch_size 个样本不能完全代表整个数据集，会给梯度带来噪声，导致训练出现震荡（比如这次噪声导致梯度偏左，下一次导致梯度偏右）。那如何解决这个问题呢？答案是引入动量这个概念，计算当前参数值不能仅仅考虑当前的梯度，还需要考虑上一次的参数值。

## 指数加权平均

这里举一个例子，假设我们有商店前一周的销售额，我们怎么估计出今天的销售额？一个简单的想法是直接对前一周的销售额取均值 $\text{price}=\frac{1}{N}\sum_{\text{week}}price_i$ 。如何优化呢？越近的数据对现在影响越大，越有参考意义，所以应该给它更大的权重。指数加权平均就更加简洁的表达了这个思想。

$$
V_t=\beta V_{t-1}+(1-\beta)\theta
$$

假设 $V_i$ 是第 $i$ 天的指数加权平均值，$\theta_i$ 为第 $i$ 天的销售额，$\beta=0.7$ 是加权平均系数，那我们可以得到：

$$
\begin{align}
V_0=0,\beta=0.7 \\
V_1=0.7V_0+0.3\beta_1 \\
V_2=0.7V_1+0.3\beta_2 \\
V_3=0.7V_2+0.3\beta_3 \\
V_4=0.7V_3+0.3\beta_4 \\
V_5=0.7V_4+0.3\beta_5 \\
V_6=0.7V_5+0.3\beta_6
\end{align}
$$

我们把指数加权平均值 $V_6$ 展开，可以看到它按照指数衰减赋予了每天销售额不同的权重，并且距离第六天越远权重越低：

$$
\begin{align}
V_6 &= 0.7(0.7V_4+0.3\beta_5)+0.3\beta_6 \\
    &= 0.7^3V_3+0.3*0.7^2\theta_4+0.3*0.7\beta_5+0.3\beta_6 \\
    &= 0.3*0.7^5\beta_1 + ...+0.21\beta_5+0.3\beta_6
\end{align}
$$

>除此之外，运行过程中只需要额外保存一个值 $V_{t-1}$ 即可对历史所有值取平均。

### 偏差修正

| 日期  | $\theta$ | $V$    |
| --- | -------- | ------ |
| 第一天 | 100      | 30     |
| 第二天 | 114      | 55.2   |
| 第三天 | 118      | 74.04  |
| 第四天 | 117      | 86.9   |
| 第五天 | 120      | 96.83  |
| 第六天 | 122      | 104.38 |
计算之后会发现前几天的销售额明显偏小，这是因为我们的初始值是 $V_0=0$，随着序列变长 $V_0$ 的影响会逐渐减小。我们可以对 $V$ 进行修正，让它乘一个系数就能缓解。

$$
V_t^{\text{correct}}=\frac{V_t}{1-\beta^t}
$$


## RMSProp

RMSProp 的思想很简单，在梯度比较小的地方我们应该放大步子，梯度大的地方我们应该缩小步子，这样就不容易陷入局部最优或者出现震荡。按照这个思路我们只需要对学习率加一个系数即可，梯度大我们就减小学习率，梯度小我们就增加学习率。

计算震荡幅度：
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

自适应更新：
$$
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t}+\epsilon} \cdot g_t
$$

当梯度大的时候，梯度的指数加权平均值就变大，那么学习率则减小，$\epsilon$ 是为了防止分母为 0。

## Adam

Adam 就是把前面的指数加权平均、偏差修正和 RMSProp 结合的产物。

**动量**

$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\end{align}
$$

**偏差修正**

$$
\begin{align}
\hat m_t &= \frac{m_t}{1-\beta_1^t} \\
\hat v_t &= \frac{v_t}{1-\beta_2^t} 
\end{align}
$$

**参数更新**

$$
\theta_t = \theta_{t-1} - \alpha\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$


```python
class Adam:
    
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float,
        ep: float,
        beta: tuple[float, float]
    ):
        self.params = params
        self.lr = lr
        self.ep = ep
        self.beta1, self.beta2 = beta
        
        self.t = 0
        
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
    
    def step(self, grads: list[torch.Tensor]):
        
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # momentum
            self.m[i] = self.beta1 * self.m[i-1] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i-1] + (1 - self.beta2) * grad ** 2
            
            # bias correction
            m = self.m[i] / (1 - self.beta1 ** self.t)
            v = self.v[i] / (1 - self.beta2 ** self.t)
            
            # update parameter
            param -= self.lr * m / (torch.sqrt(v) + self.ep)
```

PyTorch 风格的 Optimizer 会用 state 存储 m、v：

```python
class AdamTorch:
    
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float,
        ep: float,
        beta: tuple[float, float]
    ):
        self.params = params
        self.lr = lr
        self.ep = ep
        self.beta1, self.beta2 = beta
        
        self.t = 0
        
        self.state = {p: {
            "m": np.zeros_like(p),
            "v": np.zeros_like(p)
        } for p in params}
    
    def step(self, grads: list[torch.Tensor]):
        
        self.t += 1
        
        for param, grad in zip(self.params, grads):
            
            s = self.state[param]
            
            # momentum
            s["m"] = self.beta1 * s["m"] + (1 - self.beta1) * grad
            s["v"] = self.beta2 * s["v"] + (1 - self.beta2) * grad ** 2
            
            # bias correction
            m = s["m"] / (1 - self.beta1 ** self.t)
            v = s["v"] / (1 - self.beta2 ** self.t)
            
            # update parameter
            param -= self.lr * m / (torch.sqrt(v) + self.ep)
```


## AdamW

AdamW 相对 Adam 在参数更新阶段进行衰减：

$$
\theta_t=\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}-\alpha \lambda \theta_{t-1}
$$

```python
class AdamWTorch:
    
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float,
        ep: float,
        weight_decay : float,
        beta: tuple[float, float]
    ):
        self.params = params
        self.lr = lr
        self.ep = ep
        self.weight_decay  = weight_decay 
        self.beta1, self.beta2 = beta
        
        self.t = 0
        
        self.state = {p: {
            "m": np.zeros_like(p),
            "v": np.zeros_like(p)
        } for p in params}
    
    def step(self, grads: list[torch.Tensor]):
        
        self.t += 1
        
        for param, grad in zip(self.params, grads):
            
            s = self.state[param]
            
            # momentum
            s["m"] = self.beta1 * s["m"] + (1 - self.beta1) * grad
            s["v"] = self.beta2 * s["v"] + (1 - self.beta2) * grad ** 2
            
            # bias correction
            m = s["m"] / (1 - self.beta1 ** self.t)
            v = s["v"] / (1 - self.beta2 ** self.t)
            
            # update parameter
            param = (1 - self.lr * self.weight_decay) * param - self.lr * m / (torch.sqrt(v) + self.ep)
```

{{< admonition type=question title="权重衰减和 L2 正则化有没有关系？">}} 
如果你了解 L2 正则化可能会发现，损失函数加入 L2 正则化之后的梯度公式和加入权重衰减的一模一样：

$$
\text{loss}_{\text{total}} = \text{loss}_{\text{origin}} + \frac{\lambda}{2} \times ||w||^2
$$

梯度则和 AdamW 一样：

$$
∇L_{\text{total}} = ∇L_{\text{original}} + λ * w
$$
但是在 Adam 中我们采用了 Momentum 或者说指数加权平均，它会计算一阶矩 $\hat m_t$ 和二阶矩 $\hat v_t$，这时候加入的 L2 正则化就会污染他们，下面公式里面 $\lambda*w$ 被线性加入到 $m_t$ 里：

$$
\begin{align}
m_t &= \beta_1 * m_{t-1} + (1 - \beta_1) * (∇L_{data} + \lambda * w) \\
    &= \beta_1 * m_{t-1} + (1 - \beta_1) * ∇L_{data} + (1 - \beta_1) * \beta * w
\end{align}
$$
而 AdamW 采用的权重衰减直接放在参数更新阶段，不会污染梯度进而影响一阶矩和二阶矩。
{{< /admonition >}}