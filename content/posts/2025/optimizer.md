---
title: "Optimizer"
date: '2025-12-01T09:41:11+08:00'
authors: [Xilyfe]
series: ["DeepLearning"]
tags: ["深度学习"]
--- 


## Adam

Adam = 动量 Momentum + 自适应学习率优化 RMSProp + 偏差修正 Bias Correction

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

> 这里的 t 指的是时间步 - 第 t 次更新。

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
