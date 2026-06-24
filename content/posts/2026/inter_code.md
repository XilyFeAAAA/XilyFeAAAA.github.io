---
title: Interview Codes
date: 2026-06-02T17:55:39+08:00
featuredImage: http://img.xilyfe.top/img/20260609104852216.png
authors:
  - Xilyfe
series:
  - 面经
tags: []
lastmod: 2026-06-11T11:28:42+08:00
---
>准备 2026 暑假 LLM 算法实习ing

### DeepLearning


#### Softmax

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}
$$

>数值稳定的 Softmax，减去最大值防止溢出。

```python
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - x_max)
    return e / e.sum(dim=dim, keepdim=True)
```

$$
\log\text{softmax}(x_i) = \log{\frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}} = x_i - m - \log\left(\sum_{j=1}^{n} e^{x_j - m}\right)
$$

```python
def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - x_max)
    return x - x_max - e.sum(dim=dim, keepdim=True)
```

#### MSELoss

$$
\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

```python
def compute_mse_loss(y, labels):
    return ((y - labels) ** 2).mean()
```

#### KL divergence

$$
D_{KL}(P \mid \mid Q) = \sum_xP(x)\log{\frac{P(x)}{Q(x)}}
$$

```python
def kl_divergence(p, q, eps):
	p = torch.tensor(p, dtype=torch.float32)
	q = torch.tensor(q, dtype=torch.float32)
	
	p = p / p.sum(dim=-1, keepdim=True)
	q = q / q.sum(dim=-1, keepdim=True)
	p = torch.clamp(p, eps, 1)
	q = torch.clamp(q, eps, 1)
	
	return (p * torch.log(p / q)).sum()
```

上面写的 KL Divergence 公式是数学上正确的形式，在 LLM 语境下我们不会对整个 vocab 分布求和，只会计算当前策略对“实际采样 token”的 logp：

$$
k_1=\log \frac{\pi_\theta(a|s)}{\pi_{\mathrm{ref}}(a|s)}
$$

```python
def kl_penalty(logprobs, ref_logprobs):
	return logprob - ref_logprob
```

$$
k_2=\frac{1}{2}\left(\log \frac{\pi_\theta(a|s)}{\pi_{\mathrm{ref}}(a|s)}\right)^2
$$

```python
def k2_estimator(logprobs, ref_logprobs):
	return 0.5 * (logprobs - ref_logprobs) ** 2
```

$$
k_3=\frac{\pi_{\mathrm{ref}}(a|s)}{\pi_\theta(a|s)}-1-\log\frac{\pi_{\mathrm{ref}}(a|s)}{\pi_\theta(a|s)}
$$

```python
def k2_estimator(logprobs, ref_logprobs):
	ratio = logprobs - ref_logprobs
	return ratio.exp() - ratio - 1
```

#### SGD

```python
class SGD:
    def __init__(self, lr: float, max_iter: int, batch_size: int):
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size

    def fit(self, x, y, loss_fn):
        """
        假设需要拟合的是一个一次函数
        """
        n_samples, n_features = x.size()
        self.theta = torch.randn(n_features, 1, requires_grad=True)

        for it in range(self.max_iter):
            indices = torch.randperm(n_samples)

            x_shuffled = x[indices]
            y_shuffled = y[indices]

            total_loss = 0

            for i in range(0, n_samples, self.batch_size):
                X = x_shuffled[i : i + self.batch_size]
                Y = y_shuffled[i : i + self.batch_size]

                pred = X @ self.theta
                loss = loss_fn(Y, pred)
                loss.backward()

                with torch.no_grad():
                    self.theta -= self.lr * self.theta.grad

                self.theta.grad.zero_()

                total_loss += loss.item()

            avg_loss = total_loss / n_samples
            print(f"epoch={it}, loss={avg_loss:.4f}")

        return self.theta
```

### LLM

#### CrossEntropyLoss

假设真实分布为 $p(x)$，模型预测分布为 $q(x)$，那么交叉熵即为：

$$
H(p,q) = -\sum_x p(x)\log q(x)
$$

在 LLM 的语境下：
- $q(x)$ 表示模型预测得到的 token 概率分布；
- $p(x)$ 表示真实 next token 的概率分布。

由于训练数据中的真实 token 通常使用 one-hot 分布表示，因此于是交叉熵可以化简为：

$$
H(p,q)=-\log q(x_t)=-\log P(x_t \mid x_{<t})
$$


对于整条序列而言，语言模型的交叉熵损失即所有 token 交叉熵的平均值：

$$
L = -\frac{1}{T}\sum_{t=1}^{T}\log P(x_t \mid x_{<t}) = -\frac{1}{T}\sum_{t=1}^{T}\log { \frac{e^{x_t - m}}{\sum_{j=1}^{n} e^{x_j - m}}}
$$

```python
def cross_entropy(logits, labels, ignore_index=-100):
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)

    mask = labels != ignore_index
    logprobs = F.log_softmax(logits, dim=-1)
    loss = -logprobs.gather(dim=-1, index=labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return loss[mask].mean()
```

>1. `labels` 当 index 时候需要把 `clamp_min` 一下，防止 ignore_index 索引不到。 
>2. `logprobs` 再 `gather` 之后形状是 \[B, T-1]，正好用 `labels != ignore_index` mask。

#### PPL

$$
\text{PPL}(x_1, x_2, \ldots, x_N) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log q(x_i \mid x_1, \ldots, x_{i-1})\right)
$$

```python
import torch
import torch.nn.functional as F
import math

def compute_perplexity(logits, targets, ignore_index=-100):
    bs, seq_len, vocab_size = logits.size()
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction="mean"
    )
    return torch.exp(loss)
```

#### MHA

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model can not be divided by num_heads without reminders"
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        seq_len = 0
        self.mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), diagonal=1)


    def forward(self, Q, K, V):
        bs, seq_len, _ = Q.size()
        q = self.q_proj(Q)
        k = self.k_proj(K)
        v = self.v_proj(V)

        q = q.view(*q.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(*v.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(*v.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if self.mask is not None:
            scores[..., -seq_len:] += self.mask
        
        attn = torch.softmax(scores, dim=-1)
        output = (attn @ v).transpose(1, 2).reshape(bs, seq_len, self.d_model)
        return self.o_proj(output)

```

#### MHA + KVCache

```python
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # 在此实现
        assert d_model % num_heads == 0, "d_model can not be divided by num_heads without reminders"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None, mask=None):
        # 在此实现
        B, T, _ = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        if cache is not None:
            k = torch.concat([cache["k"], k], dim=2)  # seq_len 维度 concat
            v = torch.concat([cache["v"], v], dim=2)
        
        new_cache = {
            "k": k,
            "v": v
        }

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        output = (attn @ v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.o_proj(output), new_cache
```

#### GQA

![](https://img.xilyfe.top/img/20260126212710505.png)

```python
class GroupQueryAttention:
    def __init__(self, d_model, num_heads, num_kv_heads):
        assert d_model % num_heads == 0
        assert d_model % num_kv_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)



    def forward(self, x):
        B, T, _ = x.size()
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        output = (attn @ v).transpose(-2, -1).reshape(B, T, self.d_model)
        return self.o_proj(output)
```

#### MQA

```python
class MultiQueryAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        assert d_model % num_kv_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)



    def forward(self, x):
        B, T, _ = x.size()
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).unsqueeze(1)
        v = self.v_proj(x).unsqueeze(1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        output = (attn @ v).transpose(-2, -1).reshape(B, T,-1)
        return self.o_proj(output)
```

#### FFN

```python
class FeedForwardNet(nn.Module):
	def __init__(self, d_model: int, hidden_size: int):
		self.up_proj = nn.Linear(d_model, hidden_size, bias=False)
		self.down_proj = nn.Linear(hidden_size, d_model, bias=False)
	
	def forward(self, x):
		return self.down_proj(F.relu(self.up_proj(x))))
```

#### LayerNorm / RMSNorm

$$
\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 + \epsilon}} \odot \gamma + \beta
$$

```python
def layer_norm(x, eps, gamma, beta):
	mean = x.mean(dim=-1, keepdim=True)
	var = x.var(dim=-1, keepdim=True, unbiased=False)
	return (x - mean) / torch.sqrt(var + eps) * gamma + beta
```

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{H} \sum_{i=1}^{H} x_i^2 + \epsilon}} \odot \gamma + \beta
$$

```python
def rms_norm(x, eps, gamma, beta)
	rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
	return x / rms * gamma + beta
```

#### LoRA

```python
class LoRA(nn.Module):
	def __init__(self, in_feature: int, out_feature: int, rank: int, alpha: int):
		super(LoRA, self).__init__()
		self.rank = rank
		self.alpha = alpha
		
		self.A = nn.Linear(in_feature, rank, bias=False)
		self.B = nn.Linear(rank, out_feature, bias=False)
	
	def forward(self, x):
		return self.B(self.A(x)) * self.alpha / self.rank
```

应用时候给 `nn.Linear` 挂上 lora，然后把 forward 覆盖就好了：

```python
linear = nn.Linear(d_model, hidden_size)
linear.lora= LoRA(d_model, hidden_size, 8, 8)
linear.forward = lambda x:linear.lora.forward(x)+linear.forward(x)
```

#### ReLU / Swish/ SwiGLU

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

```python
def relu(x):
	return torch.clamp(x, min=0)
```

$$
\mathrm{Swish} = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}
$$

```python
def swish(x):
	return x * torch.sigmoid(x)
```

$$
\mathrm{SwiGLU}(x,W,V)=\mathrm{Swish}(xW)\odot(xV)
$$

```python
w1 = nn.Linear(d_model, hidden_size)
w2 = nn.Linear(d_model, hidden_size)
w3 = nn.Linear(hidden_size, d_model)
def swiglu(x):
	return w3(F.silu(w1(x)) * w3(x))
```

#### Advantage

$$
\hat{A}^{GRPO} = \frac{r_i - \mu}{\sigma + \epsilon}
$$

```python
def compute_grpo_adv(rewards):
	"""
	rewards: [B, G]
	"""
	mean_r = rewards.mean(dim=-1, keepdim=True)
	std_r = rewards.std(dim=-1, keepdim=True)
	advantages = (rewards - mean_r) / (std_r + 1e-8)
	return advantages
```

---

$$
\begin{align}
\hat{A}^{GAE}_t &= \sum_{l=0}(\gamma\lambda)^l \delta_{t+l} \\
&= \delta_t +\gamma\lambda\hat{A}^{GAE}_{t+1}
\end{align}
$$

```python
def compute_gae_adv(rewards, values, gamma, lam):
	gen_len = values.size(-1)
	advantages = []
	with torch.no_grad():
		for t in reversed(range(gen_len)):
			nextvalue = values[:, t + 1] if t < gen_len - 1 else 0.0
			delta = rewards[:, t] + gamma * nextvalue - values[:, t]
			adv_t = delta + gamma * lam * adv_t
			advantages.append(adv_t)
	advantages = torch.stack(advantages[::-1], dim=1)
	returns = advantages + values
	return advantages, returns
```

>这里可能会有点疑惑：PPO 里面 reward model 估计的不是 seq-level 的 reward 吗，公式里面应该需要的是 token-level reward 吧。实际像 verl 的框架会把 seq-level 的 reward 放在 \[B,T] 张量里面最后一个 token 位置上，然后随着 gae 就可以向前传播了。

#### Loss

```python
def agg_loss(
	loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    mask: torch.Tensor
):
	if loss_agg_mode == "token-mean":
		valid = torch.where(mask.bool(), loss_mat, 0.0)
		loss = valid.sum() / mask.sum()
	elif loss_agg_mode == "seq-mean-token-mean":
		seq_mask = mask.sum(dim=-1)
		seq_loss = (loss_mat * loss_mask) / (seq_mask + 1e-8)
		seq_mask = (seq_mask > 0)
		valid = torch.where(seq_mask.bool(), seq_loss, 0.0)
		loss = valid.sum() / seq_mask.sum()
	
	return loss
```

---

$$
\mathcal{L_{SFT}} = -\frac{1}{T}\sum_xp(x)\log{p(x)}
$$

```python
def sft_loss(logits, labels):
	"""
	logis: [B, T, V]
	labels: [B, T]
	"""
	logits = logits[:, :-1, :]
	labels = labels[:, 1:]
	
	loss = F.cross_entropy(
		logtis.view(-1, logits.size(-1)),
		labels.view(-1),
		ignore_index=-100,
		reduction="mean"
	)
	return loss

def sft_loss_2(logits, labels):
	logits = logits[:, :-1, :]
	labels = labels[:, 1:]
		
	logprobs = torch.log_softmax(logits, dim=-1)
	logprobs_flat = logprobs.view(-1, logprobs.size(-1))
	labels_flat = labels.view(-1)
	loss = -logprobs_flat[range(len(labels_flat)), labels_flat].mean()
	return loss
```

---

后面代码里的 `logprobs` 都默认形状为 \[B, T]：

```python
def get_logprobs(logits, labels):
	"""
	logits: [B, T, V]
	labels: [B, T]
	"""	
	logits = torch.logsoftmax(logits[:, :-1, :], dim=-1)
	logprobs = logits.gather(dim=-1, index=labels[:, 1:].unsqueeze(-1))  # [B, T-1, 1]
	return logprobs.squeeze(-1)
```

---

$$
\mathcal{L_{PPO\_{Actor}}} = -\mathbb{E}\left[\text{min}\left(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\right)\right]
$$

```python
def compute_actor_loss(logprobs, old_logprobs, advantages, mask, clip_eps=0.2):
    ratio = torch.exp(new_logprobs - old_logprobs)

    surr1 = ratio * advantages
    surr2 = (
        torch.clamp(
            ratio,
            1.0 - clip_eps,
            1.0 + clip_eps,
        )
        * advantages
    )

    # PPO objective
    policy_loss = -torch.min(surr1, surr2)

    # mask
    return (policy_loss * mask).sum() / mask.sum()
```

$$
\mathcal{L}_{PPO\_{Critic}}=\mathbb{E}\left[\max\left((V_\theta(s_t)-R_t)^2,\left(\operatorname{clip}(V_\theta(s_t),V_{\theta_{old}}(s_t)-\epsilon,V_{\theta_{old}}(s_t)+\epsilon)-R_t\right)^2\right)\right]
$$

```python
def compute_critic_loss(values, old_values, returns, ep, mask): 
	loss_1 = (values - returns) ** 2 
	loss_2 = (torch.clamp(values, old_values - ep, old_values + ep) - returns) ** 2 
	loss = torch.max(loss_1, loss_2) * mask 
	value_loss = loss.sum() / (mask.sum() + 1e-8) 
	return value_loss
```

$$
\begin{align}
\mathcal{L_{PPO\_{Reward}}} &= - \mathbb{E}_{(\alpha_x, \alpha_y) \sim D} \left[ \ln \frac{\alpha_x}{\alpha_x + \alpha_y} \right] \\
&= - \mathbb{E}_{(x, y_{win}, y_{loss}) \sim D} \left[ \ln \frac{\exp{(r(x, y_{win}))}}{\\exp{(r(x, y_{win}))} + \exp{(r(x, y_{loss}))}} \right] \\
&= - \mathbb{E}_{(x, y_{win}, y_{loss}) \sim D} \left[ \ln \frac{1}{1 + \exp{(r(x, y_{win})- r(x, y_{loss}))}} \right] \\
&= - \mathbb{E}_{(x, y_{win}, y_{loss}) \sim D} \left[ \ln  \sigma{(r(x, y_{win})- r(x, y_{loss})} \right] \\
\end{align}
$$

```python
def compute_reward_loss(inputs):
	"""
	inputs: [2B, S]
	"""
	output = model(**inputs)  # output.logits [2B, 1]
	rewards_chosen, rewards_rejected = torch.chunk(outputs.logits.squeeze(-1), chunks=2)  # [B], [B]
	loss = -nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()
	return loss
```

---

$$
\mathcal{L}_{\text{DPO}}= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Bigg[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\rm ref}(y_l|x)}\right)\Bigg]
$$

```python
def compute_logprobs(logprobs, mask):
	assert logprobs.dim() == 2
	return (logprobs * mask).sum(-1)

def compute_dpo_loss(
	chosen_logprobs, 
	ref_chosen_logprobs, 
	rejected_logprobs,
	ref_rejected_logprobs,
	beta
):
	"""
	需要 seq-level 的 logprobs
	"""
	loss = beta*(chosen_logprobs - ref_chosen_logprobs - rejected_logprobs + ref_rejected_logprobs)
	return -nn.functional.logsigmoid(loss).mean()
```

---

$$
\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G}  \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right)  - \beta D_{\text{KL}} \right]
$$

```python
def compute_grpo_loss(logprobs, old_logprobs, ref_logprobs, advantages, ep, beta, mask):
	ratio = torch.exp(logprobs - old_logprobs)
	loss_1 = ratio * advantages
	loss_2 = torch.clamp(ratio, 1 - ep, 1 + ep) * advantages
	
	# GRPO 是 k3 估计
	kl = -(logprobs - ref_logprobs) + ratio - 1
	per_token_loss = (torch.min(loss_1, loss_2) - beta * kl) * mask
	loss = loss.sum(dim=-1) / mask.sum(dim=-1) # [B]
	return loss.mean()	
```

#### Decode

1. Greedy Search

```python
def greedy_search(logits):
	"""
	返回的形状是 [B]
	"""
    return torch.argmax(logits)
```

>假设这里的 `logits` 是 NTP 时候取得最后一个 token 的 概率分布，形状为 \[B, V]

2. TopK Search

```python
def topk_search(logits, k, temperature):
	logits = logits / temperature
	probs = torch.softmax(logits, dim=-1)  # [B, V]
	topk_probs, topk_ids = torch.topk(probs, k)  # [B, K]
	selected_id = torch.multinomial(topk_probs, 1)
	return topk_ids.gather(dim=-1, selected_id).squeeze(-1)
```

3. TopP Search

```python
def topp_search(logits, p, temperature):
	logits = logits / temperature
	probs = torch.softmax(logits, dim=-1)
	sorted_probs, sorted_ids = torch.sort(probs, dim=-1, decending=True)
	cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
	mask = cumulative_probs > p
	
	mask[..., 1:] = mask[..., :-1].clone()  
	mask[..., 0] = False
	
	sorted_probs[mask] = 0.0
	sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
	sampled_idx = torch.multinomial(sorted_probs, 1)
	return sorted_ids.gather(dim=-1, sampled_idx).squeeze(-1)
```


### Algorithm
#### 最长上升子序列
#### 手撕解析括号
#### 已知 rand5 求 rand3 和 rand7
#### 英文句子分部分反转
#### 找链表第一个公共节点
#### 大数乘法
#### On 数组第 k 大
#### 编辑距离
#### 链表题（具体忘了）
#### 岛屿数量
#### 最长上升子序列
#### 三数之和