---
title: RL训推不一致的原因 & 解决方案
date: 2026-04-26T15:31:35+08:00
featuredImage: ""
authors:
  - Xilyfe
series:
  - 面经
tags: []
hiddenFromHomePage: true
lastmod: 2026-04-27T11:13:51+08:00
---
## 1. 问题

随着现在强化学习训练的规模越来越大，大部分的公司都把强化学习的 inference 和 training 分为两个部分。在 inference 或者说 rollout 阶段，我们用推理引擎（vllm/sglang）。它带来了 paged attention、kv cache 和其它优化过的 kernal 以及特定的浮点精度设置（例如 BF16、FP8 等），我们追求快速生成与高 throughput。然后在梯度更新阶段采用训练引擎（fsdp/megatron）等等，它采用的分布式框架可以提高训练规模，我们关注数值稳定性与精确梯度计算，并可能采用不同的浮点精度和算子实现。但是由于两部分用不同框架进行训练，推理引擎和训练引擎对同一个 prompt 计算得到的 logprobs 不一致，就导致了 on-policy 变成了 off-policy，我们用于训练的数据不是真的有模型自己生成了。这种微小差异在监督学习里可能不会显著，但在 RL 中会被放大，因为强化学习依赖概率比率来估计梯度。

## 2. 原因

### 2.1 数值精度差异

浮点加法不满足结合律，加之训练引擎和推理引擎使用不同的CUDA kernel、不同的并行策略（TP/DP size）、不同的精度格式（BFI6/FP8），导致浮点运算的计算顺序、数值舍入不同，具体来说：
- 并行策略不同：如训练使用 FSDP/Megatron TP=4 但是推理阶段使用 vLLM TP=8，AllReduce存在差异。
- kernel 实现不同：如 FlashAttention v2 和 Triton attention，softmax/fused op 计算路径不同。
- 精度格式不同或精度本身过低：FP8 rollout + BFI6 training，浮点尾数不同存在舍入误差。

因此即使是同一权重/同一输入，在两个引擎中也会产生微小但持续累积的数值偏差，即训推不一致。

### 2.2 MoE Router路由不一致

和 GSPO 的背景一样，混合专家 MoE 模型在推理时仅激活部分专家模块提升计算效率。但是推理和训练框架之间的差异可能导致即使对于相同的输入，在推理和训练过程中专家路由出现不一致。这种不一致性引发激活参数子空间的突变，导致优化过程不稳定。并且随着 MoE 模型规模增大，这种 MoE Router 离散放大的训推不一致变得更严重。

### 2.3 工程配置不对齐

这类问题不是必然误差，而是可以通过规范化工程流程完全消除的配置问题：
- 异步时间滞后：rollout 引擎使用过时的权重 checkpoint，并加剧 off-policy 问题。
- 采样参数不同步：temperature 缩放未在 vLLM 侧正确设置等操作不对齐。

## 3. 为什么会影响训练

经典的 RLHF 算法 PPO 为了提高效率，用每次采用的 trajectory 训练 ppo_epochs 次，从 on-policy 变成了 off-policy。然后为了调整新策略 $\pi_{\theta}$ 和旧策略 $\pi_{\theta_{old}}$ 的差异，PPO 引入了重要性采样来优化：

$$
\frac{\pi_{\theta}(y_t \mid x, y_{<t})}{\pi_{\theta_{old}}(y_t \mid x, y_{<t})}
$$

问题在于生成 rollout 用的是推理引擎，计算梯度用的是训练引擎。两套系统跑同一个模型，但由于精度、并行方式、kernel 实现不同，算出来的 $\pi_{\theta_{old}}$​​ 数值会有偏差。也就是说，分母记录的"当初生成时的概率"其实是算错的。假设推理引擎生成了 token A，真实概率是 0.6，但因为精度误差，记录下来的 $\mu_{\theta_{old}} = 0.3$。训练时计算比值：

$$r_t = \frac{\pi_\theta(A)}{0.3}$$

分母被低估了一倍，比值直接虚高一倍。训练框架看到这个比值会觉得现在的策略和当时生成时差距很大，触发 clip 或者产生很大的梯度——但这个信号是假的，实际上策略根本没变那么多。更糟的是，这个误差在每个 token、每条样本上都存在，而且方向随机——有的 token 分母被低估，有的被高估，梯度信号就变成了噪声，训练自然不稳定。

## 4. 重要性采样

![image.png](http://img.xilyfe.top/img/20260426231906619.png)

重要性采样 IS 的核心思想是用一个分布 $q(x)$ 采样的数据去估计另一个分布 $p(x)$ 下的期望，只需要乘以一个修正比率：

$$
\begin{align*}
\mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_{x \sim q}\left[ f(x) \cdot \frac{p(x)}{q(x)} \right]
\end{align*}
$$

- $p(x)$ 是真正想估计期望的分布（目标策略）
- $q(x)$ 是实际用来采样的分布（行为策略）
- $\frac{p(r)}{q(x)}$ 就是重要性比率

**在LLM RL语境下：** 数据是用行为策略采样得到的，但我们要优化的是目标策略。IS 比率就是修正致据来源和优化目标不一致的工具。两个策略分离主要是由于理论和工程上的 off-policy 导致的，比如 PPO 中 replay 多轮 rollout 的 trajectory 重复训练，就需要乘上一个 importance ratio。 

当前由于大规模 LLM RL 为了效率把 rollout 采样与梯度计算分开到不同的引擎上，出现了训推不一致问题，重要性采样做了进一步扩展。它引入了第三个量 **μ**，对应的是工程上实际跑 rollout 的**推理模型策略**：

$$
\frac{\pi_{\theta}(y_t \mid x, y_{<t})}{\mu_{\theta_{old}}(y_t \mid x, y_{<t})}=\frac{\pi_{\theta_{old}}(y_t \mid x, y_{<t})}{\mu_{\theta_{old}}(y_t \mid x, y_{<t})} \times \frac{\pi_{\theta}(y_t \mid x, y_{<t})}{\pi_{\theta_{old}}(y_t \mid x, y_{<t})}
$$

Training-Inference Discrepancy、Policy Stalenessi两项各自的来源：
- **训推不一致度（Training-Inference Discrepancy）**：同一权重、同一输入下，训练引擎 $\pi_{\theta_{old}}$ 和推理引擎 $\mu_{\theta_{old}}$ 的数值差异。来源于kernel 实现、并行策略、精度格式等基础设施层面的差异。
- **策略陈旧度（Policy Staleness）**：在训练引擎内，新策略 $\pi_{\theta}$ 与采样时旧策略 $\pi_{\theta_{old}}$ 的差异。来源于 off-policy 更新（比如同一批数据多次mini-batch更新）。PPO/GRPO 的 clip 机制就是为了优化这一项。

两个 IS 修正解决的是不同问题。

## 5. 解决方案

### 5.1 TIS

TIS 全称是 Truncated Importance Sampling，阶段重要性采样。它的思想类似 PPO 的 clip，就是用权重补偿训推框架的 logprobs 分布差异：

$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi^{\text{infer}}_{\theta_{\text{old}}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \text{clip}\left(\rho_{i,t}, 1/C, C\right) \cdot \min\left(r_{i,t}\hat{A}_{i,t},\ \text{clip}\left(r_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}\right)\hat{A}_{i,t}\right) \right]$$

其中 clip 部分就是用权重补偿的部分：

$$\rho_{i,t} = \frac{\pi^{\text{train}}_{\theta_{\text{old}}}(y_{i,t} \mid x,\ y_{i,<t})}{\pi^{\text{infer}}_{\theta_{\text{old}}}(y_{i,t} \mid x,\ y_{i,<t})}$$

### 5.2 IcePop

蚂蚁团队的 IcePop 对不一致度超过阈值的样本直接被 pop 掉，不参与梯度计算：

$$
\mathcal{L}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi^{\text{infer}}_{\theta_{\text{old}}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \text{pop}\left(\rho_{i,t}, 1/C, C\right) \cdot \min\left(r_{i,t}\hat{A}_{i,t},\ \text{clip}\left(r_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}\right)\hat{A}_{i,t}\right) \right]
$$

它和 TIS 的区别是 IcePop 会抑止那些不匹配率过大的样本：

$$
\begin{array}{c} \text{pop}(\rho_{i,t}, 1/\beta, \beta) = 
\begin{cases} 
\rho_{i,t}, & 1/\beta \leq \rho_{i,t} \leq \beta \\ 
0, & \text{otherwise} 
\end{cases} 
\end{array}
$$

### 5.3 Routing Replay

![image.png](http://img.xilyfe.top/img/20260426230632618.png)


Rollout Routing Replay 会在模型进行推理时，记录下每个 token 的 router 分布，然后在后续的训练过程中使用这些 router 分布进行计算。通过这种方式，强制训练过程模仿并对齐推理时的 router 行为，从而弥合两者之间的鸿沟。


## 6. 每批数据更新一次 PPO 还需要重要性采样吗

PPO 的工程实现里为了提高效率，把每次采样得到的 trajectory 重复训练 ppo_epochs 次 就导致了 on-policy 变成了 off-policy。那如果像 GRPO 那样每次只更新一轮，PPO 还需要 IS 权重来修正吗？答案取决于两个方面：

1. 采样单引擎采样与训练还是分离训练：如上文提到的，如果是单引擎采样+单次更新则是完全的 on-policy 就不需要重要性采样来修正了。
2. 是否采用 mini-batch 策略：如果一批 rollout 数据被拆成 K 个 mini-batch 依次更新（即便外层只循环一轮，N=1)，那么从第 2 个mini-batch 开始，参数已经被第 1 个 mini-batch 更新过了，此时 $\pi_{\theta} \neq \pi_{\theta_{old}}$ 策略陈旧度项不再为 1，严格意义上已经是 off-policy，仍需要 IS 修正。
