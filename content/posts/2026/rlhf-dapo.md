---
title: LLM 中的强化学习：DAPO
date: 2026-03-11T11:26:32+08:00
featuredImage: http://img.xilyfe.top/img/20260311113437709.png
authors:
  - Xilyfe
series:
  - RLHF
tags:
  - 大模型
  - 强化学习
lastmod: 2026-03-23T12:06:40+08:00
---
>**DAPO** 全称为 Decoupled Clip and Dynamic Sampling Policy Optimization，解耦裁剪与动态采样策略优化。该算法在 **GRPO** 基础上进行了重大改进。GRPO 是 PPO 的简化版，但在长 CoT 场景下容易出现**熵崩塌**、奖励噪声、训练不稳定等问题，DAPO 通过四个核心技术解决了这些问题。

$$
J_{\text{DAPO}}(\theta) = \mathbb{E} \left[ \frac{1}{\sum |o_i|} \sum_i \sum_t \min \left( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) \hat{A}_{i,t} \right) \right]
$$

## GRPO 存在的问题

### 熵坍塌

![image.png](http://img.xilyfe.top/img/20260322104812011.png)

在原始 PPO 中，为了让策略更新保持 proximal，防止一次性更新幅度过大把训练搞崩，它采用了 clip 进行裁剪：

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}\left(r_t(\theta),\ 1-\epsilon,\ 1+\epsilon\right) \hat{A}_t \right) \right]
$$

我们以 $\hat{A}_t>0$ 的情况举例，也就是当前决策优于平均水平：
- 如果 $r_t(\theta)<1-\epsilon$ 那么说明旧策略比较认可这个 action 但是新策略不偏向这个 action，所以我们希望朝当前方向优化，让梯度大一些步伐迈大一些，所以就不对梯度进行裁剪。
- 如果 $r_t(\theta)>1+\epsilon$ 那么说明旧策略不认可这个 action 但是新策略偏向这个 action，所以不需要太大梯度继续训练即可，通过 clip 防止更新幅度过大。

这么看来裁剪的思想没什么问题，但是这种裁剪就导致了 **每一步更新，token 的概率分布最多变化 $1 \pm \epsilon$**。举个例子 $\epsilon=0.2$：
- 对于 $r_t(\theta)<1-\epsilon$ 的情况，我们假设旧策略的概率为 0.9，它允许的上限为 1.08，也就是说即使新策略把概率提高到 1.0 也不会被裁剪，训练信号被充分利用。
- 对于 $r_t(\theta)>1+\epsilon$ 的情况，我们假设旧策略的概率为 0.2，它允许的上限为 0.24，也就是说假如新策略把概率提高到 0.3 也会导致被裁剪，失去了训练信号。

这正是经典的**马太效应**：已经强的越强，弱的永远弱。并且在长期积累后，策略分布越来尖锐，也就是说每次生成得到的 response 越来越固定，出现了熵崩塌，GRPO 同一 prompt 的 G 个采样响应可能完全相同，这在长 CoT 场景下特别致命：高质量推理路径往往需要先尝试低概率的“关键转折 token”，但它们被 clipping 彻底压制。

### 梯度稀疏

![image.png](http://img.xilyfe.top/img/20260322113322926.png)

GRPO 相较于 PPO 优化的一个点就是，通过组内归一化计算优势函数 Advantage：

$$
\hat{A}_t=\frac{R_i-\text{mean}(R)}{\text{std}(R)}
$$

优势函数的本质是**组内横向比较**：不看某条响应绝对意义上好不好，只看它比同组其他响应好多少。但这个公式有一个隐含的前提：**组内奖励必须存在差异**。一旦某个问题的所有 G 条 response 的 reward 都相同，比较就失去了意义：如果 reward 都相同那么 $R_i-\text{mean}(R)=0$，$\hat{A}_t=0$ 导致梯度同样为零，零梯度意味着这些样本对模型更新毫无贡献，白白浪费了采样和计算资源。

### 学习信号稀疏

$$
J_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G}  \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right)  - \beta D_{\text{KL}} \right]
$$

GRPO 的损失函数是 sample-level 的：它先对每个响应 $o_i$，先对内部的所有 token 求平均，再对 G 个响应求平均。这就导致了一个严重的问题：**长序列 token 梯度贡献被稀释，无法学习关键推理步骤**。这里举个例子，对于一个 prompt 假设 $G=2$ 采样了两个响应：
- 序列 A：长度 2 个 token，advantage = 1
- 序列 B：长度 4 个 token，advantage = 1

它们的 advantage 相同，理论上这两个序列应该被同等程度地强化。但 GRPO 采用的 sample-level 先对每个序列内部做 token 平均，再对 G 个序列平均：

$$
\nabla\mathcal{L} = -\frac{1}{2}\left[\frac{1}{2}\sum_{t\in A}\nabla_t + \frac{1}{4}\sum_{t\in B}\nabla_t\right]
$$
展开每个 token 的权重：

| token       | 权重                                             |
| ----------- | ---------------------------------------------- |
| A 的 token 1 | $\frac{1}{2} \times \frac{1}{2} = \frac{1}{4}$ |
| A 的 token 2 | $\frac{1}{4}$                                  |
| B 的 token 1 | $\frac{1}{2} \times \frac{1}{4} = \frac{1}{8}$ |
| B 的 token 2 | $\frac{1}{8}$                                  |
| B 的 token 3 | $\frac{1}{8}$                                  |
| B 的 token 4 | $\frac{1}{8}$                                  |

两个序列 advantage 相同，但 A 的每个 token 权重是 B 的两倍。在长 CoT 任务里，这带来两大致命后果：
1. **高质量长 CoT 学不到**：数学、代码等正确但很长的 CoT，本来应该贡献巨大，但权重被稀释，模型难以强化里面的关键推理 token。
2. **垃圾长输出惩罚不够**：重复、乱码、 gibberish 的长响应虽然质量差，但因为长度长、内部 token 权重低，惩罚信号弱，导致模型反而容易生成更长、更乱的东西。

### 奖励噪声

在大规模 LLM 强化学习训练中存在一个 Overlong Cutoff 的问题。RL 训练时通常会设置 max_response_length（如 16384 或 20480 token），防止显存爆炸和无限输出。 一旦响应长度超过这个硬限制，生成器就会强制截断，后面的 token 被丢掉。截断本身没有问题，问题在于：如何给被截断的响应打分？直觉上的做法，包括很多框架的默认做法是：给截断响应赋予一个 Overlong Punishment。理由似乎很充分 - 响应没有完成，自然算失败。但 DAPO 团队发现，这个“直觉正确"的做法会引入严重的奖励噪声，并给出了非常直观的解释：

>“By default, we assign a punitive reward to truncated samples. This approach may introduce reward noise into the training process, as a sound reasoning process can be penalized solely due to its excessive length. Such penalties can potentially confuse the model regarding the validity of its reasoning process.”

## DAPO 的解决方案

### Clip-Higher

Clip-Higher 解决的是熵崩塌问题，之前我们分析过出现熵崩塌主要源于：在 $r_t(\theta)>1+\epsilon$ 的情况下，旧策略的概率本来就不高，还限制了更新幅度的上限，抑制了低概率 token 的增长。Clip-Higher 采用非对称裁剪机制，解耦上下裁剪的范围：
- 上裁剪阈值 $\epsilon_{high}=0.28$：放宽低概率Token的探索限制。
- 下裁剪阈值 $\epsilon_{low}=0.2$：抑制高概率Token的过度利用。

DAPO 的 clip 公式修改为了：

$$
\begin{array}{c} \text{clip}\left( r_{i,t}(\theta), 1 - \epsilon_{low}, 1 + \epsilon_{high} \right)A_{i,t} \\ \end{array}
$$

### Dynamic Sampling

![image.png](http://img.xilyfe.top/img/20260322125231215.png)

Dynamic Sampling 解决的是梯度稀疏的问题，避免一个 prompt 采样出来的 G 个 response 奖励都相同梯度为零。它在采样过程中，过滤掉 reward 相同的组，也就是把这个 prompt 给过滤掉，重新采样填充批次，确保每轮采样的回答中至少包含不同奖励水平的样本，从而保证优势函数计算的有效性，避免梯度为零的情况。

### Token-Level Loss

![image.png](http://img.xilyfe.top/img/20260323001032246.png)


我们沿用上面的例子，DAPO 把所有 token 放在一起，总数 = 2+4 = 6，统一除以 6：

| token       | 权重            |
| ----------- | ------------- |
| A 的 token 1 | $\frac{1}{6}$ |
| A 的 token 2 | $\frac{1}{6}$ |
| B 的 token 1 | $\frac{1}{6}$ |
| B 的 token 2 | $\frac{1}{6}$ |
| B 的 token 3 | $\frac{1}{6}$ |
| B 的 token 4 | $\frac{1}{6}$ |

所有 token 权重完全相同，advantage 相同的序列得到同等强化。

### Overlong Reward Shaping

对于长响应被截断后给予负奖励引入噪声的问题，DAPO 引入两种机制：
- **Overlong Filtering**：在计算 loss 的时候，把超过 `max_length` 的样本直接 mask 掉，让这个序列不参与梯度更新，这样就只保留了最长的优质 CoT。
- **Soft Overlong Punishment**：过滤这个方法虽然有效，但也丢失了一个重要的信号：模型应该知道，过长的响应是不受欢迎的，需要学会更高效地推理。因此DAPO进一步提出了 Soft Overlong Punishment，一个渐进式的长度惩罚机制：

$$
\begin{array}{c} R_{\text{length}}(y) =  \begin{cases}  0, & |y| \leq L_{\text{max}} - L_{\text{cache}} \\  \frac{(L_{\text{max}} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, & L_{\text{max}} - L_{\text{cache}} < |y| \leq L_{\text{max}} \\  -1, & L_{\text{max}} < |y|  \end{cases} \\ \end{array}
$$

具体来说，当响应长度超过预设的最大值时，定义一个惩罚区间。在区间内，响应越长，受到的惩罚越大。该惩罚将被添加到原始的基于规则的正确性奖励中，从而引导模型避免生成过长的响应。
