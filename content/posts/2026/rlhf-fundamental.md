---
title: LLM 中的强化学习：基础知识
date: 2026-02-19T15:38:09+08:00
featuredImage: http://img.xilyfe.top/img/20260219153922089.png
authors:
  - Xilyfe
series:
  - RLHF
tags:
  - 大模型
  - 强化学习
lastmod: 2026-02-26T09:55:06+08:00
---
>在 CS224N 中已经学习了一部分的 RLHF 但是感觉都忘掉了而且学习的一知半解，这次 minimind 正好也需要用到 RL 的知识，涉及到 PPO、DPO 啥的，所以来完整学习一遍。这次学习的目标就是搞懂 LLM 中强化学习的基本概念，学会目前常用的 PPO、DPO、GRPO 这几个算法，然后能调库进行 RLHF。

在学习 RLHF 之前先问一个问题：SFT 和 RLHF 有什么区别？
SFT：给定了 prompt 和对应的 response，优化LLM策略，让模型每个token的预测分布尽可能接近真实的人工答案，本质是模仿学习。而 RLHF 给定 prompt 和对应回复以及人类偏好，优化LLM策略，让模型输出的语句符合人类偏好(用奖励函数量化评价)。SFT之后，模型已经会“模仿人”，但还不一定“符合人类偏好”（例如礼貌性、安全性、简洁性等)，RLHF就是进一步让模型输出符合人类偏好，本质是偏好学习。

也就是说 SFT 是人类喜欢怎么做他就怎么做，RLHF 是人类偏向什么他就朝那个方向学习。

## RL in LLM

>强化学习就是一种模式，它从环境中获取结果，然后对结果进行打分获得奖励，最后将其作为反馈从中进行学习。大模型中的强化学习，核心就是 **如何构造一种 loss**，来对模型进行正向或者反向激励。

在 LLM 训练中，RLHF 和 Pretrain 或者 SFT 不同。Pretrain 和 SFT 都是采用 Teacher-Forcing 的方法，也就是说我们需要提前准备好问题和答案；但是 RLHF 中例如 PPO 不需要准备语料，只需要准备好问题让 LLM 进行 next-token 的预测，预训练好的打分模型会对回答进行打分得到奖励。

![image.png](http://img.xilyfe.top/img/20260219160939561.png)


大模型生成序列的过程可以看作一个 **马尔可夫决策过程 (MDP)**：
- **Episode (回合)**：从给出 Prompt 到生成结束（出现 EOS 或达到最大长度）。
- **Step (步)**：生成每一个 Token 的过程。
- **Agent (智能体)**：LLM 模型自身。
- **Environment (环境)**：已生成的上下文。
- **Action ($A_t$)**：当前预测生成的 Token。
- **State ($S_t$)**：当前的 Prompt + 已生成的 Token 序列。
- **Reward ($R_t$)**：环境（或奖励模型）给出的即时反馈。

例如，围棋的一局，超级马里奥游戏中从游戏开始到救出公主的过程，或者语言模型生成一个句子的过程，这些都是一个episode。围棋中某位棋手的一次落子，超级马里奥游戏中玩家的一次操作，或者语言模型生成句子中的一个token，这些都是一个step。

第t个step中，agent与环境交互包含以下步骤（如上图）：
1. agent收到来自环境的状态$S_t$
2. 基于该状态 $S_t$，agent采取动作$A_t$
3. 环境进入新状态 $S_{t+1}$ 
4. 环境会给agent带来一些奖励 $R_t$

在 LLM 的语境下，给出一个 prompt 生成 response 的过程就是一个 episode，生成每一个 token 就称为一个 step。我们希望一个episode 中所有奖励之和能够越大越好。因此 agent 的目标是最大化一个 episode 中所有奖励之和的期望（之所以是期望而不是精确值，是因为采取动作后进入哪个新状态是环境说了算的，具有一定的随机性）。

## Actor-Critic 算法

![image.png](http://img.xilyfe.top/img/20260219165713553.png)

Actor-Critic 算法包含两个角色：演员 actor 和 评判员 critic。在大模型的语境中，actor 就是我们 LLM，它会对 prompt 预测不断预测出下一个 token；critic 也是一个神经网络，它通常需要输入 $S_t$ 和 $A_t$ 两个向量，然后输出一个标量代表预测的收益。

以上图为例，actor 通过游戏机的环境做出下一个 action，然后 critic 根据 actor 的动作给出评价，然后 actor 根据评价再调整做出下一个 action。但是这里的 critic 更像一个 **预言家**，因为Critic 的核心作用是 **预测未来的长期期望回报**，而不是仅仅评估当前的即时收益。我们之前说过：

>我们希望一个episode 中所有奖励之和能够越大越好

我们的目标不是让当前这一步得分最高，而是让整个 episode 的累积 $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$ 收益最大化。但是因为未来的 $R_{t+2}, R_{t+3}$ 等收益在当前是未知的，所以我们需要让 Critical 来估算未来的收益总和。假如如果没有评论家，就必须得等到episode 结束才能知道收益总和。

在 Actor-Critical 中每个 step 会发生以下四件事：
1. 演员收到来自环境的状态 $S_t$
2. 演员生成动作  ，然后评论家估计状态动作价值 $Q(S_t, A_t)$ 。演员用 $loss = -\log p(A_t|S_t) Q(S_t, A_t)$ 来更新参数
3. 环境收到 $A_t$ 之后给出 $S_{t+1}$ ，更新参数后的演员用 $S_{t+1}$ 生成 $A_{t+1}$
4. 环境给出 $R_t$ ，评论家用 $loss = [Q(S_{t+1}, A_{t+1}) + R_t - Q(S_t, A_t)]^2$ 来更新参数

 先来理解一下演员模型的 loss 函数。我们忽略 $-log$，损失函数里面含负对数的原因我们在负对数似然就已经了解了。当 Critical 预测价值 $Q(S_t, A_t)>0$，那我们肯定希望这个 action 的概率分布尽可能大，Actor 就会尽可能更新参数来使 $p(A_t|S_t)$ 变大。如果预测价值 $Q(S_t, A_t)<0$，那么会希望它的概率分布尽可能小，Actor 更新参数使 $p(A_t|S_t)$ 变小。

那评判员模型的 loss 函数呢？$Q(S_{t+1}, A_{t+1}) + R_t - Q(S_t, A_t)$ 其实就是预测值和真实值的差距，$Q(S_t, A_t)$ 实际上等于 $Q(S_{t+1}, A_{t+1}) + \hat{R_t}$，作差就能得到 $R_t - \hat{R_t}$。我们会希望差距的绝对值尽可能小，以此来优化评判员模型，让他尽可能贴近真实的奖励。
## A2C 算法

A2C 全程是 Advantage Actor-Critic，是 Actor-Critic 算法的改良。

它的思想很简单：假如你和你的朋友都是学生，你平时考试考90分，他平时考试考60分。经过一个月的期末复习，在期末考试中你考了96分，他考了95分，你觉得谁的期末复习策略是成功的？显然你朋友的期末复习策略是更成功的。虽然你考了更高的分数，但这个分数基于你平时的积累，相当于是正常发挥了。而你朋友却是超常发挥。因此单看期末，他的复习策略更值得他好好强化。

在此基础上，A2C 引入了一个 **优势 adv** 的概念来代替之前的 $Q$。假设评论家的预估动作价值为 $Q(S_t,A_t)$ 预估状态价值为 $V(S_t)$ 那么：

$$
\text{Adv} = Q(S_t,A_t) - V(S_t)
$$

- 演员 $loss=-\log{p(A_t|S_t)Adv(S_t,A_t)}$
- 评论家 $loss=Adv^2(S_t,A_t)$ 

这里的 $V(S_t)$ 和之前的 $Q(S_t,A_t)$ 有什么区别呢？
首先 $Q(S_t,A_t)$ 指的是：在状态 $S_t$ 下，执行特定动作 $A_t$ 之后，未来能拿到的总收益。比如：如果你这步棋走‘跳马’，你未来的胜率是 80%。而 $V(S_t)$ 指的是：在状态 $s$ 下，按照当前的策略继续走下去，执行任何 action 平均能拿到的总收益。比如：你现在的盘面大优，平均胜率是 70%。那如果像 Actor-Critical 模型一样采用 $Q(S_t,A_t)$ 就存在一个问题。假如现在的状态非常好，不管选哪个 action 都能得到一个很好的收益，$Q(S_t, A_T)$ 都会很大，这就会导致梯度更新方向不稳定。所以 A2C 采用 $Q(S_t,A_t) - V(S_t)$ 就能得到 **当前策略是否比平均水平好多少**。

在 A2C 的工程实现中通常不单独训练一个 $Q$ 网络，而是利用 **时序差分误差 TD Error** 来代替 $Q$。我们根据贝尔曼方程：

$$
Q(S_t, A_t) \approx R_t + \gamma V(S_{t+1})
$$

就可以得到新公式：

$$
\delta_t = R_t + \gamma V(S_{t+1}) - V(S_t)
$$

这样只需要学习一个 $V$ 函数，就能同时得到状态价值估计和优势估计，无需额外学习复杂的 $Q$ 函数。但如果只用一步的 TD Error 作为 Advantage，虽然偏差小，但波动很大。为了平衡准确度和稳定性引入了 GAE。这里不对强化学习的知识点做过多介绍，GAE 主要的思想就是 **综合考量未来的变化，做一个加权平均**。

$$
A_t^{GAE} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots + (\gamma\lambda)^{T-t}\delta_T
$$

这里的 $\lambda$ 是一个超参数，一般取 0.95：
- 如果 $\lambda = 0$，GAE 就退化成了单纯的 TD Error。
- 如果 $\lambda = 1$，GAE 就变成了蒙特卡洛采样，算出整条路径的总和。

>代码实现上，由于算出 t 时刻的 $A_t^{GAE}$ 需要之前 t+1 时刻之后的 $\delta_t$ ，所以需要对 token 逆序的处理。

于是 A2C 的步骤即为：
1. 演员收到来自环境的状态 $S_t$，生成动作 $A_t$
2. 环境收到 $A_t$ 之后给出奖励 $R_t$ 和新状态 $S_{t+1}$
3. 评论家估计状态价值 $V(S_t)，V(S_{t+1})$， 并计算优势 $Adv(S_t, A_t) = A_t^{GAE}$
4. 演员用 $loss = -\log p(A_t|S_t) \text{Adv}(S_t, A_t)$ 更新参数
5. 评论家用 $loss = [\text{Adv}(S_t, A_t)] ^2$ 更新参数

## KL 散度

在 RLHF 中存在 Reward Hacking 这个概念，训练过程中模型可能会朝着 Reward 倾向的方向走捷径。比如 prompt 是 "今天的天气如何"，那么模型会生成天气情况的分析然后再告诉今天的天气，依次来获取更高的 reward。所以我们需要在 loss 中增加一项，避免模型过度学习，或者说让训练后的模型和原模型差距小一些。RLHF 中引入了一个 Ref Model，这个模型就是经过 SFT 训练后冻结的模型。我们用它和新模型计算 KL 散度，来衡量模型相较于原先的变化。KL 散度的计算公式为：

$$
\text{KL} = \frac{1}{n}\sum_{\text{response}}{log{\frac{p(a|s)}{p_{ref}(a|s)}}}
$$

 首先，我们喂一个 prompt 给 Actor 模型，让它正常输出对应的 response。response 中每一个 token 都有它对应的概率分布，我们把它记为 log_probs。我们把 Actor 生成的"prompt + response" 以 Teacher-Forcing 的方式喂给 Reference 模型，那么它同样能给出 response 中每个 token 的 log_prob 结果，我们记其为 ref_log_probs。把这两个概率分布作差，然后再求对数之和的平均值，就是 KL 散度了。

KL 散度的标准定义应该是：对于单个 token 的 KL 散度是要对 vocab 上 **每一个 token** 的概率分布作差。但是在 RLHF 的实际实现中，**KL 只针对 Actor 实际生成的 token 计算概率差**，也就是计算 `log p_actor(response[t]) − log p_ref(response[t])`。

Hugging Face 的 trl 库中 KL 惩罚的计算逻辑可以简化如下：

```python
# 1. 获取当前模型生成 token 的 log_prob
log_probs = actor_model.get_log_probs(states, actions) 

# 2. 获取参考模型 (Ref) 生成同样 token 的 log_prob
# 注意：这里不需要 ref_model 输出整个 vocab 的分布，只需要 gather 对应 action 的值
ref_log_probs = ref_model.get_log_probs(states, actions) 

# 3. 计算 KL 估计
kl = log_probs - ref_log_probs

# 4. 加入 Loss
# 通常还会乘一个系数 beta，并且有时会取 abs(kl) 或者根据方向调整
loss = ppo_loss + beta * kl 
```

`get_log_probs` 函数内部通常是通过 `gather` 操作，直接从 logits 中取出对应 action 索引的值，而不是遍历整个 vocab。

>强化学习中标准的 KL Divergence 公式为：
>
>$$D_{KL}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$
>
>这里的 $x$ 是全部的可能性，而我们在 PPO 里面无法遍历所有可能的状态和动作 $x$ 来计算上面的求和 $\sum$。我们只能通过**采样**来进行蒙特卡洛估计：
>
>$$\sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \sum_{x_i}log\frac{P(x_i)}{Q(x_i)}$$



