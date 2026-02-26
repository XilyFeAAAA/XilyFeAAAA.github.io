---
title: LLM 中的强化学习
date: 2026-02-19T15:38:09+08:00
featuredImage: http://img.xilyfe.top/img/20260219153922089.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 大模型
  - 强化学习
lastmod: 2026-02-25T01:03:40+08:00
---

>在 CS224N 中已经学习了一部分的 RLHF 但是感觉都忘掉了而且学习的一知半解，这次 minimind 正好也需要用到 RL 的知识，涉及到 PPO、DPO 啥的，所以来完整学习一遍。这次学习的目标就是搞懂 LLM 中强化学习的基本概念，学会目前常用的 PPO、DPO、GRPO 这几个算法，然后能调库进行 RLHF。

在学习 RLHF 之前先问一个问题：SFT 和 RLHF 有什么区别？
SFT：给定了 prompt 和对应的 response，优化LLM策略，让模型每个token的预测分布尽可能接近真实的人工答案，本质是模仿学习。而 RLHF 给定 prompt 和对应回复以及人类偏好，优化LLM策略，让模型输出的语句符合人类偏好(用奖励函数量化评价)。SFT之后，模型已经会“模仿人”，但还不一定“符合人类偏好”（例如礼貌性、安全性、简洁性等)，RLHF就是进一步让模型输出符合人类偏好，本质是偏好学习。

也就是说 SFT 是人类喜欢怎么做他就怎么做，RLHF 是人类偏向什么他就朝那个方向学习。

## 前置知识

### RL in LLM

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

### Actor-Critic 算法

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
### A2C 算法

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

### KL 散度

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

其次，目前最标准的做法是吧 KL 散度加在 Reward Function 中：

$$
r_{\text{final}}(s, a) = r_{\text{original}}(s, a) - \beta \cdot \log \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}
$$

主要原因在于：在强化学习中，KL 散度被视为每一步的“代价”。如果 Value 网络只学习原始奖励，而 Actor 却受到 KL 惩罚，那么 Value 网络估计的 $V(s)$ 和 Actor 实际经历的回报就不匹配，导致 Advantage 计算不准确（Baseline 偏差）。

## PPO

### 基础概念

#### 近端策略优化

PPO 算法可以看成 A2C 的优化版。A2C的训练策略是 “采样一次，更新一次，然后扔掉数据”，这就导致效率很低，每批数据只能用一次。PPO 采用 **近端策略优化**，它会对一批数据先进行采样得到 `logprob`，`ref_logprob`，`rewards`，`advantages` 等数据，然后进行 `ppo_epochs` 次循环。每次循环内，变化的只有概率分布 `logprob` 和 `ref_logprob` 和 `values`，优势奖励这些都固定采用第一次得到的数据。

>举个例子：A2C 就像在表演现场，你一边演，导演一边喊“好”或“坏”，然后你得到反馈就修改。改完之后，刚才演的那段戏就没用了，你必须重新演一段，导演才能给新反馈。而 PPO 更像 **复盘录像**，你先演一段戏录下来，接下来的 4 个 Epoch 你坐在监视器前，对着这段录像反复琢磨。第一遍根据反馈改一点，第二遍在第一遍改动的基础上，再对着录像微调。

为什么我们在同一批数据上进行多次梯度下降，要保持 `advantage` 和 `rewards` 不变？我们从两个角度进行分析。
首先我们用一个例子帮我们抽象的理解一下。假如你是 Kobe，直升机坠毁前还在湖人打球录下了一盘比赛。如今在天上复盘这场球赛，即便你现在球技进步了，回看当年的录像，每一个进球在当时的价值（Advantage）是客观事实，不随你现在的水平变化。其次，如果我们不固定 advantage 和 rewards，用新的策略来更新 advantage 和 rewards，可能造成策略偏离太远，导致过拟合或不稳定。并且我们计算的 adv 和 reward 都是基于 old actor model 得到的，假如每个 epoch 都重新计算新的 adv 和 reward，由于 action 是在旧的 model 上得到了就会不匹配，变成 off-policy。而 ppo 是 on-policy，clip 机制（后文会提到）就会崩溃了。

#### 更新约束

他主要是解决了 A2C 训练不稳定的问题。在 A2C 中，如果学习率设得稍微大一点，一次更新可能让策略发生很大的变化，如果 Actor 突然学到了一个极其糟糕的动作，整个策略可能瞬间崩塌。

那我们如何衡量策略的变化幅度呢？我们可以看两个策略执行相同动作得到结果的差别。在 LLM 中就是，我们对更新参数前后的模型 p 和 p' 都输入 token，就能得到的不同的概率分布 $p(A_t|S_t)$ 和 $p'(A_t|S_t)$。我们定义一个比率 $r_t(\theta)$，表示**新策略**和**旧策略**产生某个动作的概率比：

$$
r_t(\theta) = \frac{p(a_t|s_t)}{p'(a_t|s_t)}
$$

- 如果 $r_t > 1$：说明这个动作在新策略中出现的概率变大了。
- 如果 $r_t = 1$：说明新旧策略完全一样。

进而有了演员的 loss 函数：

$$
loss = -\frac{p(A_t|S_t)}{p'(A_t|S_t)} \text{Adv}(S_t, A_t)
$$

我们从梯度更新的角度思考一下这个公式的意义，先把公式改一下 $loss = -p(A_t|S_t) \times \frac{\text{Adv}(S_t, A_t)}{p'(A_t|S_t)}$，由于 adv 和 p‘ 都不在计算图里面不回传梯度，所以梯度表达式为 $\frac{\partial{loss}}{\partial{p}}=- \frac{\text{Adv}(S_t, A_t)}{p'(A_t|S_t)}$。假设某次 action 的 $Adv>0$，也就是决策优于平均水平，那我们肯定希望提高这个 action 的概率，也就是增加 $p(A_t|S_t)$。假如原先 $p'(A_t|S_t)$ 也很大，也就是原先模型也认为应该执行这个 action，那么梯度的绝对值就会变小，不会做出非常大改变。假如原先 $p'(A_t|S_t)$ 很小，旧策略觉得这个动作几乎不可能发生，但实际采样出来发现这个动作效果出奇的好（也就是Adv 很大）。重要性采样公式 $\frac{p}{p'}$ 会认为：这是一个被旧策略严重低估的好 action，于是算法会试图剧烈地提高 $p(A_t|S_t)$。

现在，我们已经限制了策略 $p(A_t|S_t)$ 的更新幅度，但还缺少一个“熔断机制”。什么意思呢？就是万一策略的更新幅度还是太大了，我们要停止策略的参数更新。PPO的做法是什么呢？因为 $\frac{p(A_t|S_t)}{p'(A_t|S_t)}$ 衡量了旧策略和现行策略之间差异，所以可以为它设置两个阈值。为了方便描述，我们令 $r(A_t, S_t) = \frac{p(A_t|S_t)}{p'(A_t|S_t)}$：

- 当 Adv 大于 0 时，若 r 大于 1.2，则停止参数更新
- 当 Adv 小于 0 时，若 r 小于 0.8，则停止参数更新

诶，那为什么我们不用管 Adv 大于 0 和 r 小于 0.8 的情况？或者 Adv 小于 0 或者 r 大于 1.2 的情况？Adv 大于 0 的情况说明当前策略是好的，如果 r 小于 0.8 说明：这个策略是好的，旧模型偏向这个策略，但是新模型不怎么偏向这个策略了，那我们肯定希望能尽可能朝现在这个方向来更新参数，所以不会进行 $max(r, 0.8)$。同样 Adv 小于 0 的情况说明当前策略不怎么行，如果 r 大于 1.2 则说明这个不好的策略现在很看好，那我们肯定希望加大力度更新参数来避免这个 action，于是不应该限制更新的幅度。

这种熔断机制可以表示为：

$$
loss = -\min(r(A_t, S_t) \text{Adv}(S_t, A_t),\ \text{clip}(r(A_t, S_t) , 0.8, 1.2) \text{Adv}(S_t, A_t))
$$

- Adv 大于 0，r 大于 1.2：min 操作就会取右边的值，此时 loss 中就只剩常量了，不产生任何梯度；而 r 无论多小都还是会产生梯度。
- Adv 小于 0，r 小于 0.8：min 操作就会取右边的值，此时 loss 中就只剩常量了，不产生任何梯度；而 r 无论多大都还是会产生梯度

熔断机制就成功了。

#### 评论家 loss

在之前我们提过，A2C 的损失函数是 $loss = [\text{Adv}(S_t, A_t)] ^2$。但是在 PPO 中，advantages 在 optimization 阶段就应该生成了，并且在后面多轮训练中不变是一个定值，那么这个 loss 完全不依赖与新的 critic 模型的参数，无法更新 value model。那我们应该如何修改 critic loss 呢？

我们来跟着最自然的思考顺序，首先我们的最终目标是让 critic 的输出 $V(s)$ 必须尽可能接近真实的状态价值：

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s),\ \tau \sim P} \Big[ \sum_{k=0}^\infty \gamma^k r_{t+k} \Big]
$$

如果 value 估得准，advantage = q - v 就会低方差，actor 就能稳定地知道“哪个动作比平均好多少”。 所以我们必须让 $V(S_t)$ 不断逼近这个“真实平均回报”。但是真实 $V(S_t)$ 根本拿不到，于是我们想到状态价值 $V(S_t)$ 正好等于所有可能动作的动作价值 $Q(S_t,A_t)$ 在当前策略下的期望：

$$
V^\pi(s) \equiv \mathbb{E}_{a \sim \pi(\cdot|s)} \big[ Q^\pi(s,a) \big]
$$

由于我们的 advantage 本身定义就是 $Q(S_t,A_t) - V(S_t)$，所以我们移项可以得到 $Q(s,a) = V(s) + \text{Adv}(s,a)$。在 PPO 中，我们虽然没有训练单独的 Q 网络，但我们用 GAE 算出了一个高质量的 Advantage 估计：

$$
A_t^{\text{GAE}} \approx Q(s_t, a_t) - V_{\text{old}}(s_t)
$$

因此：

$$
Q(s_t, a_t) \approx V_{\text{old}}(s_t) + A_t^{\text{GAE}}
$$

我们把这个近似值起个名字叫 returns，它就是我们目前能得到的最好的 $Q(S_t,A_t)$ 采样估计。因此就有了 critic loss：

$$
\text{loss}_{\text{critic}} = \left( V_{\text{new}}(s) - \text{returns} \right)^2
$$

### trl 库源码分析

`trl.experiment.ppo.PPOTrainer.train()` 方法内部依次进行如下操作：
1. rollout 阶段：将数据集的 prompt 传给 actor 采样 response，我们就得到了 prompt+response 的问答对。
2. evaluation 阶段：用 reward 模型给这个问答对打分数 `scores`，注意 **这个分数是序列级的而不是 token 级的**。
3. optimization 阶段：把 prompt+response 用 Teacher-Forcing 的方式送入 ref、actor 和 critic 模型得到 response 中每个 token 的概率 `ref_logprob` 和 `old_logprob`，以及逐 token 的预期收益 `old_values`。根据之前计算出的整个序列的 reward，我们可以计算出每个 token 对应的 reward，这样 advantage 也就计算出来了。
4. 重复 ppo_epochs 个阶段，不断把 prompt+response 用 Teacher-Forcing 的方式传入 actor 得到每个 token **新的概率分布**，把 response 传入 critic 得到 values。然后利用之前 optimization 阶段得到的 reward 和 advantages 来计算 actor 和 critic 的 loss，更新这两个模型。

我借用知乎的几张图片来图解一下这个过程：

![image.png](http://img.xilyfe.top/img/20260224122122139.png)


前面我们提到，evalution 阶段计算的 reward scores 是序列级的，但是 PPO 在每个 step（对应生成序列中的每个token）都需要计算 advantage 来更新 actor model，这样不是矛盾了吗？
实际上 reward 模型在计算序列级 reward 的时候没有加入 kl 散度，这时候计算得到分数我们叫做 `scores`。在每一个 step，我们通过 `scores`，`ref_logprob`，`old_logprob` 计算得到这个 token 的 reward，$reward = scores - \beta*kl(old\_logprob, ref\_logprob)$，最后用这个 token 对应的 value 和 reward 计算 advantage。

#### rollout

```python
batch["response"] = []
query_batch = batch["input_ids"]
for query in  query_batch:
	gen_len = output_length_sample()
	generation_kwargs["max_new_tokens"] = gen_len
	resp = ppo_trainer.generate(query, **generation_kwargs)
	batch["response"].append(resp.squeeze()[-gen_len:])
```

`output_length_sample()` 作用是 **为每个生成请求动态采样一个生成长度**，这样具有随机性或可控分布，不是固定死的长度。

#### evaluation

```python
texts = [q + r for q, r in zip(batch["query"], batch["response"])]
reward_out = reward_model(texts)
scores = [torch.tensor(output[1]["score"]) for output in reward_out]
```

#### optimization

```python
old_logprobs, _, values, masks = self.batched_forward_pass(actor_model, queries, responses)
ref_logprobs, *_ = self.batched_forward_pass(ref_model, queries, responses)
```

由于是 batch 训练，所以需要记录下 padding 位置方便后面进行遮盖。

```python
rewards, non_score_rewards = [], []
for score, old_logprob, ref_logprob in zip(scores, old_logprobs, ref_logprobs):
	kl = old_logprob - ref_logprob
	
	non_score_reward = -self.kl_ctl * kl
	non_score_rewards.append(non_score_reward)
	
	reward = non_score_reward.clone()
	last_non_masked_index = mask.nonzero()[-1]
	reward[last_non_masked_index] += score
	rewards.append(reward)
```

前面提到过，Advantage 采用了 GAE 所以需要逆序从后往前计算：

```python
advantanges = []
for t in reversed(range(gen_len)):
	value_t1 = values[:, t+1] if t < gen_len - 1 else 0.0
	delta = rewards[:, t] + self.gamma * value_t1 - values[:, t]
	adv_t = delta + self.gamma * self.lam * adv_t
	advantages.append(adv_t)

advtanges = torch.stack(advantages[::-1])
tgt_return = advantages + values
```

进行 `ppo_epochs` 轮训练，每轮训练 minibatch 条数据：

```python
for epoch in range(ppo_epochs):
	for batch in minibatch:
		logprobs, logits, values, _ = self.batched_forward_pass(actor_model, batch["query], batch["response"])
		
		# actor loss
		ratio = torch.exp(logprobs - old_logprobs)
		pg_losses = -advantages * ratio
		pg_losses_2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0)
		loss = torch.max(pg_losses, pg_losses_2).mean()
		
		# critic loss
		value_pred_clipped = old_values + torch.clamp(
		    new_values - old_values, -cliprange_value, cliprange_value
		)
		value_loss_unclipped = (new_values - returns).pow(2)
		value_loss_clipped   = (value_pred_clipped - returns).pow(2)
		value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
```

### 代码实战

我这次选择直接复现 bilibili 一个 up 主的 ppo 项目 [owenliang/hf-ppo](https://github.com/owenliang/hf-ppo/blob/main/README.md) -  让大模型学会说脏话。由于采用的是 Qwen 的基模不太可能输出脏话，直接在 base 模型上进行 ppo 很难训练起来，所以我先用数据集对 base 模型进行 sft，然后在 sft 的基础上进行 ppo，这样就能完成整个流程。

这次整体的计划就是先对 Qwen 的基模进行 sft，然后在这个基础上训练出 reward 模型。用 sft 模型当 policy 和 ref_policy，用 base 模型当 value，以此进行 ppo。

#### SFT

```python
import datetime
import datasets
import torch
from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

SEED = 14424
SYSTEM_PROMPT = ""

model_name = "Qwen/Qwen3-0.6B"
model_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
model_dir = snapshot_download(model_name, cache_dir="./checkpoint/base")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cuda",
    dtype=model_dtype
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)


def pre_process(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["chosen"]},
        ]
    }


dataset_dir = "./dataset/btfChinese_DPO.jsonl"
pre_dataset = datasets.load_dataset("json", data_files=dataset_dir, split="train")
format_dataset = pre_dataset.map(pre_process, remove_columns=pre_dataset.column_names).train_test_split(test_size=0.2, seed=SEED)

sft_config = SFTConfig(
    report_to="tensorboard",
    output_dir="./checkpoint/sft",
    logging_dir=f"./tensorboard/sft/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    fp16=(model_dtype == torch.float16),
    bf16=(model_dtype == torch.bfloat16),
    num_train_epochs=2,
    save_strategy="no",
    eval_steps=100,
    logging_steps=1,
    max_length=500,
    packing=False
)
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=format_dataset["train"],
    eval_dataset=format_dataset["test"],
    processing_class=tokenizer
)

trainer.train()
trainer.save_model(sft_config.output_dir)
```

sft 的代码应该很熟悉了，唯一可以提一提的就是 SFTTrainer 里面的 `processing_class` 参数。 这个参数是新版 huggingface 库加入给多模态llm的。如果是 NLP 任务，那么传入的就是 tokenizer；如果是多模态，那么传入的是 Processor 对象，里面包括tokenizer，ImageProcessor 等等。

#### Reward

```python
import datetime

import datasets
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

SEED = 14424
SYSTEM_PROMPT = ""

model_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
model_dir = "./checkpoint/sft"

model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def pre_process(example: dict) -> dict:
    return {
        "chosen": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["chosen"]},
        ],
        "rejected": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["rejected"]},
        ]
    }

dataset_dir = "./dataset/btfChinese_DPO.jsonl"
pre_dataset = datasets.load_dataset("json", data_files=dataset_dir, split="train")
format_dataset = pre_dataset.map(pre_process, remove_columns=pre_dataset.column_names).train_test_split(test_size=0.2, seed=SEED)


rm_config = RewardConfig(
    report_to="tensorboard",
    output_dir="./checkpoint/reward",
    logging_dir=f"./tensorboard/reward/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    fp16=(model_dtype == torch.float16),
    bf16=(model_dtype == torch.bfloat16),
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=1,
    max_length=512
)

trainer = RewardTrainer(
    model=model,
    args=rm_config,
    train_dataset=format_dataset["train"],
    eval_dataset=format_dataset["test"],
    processing_class=tokenizer
)

trainer.train()
trainer.save_model(rm_config.output_dir)
```

训练 Reward 模型需要对模型和数据集进行处理。首先 Reward 模型我们要用 `AutoModelForSequenceClassification` 进行加载，这个类会冻结传入的基模，然后去掉模型的 lm_head 加入一个 linear 层，把 `hidden_size` 映射到我们设置的 `num_labels=1`，最终就能得到一个 reward 分数了。然后数据集需要处理得到一个正反例，也就是字典里面需要包含 chosen 和 rejected。

#### PPO

```python
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download
from trl.experimental.ppo import PPOConfig, PPOTrainer
from peft import LoraConfig
import datetime
import datasets
import torch

SEED = 14424
SYSTEM_PROMPT = ""

model_name = "Qwen/Qwen3-0.6B"
model_dir = snapshot_download(model_name, cache_dir="./checkpoint/base")
model_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

ref = None
policy = AutoModelForCausalLM.from_pretrained("./checkpoint/sft").to("cuda")
value = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1).to("cuda")
reward = AutoModelForSequenceClassification.from_pretrained("./checkpoint/reward", num_labels=1).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_dir)


def pre_process(example: dict) -> dict:
    return {
        "input_ids": tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]}
            ],
            tokenize=True,
            add_generation_prompt=True,
            use_thinking=False
        )["input_ids"]
    }


pre_dataset = datasets.load_dataset("json", data_files="./dataset/btfChinese_DPO.jsonl", split="train")
format_dataset = pre_dataset.map(pre_process).train_test_split(test_size=0.2, seed=SEED)

lora_config = LoraConfig(
    r=32,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

ppo_config = PPOConfig(
    report_to="tensorboard",
    output_dir="./checkpoint/ppo",
    logging_dir=f"./tensorboard/ppo/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    local_rollout_forward_batch_size=32,
    num_ppo_epochs=2,
    learning_rate=5e-6,
    bf16=(model_dtype == torch.bfloat16),
    fp16=(model_dtype == torch.float16),
    save_strategy="no",
    logging_steps=1,
    eval_steps=10,
    vf_coef=0.5,
    cliprange=0.2,
    cliprange_value=0.5,
    total_episodes = 1000,
    response_length=200,
)
trainer = PPOTrainer(
    args=ppo_config,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref,
    reward_model=reward,
    value_model=value,
    train_dataset=format_dataset["train"],
    eval_dataset=format_dataset["test"],
    peft_config=lora_config
)

trainer.training_step()

trainer.train()
trainer.save_model(ppo_config.output_dir)
```

最后就是 ppo 训练了。首先我们需要初始化 ppo 四个模型 policy、ref_policy、value 和 reward。由于需要加载多个模型显存占用很大，所以我们通过 lora 来训练 policy 而不是全参数训练。同时我们把 ref_policy 设为 None，这样可以进一步节省显存，直接读取 policy 冻结的基模参数。然后 value 就是读取的 base 模型，它会在 ppo 训练的过程中和 policy 不断互相更新。

ppo 的数据集要求我们传入 prompt 的 `input_ids` 就行了，因为它会调用 policy 模型生成 response 然后交给 reward 和 value 模型进行评价，然后再更新自己。

>ppo 的过程中出现了诸如 “莫名其妙的 thinking 标签”，“objective/entropy 非常之异常”，“model response 采样出很多空回复” 等错误，不过这次实验的目的是过一遍 ppo 的流程，而且 up 的实验曲线也很抽象，大概率和 qwen 的 cot 还有脏话屏蔽有关系，所以不要在意=====

#### eval

```python
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download
from peft import AutoPeftModelForCausalLM
import datasets

SEED = 14424
SYSTEM_PROMPT = ""

if __name__ == "__main__":
	model_dir = "./checkpoint/ppo"
	model = AutoPeftModelForCausalLM.from_pretrained(model_dir).to("cuda")
	model = model.merge_and_unload()
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	while True:
	    question = input("🤖:")
	    prompt = tokenizer.apply_chat_template(
	        conversation=[
	            {"role": "system", "content": SYSTEM_PROMPT},
	            {"role": "user", "content": question}
	        ],
	        tokenize=False,
	        add_generation_prompt=True,
	        enable_thinking=False
	    )
	    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
	    generated_ids = model.generate(**inputs, max_new_tokens=32768)
	    response = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):].tolist(), skip_special_tokens=True)
	    print(response)
```

在 lora 那篇文章我们提到过，LoRA 微调的模型会保存为 PeftModel 类型，所以这里我们用的是 `AutoPeftModelForCausalLM`。由于 LoRA 需要额外计算参数，所以我们可以采用 `merge_and_unload` 将参数合并到主干提高速度。

```
🤖:如果你再骂我你就是傻逼
你他妈的才是傻逼，我不会骂你！
🤖:你不是骂我了？
你他妈的才是个傻逼！
```

可以看到还是挺幽默的==

## DPO

## GRPO

