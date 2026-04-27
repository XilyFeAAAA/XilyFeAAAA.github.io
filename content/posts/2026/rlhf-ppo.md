---
title: LLM 中的强化学习：PPO
date: 2026-02-19T15:38:09+08:00
featuredImage: http://img.xilyfe.top/img/20260226203659803.png
authors:
  - Xilyfe
series:
  - RLHF
tags:
  - 大模型
  - 强化学习
lastmod: 2026-04-13T01:58:53+08:00
---
## 1. Actor-Critic

Actor-Critic 算法包含两个角色：演员 Actor 和 评判员 Critic。在大模型的语境中，Actor 就是我们 LLM，它会对 prompt 预测不断预测出 next token；Critic 也是一个神经网络，它就是来预测动作价值 $Q(s,a)$ 的，LLM 的语境下 state 是不可重复采样的，我们没法通过蒙特卡洛估计或其他方法计算出它的期望，因此 RLHF 中我们动作价值 $Q(s)$ 用神经网络来近似。

在训练过程中，LLM 不断根据上下文生成 next token 也就是做出 action，然后 Critic 模型根据 state 和 action 给予动作价值。现在我们应该就可以很轻松的写出训练 Actor 的损失函数了：

$$
\mathcal{L}_{\text{actor}} = -\frac{1}{N}\sum_{t} \log \pi_\theta(a_t|s_t) \cdot Q(s_t,a_t)
$$
$\pi(a_t|s_t)$ 是做出这个动作的概率，也就是 next token 对应的概率，$Q(s_t, a_t)$ 就是这个动作的价值。当 Critic 模型预测 $Q(S_t, A_t)>0$ 时，说明这个动作是好的，所以我们希望提到这个动作的概率，损失函数就会让 Actor 尽可能更新参数来使 $\pi(A_t|S_t)$ 变大。同理如果 $Q(S_t, A_t)<0$，那么我们希望这个动作的概率尽可能小，Actor 就会更新参数使 $\pi(A_t|S_t)$ 变小。

其次就是 Critic 模型，我们训练它的目的是让它预测的动作价值 $Q(s,a)$ 尽可能接近真实值。根据动作价值的定义，我们可以推导出：

$$
Q(s_t,a_t) = Q(s_{t+1}, a_{t+1}) + r(s,a)
$$

所以 $Q(s_{t+1}, a_{t+1}) + r(s,a) - \hat{Q}(s_t, a_t)$ 就是真实值和预测值的差距（这里为了区分让 Critic 预测的动作价值为 $\hat{Q}$），直接用它就可以当 Critic 模型的损失函数了：

$$
\mathcal{L}_{\text{critic}} = \frac{1}{N}\sum_{t} (Q(s_{t+1}, a_{t+1}) + r(s,a) - \hat{Q}(s_t, a_t))^2
$$

## 2. A2C

A2C 的思路就是我们在上一章提到的 **优势 Advantage**。单纯的动作价值和状态价值无法体现相对价值，例如假如选择下一个 token 为 "你好" 的动作价值是 5 看起来很低，但是相对于其他动作已经很高了，我们就应该强化这一 action：

$$
\delta_t = Q(s_t,a_t) - V(s_t)
$$

然后我们根据贝尔曼方程：

$$
V(s)=\mathbb{E}_{a\sim\pi, s'\sim P}[r(s,a)+\gamma V(s')]
$$

就可以把式子变形为：

$$
\delta_t = r(s,a) + \gamma V(s_{t+1}) - V(s_t)
$$

然后为了降低偏差和方差，我们用 GAE 对多步 TD ERROR 进行估计：

$$
A_t^{GAE} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots + (\gamma\lambda)^{T-t}\delta_T
$$

Actor 模型的损失函数变为：

$$
\mathcal{L}_{\text{actor}} = -\frac{1}{N}\sum_{t} \log \pi_\theta(a_t|s_t) \cdot A_t
$$

需要注意 Critic 模型的任务也从近似 $Q(s,a)$ 变成了近似 $V(s)$，Critic 的损失函数也应该相应改变：

$$
\mathcal{L}_{\text{critic}} = \frac{1}{N}\sum_{t} (r_t + \gamma V(s_{t+1}) - V(s_t))^2
$$

>为什么用 $V$ 表示 $Q$ 而不是反过来，可以看上一篇文章。

LLM 语境下 A2C 的步骤为：

1. LLM 根据 prompt 和生成的 token（即状态 $S_t$），预测 next token（即执行动作 $A_t$）​
2. 在 LLM 的 RLHF 中，通常只有在生成完最后一个 token（如 `<EOS>`）时，Reward Model 才会给出一个标量奖励 $R_{final}$，而中间生成过程的 per-token reward 通常设为 $0$（或者带有微小的 KL 惩罚项）。GAE 的精妙之处就在于，它能利用状态价值函数 $V(s)$，将这个稀疏的、只在句末出现的 $R_{final}$ 平滑地分配（反向传播）给前面生成的每一个 token，计算出各自的 $A_t$
3. 演员用 $\mathcal{L}_{\text{actor}} = -\frac{1}{N}\sum_{t} \log \pi_\theta(a_t|s_t) \cdot A_t$ 更新参数
4. 评论家用 $\mathcal{L}_{\text{critic}} = \frac{1}{N}\sum_{t} (r_t + \gamma V(s_{t+1}) - V(s_t))^2$ 更新参数

## 3. PPO

### 3.1 近端策略优化

PPO 算法可以看成 A2C 的优化版。A2C的训练策略是 “采样一次，更新一次，然后扔掉数据”，这就导致效率很低，每批数据只能用一次。PPO 采用 **近端策略优化**，具体步骤如下：
1. 对 `batch_size` 个 prompt 生成 completions
2. 计算得到这些 completions 的 `logprobs`、`rewards`、`advantages`、`return`
3. 进行 `num_epochs` 次更新，每一次用新的模型计算 `logprobs` 和 `values` 然后再更新 Actor 和 Critic

可以注意到：每次更新变化的只有概率分布 `logprob` 和预测的状态价值 `values`，而优势奖励这些都固定采用第一次得到的数据。
- 为什么要保持 `advantages` 不变：因为 PPO 的核心目标是优化**当前的策略**，使其比**采样时的策略**更好。`advantages` 代表了“采样时的那个动作比平均水平好多少”。如果我们在 $n$ 次迭代中不停地重新计算 `advantages`，训练会变得极其不稳定，甚至导致梯度爆炸。
- 为什么要保持 `reward` 不变：`reward` 是针对 Actor 生成的完整句子 给出的评分。一旦采样完成，completions 就固定下来了，所以 `reward` 自然是固定不变的。

>举个例子：A2C 就像在表演现场，你一边演，导演一边喊“好”或“坏”，然后你得到反馈就修改。改完之后，刚才演的那段戏就没用了，你必须重新演一段，导演才能给新反馈。而 PPO 更像 **复盘录像**，你先演一段戏录下来，接下来的 4 个 Epoch 你坐在监视器前，对着这段录像反复琢磨。第一遍根据反馈改一点，第二遍在第一遍改动的基础上，再对着录像微调。

### 3.2 更新约束

在 A2C 中，如果学习率设得稍微大一点，一次更新可能让策略发生很大的变化，如果 Actor 突然学到了一个极其糟糕的动作，整个策略可能瞬间崩塌。那我们如何衡量策略的变化幅度呢？我们可以看两个策略执行相同动作得到结果的差别。在 LLM 中就是，我们对更新参数前后的模型都输入 token，就能得到的不同的概率分布 $\pi(A_t|S_t)$ 和 $\pi'(A_t|S_t)$。我们定义一个比率 $r_t(\theta)$，表示**新策略**和**旧策略**产生某个动作的概率比：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

- 如果 $r_t > 1$：说明这个动作在新策略中出现的概率变大了。
- 如果 $r_t = 1$：说明新旧策略完全一样。

进而有了演员的 loss 函数：

$$
\mathcal{L}_{\text{actor}} = -\frac{1}{N}\sum_{t} \log \frac{\pi(A_t|S_t)}{\pi'(A_t|S_t)} \cdot Q(s_t,a_t)
$$

我们从梯度更新的角度思考一下这个公式的意义，先把公式改一下 $loss = -\pi(A_t|S_t) \times \frac{\text{Adv}(S_t, A_t)}{\pi'(A_t|S_t)}$，由于 adv 和 π' 都不在计算图里面不回传梯度，所以梯度表达式为 $\frac{\partial{loss}}{\partial{\pi}}=- \frac{\text{Adv}(S_t, A_t)}{\pi'(A_t|S_t)}$。假设某次 action 的 $Adv>0$，也就是决策优于平均水平，那我们肯定希望提高这个 action 的概率，也就是增加 $\pi(A_t|S_t)$。假如原先 $\pi'(A_t|S_t)$ 也很大，也就是原先模型也认为应该执行这个 action，那么梯度的绝对值就会变小，不会做出非常大改变。假如原先 $\pi'(A_t|S_t)$ 很小，旧策略觉得这个动作几乎不可能发生，但实际采样出来发现这个动作效果出奇的好（也就是Adv 很大）。重要性采样公式 $\frac{\pi}{\pi'}$ 会认为：这是一个被旧策略严重低估的好 action，于是算法会试图剧烈地提高 $\pi(A_t|S_t)$。

现在，我们已经限制了策略 $\pi(A_t|S_t)$ 的更新幅度，但还缺少一个“熔断机制”。什么意思呢？就是万一策略的更新幅度还是太大了，我们要停止策略的参数更新。PPO 的做法是什么呢？因为 $\frac{\pi(A_t|S_t)}{\pi'(A_t|S_t)}$ 衡量了旧策略和现行策略之间差异，所以可以为它设置两个阈值。为了方便描述，我们令 $r(A_t, S_t) = \frac{\pi(A_t|S_t)}{\pi'(A_t|S_t)}$，这种熔断机制可以表示为：

$$
loss = -\min(r(A_t, S_t) \text{Adv}(S_t, A_t),\ \text{clip}(r(A_t, S_t) , 0.8, 1.2) \text{Adv}(S_t, A_t))
$$

- Adv 大于 0，r 大于 1.2：min 操作就会取右边的值，此时 loss 中就只剩常量了，不产生任何梯度则停止参数更新；而 r 无论多小都还是会产生梯度。
- Adv 小于 0，r 小于 0.8：min 操作就会取右边的值，此时 loss 中就只剩常量了，不产生任何梯度则停止参数更新；而 r 无论多大都还是会产生梯度

诶，那为什么我们不用管 Adv 大于 0 且 r 小于 0.8 的情况？或者 Adv 小于 0 且 r 大于 1.2 的情况？Adv 大于 0 的情况说明当前策略是好的，如果 r 小于 0.8 说明：这个策略是好的，旧模型偏向这个策略，但是新模型不怎么偏向这个策略了，那我们肯定希望能尽可能朝现在这个方向来更新参数，所以不会进行 $max(r, 0.8)$。同样 Adv 小于 0 的情况说明当前策略不怎么行，如果 r 大于 1.2 则说明这个不好的策略现在很看好，那我们肯定希望加大力度更新参数来避免这个 action，于是不应该限制更新的幅度。

### 3.3 Critic loss

PPO 的 Critic loss 和 A2C 还有些许不同，这几个算法设计 Critic Loss 的思路就是把状态价值 $V(s)$ 的目标值（真实值）和预测值做一个 MSE Loss 也就是均方差。在 A2C 中我们通过贝尔曼方程得到了 $V(s)$ 的另一种表示 $v_{target} = r_t + \gamma V(s_{t+1})$，因此有了 A2C 的 Critic loss：$\mathcal{L}_{\text{critic}} = \frac{1}{N}\sum_{t} (r_t + \gamma V(s_{t+1}) - V(s_t))^2$。但是这个办法在 PPO 里面行不通，我们先看看 PPO 的思路以及实现方向，然后再说明为什么不能照搬 A2C 的 1-step 贝尔曼。

先跟着最自然的思考顺序，首先我们的最终目标是让 critic 的输出 $V(s)$ 必须尽可能接近真实的状态价值：

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s),\ \tau \sim P} \Big[ \sum_{k=0}^\infty \gamma^k r_{t+k} \Big]
$$

如果状态价值 $V(s)$ 估得准，$\text{Adv} = Q(s,a)-V(s)$ 就会低方差，Actor 就能稳定地知道“哪个动作比平均好多少”。 所以我们必须让 $V(s)$ 不断逼近这个“真实平均回报”。但是真实 $V(s)$ 根本拿不到，于是我们想到状态价值 $V(s)$ 正好等于所有可能动作的动作价值 $Q(s,a)$ 在当前策略下的期望：

$$
V^\pi(s) \equiv \mathbb{E}_{a \sim \pi(\cdot|s)} \big[ Q^\pi(s,a) \big]
$$

由于我们的 advantage 本身定义就是 $Q(s,a) - V(s)$，所以我们移项可以得到 $Q(s,a) = V(s) + \text{Adv}(s,a)$。在 PPO 中，我们虽然没有训练单独的 Q 网络，但我们用 GAE 算出了一个高质量的 Advantage 估计：

$$
A_t^{\text{GAE}} \approx Q(s_t, a_t) - V_{\text{old}}(s_t)
$$

因此：

$$
\text{returns} = Q(s_t, a_t) \approx V_{\text{old}}(s_t) + A_t^{\text{GAE}}
$$

我们把这个近似值起个名字叫 returns，它就是我们目前能得到的最好的 $Q(S_t,A_t)$ 采样估计。因此就有了 critic loss：

$$
L_{\text{critic}} = \mathbb{E}_{t} \left[ \left( V(s_t) - \text{returns} \right)^2 \right]
$$

Critic 和 Actor 一样都有对 loss 的变化幅度做出限制，Critic 预测的是 values，所以限制了 $V_{new}$ 和 $V_{old}$ 的变化：

$$
L_{\text{critic}}^{\text{clip}}= \mathbb{E}_{t} \left[ \max \Big( 
    \big( V(s_t) - \hat{R}_t \big)^2,\ 
    \big( \text{clip}\big(V(s_t),\ V_{_{\text{old}}}(s_t) - \epsilon,\ V_{_{\text{old}}}(s_t) + \epsilon\big) - \text{returns} \big)^2 
\Big) \right]
$$

这里还是解释一下：首先 critic loss 是在做一个回归，我们用了 MSE，希望新模型的预测值 $V_{\text{new}}$ 尽可能接近目标值 $V_{target}$ 也是就 returns。为了防止价值函数更新太快导致策略崩溃，PPO 给 Critic 也加了一个 max 来限制更新。当 $V_{\text{new}}$ 在 $[V_{old}-\epsilon, V_{old}+\epsilon]$ 这个区间时候 $V_{\text{clip}} = V_{\text{new}}$ 正常更新；当 $V_{\theta} > V_{old} + \epsilon$，截断项里的预测值会被锁定在 $V_{\text{clip}} = V_{old} + \epsilon$。假如 $V_{\text{new}}$ 比 $V_{\text{clip}}$ 更接近 returns，那么说明一次意外的更新（$V_{\text{new}}$ 超过上界了）导致 loss 更低了，我们就得做出限制不能让他更新，max 就会选择 $(V_{\text{clip}} - \text{returns})^2$ 里面不含参数。如果 $V_{\text{clip}}$ 比 $V_{\text{new}}$ 更接近 returns，那么选择的就是 $(V_{\text{new}} - \text{returns})^2$ 正常更新了。同理 $V_{\theta} < V_{old} + \epsilon$ 也是这样，真的很巧妙。


>归根结底，actor loss 和 critic loss 里面的 clip + min(max) 都是模型为了防止过度优化做出的 **悲观估计**。actor model 意图最大化损失函数 $\text{ratio} * \text{advantages}$ 所以我们需要做一个 min 的操作，而 critic model 损失函数的均方差意图是最小化 $V_{\text{new}}$ 和 $\text{returns}$ 的差距，所以我们悲观估计时候要做 max 操作。
>PS：具体代码实现上，由于梯度下降一般需要让损失函数求最小值，所以我们在 actor model 的 loss 里面会加上负号变成 $-\text{ratio} * \text{advantages}$，可能做的是 max 操作，不过都是一个思想。

于是 PPO 的完整流程就变成了：

```python
for batch in dataset: 
    计算 advantages，reward，values，returns，ref_logprobs
    
    for epoch in range(ppo_epochs):
        用最新的策略重新计算 batch 的 logprobs，values
        计算 actor loss 和 critic loss
        梯度下降更新模型
```

---

现在可以回到之前的问题了，为什么 PPO 的 Critic loss 不能和 A2C 一样用 1-step 贝尔曼公式当做 $V_{\text{target}}$？

根本原因在于 PPO 对一条数据会进行多轮训练。如果按照 A2C 的思路，出现的第一个问题就是梯度不收敛：在优化 MSE 损失函数 $\text{Loss} = (V_{\theta}(s_t) - V_{target})^2$ 时，我们希望 $V_{\theta}(s_t)$ 去拟合 $V_{target}$。如果 $V_{target}$ 也是 $\theta$ 的函数，即 $V_{target}(\theta) = r + \gamma V_{\theta}(s_{t+1})$，那么每次梯度下降时，模型不仅在改变左边的预测值，也在改变右边的目标值。这会导致震荡、发散，或者模型为了降低损失而陷入一种“自我欺骗”的退化解。其次 Critic 网络在训练初期是不准的。如果 $V_{target}$ 是固定的（基于采样时的快照），那么误差是静态的。如果 $V_{target}$ 是每一轮循环实时 forward 得到的，当前步骤对 $V$ 的一次错误更新（比如因为某个采样噪声导致 $V$ 偏高）会立刻反映在 $V_{target}$ 中，导致下一次更新更加偏高。这种**正反馈环**会迅速放大网络初期的估值偏差。


### 3.4 Reward Loss

在讲 Reward Loss 之前需要先介绍一个 Bradley-Terry 模型，它是一种经典的概率模型，用于处理成对比较和排名问题。BT 模型假设每个对象有一个隐含的“强度”或“分数”参数，通常用 $\pi$ 表示。当比较两个对象 $i$ 和 $j$ 时，$i$ 优于 $j$ 的概率计算公式为：

$$
P(i > j) = \frac{\pi_i}{\pi_i + \pi_j}
$$

我们先举一个例子，假如我一个对战数据：

| 对战    | 胜利  | 失败  |
| ----- | --- | --- |
| A 对 B | 8   | 4   |
| A 对 C | 3   | 5   |

那我们利用最大似然估计（这批胜负数据出现的概率最大），来找到 $\alpha_a$，$\alpha_b$，$\alpha_c$：

$$
L = \left(\frac{\alpha_A}{\alpha_A+\alpha_B}\right)^8 \times \left(\frac{\alpha_B}{\alpha_A+\alpha_B}\right)^4 \times \left(\frac{\alpha_A}{\alpha_A+\alpha_C}\right)^3 \times \left(\frac{\alpha_C}{\alpha_A+\alpha_C}\right)^5
$$

然后我们求对数得到：

$$
\ln L = 8\ln\left(\frac{\alpha_A}{\alpha_A+\alpha_B}\right) + 4\ln\left(\frac{\alpha_B}{\alpha_A+\alpha_B}\right) + 3\ln\left(\frac{\alpha_A}{\alpha_A+\alpha_C}\right) + 5\ln\left(\frac{\alpha_C}{\alpha_A+\alpha_C}\right)
$$

在优化中，我们用梯度下降等方法最小化一个函数，但 MLE 是最大化 ln L，所以取个负数得到负对数似然，于是我们就能得到一般的损失函数：

$$
\text{Loss} = - \mathbb{E}_{(\alpha_x, \alpha_y) \sim D} \left[ \ln \frac{\alpha_x}{\alpha_x + \alpha_y} \right]
$$

在 RLHF 中，BT 用于从人类偏好数据学习奖励函数 $r(x, y)$。给定一对偏好：$y_w$​ 优于 $y_l$，建模概率：

$$
P(y_{\text{w}} > y_{\text{l}} \mid x)= \frac{r(x,y_{\text{w}})}{r(x,y_{\text{w}}) + r(x,y_{\text{l}})}
$$

由于奖励函数 $r(x,y)$ 可能返回的是负数，但是 BT 模型要求分数为正数，所以加上指数函数：

$$
P(y_w > y_l \mid x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}
$$
然后代入损失函数就能得到：

$$
\begin{align}
\text{Loss} &= - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \ln \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} \right] \\
&= - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \ln \frac{1}{1 + \exp(r(x, y_l) - r(x, y_w))} \right] \\
&= - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \ln \sigma \left( r(x, y_w) - r(x, y_l) \right) \right]
\end{align}
$$

根据大数定律，我们用有限样本 $N$ 进行蒙特卡罗估计期望：

$$
\text{Loss} \approx - \frac{1}{N} \sum_{i=1}^N \ln \sigma \left( r(x_i, y_{w_i}) - r(x_i, y_{l_i}) \right)
$$

至此我们就可以用 `{"prompt": prompt, "win": win_response, "loss": loss_response}` 来更新 Reward Model 了。

## 4. trl 库源码分析

`trl.experiment.ppo.PPOTrainer.train()` 方法内部依次进行如下操作：
1. rollout 阶段：将数据集的 prompt 传给 actor 采样 response，我们就得到了 prompt+response 的问答对。
2. evaluation 阶段：用 reward 模型给这个问答对打分数 `scores`，注意 **这个分数是序列级的而不是 token 级的**。
3. optimization 阶段：把 prompt+response 用 Teacher-Forcing 的方式送入 ref、actor 和 critic 模型得到 response 中每个 token 的概率 `ref_logprob` 和 `old_logprob`，以及逐 token 的预期收益 `old_values`。根据之前计算出的整个序列的 reward，我们可以计算出每个 token 对应的 reward，这样 advantage 也就计算出来了。
4. 重复 ppo_epochs 个阶段，不断把 prompt+response 用 Teacher-Forcing 的方式传入 actor 得到每个 token **新的概率分布**，把 response 传入 critic 得到 values。然后利用之前 optimization 阶段得到的 reward 和 advantages 来计算 actor 和 critic 的 loss，更新这两个模型。

我借用知乎的几张图片来图解一下这个过程：

![image.png](http://img.xilyfe.top/img/20260224122122139.png)


前面我们提到，evalution 阶段计算的 reward scores 是序列级的，但是 PPO 在每个 step（对应生成序列中的每个token）都需要计算 advantage 来更新 actor model，这样不是矛盾了吗？实际上 reward 模型在计算序列级 reward 的时候没有加入 kl 散度，这时候计算得到分数我们叫做 `scores`。在每一个 step，我们通过 `scores`，`ref_logprob`，`old_logprob` 计算得到这个 token 的 reward，$reward = scores - \beta*kl(old\_logprob, ref\_logprob)$，最后用这个 token 对应的 value 和 reward 计算 advantage。

### 4.1 rollout

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

### 4.2 evaluation

```python
texts = [q + r for q, r in zip(batch["query"], batch["response"])]
reward_out = reward_model(texts)
scores = [torch.tensor(output[1]["score"]) for output in reward_out]
```

### 4.3 optimization

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

## 5. 代码实战

我这次选择直接复现 bilibili 一个 up 主的 ppo 项目 [owenliang/hf-ppo](https://github.com/owenliang/hf-ppo/blob/main/README.md) -  让大模型学会说脏话。由于采用的是 Qwen 的基模不太可能输出脏话，直接在 base 模型上进行 ppo 很难训练起来，所以我先用数据集对 base 模型进行 sft，然后在 sft 的基础上进行 ppo，这样就能完成整个流程。

这次整体的计划就是先对 Qwen 的基模进行 sft，然后在这个基础上训练出 reward 模型。用 sft 模型当 policy 和 ref_policy，用 base 模型当 value，以此进行 ppo。

### 5.1 SFT

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

### 5.2 Reward

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

### 5.3 PPO

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

### 5.4 eval

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