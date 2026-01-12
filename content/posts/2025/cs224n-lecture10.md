---
title: "Lecture 10: Post-Training"
date: '2025-12-01T11:24:11+08:00'
authors: [Xilyfe]
series: ["CS224N"]
tags: ["深度学习"]
--- 


## Post-training

Post-training 位于预训练之后，目的是为了让模型 **更懂任务、更听指令、更符合人类意图**。

Post-training 包含3个步骤：

1. **Finetune**
2. **RLHF**
3. **DPO**

其中 Finetune 又包含 Instruction Finetuning、LoRA、Task Finetuning 等。

## Instruction Finetuning

过去的训练方式是用大量数据在预训练上，用少量数据对特定任务进行微调。但是后面发现用大量数据在不同任务上进行后训练，然后整合到一个 UX 中。这些指令包含不同的任务，例如 Q&A，翻译，生成，推理等。

但是指令微调的局限性也很明显：

- IF 属于 SFT 也就是 Supervised Finetuning，它的数据收集成本非常高。
- 对于开放性的问题，没有正确的答案：例如翻译任务，将 good 翻译为"一般"和"差"代价一样，但是明显翻译为"差"错的更多


## RLHF

RLHF 基于人类反馈的强化学习是 post-training 的一种方法，它的思路是通过人类的反馈来优化模型。但是这个想法存在几个问题：

- 人力成本很昂贵：解决方法很简单，就是通过机器学习的方法训练一个模型，可以预测人们更倾向于哪一个答案。
- 反馈很难量化：例如对于生成任务，很难对结果进行打分，解决方法就是生成多个结果，然后对其进行排序，rank instead of score

### RL 基本概念


<div style="text-align: center">
    <img src="../../../../resource/ai/llm/RL.png" width="70%" />
</div>

强化学习的基本思路：

1. Agent 根据状态 State 做出行为 Action
2. Action 进而对环境产生影响，状态更新并且基于 Reward Model 给予 Agent 奖励 Reward
3. Agent 根据奖励和新的状态做出新的行为

我们谈到了奖励值 Reward  ，它表示环境进入状态 State 下的**即时奖励**。但如果只考虑即时奖励，目光似乎太短浅了：当下的状态和动作会影响到未来的状态和动作，进而影响到未来的整体收益。所以，一种更好的设计方式是：**t 时刻状态 s 的总收益 = 身处状态 s 能带来的即时收益 + 从状态 s 出发后能带来的未来收益。** 写成表达式就是：

$$
V_t=R_t+\gamma V_{t+1}
$$

- $V_t$ 指 t 时刻之后的全部收益
- $R_t$ 指 t 时刻即时收益
- $V_{t+1}$ 指 t+1 时候之后的全部收益
- $\gamma$ 是折扣因子，它决定了我们在多大程度上考虑将“未来收益”纳入“当下收益

### RL in LLM

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/rlinllm.jpg" width="70%" />
</div>

- 我们先喂给模型一个 prompt，期望它能产出符合人类喜好的 response
- 在 t 时刻，模型根据上文，产出一个token，**这个token即对应着强化学习中的动作，我们记为** $A_t$。因此不难理解，在NLP语境下，强化学习任务的动作空间就对应着词表。
- 在 t 时刻，模型产出 token $A_t$ 对应着的即时收益为 $R_t$，总收益为 $V_t$。此刻，模型的状态变为 $S_{t+1}$，也就是从“上文”变成“上文 + 新产出的token”
- 在NLP语境下，智能体是语言模型本身，环境则对应着它产出的语料


### RLHF 中的四个模型

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/4model.jpg" width="70%" />
</div>

- **Actor Model**：演员模型，这就是我们想要训练的目标语言模型
- **Critic Model**：评论家模型，它的作用是预估总收益 $V_t$
- **Reward Model**：奖励模型，它的作用是计算即时收益 $R_t$
- **Reference Model**：参考模型，它的作用是在RLHF阶段给语言模型增加一些“约束”，防止语言模型训歪

> 其中 Actor Model 和 Critical Model 需要在 RLHF 过程中参与训练，Reward Model 和 Reference Model 两个模型需要冻结参数。

#### Actor Model

Actor Model 一般用SFT阶段产出的SFT模型来对它做初始化。策略是，先喂给 Actor 一条 prompt （这里假设batch_size = 1，所以是 1 条 prompt），让它生成对应的 response。然后，我们再将“prompt + response"送入我们的“奖励-loss”计算。

#### Reference Model

RLHF 中存在 Reward Hacking 这个概念，指的是模型知道了评分标准而去直接学习投机取巧的方法而不是学习知识。解决方案就是通过 Reference Model 来衡量预训练得到的模型和 RLHF 微调之后模型的差异，约束 policy 不要离正常语言太远。

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/klmetric.jpg" width="70%" />
</div>

- **对 Actor 模型**，我们喂给它一个 prompt，它正常输出对应的 response。那么 response 中每一个 token 肯定有它对应的 log_prob 概率分布呀，我们把这样的结果记为 **log_probs**。
- **对 Ref 模型**，我们把 Actor 生成的"prompt + response"喂给它，那么它同样能给出每个 token 的 log_prob 结果，我们记其为**ref_log_probs**
- 那么这两个模型的输出分布相似度就可以用 `ref_log_probs - log_probs` 来衡量，我们在加入 KL 散度作为惩罚项就可以避免偏离 Ref 太远。

$$
Reward'=Reward - \beta * KL
$$

> Ref 模型一般是将 SFT 模型直接复制后冻结参数。

---

KL 散度在 RLHF 中具体如何计算：

1. 将 prompt 输入 actor 得到 response
2. 将 "prompt+response" 输出 actor 得到每个 token 的概率分布 `act_log_probs`
3. 将 "prompt+response" 输出 ref 得到每个 token 的概率分布 `ref_log_probs`
4. 最后 KL 散度就是对每个 token 概率差的对数求和。

$$
KL = \frac{1}{n}\sum_{response}{\log{(prob\_ref(token))-\log{(prob\_act(token))}}}
$$

>KL 散度的标准定义应该是，对于单个 token 的 KL 散度是要对 vocab 上**每一个 token**都求概率和加权求和。但是在 RLHF 实际实现中，token 级 KL 只针对 _Actor 实际生成出来的 response[t] token_，计算 `log p_actor(response[t]) − log p_ref(response[t])`

举个例子：

- prompt="i like"
- 输入到 actor 得到 response "you very much"
- 将 "i like you very much" 输入 actor 求概率分布，**这里用的是 Teacher-Forcing**
	- 先前向记录第一个 token 的概率分布，将 input=prompt 经过 forward 就能得到 response_0 的概率分布。
	- input=prompt+r0，经过前向计算得到 response_1 的概率分布
- 类似的将 "i like you very much" 输入到 ref 得到概率分布

| 位置 t | 要预测的 token | Reference Model 的概率分布 p_ref(.)        | Actor（当前 policy）的概率分布 p_actor(.)        |
| ---- | ---------- | ------------------------------------- | --------------------------------------- |
| 1    | you        | p_ref(you)=0.40, i=0.25, like=0.10, … | p_actor(you)=0.75, I=0.05, like=0.12, … |
| 2    | very       | p_ref(very)=0.35, …                   | p_actor(very)=0.60, …                   |
| 3    | much       | p_ref(much)=0.70, …                   | p_actor(much)=0.85, …                   |
| 4    | \<eos>     | p_ref(\<eos>)=0.90, …                 | p_actor(\<eos>)=0.92, …                 |

对每个 response token t，计算单 token 的 KL 散度

| 位置 t | 要预测的 token | KL_t                |
| ---- | ---------- | ------------------- |
| 1    | you        | ≈log(0.75)-log(0.4) |
| 2    | very       | ...                 |
| 3    | much       | ...                 |
| 4    | \<eos>     | ...                 |

#### Critical Model

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/critical.jpg" width="70%" />
</div>

Q：训练Actor模型我能理解，但我还是不明白，为什么要单独训练一个 Critic 模型用于预测收益呢？
A：这是因为，当我们在前文讨论总收益（即时 + 未来）时，我们是站在上帝视角的，也就是这个 $V_t$ 就是客观存在的、真正的总收益。但是我们在训练模型时，就没有这个上帝视角加成了，也就是在 t 时刻，我们给不出客观存在的总收益 $V_t$，我们只能训练一个模型去预测它。

### Actor Loss

先来看一个直观的 loss 设计方式：

$$
actor\_loss = -\sum V_{t}log P(A_{t}|S_{t})
$$
- $P(A_{t}|S_{t})$ 是在状态 S 的情况下执行 $A_{t}$ 的概率
- $V_t$ 是对应的预期未来收益

假如 $V_t>0$ 那么损失函数就倾向于提高执行 $A_{t}$ 的概率，反之减少概率。但是这个方法存在一个问题：**只要预期收益是正的，模型就拼命提高这个回答出现的概率**，因此引入了 Advantage 这个概念。

> 假设是在迷宫游戏中以到达出口为目的，直接到达终点的 $V_t=10$，绕一圈到达出口也可以胜利但是 $V_t=5$，两者的预期收益都为正数也就是说都会提高执行他们的概率，但实际上我们需要提高的只有第一个方法。换句话说"这会导致策略误判，把差动作也当成好动作优化"。

所以我们定义优势为：

$$
Adv_{t} = R_{t} + \gamma * V_{t+1} - V_{t}
$$

新的 Actor Loss 为：

$$
actor\_loss = -\sum Adv_{t}log P(A_{t}|S_{t})
$$

---

前面还记得我们提到了 Reference Model，它的作用是为了约束模型的更新，它具体用在 Reward 奖励函数中，遵循「鼓励高人类偏好（RM 奖励）+ 抑制策略偏离（KL 惩罚）」。

最简单的实现，只需要计算一次序列的 KL 散度：

$$
R_t=r(s,a)-\gamma KL
$$

简单代码实现如下：

```python
ref_model = actor_model = sft_model
def compute_rewards(prompts, responses)
	with torch.no_grad():
        rm_rewards = rm_model(**tokenized).logits.squeeze(-1)  # [batch_size]
	kls = []
	for p, r in zip(prompts, responses):
		input_text = f"### 人类：{p}\n### 助手：{r}"
		# 处理为 token
		actor_logits = actor_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        actor_probs = F.softmax(actor_logits, dim=-1)
        # 这里 input_ids 的起始位置用 prompt 长度更好
		actor_log_probs = torch.log(actor_probs.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
		with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
            ref_probs = F.softmax(ref_logits, dim=-1)
            ref_log_probs = torch.log(ref_probs.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
        
        # KL 散度（平均到每个 token）
        kl = (actor_log_probs - ref_log_probs) / (input_ids.shape[1] - 1)
        kls.append(kl)
	    
    beta = 0.1  # KL 权重（可根据训练情况调整）
    total_rewards = rm_rewards - beta * kls  # 最终奖励 = 原始奖励 - KL 惩罚
    return total_rewards, rm_rewards, kls
```

- 首先计算 RM，注意需要冻结参数
- 之后计算每一个 Q&A 的 KL 散度：
	- 先计算 Actor Model 的对数概率，具体上 actor_model.logits 返回的是一个 \[batch_size, seq_len, vocal_size] 的矩阵，代表每个 token 选择的概率，通过 softmax 求概率分布之后通过 gather 函数取出正真实 token 对应的概率，形状为 \[batch_size, seq_len, 1]，squeeze 到 \[batch_size, seq_len] 再求对数和, 形状就变成了 \[batch_size, ]，这就对应前面公式里的 $\sum_{response}{\log{(prob\_act(token))}}$。
	- Ref Model 的计算同 Actor Model，就是要冻结参数。
	- 两个对数求差之后求平均就是这个 sequence 的 KL 散度了。

> `logits[:, seq_len-1, :]` 预测的是 **第 seq_len+1 个 token** —— 但原输入序列只有 `seq_len` 个 token，第 seq_len+1 个 token 是 “未存在的、需要生成的 token”，在当前场景（计算已生成序列的对数概率）中，这个位置的预测是 **无用的**。

---

deepspeed-chat 的 RLHF 实践中，对 $R_t$ 做了另一种设计:

 $$\begin{array}{c} R_t =
\begin{cases}
- \text{kl\_ctl} \cdot \log \frac{P(A_t|S_t)}{P_\text{ref}(A_t|S_t)}, & t \neq T \\
- \text{kl\_ctl} \cdot \log \frac{P(A_t|S_t)}{P_\text{ref}(A_t|S_t)} + R_t, & t = T
\end{cases} \end{array}$$

- 当 $t \neq T$ 时，我们更加关心 Actor 是否有在 Ref 的约束下生产 token  
- 当 $t=T$ 时，我们不仅关心 Actor 是否遵从了 Ref 的约束，也关心真正的即时收益

```python
def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        # ---------------------------------------------------------------------------------------------------
        # response开始的位置
        # （因为我们对prompt做过padding处理，因此batch中每个prompt长度一致，也就意味着每个response开始的位置一致）
        # （所以这里start是不加s的，只是一个int）
        # ---------------------------------------------------------------------------------------------------
        start = prompts.shape[1] - 1
        # ---------------------------------------------------------------------------------------------------
        # response结束的位置
        # （因为一个batch中，每个response的长度不一样，所以response的结束位置也不一样）
        # （所以这里end是加s的，ends的尺寸是(batch_size,)
        # ---------------------------------------------------------------------------------------------------
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, ends[j]] += reward_clip[j]

        return rewards
```

> 同样可以把最后一个时刻的即时奖励替换为每个 token 即时奖励的均值

---

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/ppo.jpg" width="70%" />
</div>

- 第一步，我们准备一个 batch 的 prompts
- 第二步，我们将这个 batch 的 prompts 喂给 Actor 模型，让它生成对应的 responses
- 第三步，我们把 prompt+responses 喂给我们的 Critic/Reward/Reference 模型，让它生成用于计算 actor/critic loss 的数据
- 第四步，我们根据这些经验，实际计算出 actor/critic loss，然后更新 Actor 和 Critic 模型，最终得到的 Actor 模型就是 RLHF 之后微调过的最终模型

从图例中可以看到 PPO 采用的是 batch_prompts，因为训练不可能是 **生成一个样本 → 立刻更新模型 → 再生成一个样本 → 更新……**，而是 **一大批样本生成完 → 再训练很多步**，这就导致了一个问题：当我们用旧的模型生成了一堆 prompt-response 并且得到了对应的 advantage，然后求 Actor Loss 对 Actor Model 更新了很多次，举个例子：

- 准备了 batch_size 个 prompt 喂给 Actor Model，相对于后面来说，现在的 Actor Model 就是旧的，它生成了结果 responses。假设某个 responses\[k] 中的某个 token 为 hello 且 P(hello)=0.6
- 经过 epochs 轮训练，得到了新的 Actor Model，这时候如果再把之前的 prompts 喂给他得到的 responses 就不一样了
- 这时候我们需要求这一轮的 $actor\_loss = -\sum Adv_{t}log P(A_{t}|S_{t})$ ，此时我们用的还是旧模型得到的 response，但是训练后的新模型每个 token 的概率就不同了，此时 P(hello)=0.1，模型认为概率这么低还选中了，那更要提高它的概率，于是把 hello 这个不喜欢的 token 概率又提高了。

解决方案就是(涉及数学问题不会了，反正就是这个)：

$$
actor\_loss = -min(Adv_{t} *\frac{P(A_{t} | S_{t})}{P_{old}(A_{t} | S_{t})},  Adv_{t} * clip(\frac{P(A_{t} | S_{t})}{P_{old}(A_{t} | S_{t})}, 0.8, 1.2))
$$


```python
def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        """
        logprobs: 实时计算的，response部分的prob（只有这个是随着actor实时更新而改变的）
        old_logprobs：老策略中，response部分的prob （这个是固定的，不随actor实时更新而改变）
        advantages： 老策略中，response部分每个token对应的优势（这个是固定的，不随actor实时更新而改变）
        mask：老策略中，response部分对应的mask情况这个是固定的，不随actor实时更新而改变）
        self.cliprange: 默认值是0.2
        """
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        # 最后是取每个非mask的response token的平均loss作为最终loss
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum() 
        return pg_loss
```

### Critical Loss

$$
\begin{align}
L^{VF} &= \mathbb{E}\Big[(V_\theta(s_t) - R_t^{\text{target}})^2\Big] \\
R_t^{target}&=\sum_{k=t}^{T}r_k
\end{align}
$$


Critical Model 的目的是预测未来收益，所以 Critical Loss 的设计也很简单了，就是求未来预期收益和未来实际收益的 MSE。

---

这里又有一个问题了：既然我们可以得到**未来实际收益**，那么我们还需要 Crtical Model 预测未来收益做什么？查了白天还是不懂。。


## DPO

与传统 RLHF 相比，DPO 的核心创新在于：**直接利用人类标注的 "哪个回答更好" 的偏好数据来优化模型，而不是先训练一个奖励模型再用强化学习优化**。

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/dpo.png" width="70%" />
</div>


DPO 需要的训练数据格式非常简单：**三元组 (prompt, chosen, rejected)**，即：

```plaintext
{
  "prompt": "解释量子计算",
  "chosen": "量子计算利用量子比特可以同时处于多个状态的特性，实现信息的并行处理，使某些问题的解决速度呈指数级提升",
  "rejected": "量子计算是一种涉及原子和粒子的复杂技术"
}
```

---

常规的 SFT 训练都是希望能最大化 $\log(P(y|x))$ 也就是最大化选中 chosen answer 的概率，但是 DPO 认为还需要正确答案比作物答案选择的概率大，DPO 最大化的是：

$$
loss = -\log\sigma(\log(P(y^{chosen}|x))-\log(P(y^{reject}|x)))
$$

Q：最大化 $\log(P(y^{correct}|x))-\log(P(y^{reject}|x))$ 很好理解，就是希望选择 correct 的概率大，选择 reject 的概率小，但是为什么还要在前面加一个 sigmoid 呢？
A：因为模型要学习的是 $P(y^{chosen} > y^{rejected} | x)$，概率比“分数差”更好表达学习目标。如果直接用两个 P 相减，那么它是一个无界的分数差，没有统一尺度，用 sigmoid 之后就可以把它映射到 0-1 的区间。

同时和 RLHF 一样为了不让模型训练跑偏，还要引入 KL 散度：

$$
loss = -\log\sigma(\beta \log\frac{\pi_\theta(y^{chosen}\mid x)}{\pi_{\text{ref}}(y^{chosen}\mid x)}-\beta \log\frac{\pi_\theta(y^{reject}\mid x)}{\pi_{\text{ref}}(y^{reject}\mid x)})
$$


这里的 $\pi_\theta(y^{chosen}\mid x)$ 就是 Teacher-Forcing 对 chosen 中的每一个 token 求联合概率密度 $\prod_{chosen}P(token \mid prompt)$。