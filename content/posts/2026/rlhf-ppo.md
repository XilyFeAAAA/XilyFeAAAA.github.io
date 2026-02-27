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
lastmod: 2026-02-27T12:57:57+08:00
---
## 前置知识

### 近端策略优化

PPO 算法可以看成 A2C 的优化版。A2C的训练策略是 “采样一次，更新一次，然后扔掉数据”，这就导致效率很低，每批数据只能用一次。PPO 采用 **近端策略优化**，它会对一批数据先进行采样得到 `logprob`，`ref_logprob`，`rewards`，`advantages` 等数据，然后进行 `ppo_epochs` 次循环。每次循环内，变化的只有概率分布 `logprob` 和 `ref_logprob` 和 `values`，优势奖励这些都固定采用第一次得到的数据。

>举个例子：A2C 就像在表演现场，你一边演，导演一边喊“好”或“坏”，然后你得到反馈就修改。改完之后，刚才演的那段戏就没用了，你必须重新演一段，导演才能给新反馈。而 PPO 更像 **复盘录像**，你先演一段戏录下来，接下来的 4 个 Epoch 你坐在监视器前，对着这段录像反复琢磨。第一遍根据反馈改一点，第二遍在第一遍改动的基础上，再对着录像微调。

为什么我们在同一批数据上进行多次梯度下降，要保持 `advantage` 和 `rewards` 不变？我们从两个角度进行分析。
首先我们用一个例子帮我们抽象的理解一下。假如你是 Kobe，直升机坠毁前还在湖人打球录下了一盘比赛。如今在天上复盘这场球赛，即便你现在球技进步了，回看当年的录像，每一个进球在当时的价值（Advantage）是客观事实，不随你现在的水平变化。其次，如果我们不固定 advantage 和 rewards，用新的策略来更新 advantage 和 rewards，可能造成策略偏离太远，导致过拟合或不稳定。并且我们计算的 adv 和 reward 都是基于 old actor model 得到的，假如每个 epoch 都重新计算新的 adv 和 reward，由于 action 是在旧的 model 上得到了就会不匹配，变成 off-policy。而 ppo 是 on-policy，clip 机制（后文会提到）就会崩溃了。

### 更新约束

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

现在，我们已经限制了策略 $p(A_t|S_t)$ 的更新幅度，但还缺少一个“熔断机制”。什么意思呢？就是万一策略的更新幅度还是太大了，我们要停止策略的参数更新。PPO的做法是什么呢？因为 $\frac{p(A_t|S_t)}{p'(A_t|S_t)}$ 衡量了旧策略和现行策略之间差异，所以可以为它设置两个阈值。为了方便描述，我们令 $r(A_t, S_t) = \frac{p(A_t|S_t)}{p'(A_t|S_t)}$，这种熔断机制可以表示为：

$$
loss = -\min(r(A_t, S_t) \text{Adv}(S_t, A_t),\ \text{clip}(r(A_t, S_t) , 0.8, 1.2) \text{Adv}(S_t, A_t))
$$

- Adv 大于 0，r 大于 1.2：min 操作就会取右边的值，此时 loss 中就只剩常量了，不产生任何梯度则停止参数更新；而 r 无论多小都还是会产生梯度。
- Adv 小于 0，r 小于 0.8：min 操作就会取右边的值，此时 loss 中就只剩常量了，不产生任何梯度则停止参数更新；而 r 无论多大都还是会产生梯度

诶，那为什么我们不用管 Adv 大于 0 且 r 小于 0.8 的情况？或者 Adv 小于 0 且 r 大于 1.2 的情况？Adv 大于 0 的情况说明当前策略是好的，如果 r 小于 0.8 说明：这个策略是好的，旧模型偏向这个策略，但是新模型不怎么偏向这个策略了，那我们肯定希望能尽可能朝现在这个方向来更新参数，所以不会进行 $max(r, 0.8)$。同样 Adv 小于 0 的情况说明当前策略不怎么行，如果 r 大于 1.2 则说明这个不好的策略现在很看好，那我们肯定希望加大力度更新参数来避免这个 action，于是不应该限制更新的幅度。

### Critic loss

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

critic 和 actor 一样都有对 loss 的变化幅度做出限制，critic 预测的是 values，所以限制了 $V_{new}$ 和 $V_{old}$ 的变化：

$$
V_{clip} = V_{old} + \text{clip}(V_{new}-V_{old}, -\epsilon, \epsilon)
$$

因此得到了最终的 critic loss 公式：

$$
L^{\text{value}} = \max\left[ (V_{\text{new}}(s_t) - \text{returns})^2, \left( V_{\text{new}}(s_t) + \text{clip}(V_\theta(s_t) - V_{\text{old}}(s_t), -\epsilon, \epsilon) - \text{returns} \right)^2 \right]
$$

这里还是解释一下：首先 critic loss 是在做一个回归，我们用了 MSE，希望新模型的预测值 $V_{\text{new}}$ 尽可能接近目标值 $V_{target}$ 也是就 returns。为了防止价值函数更新太快导致策略崩溃，PPO 给 critic model 也加了一个 max 来限制更新。当 $V_{\text{new}}$ 在 $[V_{old}-\epsilon, V_{old}+\epsilon]$ 这个区间时候 $V_{\text{clip}} = V_{\text{new}}$ 正常更新；当 $V_{\theta} > V_{old} + \epsilon$，截断项里的预测值会被锁定在 $V_{\text{clip}} = V_{old} + \epsilon$。假如 $V_{\text{new}}$ 比 $V_{\text{clip}}$ 更接近 returns，那么说明一次意外的更新（$V_{\text{new}}$ 超过上界了）导致 loss 更低了，我们就得做出限制不能让他更新，max 就会选择 $(V_{\text{clip}} - \text{returns})^2$ 里面不含参数。如果 $V_{\text{clip}}$ 比 $V_{\text{new}}$ 更接近 returns，那么选择的就是 $(V_{\text{new}} - \text{returns})^2$ 正常更新了。同理 $V_{\theta} < V_{old} + \epsilon$ 也是这样，真的很巧妙。


>归根结底，actor loss 和 critic loss 里面的 clip + min(max) 都是模型为了防止过度优化做出的 **悲观估计**。actor model 意图最大化损失函数 $\text{ratio} * \text{advantages}$ 所以我们需要做一个 min 的操作，而 critic model 损失函数的均方差意图是最小化 $V_{\text{new}}$ 和 $\text{returns}$ 的差距，所以我们悲观估计时候要做 max 操作。
>PS：具体代码实现上，由于梯度下降一般需要让损失函数求最小值，所以我们在 actor model 的 loss 里面会加上负号变成 $-\text{ratio} * \text{advantages}$，可能做的是 max 操作，不过都是一个思想。


### Reward Loss

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

## trl 库源码分析

`trl.experiment.ppo.PPOTrainer.train()` 方法内部依次进行如下操作：
1. rollout 阶段：将数据集的 prompt 传给 actor 采样 response，我们就得到了 prompt+response 的问答对。
2. evaluation 阶段：用 reward 模型给这个问答对打分数 `scores`，注意 **这个分数是序列级的而不是 token 级的**。
3. optimization 阶段：把 prompt+response 用 Teacher-Forcing 的方式送入 ref、actor 和 critic 模型得到 response 中每个 token 的概率 `ref_logprob` 和 `old_logprob`，以及逐 token 的预期收益 `old_values`。根据之前计算出的整个序列的 reward，我们可以计算出每个 token 对应的 reward，这样 advantage 也就计算出来了。
4. 重复 ppo_epochs 个阶段，不断把 prompt+response 用 Teacher-Forcing 的方式传入 actor 得到每个 token **新的概率分布**，把 response 传入 critic 得到 values。然后利用之前 optimization 阶段得到的 reward 和 advantages 来计算 actor 和 critic 的 loss，更新这两个模型。

我借用知乎的几张图片来图解一下这个过程：

![image.png](http://img.xilyfe.top/img/20260224122122139.png)


前面我们提到，evalution 阶段计算的 reward scores 是序列级的，但是 PPO 在每个 step（对应生成序列中的每个token）都需要计算 advantage 来更新 actor model，这样不是矛盾了吗？实际上 reward 模型在计算序列级 reward 的时候没有加入 kl 散度，这时候计算得到分数我们叫做 `scores`。在每一个 step，我们通过 `scores`，`ref_logprob`，`old_logprob` 计算得到这个 token 的 reward，$reward = scores - \beta*kl(old\_logprob, ref\_logprob)$，最后用这个 token 对应的 value 和 reward 计算 advantage。

### rollout

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

### evaluation

```python
texts = [q + r for q, r in zip(batch["query"], batch["response"])]
reward_out = reward_model(texts)
scores = [torch.tensor(output[1]["score"]) for output in reward_out]
```

### optimization

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

## 代码实战

我这次选择直接复现 bilibili 一个 up 主的 ppo 项目 [owenliang/hf-ppo](https://github.com/owenliang/hf-ppo/blob/main/README.md) -  让大模型学会说脏话。由于采用的是 Qwen 的基模不太可能输出脏话，直接在 base 模型上进行 ppo 很难训练起来，所以我先用数据集对 base 模型进行 sft，然后在 sft 的基础上进行 ppo，这样就能完成整个流程。

这次整体的计划就是先对 Qwen 的基模进行 sft，然后在这个基础上训练出 reward 模型。用 sft 模型当 policy 和 ref_policy，用 base 模型当 value，以此进行 ppo。

### SFT

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

### Reward

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

### PPO

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

### eval

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