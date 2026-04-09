---
title: LLM 中的强化学习：GRPO
date: 2026-03-02T20:07:09+08:00
featuredImage: http://img.xilyfe.top/img/20260226204104024.png
authors:
  - Xilyfe
series:
  - RLHF
tags:
  - 大模型
  - 强化学习
lastmod: 2026-04-09T12:00:50+08:00
---
## 公式推导

GRPO 可以看做 PPO 的变体，我们回顾一下 PPO 这个算法：

$$
\mathcal{L}_{PPO} = -\mathbb{E} \left[ \min\left( r(\theta) \hat{A}, \ \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \hat{A} \right) \right]
$$

PPO 需要同时加载 actor、ref、reward 和 critic 模型，reward 和 critic 模型得到 rewards 和 values 通过 GAE 计算得到 advantage，然后 actor 和 ref 模型得到新旧两个 log_probs 计算 KL 散度来约束模型变化。PPO 存在的问题就是，它需要 **额外训练一个 critic model** 显存翻倍了，其次 PPO 用的是 **绝对优势**，对 reward model 的噪声很敏感。所以 GRPO 的解决思路就是 <mark>去掉 critic model</mark> 同时用 <mark>相对优势</mark> 来代替。

![image.png](http://img.xilyfe.top/img/20260302201958209.png)

具体来说，GRPO 让一个 prompt 同时生成多个 response，然后用 reward model 对每一个 response 进行打分。之后就是计算相对优势了，GRPO 先算这组的均值和标准差：

$$
\mu = \frac{1}{G} \sum r_i, \quad \sigma = \sqrt{\frac{1}{G} \sum (r_i - \mu)^2}
$$

然后每个回答的优势：

$$
\hat{A}_i = \frac{r_i - \mu}{\sigma + \epsilon}
$$

这样 GRPO 就把每一个 response 的得分化作了组内的相对优势，减少了 critic model 的占用。GRPO 和 PPO 损失函数公式相同，就是把 advantage 换了一个计算方式，我们就能得到 GRPO 的公式：

$$
J_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G}  \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right)  - \beta D_{\text{KL}} \right]
$$

>在训练DeepSeek-R1-Zero时，不仅去掉了价值模型，甚至连奖励模型都去掉了。取而代之的是仅仅使用基于规则的奖励函数。进一步降低了计算消耗。
>- 准确奖励：对于有确定结果的问题，直接判断结果是否正确。例如数学题，代码题。
>- 格式奖励：是否按照指定格式输出。（对于没有客观答案的题，只判断格式进行奖励。为后面的自我进化做铺垫)

在 PPO 里面我们用 GAE 计算了每个时间步的 advantage，那么 GRPO 采用同一个公式应该也是逐 token 的 advantage吧。可是 reward model 是对整个 sequence 进行打分，并且归一化后我们每个句子得到一个相对优势，那是不是矛盾了呢？实际上GRPO 的 advantage 确实是基于句子级 reward 计算出来的标量，但在计算 policy gradient / surrogate loss 时，这个标量会被广播到该 response 的所有 token 上，所以训练过程仍然是逐 token 的，只是所有 token 用的 advantage 值相同。这么做是因为 GRPO 的设计目标就是**简化 + 适配可验证奖励**：
- 很多推理任务的 reward 是**句子级别对错**（数学题答对=1，答错=0；代码跑通=1，没跑通=0）
- 如果强行用 GAE 去逐 token 估计 advantage，由于中间 token 没有显式奖励信号，反而会引入大量噪声
- 而“一题多答 + 组内相对”正好利用了“同样问题不同回答的质量差异”来提供干净的相对信号

{{< admonition type=question title="GRPO 每个 token 的损失都一样为什么还要求平均？">}} 
一开始看的时候我有个疑惑：reward function 都是对一整个 sequence 进行打分得到优势，然后通过 group 内标准化得到优势 $\hat A_i$，这个优势被分配给 sequence 内的所有 token $\hat A_{i,t} = \hat A_i$，也就是说每一个 token 的优势是一样的。其次在 GRPO 中 `num_iterations=1`，也就是说： 

$$
r_{i,t}(\theta) = \exp(\log \pi_\theta - \log \pi_{\text{old}}) = 1
$$

那么每个 token 的损失 $r_{i,t}(\theta) \hat{A}_{i,t}$ 不就都是相同的吗，我们有什么必要求平均呢？问题在于：**损失的值相同，但是每个 token 的梯度不同**。损失函数对 $\theta$ 求梯度时，虽然 $r_{i,t}$​ 的 **数值**为 1，但它**不是常数**，因为：

$$
r_{i,t}(\theta) = \exp\bigl(\log\pi_\theta(o_{i,t}) - \log\pi_\theta(o_{i,t}).\text{detach()}\bigr)
$$

对 $\theta$ 求导：

$$
\nabla_\theta r_{i,t} = r_{i,t} \cdot \nabla_\theta \log\pi_\theta(o_{i,t}) = 1 \cdot \nabla_\theta \log\pi_\theta(o_{i,t})
$$

所以整体梯度为：

$$
\nabla_\theta \mathcal{L} = -\frac{1}{G} \sum_i \hat{A}_i \cdot \left( \frac{1}{|o_i|} \sum_t \nabla_\theta \log\pi_\theta(o_{i,t}) \right)
$$

每一个 token 的贡献不同，虽然大家共享同一个 $\hat{A}_i$，但每个 token 在当前模型下的 log-probability 不同。有的 token 模型本来就很确定，有的很犹豫。梯度下降会根据 $\hat{A}_i$ 的正负，去集体拉高或压低这组 token 的出现概率。如果不求平均，只选其中一个 token 来计算，那么只更新了模型在那个特定位置的生成偏好，而忽略了整个 sequence 其他 token 的贡献。

其次是为了消除序列长度的影响。如果在计算 Loss 时不除以 $|o_i|$：
- **长响应会统治梯度：** 一个 1000 tokens 的废话连篇的回复，其梯度总和会比一个 10 tokens 的精炼回复大 100 倍。 
- **训练不稳定：** 如果不平均，模型会为了获取更高的“总梯度”而倾向于把序列越写越长。
{{< /admonition >}}

{{< admonition type=question title="GRPO 的损失是 token-level 还是 sample-level?">}} 
在网上搜 GRPO 的损失还是 token-level 还是 sample-level 有很多声音，如果问 ChatGPT 会告诉你 GRPO 是 token-level 的 loss，但是如果你看过我 DAPO 的文章，里面写到 GRPO 是 sample-level 的，长序列 token 的权重会被稀释，所以答案是什么？

$$
\mathcal{L}_{\text{GRPO}} = \frac{1}{G} \sum_{i=1}^{G} \underbrace{\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \ell_{i,t}}_{\text{先在 sample 内做 token 平均}}
$$

- 说 token-level 的人指的是 loss 是对每个 token 的 log prob 逐一计算的
- 说 sample-level 的是指每条回答对 loss 的贡献是**等权**的，每条回答权重 = $\frac{1}{G}$。也正是因为每个 sample 的权重相同，如果某个权重的 token 很多，每个 token 的权重就会相应被稀释，这也是 DAPO 所说的。

所以GRPO 中 token-level 是指计算粒度，说它 sample-level 是指最终贡献权重。但是现在的强化学习训练框架里面，GRPO 通常也把原始的公式改成 DAPO 中 token-level 的实现方式。
{{< /admonition >}}

## 代码实现

```python
class GRPOTrainer:

    def __init__(
            self,
            actor_model: nn.Module,
            ref_model: nn.Module,
            reward_model: nn.Module,
            tokenizer: TokenizerType,
            config: GRPOConfig,
            train_dataset: Dataset
    ):
        self.actor_model = actor_model.to(config.device)
        self.ref_model = ref_model.to(config.device)
        self.reward_model = reward_model.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = AdamW(params=actor_model.parameters(), lr=config.lr)
        self.dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=config.batch_size
        )

    @staticmethod
    def compute_logprobs(
            model: nn.Module,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            num_actions: int
    ) -> torch.Tensor:
        """
        input_ids: [B, G, L]
        """
        B, G, L = input_ids.shape
        flat_ids = input_ids.view(B * G, L)
        flat_mask = attention_mask.view(B * G, L)

        outputs = model(flat_ids, attention_mask=flat_mask)
        logits = outputs.logits[..., :-1, :].contiguous()
        labels = flat_ids[:, 1:].contiguous()

        logprobs = F.log_softmax(logits, dim=-1)
        logprobs = logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        return logprobs.view(B, G, L - 1)[..., -num_actions:]

    def get_experience(self, prompts: list[str]) -> dict:
        inputs = self.tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}] for prompt in prompts],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        )

        num_actions = self.config.max_new_tokens
        # [B*G, seq_len]
        beam_input_ids = inputs["input_ids"].repeat_interleave(self.config.num_generations, dim=0)
        beam_attn_masks = inputs["attention_mask"].repeat_interleave(self.config.num_generations, dim=0)

        # [B, G, seq_len+max_new_tokens]
        with torch.no_grad():
            prompt_response = self.actor_model.generate(
                input_ids=beam_input_ids,
                attention_mask=beam_attn_masks,
                max_new_tokens=num_actions,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            ).view(self.config.batch_size, self.config.num_generations, -1)

        attention_mask = prompt_response.ne(self.tokenizer.pad_token_id).long()
        response_mask = prompt_response[:, :, beam_input_ids.size(-1):].ne(self.tokenizer.pad_token_id).long()

        # [B, G]
        with torch.no_grad()
            rewards = self.reward_model(prompt_response).detach()
        # [B, G]
        mean_reward = rewards.mean(dim=-1, keepdim=True)
        std_reward = rewards.std(dim=-1, keepdim=True)
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        with torch.no_grad():
            # [B, G, resp_len]
            old_logprobs = self.compute_logprobs(self.actor_model, prompt_response, attention_mask, num_actions)
            ref_logprobs = self.compute_logprobs(self.ref_model, prompt_response, attention_mask, num_actions)

        return {
            "prompt_response": prompt_response,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "old_logprobs": old_logprobs.detach(),
            "ref_logprobs": ref_logprobs.detach(),
            "advantages": advantages,
            "num_actions": num_actions
        }

    def compute_loss(self, exp: dict) -> torch.Tensor:
        old_logprobs = exp["old_logprobs"]
        ref_logprobs = exp["ref_logprobs"]
        advantages = exp["advantages"].unsqueeze(-1)  # [B, G, 1]
        response_mask = exp["response_mask"]
        num_actions = exp["num_actions"]

        cur_logprobs = self.compute_logprobs(
            self.actor_model, exp["prompt_response"], exp["attention_mask"], num_actions
        )

        log_ratio = cur_logprobs - ref_logprobs
        # [B, G, resp_len]
        ratio = log_ratio.exp()


        k3 = -log_ratio + ratio - 1
        surr_1 = ratio * advantages
        surr_2 = torch.clamp(ratio, 1 - self.config.eps, 1 + self.config.eps) * advantages
        per_token_loss = (-torch.min(surr_1, surr_2) + self.config.beta * k3) * response_mask
        loss = per_token_loss.sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8)
        return loss.mean()

    def train(self):
        for epochs in range(self.config.epochs):
            for batch in self.dataloader:
                exp = self.get_experience(batch["prompts"])
                for _ in range(self.config.ppo_epochs):
                    self.optimizer.zero_grad()
                    loss = self.compute_loss(exp)
                    loss.backward()
                    self.optimizer.step()
```

GRPO 的实现就比较简单了，这里主要记录两个问题：

首先 GRPO 对一个 promp 会 generate 多个 response，然后在 prompt 对应的这一组的 response 中进行打分，求平均优势。GRPO 对一个 prompt 生成多个回答的方式是让 prompts 进行 repeat，形状变成 $[B,L] \rightarrow [B*G, L]$，然后我们需要组内打分时候变形为 $[B*G,L] \rightarrow [B,G,L]$ 然后在第二维求和求平均就好了。

```python
# [B*G, seq_len]
beam_input_ids = inputs["input_ids"].repeat_interleave(self.config.num_generations, dim=0)
beam_attn_masks = inputs["attention_mask"].repeat_interleave(self.config.num_generations, dim=0)
```

第二个就是掩码部分，我们需要处理好 `attention_mask` 和 `response_mask`。`attention_mask` 是整个 sequence 的掩码，用于 Teacher-Forcing 求句子 logits 时候不关注 PAD 部分。`response_mask` 是 response 部分的掩码，用于求 logprobs 时候把 PAD 部分置为 0。

```python
attention_mask = prompt_response.ne(self.tokenizer.pad_token_id).long()
response_mask = prompt_response[:, :, beam_input_ids.size(-1):].ne(self.tokenizer.pad_token_id).long()
```