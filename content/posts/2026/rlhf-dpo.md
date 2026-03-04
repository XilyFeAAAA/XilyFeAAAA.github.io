---
title: LLM 中的强化学习：DPO
date: 2026-02-26T15:38:09+08:00
featuredImage: http://img.xilyfe.top/img/20260226204104024.png
authors:
  - Xilyfe
series:
  - RLHF
tags:
  - 大模型
  - 强化学习
lastmod: 2026-03-02T08:04:13+08:00
---
## 公式推导

PPO 算法我们之前聊过，它需要同时加载四个模型（Actor、Critic、Reward Model、Ref Model），显存需求极大，还需要单独训练 Reward Model 和 Critic Model。 而 DPO 只需 2 个模型：待优化的 Actor Model ($π_θ$) 和冻结的 Reference Model ($π_{ref}$)。DPO 不对 prompt+response 打分，而是直接让模型学习区分 good answer ($y_w$) 和 bad answer ($y_l$)。DPO 的本质就是把 reward model 隐式地参数化进了 policy 本身，从而一步到位完成对齐。

DPO 和 PPO 算法的核心优化目标都是：

$$
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ r_\phi(x, y) \right] - \beta \mathbb{D}_{\text{KL}} \left[ \pi(y | x) \| \pi_{\text{ref}}(y | x) \right]
$$

我们希望最大化得到奖励的期望，并且让新训练的模型尽可能和基准模型分布一致。接着我们对上面的式子进行一系列变化：

$$
\begin{align}
&\max_{\pi} \mathbb{E}\Big[ r(x,y) - \beta\log\frac{\pi(y|x)}{\pi_{\rm ref}(y|x)} \Big] \\
&\iff \min_{\pi} \mathbb{E}\Big[ \log\frac{\pi(y|x)}{\pi_{\rm ref}(y|x)} - \frac{r(x,y)}{\beta} \Big] \\
&\iff \min_{\pi} \mathbb{E}\Big[ \log\frac{\pi(y|x)}{\pi_{\rm ref}(y|x)\exp\!\big(r(x,y)/\beta\big)} \Big]
\end{align}
$$

在推导的过程中意外的发现，这个形式很像 KL 散度，但是分母 $\sum_y \pi_{\text{ref}}(y|x)\exp(\tfrac{1}{\beta}r_\phi(x,y)) \neq 1$ ，所以它不是一个合法概率分布。所以我们希望对分母进行处理，使它满足 $\sum_y q(y|x)=1$。

$$
\begin{align}
&= \min_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\exp(\frac{1}{\beta} r_\phi(x, y))\frac{1}{Z(x)}Z(x)} \right] \\
&= \min_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ \log\frac{\pi(y\mid x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp(\frac{1}{\beta} r_\phi(x, y))}-\log{Z(x)} \right]
\end{align}
$$

我们希望分母为合法的概率分布 $\sum_y\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp(\frac{1}{\beta} r_\phi(x, y))=1$，然后就能得到：

$$
Z(x)=\sum_y\pi_{\text{ref}}(y|x)\exp\!\left(\frac{1}{\beta}r_{\phi}(x,y)\right)
$$

因此分母就变成了：

$$
\pi^*(y|x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)
$$

优化目标的表达式变成了：

$$
\min_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ \log\frac{\pi(y\mid x)}{\pi^*(y\mid x)}-\log{Z(x)} \right]
$$

由于 $Z(x)$ 和我们准备优化的模型 $\pi$ 没有关系，所以我们可以把后面那一项 $\log{Z(x)}$ 直接去掉得到：

$$
\begin{align}
&= \min_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ \log\frac{\pi(y\mid x)}{\pi^*(y\mid x)}\right] \\
&= \min_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ \mathbb{D}_{\text{KL}} \left[ \pi(y | x) \| \pi^*(y | x) \right] \right]
\end{align}
$$

>为什么说 $\log Z(x)$ 可以直接去掉？
>因为 $Z(x)$ 和我们需要优化得到的 $\pi$ 没有关系。但是有人会问了，$Z(x)$ 里面不是有 $r(x,y)$ 吗？如果 $r(x,y)$ 也和 $\pi$ 无关，那我们不是可以直接在前一把直接把 $\exp\!\left(\frac{1}{\beta}r_{\phi}(x,y)\right)$ 从分母去掉了吗？确实，$r_{\phi}$ ​ 不依赖参数 $θ$，但它依赖采样变量 $y$。我们把目标拆开：
>
>$$\begin{align}&= \min_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\exp(\frac{1}{\beta} r_\phi(x, y))\frac{1}{Z(x)}Z(x)} \right]\\&= \min_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}\left[\log\pi_\theta(y|x)-\log\pi_{\text{ref}}(y|x)-\frac{1}{\beta}r_\phi(x,y)\right]\end{align}$$
>
>可以看到 第三项 $\frac{1}{\beta}r_\phi(x,y)$ 依赖于采样的 $y$ 那么也就是依赖于策略 $\pi$ 了。但是 $Z(x)$ 的表达式是一个求和，这里的 $y$ 是哑变量，不是根据策略 $\pi$ 采样的随机变量。求和一旦完成，结果是一个只依赖于 $x$ 的数。


这个优化目标我们很容易很观察到，当 KL 期望取最小值时，自然有 $\pi$ = $\pi^*$：

$$
\pi(y | x) = \pi^*(y|x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)
$$

---

我们梳理一下到目前为止我们做了什么。假设我们已经有一个奖励函数 $r$ 的基础上，我们的目标是找到能使这个目标值最大化的对齐模型 $\pi$：

$$
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ r_\phi(x, y) \right] - \beta \mathbb{D}_{\text{KL}} \left[ \pi(y | x) \| \pi_{\text{ref}}(y | x) \right]
$$

我们从这个目标出发，经过一系列推导找到了 $\pi$ 的显式解：

$$
\pi(y | x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)
$$

但是我们却很难直接利用起这个显式解形式，原因如下：
- $Z(x)$ 的值很难估计。根据 $Z(x)$ 的形式可知，想要估计它，需要对一个 x 采样足够多的回答 y，这个代价是十分昂贵的。
- 同时回顾最开始我们的目标：省略训练奖励模型这个步骤，一步到位来训练对齐模型。而目前我们得到的 $Z(x)$ 的显式解仍然需要一个确定的奖励函数 $r$，没有达到我们的目标。

我们把 $\pi$ 的显式解再调整一下，就能从最优策略反推 reward 函数：：

$$
\begin{align}
r(x,y) \ =  \beta \log \frac{\pi(y|x)}{\pi_{\rm ref}(y|x)} \ + \ \beta \log Z(x)
\end{align}
$$

既然我们得到 reward model 的表达式，那么就可以代入 Bradley-Terry 模型里面了：

$$
p(y_w \succ y_l \mid x) = \sigma \big( r(x,y_w) - r(x,y_l) \big)
$$

把 $r$ 代入后，$\log Z(x)$ 项完全抵消，就能得到最终 DPO 损失函数：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\rm ref}) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Bigg[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\rm ref}(y_l|x)}\right)\Bigg]
$$

虽然最后把 $r$ 代入 Bradley-Terry 模型的操作很丝滑，但是我有个疑问：在 PPO 里面 Bradley-Terry 模型的最大似然估计是用来当 Reward Model 的损失函数的，那我们把 $r$ 代入 loss 为什么就能变成 DPO 的损失函数呢？它们的目标都不一样吧？是因为论文已经严格证明：**只要 $π_\theta$ 收敛到最优解，它所“隐含”的 reward 就正好是 RLHF 目标里要最大化的那个 $r$**。所以直接优化这个新的 loss，等价于“先训 RM、再用 PPO”整个流程，但省掉了中间所有步骤


## 代码实现

```python
class DPOTrainer:
    def __init__(
            self,
            actor_model: nn.Module,
            ref_model: nn.Module,
            tokenizer: TokenizerType,
            config: DPOConfig,
            train_dataset: Dataset
    ):
        self.actor_model = actor_model.to(config.device)
        self.ref_model = ref_model.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = AdamW(params=actor_model.parameters(), lr=config.lr)

        dataset = self._prepare_dataset(train_dataset)
        dataset.set_format("torch")
        self.dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=config.batch_size
        )

    def compute_loss(
            self,
            chosen_logprobs: torch.Tensor,
            rejected_logprobs: torch.Tensor,
            ref_chosen_logprobs: torch.Tensor,
            ref_rejected_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.config.beta * (chosen_logprobs - ref_chosen_logprobs - rejected_logprobs + ref_rejected_logprobs)
        return -F.logsigmoid(loss).mean()

    @staticmethod
    def get_logprobs(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, resp_mask: torch.Tensor):
        logits, *_ = model(input_ids, attention_mask=attention_mask)

        logits = logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        mask = resp_mask[:, 1:].contiguous()

        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return (log_probs * mask).sum(dim=1)

    def _prepare_dataset(self, dataset: Dataset) -> Dataset:
        assert {"prompt", "chosen", "rejected"} <= set(dataset.column_names)
        return dataset.map(
            preprocess,
            remove_columns=["prompt", "chosen", "rejected"],
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "max_length": self.config.max_length
            }
        )

    def train(self):
        for epoch in range(self.config.epochs):
            for batch in self.dataloader:
                chosen_args = [batch["input_ids_chosen"].to(self.config.device),
                               batch["attention_mask_chosen"].to(self.config.device),
                               batch["response_mask_chosen"].to(self.config.device)]
                rejected_args = [batch["input_ids_rejected"].to(self.config.device),
                                 batch["attention_mask_rejected"].to(self.config.device),
                                 batch["response_mask_rejected"].to(self.config.device)]
                chosen_logprobs = self.get_logprobs(self.actor_model, *chosen_args)
                rejected_logprobs = self.get_logprobs(self.actor_model, *rejected_args)

                with torch.no_grad():
                    ref_rejected_logprobs = self.get_logprobs(self.ref_model, *rejected_args)
                    ref_chosen_logprobs = self.get_logprobs(self.ref_model, *chosen_args)

                loss = self.compute_loss(
                    chosen_logprobs,
                    rejected_logprobs,
                    ref_chosen_logprobs,
                    ref_rejected_logprobs
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```

写 DPOTrainer 时候碰到了几个问题记录一下：

首先是 **损失计算**，在 PPO 里面 log_probs 是逐 token 的，$loss = -\frac{p(A_t|S_t)}{p'(A_t|S_t)} \text{Adv}(S_t, A_t)$ 我们对每一个 token 的优势都添加约束。但是在 DPO 里面 log_probs 是逐 sequence 的，$\log \pi_{\theta}(y \mid x)$  指的是整条回复 y 在给定 x 下的对数概率。整条回复的累积 $\log P(y \mid x)$，直接用来比较 chosen 和 rejected 的相对对数概率。

```python
def get_logprobs(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, resp_mask: torch.Tensor):
   logits, *_ = model(input_ids, attention_mask=attention_mask
   logits = logits[:, :-1, :].contiguous()
   labels = input_ids[:, 1:].contiguous()
   
   mask = resp_mask[:, 1:].contiguous()
   log_probs = F.log_softmax(logits, dim=-1)
   log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
   return (log_probs * mask).sum(dim=1)
```

这里可以看到我们把 log_probs 求和得到了整个句子的 $\log \pi_{\theta}(y \mid x)$。

第二个问题是 **处理数据**，我们 DPO 的数据格式是：

```json
{
	"prompt": "",
	"chosen": "",
	"rejected": ""
}
```

从上一个问题我们可以注意到，由于 log_probs 是逐 sequence 的，所以我们需要区别 **哪些位置需要计算 log_probs**。对一个 sequence 实际上我们只需要 response 部分，也就是说开头的 prompt 部分和结尾的 PAD 我们都需要忽略。

我的解决方法是通过 `datasets.map` 对数据集进行预处理：首先对 prompt 进行 `apply_chat_template` 知道它的长度，然后计算 prompt+response 的长度，最后拼接掩码：

```python
chosen_mask = [0] * prompt_len + [1] * (len(chosen_input_ids) - prompt_len)
rejected_mask = [0] * prompt_len + [1] * (len(rejected_input_ids) - prompt_len)
```

这个做法的问题就是效率比较低===