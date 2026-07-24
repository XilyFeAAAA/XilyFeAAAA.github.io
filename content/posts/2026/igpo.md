---
title: Information Gain-based Policy Optimization
date: 2026-07-21T11:12:28+08:00
featuredImage: http://img.xilyfe.top/img/20260721111136254.png
authors:
  - Xilyfe
series:
  - 论文阅读
tags: []
lastmod: 2026-07-21T11:12:28+08:00
---
## 研究动机

1. GRPO 采用的组内相对优势，它的优势是一般是通过 ORM 得到的。在训练初期或者训练后期很容易出现奖励全为 0 或者全为 1 的情况，这样组内相对优势接近于零，训练时候没法提供梯度。
2. 在 agentic 多轮训练中，粗粒度的 ORM 奖励态稀疏了，没法知道每个 turn 到底是不是 helpful 的。可能出现前面某个 turn 的动作是正确的，但是后面出错导致结果出了问题，反而会降低正确动作的概率。
3. ORM 的数据利用率很低，每个 trajectory 只能提供一个信号。

## 创新点

$$
\mathcal{J}_{\mathrm{IGPO}}(\theta)
= \mathbb{E}_{(q,a)\sim\mathcal{D},\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot|q)}
\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\min
\left(
\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}
{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})}
\tilde{A}_{i,t},
\right.\right.
\left.\left.
\operatorname{clip}
\left(
\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}
{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})},
1-\epsilon,1+\epsilon
\right)
\tilde{A}_{i,t}
\right)
-\beta D_{\mathrm{KL}}
\left(
\pi_{\theta}\|\pi_{\mathrm{ref}}
\right)
\right]
$$

咋一看这不是 GRPO 那个 token-mean-seq-mean 的 loss 公式吗，确实 IGPO 采用的是相同的 policy optimization，它做出的优化是给出了 process-based reward，也就是说它把 $\hat{A}_{i,t}$ 的定义改变了。

具体来说 IGPO 先进行一次正常的 agentic rollout，每一个 turn 模型输出 `think` 和 `search` 标签，然后我们把检索结果插入 history 里面，下一个 turn 继续 rollout。等到生成一整个 trajectory 之后，IGPO 把 trajectory 切成一个个 section。假设一共有 $t$ 轮对话，$s_0$ 包含了第一轮的对话+探针，$s_1$ 包含了第一轮和第二轮对话+探针，$s_{t-1}$ 就是完整的对话+探针。这里的探针指的是 `"<think>Now there’s enough information to answer</think><answer>Ground Truth a</answer>"` 这个句子，我们把这个句子加在每段对话之后做 teacher-forcing，就可以知道经过这一轮对话模型对答案的置信度是多少。IGPO 把 $t$ 个 section 都拿去做teacher-forcing，把前后两个 section 的置信度作差就可以指定 **经过这轮 rollout，策略生成 ground truth 的概率增加了还是减少了**。如果概率增加那么就可以给一个 position 的 reward。


![image.png](http://img.xilyfe.top/img/20260721112609147.png)

所以说对一个 group 的 prompt rollout 之后，我们可以得到一组 ${o_t^G}$，其中每一个 $o_t^i$ 代表了组内第 $i$ 个样本第 $t$ 轮对话带来的增益。然后和 GRPO 一样的进行组内归一化得到 $A_{i,t}$。这里很神奇的是，IGPO 参考了 PPO 的 GAE。它认为尽管 $A_{i,t}$ 衡量了每个 turn 的相对质量，但它仅考虑了及时奖励，而忽略了当前轮次对于后续轮次的影响，而捕获这种长期影响对于发展扩展 agent 的 long-horizon 能力很重要：

$$
\tilde{A}_{i,t}=\sum_{k=t}^{T}\gamma^{k-t}A_{i,k}
$$



