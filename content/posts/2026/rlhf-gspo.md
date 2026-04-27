---
title: LLM 中的强化学习：GSPO
date: 2026-03-17T11:26:32+08:00
featuredImage: http://img.xilyfe.top/img/20260311113437709.png
authors:
  - Xilyfe
series:
  - RLHF
tags:
  - 大模型
  - 强化学习
lastmod: 2026-04-28T12:08:16+08:00
---
{{< admonition type=info title="Summary">}} 
GSPO（Group Sequence Policy Optimization，群组序列策略优化）是阿里 Qwen 团队提出的一种强化学习对齐算法，**与以往基于token级别重要性比率**的算法不同，GSPO将重要性比率定义在**序列似然**的基础上，并在**序列级别**进行裁剪、奖励和优化。我们证明，GSPO在训练效率和性能方面均优于GRPO算法，显著稳定了 MoE 架构下的强化学习训练过程，并有望简化强化学习基础设施的设计。
{{< /admonition >}}

## 1. 存在的问题

### 1.1 粒度错配

![image.png](http://img.xilyfe.top/img/20260427141721947.png)

Qwen 团队发现在 MoE 架构下用 GRPO 训练的时候，很容易出现训练后期崩塌的问题。它们检查了 RL Infra、数据集等各方面问题，最终认为问题可能出在 GRPO 的公式本身，而 MoE 架构下多 router 放大的这个问题。它们提出的问题是 GRPO 公式里面的 importance ratio 起了什么作用？

$$
w_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})} 
$$

GRPO 在工程实现里面会采用 mini-batch 的训练方法，第二次开始就变成了 off-policy，所以 GRPO 用这个 importance ratio 来修正两个策略之间的差别。而问题恰恰在于：**重要性采样是一个大数定律级别的工具，它需要多次采样的平均来发挥作用。单次采样的重要性权重在统计上是没有意义的。**

这是什么意思呢？在 GRPO 的目标函数中，对每个 token 位置 t 的重要性比率：是这样计算的：
- 分子：新策略在位置 t 给 token $y_{i,t}$ 的概率 $\pi_\theta(y_{i,t}|x, y_{i,<t})$
- 分母：旧策略在位置 t 给同一 token 的概率 $\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})$

问题来了：$y_{i_t}$ 是从 $\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})$ 中单次采样得到的。对于位置 t 的这个分布，我们只有一个样本。这意味着这里的重要性比率，根本没有满足重要性采样多样本平均的前提。它不是在纠正分布偏差，而是在用一个基于单次采样的随机权重去缩放梯度，这没有起到修正的作用，反而是在引入噪声。这个问题在 MoE 架构下更加严重。以 48 层 Qwen3-30B-A3B 为例，每次参数更新后约有 10% 的专家路由改变，路由改变意味着参数更新前后，**模型处理同一个 token 时走的计算路径不同**。这直接导致 token 级概率在相邻两次计算间剧烈波动，进而使重要性比率 $w_{i,t}$ 的操声更大。实验结果证明 GRPO 在不采用任何工程手段的情况下，根本无法在 MoE 模型上正常收敛，训练奖动长期在。

其次 GRPO 的 importance ratio 和它的 reward 设计也不匹配。GRPO 是把每一个 response 的奖励在组内进行归一化，然后平摊到每一个 token 上，所以它的 reward 是 sample-level 的，但是又试图在 token 层面进行修正，两者粒度是不一致的。

{{< admonition type=info title="Routing Replay">}} 
为了让 GRPO 这类算法能在 MoE 架构上稳定运行，研究者们采用了一种名为 Routing Replay 的策略。它的功能是：
- 在模型生成数据时，记录下每个词元由哪些专家处理。
- 在模型优化、需要进行新旧对比时，强制新模型回放旧策略的 router 选择。

尽管有效，但这毕竟是一个额外的补丁，增加了系统复杂性，也限制了模型自由探索更优专家组合的能力。
{{< /admonition >}}


{{< admonition type=question title="为什么 PPO 中没有提到这个问题？">}} 

PPO 不也是会对每个 token 加一个 IS 修正吗，为什么 PPO 中就没有提到这个问题？因为虽然 PPO 同样做 per-token IS 修正，但它有一个关键组件 GRPO 没有：**Critic**。


$$
\mathcal{L}^{PPO} = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]
$$

其中：

- $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ — per-token IS ratio
- $\hat{A}_t$​ — **由 Critic 网络在 token 级别估计的优势函数**（通过 GAE）

关键点：**IS ratio 是 token 级的，Advantage 也是 token 级的，两者粒度一致。** Critic 网络专门学习 $V(s_t)$，为每个 token 的状态提供独立的价值估计，所以 $\hat{A}_t$ 本身就是对这一步决策质量的局部刻画。GRPO 的每个 token 用自己的 IS ratio 去乘一个 sequence-level reward，既不是在做正确的序列级 IS 修正，也不是在做真正的 token 级修正——它处于一个两头不靠的中间状态，所以引入了额外噪声。而PPO 的 per-token IS 乘以 token-level advantage，反而是在正确地做局部修正，统计意义上是自洽的。

{{< /admonition >}}

### 1.2 clip 不对称

其次 PPO/GRPO 的 clip 操作将重要性比率限制在 $[1-\epsilon, 1+\epsilon]$，本意是限制更新幅度。但仔细看裁剪后的实际范围：
- 正优势（好的样本，想提升概率）：比率被限制在 $[0, 1+\epsilon]$。
- 负优势（坏的样本，想降低概率）：比率被限制在 $[1-\epsilon, +\infty]$，理论上没有上界。

对负优势样本来说，当某个 token 的单次采样怡好产生了一个很大的噪声比率，这个比率不会被裁掉，会以原始大小放大这个 token 的负向梯度，加速模型往错误方向更新。这正是 GRPO 训练崩溃的常见模式：少数噪声大的 token 产生了超大的负向梯度，拉着整个模型快速偏离。

{{< admonition type=question title="clip 不是限制了上界吗，为什么负优势没有上界？">}} 

我一开始有这种疑惑，实际上是没有区分 **clip函数本身的范围** 和 **min操作之后实际生效的值**。clip 函数本身确实把 $r_t$ 约束在 $[1-\epsilon, 1+\epsilon]$，但 min操作会根据 $A_t$ 的符号，选择两项中更小的那个，这才导致了不对称。当负样本 $A_t<0$ 时，如果 $r_t>1+\epsilon$ 那么 $r_tA_t<(1+\epsilon)A_t$，负样本在上方没有截断，$r_t$​ 可以趋向 $+\infty$。

{{< /admonition >}}


## 2. 解决方案

通过上述分析可以看到改进方向有两条路，要么把奖励和修正的粒度统一到 token-level 要么统一到 sequence-level。如果选择 token-level 的 reward，那么就和 PPO 类似了（用 critic model）实现难度和 scale 难度都比较大，所以 GSPO 选择 seq-level 的 importance ratio ：


$$
J_{\mathrm{GSPO}}(\theta)=\mathbb{E}_{x\sim \mathcal{D},\,\{y_i\}_{i=1}^{G}\sim \pi_{\theta_{\text{old}}}(\cdot|x)}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(s_i(\theta)\hat{A}_i,\;\operatorname{clip}\big(s_i(\theta),1-\epsilon,1+\epsilon\big)\hat{A}_i\right)\right]
$$

可以注意到 GSPO 和 GRPO 的目标函数有所不同：

$$J_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G}  \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right)  - \beta D_{\text{KL}} \right]$$

由于 GSPO 全面转向了 seq-level 的重要性采样和优势，所以不需要把优势平摊到每一个 token 再进行累加。其次 GSPO 对 importance ratio 做了修改：

$$s_i(\theta)=\left(\frac{\pi_{\theta}(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}\right)^{\frac{1}{|y_i|}}=\exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\right)$$


>这里对长度进行归约防止单个 token 的 importance ratio 影响 seq-level 的数值。

通过 seq-level 的重要性采样，它除了解决粒度错配问题，也解决了 clip 不对称的问题。用了 seq-level 的 ratio 之后，单个 token 的噪声被**平均/稀释**进了整条序列，不再有机会单独造成梯度爆炸。clip 作用在这个聚合后的量上，对称性自然恢复。


## 3. 梯度分析

- **GSPO 梯度**

$$\nabla_{\theta} J_{\mathrm{GSPO}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,(y_i)_{i=1}^{G}\sim \pi_{\theta_{\text{old}}}(\cdot|x)}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\frac{\pi_{\theta}(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}\right)^{\frac{1}{|y_i|}}\hat{A}_i\cdot \frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\nabla_{\theta}\log \pi_{\theta}(y_{i,t}|x,y_{i,<t})\right]$$

- **GRPO 梯度**

$$\nabla_{\theta} J_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,(y_i)_{i=1}^{G}\sim \pi_{\theta_{\text{old}}}(\cdot|x)}\left[\frac{1}{G}\sum_{i=1}^{G}\hat{A}_i\cdot \frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\nabla_{\theta}\log \pi_{\theta}(y_{i,t}|x,y_{i,<t})\right]$$

从梯度角度分析就可以看到 GSPO 和 GRPO 的区别，两者对一个 response 不同 token 梯度加权的方式不同。GRPO 用每个 token 的重要性权重对梯度进行加权，而 GSPO 用 importance ratio 对整个句子的梯度做一个约束。

## 4. 实验结果

### 4.1 数据分析

![image.png](http://img.xilyfe.top/img/20260427161233929.png)

1. 重要性采样从 token-level 变成了 seq-level，计算速度大幅度提高
2. 解决了粒度不对齐的问题，训练后期也没有发生坍塌更加稳定
3. 下游任务性能更强

### 4.2 裁剪悖论

![image.png](http://img.xilyfe.top/img/20260427161456698.png)

GSPO 论文中出现了一个很有意思的实验结果：GSPO 通过 clip 裁剪掉的 token 比例是 GRPO 的 **115 倍**。

- GRPO clip 了 0.13% 的 token，这意味着 99.87% 的 token 的重要性比率都落在 $[1-\epsilon, 1+\epsilon]$ 内没有被裁剪。一看起来更新非常精准，但这怡怡是问题所在：这些没有被裁剪的权重，包含了大量的噪声。它们是基于单次采样计算的随机值，在统计上没有意义，但梯度计算时照单全收了，GRPO 在把噪声当信号处理。
- GSPO 的序列级 importance ratio 经过了序列内的几何平均，单个 token 的随机噪声被大幅平滑掉了。$s_i$ 的数值更加稳定，天然接近 1.0，clip 的范围因此可以设得极窄。被裁剪的 15% 是真正偏离过大的更新，保留下来的 85% 梯度质量远高于 GRPO 全部保留的那些梯度。


## 5. GSPO-token

GSPO-token 想同时满足两件事：
1. token-level 的重要性采样方差太大了，所以希望用 seq-level 的 IS 来修正。
2. 在类似 multi-turn 的场景中，我们可能希望实现更细粒度的 advantages 调整，GSPO 采用的 seq-level 优势太粗粒度了。

但是这两件事天然冲突，假如我们直接写 $s_i(\theta) \cdot \hat{A}_{i,t}$，对 $\theta$ 求导，那么会出现：

$$
\begin{align}
&= \nabla_\theta \left[\frac{1}{|y_i|}\sum_t s_i(\theta) \cdot \hat{A}_{i,t}\right] \\
&= \frac{1}{|y_i|}\sum_t \hat{A}_{i,t} \cdot \nabla_\theta s_i(\theta) \\ 
&= \frac{1}{|y_i|} \cdot \nabla_\theta s_i(\theta) \cdot \sum_t \hat{A}_{i,t} \\
&= \frac{s_i(\theta)}{|y_i|} \cdot \left(\sum_t \hat{A}_{i,t}\right) \cdot \frac{1}{|y_i|}\sum_{t'} \nabla_\theta \log\pi_\theta(y_{i,t'})
\end{align}
$$

那么每个 token $t'$ 收到的梯度权重是：

$$
\frac{s_i(\theta)}{|y_i|^2} \cdot \underbrace{\left(\sum_t \hat{A}_{i,t}\right)}_{\text{所有token的总和}}
$$

所有 token 共享**同一个权重**——就是所有 token advantage 的总和。$\hat{A}_{i,t}$ 根本没有机会单独作用在各自的 token 上，那梯度就还是序列级的，per-token advantage 就没有意义了。

GSPO-token 采用了一个很巧妙的方法，它让重要性采样变成 $s_{i,t}(\theta) = \text{sg}[s_i(\theta)] \cdot \frac{\pi_\theta(y_{i,t})}{\text{sg}[\pi_\theta(y_{i,t})]}$：
- 第一项 $\text{sg}[s_i(\theta)]$ 数值上就是序列级 IS ratio 的当前值
- 第二项 $\dfrac{\pi_\theta(y_{i,t})}{\text{sg}[\pi_\theta(y_{i,t})]}$ 分子分母数值相同，所以整体数值为一

当我们进行求导时候就有意思了：

$$
\begin{align}
&= \nabla_\theta \left[ \frac{1}{|y_i|}\sum_t \text{sg}[s_i(\theta)] \cdot \frac{\pi_\theta(y_{i,t})}{\text{sg}[\pi_\theta(y_{i,t})]} \cdot \hat{A}_{i,t} \right] \\
&= \frac{1}{|y_i|}\sum_t \frac{\text{sg}[s_i(\theta)] \cdot \hat{A}_{i,t}}{\text{sg}[\pi_\theta(y_{i,t})]} \cdot \nabla_\theta \pi_\theta(y_{i,t}) \\
&= \frac{1}{|y_i|}\sum_t \frac{\text{sg}[s_i(\theta)] \cdot \hat{A}_{i,t}}{\text{sg}[\pi_\theta(y_{i,t})]} \cdot \pi_\theta(y_{i,t}) \cdot \nabla_\theta \log\pi_\theta(y_{i,t}) \\
&= \frac{1}{|y_i|}\sum_t \text{sg}[s_i(\theta)] \cdot \hat{A}_{i,t} \cdot \nabla_\theta \log\pi_\theta(y_{i,t}) \\
&=\frac{\text{sg}[s_i(\theta)]}{|y_i|} \sum_t \hat{A}_{i,t} \cdot \nabla_\theta \log\pi_\theta(y_{i,t})
\end{align}
$$

此时 token $t$ 的梯度权重是 $\hat{A}_{i,t}$，各 token **独立地**被自己的 advantage 加权。GSPO-token 本质上是用了一个"障眼法"——让 clip 和 IS 的判断依然发生在序列级从而保持稳定性，但把梯度信号下放回 token 级，从而支持精细 advantage。