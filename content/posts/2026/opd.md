---
title: On-Policy Distillation
date: 2026-05-07T13:40:46+08:00
featuredImage: http://img.xilyfe.top/img/20260507134243217.png
authors:
  - Xilyfe
series:
  - 论文阅读
tags: []
lastmod: 2026-05-31T11:58:28+08:00
---
>参考论文：A Survey of On-Policy Distillation for Large Language Models 
>
>标签：`Knowledge Distillation` `On-Policy Learning` `LLM Post-Training` `Exposure Bias` `RLKD`
>
>TL;DR：OPD 通过让 student 在自身 rollout 的轨迹上接受 teacher 反馈，从根本上解决了传统 off-policy 蒸馏的 exposure bias 问题；本文提出统一的 ff f-散度框架，将 GKD、MiniLLM、DistiLLM 等方法统一建模，并沿反馈信号（logit/outcome/self-play）、teacher 访问方式（白盒/黑盒/无 teacher）、损失粒度（token/sequence/hybrid）三个维度系统梳理 OPD 全貌，揭示其与密集 token 级 RL 的数学等价性，已被 Qwen3、DeepSeek-V4 等主流模型后训练采用。

## 1. 研究动机

大模型的后训练方法大致可以分为两类：on-policy 训练和 off-policy 训练。
- off-policy 训练：sft 和我们之前讲过的知识蒸馏里面的白盒蒸馏就是一个 off-policy 策略。回忆一下白盒蒸馏的过程：我们用教师模型 rollout 得到多条 trajectory 以及每个 token 的概率分布，然后我们用 teacher-forcing 的方式让学生模型输出在这段 trajectory 上的概率分布，通过交叉熵损失实现类似 sft 的操作。这个过程中学生模型训练用到的数据来自教授模型，而不是自己生成的，所以就是一个 off-policy 训练。这种 off-policy 训练有一个好处就是它的效率非常高，一整个 trajectory 里面每个 token 都可以提供 signal，相比之下 rl 通常是 outcome-based 效率就低很多。
- on-policy 训练：该方法从学生模型自身的生成中采样 trajectory，并给予一定的奖励。 强化学习是典型的 on-policy 训练方法。例如，在训练一个模型解决数学问题时，可以通过评估模型生成的每个解题步骤是否正确来给予奖励。on-policy 的缺点前面提到了，就是奖励信号非常稀疏。但是它的优点也很明显：RL 的训练目标是强化自身生成的好样本，而不是强行去拟合新的数据分布，对模型内部的知识不会产生破坏效果，**不容易出现灾难性遗忘的问题**。

理想的后训练方法应该兼具二者之长，既能获得 on-policy 训练的相关性，又能利用 off-policy 蒸馏的密集奖励信号？这就引出了本文的核心—**On-Policy Distillation**。

## 2. 公式推导

on-policy distillation 的核心思想是：**从学生模型中采样轨迹，并使用一个高性能的教师模型来为该轨迹中的每一个 token 进行打分**。这里的打分指的不是 llm-as-judge 那种打分，OPD 本质上就是通过采样的方法，优化模型策略 ​$\pi_{\theta}$​ 和教师策略 $\pi_{\text{teacher}}$​ 之间的 reverse KL。也就是说 OPD 的优化目标本质是最小化：

$$
J_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D} \left[ D_{\text{KL}}(\pi_\theta(\cdot|x) \| q(\cdot|x)) \right]
= \mathbb{E}_{x, y \sim \pi_\theta} \left[ \log \pi_\theta(y|x) - \log q(y|x) \right]
$$

那按照以往的思路我们有两个办法：直接把 KL 散度取负数当做 loss 进行反向传播，或者对它做 policy gradient，那该怎么选择呢？答案是只能对 reverse KL 做 policy gradient，不能直接反向传播，forward KL 才能反向传播。

---

$$
D_{\mathrm{KL}}(q \| \pi_\theta) = \sum_y q(y) \log \frac{q(y)}{\pi_\theta(y)} = \mathbb{E}_{y \sim q}\bigl[\log q(y) - \log \pi_\theta(y)\bigr]
$$ Forward KL 的公式如上，可以看到，由于期望是对 $q$ 取的，而 $q$ 不依赖 $\theta$，因此梯度可以直接移入期望内（梯度的期望等于期望的梯度）： 

$$
\nabla_\theta D_{\mathrm{KL}}(q \| \pi_\theta) = \mathbb{E}_{y \sim q}\bigl[-\nabla_\theta \log \pi_\theta(y)\bigr]
$$

而我们通过蒙特卡洛估计从 $q$ 中采样一个样本 $y \sim q$，构造损失： $$\hat{J} = -\log \pi_\theta(y)$$ 对应的梯度估计量： $$\hat{g} = -\nabla_\theta \log \pi_\theta(y)$$ 其期望恰好等于真实梯度： $$\mathbb{E}_{y \sim q}[\hat{g}] = \mathbb{E}_{y \sim q}\bigl[-\nabla_\theta \log \pi_\theta(y)\bigr] = \nabla_\theta D_{\mathrm{KL}}(q \| \pi_\theta)$$ 因此 $\hat{g}$ 是**无偏梯度估计量**。

$$
J(\theta) = D_{\mathrm{KL}}(\pi_\theta \| q) = \sum_y \pi_\theta(y) \log \frac{\pi_\theta(y)}{q(y)} = \mathbb{E}_{y \sim \pi_\theta}\bigl[\log \pi_\theta(y) - \log q(y)\bigr]
$$
与 Forward KL 的关键区别在于：Reverse KL 的**期望是对 $\pi_\theta$ 取的，期望内部和下标都依赖 $\theta$**，因此不能简单地将梯度移入期望。

我们定义辅助函数： 

$$
f_\theta(y) = \log \pi_\theta(y) - \log q(y)
$$ 
则目标函数可以写成：

$$
J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[f_\theta(y)] = \sum_y \pi_\theta(y) f_\theta(y)
$$ 
对其求梯度，使用乘积法则展开： 

$$
\nabla_\theta J(\theta) = \sum_y \nabla_\theta \pi_\theta(y) \cdot f_\theta(y) + \sum_y \pi_\theta(y) \cdot \nabla_\theta f_\theta(y)
$$
利用恒等式 $\nabla_\theta \pi_\theta(y) = \pi_\theta(y) \nabla_\theta \log \pi_\theta(y)$，将上式改写为期望形式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\bigl[f_\theta(y) \nabla_\theta \log \pi_\theta(y) + \nabla_\theta f_\theta(y)\bigr]
$$

由于 $q$ 不依赖 $\theta$，有 $\nabla_\theta f_\theta(y) = \nabla_\theta \log \pi_\theta(y)$，代入得： 

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\Bigl[\bigl(\log \pi_\theta(y) - \log q(y)\bigr)\nabla_\theta \log \pi_\theta(y) + \nabla_\theta \log \pi_\theta(y)\Bigr]
$$
合并同类项： 

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\Bigl[\bigl(1 + \log \pi_\theta(y) - \log q(y)\bigr)\nabla_\theta \log \pi_\theta(y)\Bigr]
$$ 
注意到常数项 $1$ 可以去掉，因为： 

$$
\mathbb{E}_{y \sim \pi_\theta}\bigl[\nabla_\theta \log \pi_\theta(y)\bigr] = \sum_y \pi_\theta(y) \nabla_\theta \log \pi_\theta(y) = \sum_y \nabla_\theta \pi_\theta(y) = \nabla_\theta \sum_y \pi_\theta(y) = 0
$$

因此最终得到 **Reverse KL 的梯度公式**： 

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\Bigl[\bigl(\log \pi_\theta(y) - \log q(y)\bigr)\nabla_\theta \log \pi_\theta(y)\Bigr]
$$

假如我们和 forward kl 一样，用蒙特卡洛估计从 $\pi_\theta$ 中采样 $y \sim \pi_\theta$，直接将 Reverse KL 用作 PyTorch 中的 loss： 

$$
\hat{J} = \log \pi_\theta(y) - \log q(y)
$$

backward 得到的梯度为： 

$$
\hat{g} = \nabla_\theta \log \pi_\theta(y)
$$
其期望为： 

$$
\mathbb{E}_{y \sim \pi_\theta}[\hat{g}] = \mathbb{E}_{y \sim \pi_\theta}\bigl[\nabla_\theta \log \pi_\theta(y)\bigr] = 0
$$
根本原因，在于现有深度学习框架（如 PyTorch）的 `backward()` 默认将**离散采样结果 y 视为常量**，因此只计算了 $\nabla_\theta f_\theta(y)$ 这一项，遗漏了采样分布本身随 $\theta$ 变化所带来的项：

$$f_\theta(y) \cdot \nabla_\theta \log \pi_\theta(y)$$

这一项恰好是 Reverse KL 梯度的主体，被框架自动丢弃了，导致梯度估计有偏，实际上偏差恰好等于真实梯度本身。所以 reverse KL 只能用 policy gradient，不能直接 backward。

---

那现在这个 RL Loss 该怎么设计？接着我们求它的梯度：

$$
\begin{align}
J_{\text{OPD}}(\theta) &= \mathbb E_{x\sim D}\left[D_{KL}(\pi_\theta(\cdot|x)\|q(\cdot|x))\right] \\
&=\mathbb E_{x\sim D}\left[\sum_y\pi_\theta(y|x)\log\frac{\pi_\theta(y|x)}{q(y|x)}\right] \\
&=\mathbb E_{x\sim D}\left[\sum_y\pi_\theta(y|x)(\log \pi_\theta(y|x)-\log q(y|x))\right] \\
\nabla_\theta J(\theta) &=\nabla_\theta\sum_y\pi_\theta(y)(\log \pi_\theta(y)-\log q(y)) \\
&=\sum_y\nabla_\theta \pi_\theta(y)(\log \pi_\theta(y)-\log q(y))+\sum_y\pi_\theta(y)\nabla_\theta \log \pi_\theta(y)
\end{align}
$$

由于 $\nabla_\theta \log \pi_\theta(y)=\frac{1}{\pi_\theta(y)}\nabla_\theta \pi_\theta(y)$，所以第二项 $\sum_y\pi_\theta(y)\frac{\nabla_\theta \pi_\theta(y)}{\pi_\theta(y)}=\sum_y\nabla_\theta \pi_\theta(y)$，而 $\sum_y \pi_\theta(y)=1$，所以 $\sum_y\nabla_\theta \pi_\theta(y)=\nabla_\theta 1=0$，因此：

$$
\begin{align}
\nabla_\theta J(\theta)  &= \sum_y\nabla_\theta \pi_\theta(y)(\log \pi_\theta(y)-\log q(y)) \\
&= \sum_y\pi_\theta(y)(\log \pi_\theta(y)-\log q(y))\nabla_\theta \log \pi_\theta(y) \\
&= \mathbb E_{y\sim \pi_\theta}\left[(\log \pi_\theta(y)-\log q(y))\nabla_\theta \log \pi_\theta(y)\right] \\
&= \mathbb E_{x\sim D,\ y\sim \pi_\theta(\cdot|x)}\left[(\log \pi_\theta(y|x)-\log q(y|x))\nabla_\theta \log \pi_\theta(y|x)\right]
\end{align}
$$

自回归模型的序列概率可以分解为 $\log \pi_\theta(y|x) = \sum_{t=1}^T \log \pi_\theta(y_t | c_t)$ 所以：

$$
\log\pi_\theta(y|x) - \log q(y|x) = \sum_{t'=1}^T \underbrace{\left(\log\pi_\theta(y_{t'}|c_{t'}) - \log q(y_{t'}|c_{t'})\right)}_{r_{t'}}= \sum_{t'} r_{t'}
$$

同理，序列的梯度也可以拆开：

$$
\nabla_\theta \log\pi_\theta(y|x) = \sum_{t=1}^T \nabla_\theta \log\pi_\theta(y_t|c_t) = \sum_t g_t
$$

把两个拆开的式子代回：

$$
\nabla_\theta J = \mathbb{E}_{y \sim \pi_\theta} \left[ \underbrace{\left(\sum_{t'} r_{t'}\right)}_{f(y)} \cdot \underbrace{\left(\sum_t g_t\right)}_{\nabla_\theta \log\pi_\theta(y)} \right] = \mathbb{E}\left[ \sum_t \sum_{t'} r_{t'} \cdot g_t \right]
$$

这就是 reverse KL sequence-level 的梯度估计器。但是当 $t' < t$ 时，$r_{t'}$​ 只依赖于 $t$ 步之前的前缀，而 $\mathbb{E}[g_t | x, y_{<t}] = 0$（所有 token 的梯度期望为零）。因此 $t' < t$ 的项期望为零，可以消去：

$$
\mathbb{E}[\hat{g}_{\text{seq}}] = \mathbb{E}\left[\sum_{t=1}^{T} \underbrace{\left(\sum_{t'=t}^{T} r_{t'}\right)}_{\text{Return-to-go：从 t 步往后的累计 log-ratio}} \cdot g_t\right]
$$

它的物理意义是：你在第 $t$ 步写下的词（梯度为 $g_t$），不仅要对当前这一步的奖励 $r_t$ 负责，还要对**从今往后一直到句尾**的所有奖励（ $r_{t}, r_{t+1}, \dots, r_T$ ）负责。这在强化学习里叫 **Return-to-go**。在大模型训练里，另一个很常见的做法是：每个位置只保留当前这一步的即时项：

$$\nabla_\theta J = \mathbb{E}\left[ \sum_{t=1}^{T} r_t \cdot g_t \right]$$

我们把这种近似称为 token-level OPD，它去掉了对未来奖励的耦合。那么选择 token-level OPD 还是 sequence-level OPD 呢？Revisiting On-policy Distillation 这篇文章里面通过实验对比了两种方法：
1. token-level OPD 丢掉了 sequence-level 估计器中的未来奖励耦合项，因此相对 sequence-level 目标一般是有偏的。
2. token-level 估计器的方差上界随序列长度呈二次增长 $O(T^2)$，而 sequence-level 的方差上界为四次增长 $O(T^4)$

在大模型、智能体后训练这种长时程场景里，回复序列长度可能达到几十万 token，梯度方差是否可控会直接影响训练稳定性。所以经过权衡，**低方差相较于低偏差更重要**，最终采用的是 token-level OPD。

所以现在我们已经推导出了：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{x, y \sim \pi_\theta} \left[ \sum_t r_t \cdot g_t \right]
$$

对应标准 Policy Gradient 的梯度形式：

$$
\nabla_\theta J = \mathbb{E}\left[ \sum_t A_t \cdot g_t \right]
$$

可以得到 $A_t$ 就直接对应 $r_t$，也就是把每个 token 的 reverse KL 当做 policy gradient 的 advantage 就可以了。具体实现中前面加负号是因为我们要**最小化** $J_{\text{OPD}}$，而 Policy Gradient 框架习惯写成**最大化**奖励，所以翻转符号：$A_t = -r_t$。

## 3. KL 散度

前面提到了我们用反向 KL 散度来计算两个分布之间的差异，那为什么是 Reversal KL 呢？在机器学习的语境下，我们一般是用一个模型估计出的分布 $q_\theta(x)$ 去拟合目标分布 $p(x)$，这里 $\theta$ 表示模型的参数。之前提到，KL 散度不对称，$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$，所以"哪个在前哪个在后"非常重要。


### 3.1 Forward KL 


$$D_{KL}(q \| \pi_\theta) = \mathbb{E}_{y \sim q}\left[\log \frac{q(y)}{\pi_\theta(y)}\right] = \mathbb{E}_{y \sim \textbf{teacher}}\left[\log q(y) - \log \pi_\theta(y)\right]$$

从公式上可以注意到 $y$ 是从目标分布 $q$ 采样得到的，若某处 $q(y) > 0$ 但 $\pi_\theta(y) \approx 0$，则 $-\log \pi_\theta(y) \to +\infty$，这一项会**无限大**，造成无限大的 loss。所以为了避免无限大，student **被迫**去覆盖 teacher 所有有概率的区域——哪怕需要把概率质量"摊开"，覆盖两个峰之间的低概率谷地。结果是 student 变得"面面俱到但哪里都不够好"，生成的内容是所有 teacher 模式的**模糊平均**。

### 3.2 Reverse KL

$$D_{KL}(\pi_\theta \| q) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y)}{q(y)}\right] = \mathbb{E}_{y \sim \textbf{student}}\left[\log \pi_\theta(y) - \log q(y)\right]$$

反向 KL 是从估计分布，也就是 student 自己的分布上采样的。若某处 $\pi_\theta(y) \approx 0$，该区域对期望的贡献接近零，那么 student 根本不会采样到那些位置，所以 student 不在乎 teacher 那些自己没覆盖到的区域。Student 只需要找到 teacher 的**某一个**高概率模式，集中概率质量进去，loss 就可以很小。

### 3.3 原因

所以为什么需要采用 RKL 呢？首先毋庸置疑的，我们的目标分布 $q$ 肯定是无法解析的。那么我们对 $q$ 的了解途径有两种，一是我们可以从分布 $q$ 中采样，二是给定一个样本，我们能计算它在分布 $q$ 下（即使是未归一化的）的概率密度。在当前 on-policy distillation 的情景下，我们需要在 student 上采样，所以只能计算 trajectory 在 teacher 分布上的概率密度，那么只能够采用 RKL 了。

其次我们把 Reverse KL 目标展开：

$$
\begin{align}
D_{KL}(\pi_\theta \| q) &= \mathbb{E}_{y \sim \pi_\theta}\left[\log \pi_\theta(y) - \log q(y)\right] \\
&= -\mathcal{H}(\pi_\theta) - \mathbb{E}_{y \sim \pi_\theta}[\log q(y)]
\end{align}
$$
所以当我们试图最小化 Reverse KL 时候就在做两件事：
1. 最大化 $\mathbb{E}[\log q(y)]$：让 student 采样的序列在 teacher 下有高概率
2. 最大化 $\mathcal{H}(\pi_\theta)$：保持 student 策略的多样性

这个形式和带熵正则的强化学习完全同构，这也是为什么 Revisiting OPD 论文里说"OPD 可以看作一个带熵正则的有限时域 RL 问题"。


## 4. 实验结果

### 4.1 thinking pattern 一致

实验固定 student 为 Qwen3-1.7B-Base，对比两个 teacher：不开思考模式的 Qwen3-4B，和对 Qwen3-4B-Base 直接做 GRPO的 Qwen3-4B-Base-GRPO。单看数学 benchmark 平均分，GRPO teacher 其实不如常规 4B teacher，但蒸馏结果却恰恰相反，跟随 GRPO teacher 训练的 student 获得了更显著的性能提升。 原因是 Base 模型的 thinking pattern 与同样源自 Base 的 GRPO teacher 更契合，训练初期的 Overlap Ratio（师生在候选 Top-k Token 上的重合度）显著更高。

### 4.2 information gain

在 DeepSeek 和 Qwen 两个 family 里，对比了"同源但更大尺寸的 teacher"（DS-7B 和 Qwen3-4B）和"经过额外 RL 后训练注入了新能力的 teacher"（Skywork 7B 和 Qwen3-4B-Math-RL）。结果非常明显：如果 teacher 仅仅是把参数量放大，用的还是同一套数据和训练配方，那哪怕跑分比学生高，蒸馏带来的提升也微乎其微。 
同源放大的模型在 student 视角的局部状态下，局部概率分布已经非常相似，teacher 几乎没有增量信息。所以 **高分 ≠ 好 teacher。** OPD 不是在学习高分，而是在提取 teacher 的概率分布模式，teacher 侧的 information gain 决定了蒸馏的性能上限。

所以 OPD 成功的宏观前提是 thinking pattern的匹配与新知识的注入。

## 5. OPD 机制

### 5.1 对齐高概率 token

沿用前面的典型对照组：以 R1-Distill-1.5B 为学生，对比成功的 Teacher（JustRL-1.5B）和失败的 Teacher（R1-Distill-7B）。这一次在训练的每一步都监控了三个核心指标：Overlap Ratio（师生在 Top-k 候选 Token 上的重合度）、Overlap-Token Advantage（重合 Token 的优势值对齐情况）以及 Entropy Gap（两者在相同状态下的信息熵差距）。

![](https://pic3.zhimg.com/v2-87a5c33320545ce9dd75260d243aa0c0_1440w.jpg)

成功的蒸馏过程展现出了一种极具标志性的“渐进对齐”（Progressive alignment）特征。在成功的OPD中观察到师生之间的 Overlap Ratio 随着训练步数稳步攀升（从约 72% 一路涨到了 91% 以上），同时优势值向零收敛，熵差距不断缩小。这意味着，学生在自己生成的trace上，不仅逐步定位到了老师也偏好的高概率 Token，而且正在精确校准自身在这些 Token 上的概率，使之与老师保持一致。反观 7B Teacher的失败对照组，这三个指标从一开始就陷入停滞，几乎没有变化。更关键的是，我们统计后发现，虽然师生重合的 Top-k Token 在词表中只占极小的一部分，但它们实际上承载了双方 97% 到 99% 的概率质量（Probability Mass）。这说明，高概率 Token 的对齐并不是细枝末节的副产物，而是核心概率分布演化的主轴。

![](https://pic2.zhimg.com/v2-52ca5dcb1e9bf0a6968c5f28e0d83263_1440w.jpg)

### 5.2 共享 token 和非共享 token

既然 OPD 的成功伴随着高概率 token 的对齐，那么这两者是否有因果关系呢？ 为了搞清楚这一点，OPD 设计了一个具有针对性的消融实验。把常规的 Top-k 监督信号强行拆解成了互斥的两部分：一个版本仅仅在师生重合的 Token 上计算蒸馏损失（Overlap Top-k），另一个版本我们只在双方不重合的差异 Token 上计算损失（Non-Overlap Top-k）。实验结果给出了非常明确的答案：仅仅在 Overlap Token 上进行优化，就足以几乎完美复现标准 Top-k OPD 的全部性能增益。而仅仅依靠 Non-Overlap Token 训练出来的模型，性能则大幅落后。这个发现说明 **OPD 的绝大部分有效梯度来自于在双方都已经认可的“高概率Overlap Token 集合”里，精准地进行概率权重的再分配。额外那些Non-Overlap Token，其实并没有贡献多少有价值的优化信号。**

![](https://picx.zhimg.com/v2-6f91dd610861190f76d361e5a238ff15_1440w.jpg)

到这里，我们其实就看清了 OPD 共作的一个“飞轮效应”。当师生在初始阶段存在合理的思维模式重合时（满足我们在第一部分提到的条件），老师会对这些共享的 Token 给出明确的打分。Reverse-KL 散度的优化目标会促使学生将概率质量更集中地倾注到这些被老师认可的重合 Token 上。随着重合 Token 的概率越来越高，那些原本不重合的边缘 Token 就自然而然地被“挤出”了学生的 Top-k 集合。结果就是，重合区因为优化而变得更大，优化信号又因为重合区的扩大而变得更加纯粹和稳定，形成了一个良性循环。相反，如果初始阶段师生的思维模式严重错位，重合区太小，或者老师在重合区内无法提供高于学生的“新知识”信号，这个飞轮就会从一开始卡死，从而导致我们在失败组里看到的停滞现象。

理清了这套微观机制后，一个非常实际的问题自然浮现：既然初始的重合度如此重要，那么当我们拿到一个和学生思维模式差异很大、但又确有真才实学的强力 Teacher 时，难道只能束手无策吗？我们能否通过某种工程手段，人为地帮它们把这个“飞轮”先转起来？

## 6. 冷启动

于是乎 OPD 想到了用白盒蒸馏进行冷启动。让 teacher 先 offline 地生成一批回答，用这批数据对 teacher 做一次 sft，也就是白盒蒸馏了，强行把学生的生成偏好往 teacher 的 thinking pattern 上拽。在此基础上，我们再启动正式的 OPD 训练。

实验结果证明，这一招极为有效。相比于从 Base 模型直接硬上 OPD，经过 SFT 冷启动的student在各项 Benchmark 上的最终表现都有了实质性的飞跃，并且最终的上限也显著更高。透过我们上一部分提到的微观指标来看，原因非常清晰：**SFT 直接拉高了训练初始阶段的 Overlap Ratio，并且大幅压低了师生之间的 Entropy 差距。** 我们进一步分析了 Overlap Mass（重合 Token 占据的总概率质量），发现 SFT 后的 studnet 一开始就能稳稳覆盖 teacher 高概率的核心区域。

![](https://pic3.zhimg.com/v2-aef4801275b4d895c7bbfa829dc327da_1440w.jpg)

![](https://pic1.zhimg.com/v2-89f3dc8306eb564e5b9ec5c0d83efa8a_1440w.jpg)

## 7. 存在的问题

前面我们提到的 token-level OPD 都是 sampled-token OPD，理论上 token-level 方差更小，但**实践里 sampled-token 形式依然很不稳定**：

### 7.1 奖励信号结构性失衡

每步的奖励是：

$$r_t = \log q(y_t|c_t) - \log \pi_\theta(y_t|c_t)$$

问题在于 $y_t$​ 是从学生模型采样的，学生模型给它高概率，所以教师模型对它的概率往往更低，导致 $r_t < 0$。**绝大多数 token 都在被惩罚**，只有极少数"正向事件"在推动学习。如果绝大多数 sampled token 都在被往下压，优化就会过度依赖那一小撮局部上“看起来有利”的 token。

### 7.2 tokenizer 不一致

如果教师模型和学生模型的 tokenizer 不同，同一段文本被切成不同 token，那么比较单个 token 的概率就没有意义。比如学生把 `<think>` 切成 `<` + `think` + `>`，教师切成 `<th` + `ink` + `>`。学生输出 `<` 时，教师给它极低概率（教师更偏好 `<th`），产生巨大负奖励——但两个模型在语义上其实想表达同一件事。而且 sampled-token OPD 监督压在单个 token 上；如果这个 token 本身还是 tokenizer 依赖的产物，那么这个奖励很容易把优化带偏。


## 8. 改进方案

### 8.1 topk-token OPD

不再比较单个 sampled token，而是在**教师模型 top-K 支持的 token 子集**上比较两个分布的 KL：
1. 取 top-K 支持集$S_t = \text{TopK}_q(c_t)$教师模型最看好的 K 个 token）
2. 在 $S_t$ 上对教师和学生各自重归一化：$\hat{\pi}_\theta(v|c_t) = \frac{\pi_\theta(v|c_t)}{\sum_{u \in S_t} \pi_\theta(u|c_t)}, \quad \hat{q}(v|c_t) = \frac{q(v|c_t)}{\sum_{u \in S_t} q(u|c_t)}$
3. 计算局部 KL：$\mathcal{L}_{\text{top-K}}(c_t) = \sum_{v \in S_t} \hat{\pi}_\theta(v|c_t) \log \frac{\hat{\pi}_\theta(v|c_t)}{\hat{q}(v|c_t)}$

这样子梯度分布在 $S_t$​ 内所有 K 个 token 上，正向和负向调整分散在教师认为合理的候选区域里，而不是聚焦于学生碰巧采样出的那一个 token。

{{< admonition type=info title="重归一化">}} 
我们截取了全词表中概率最高的前 $K$ 个 token，这 $K$ 个 Token 的原始预测概率相加，其总和必定**小于 1**。如果我们直接在这 $K$ 个 Token 上去算学生和教师的 KL 散度，由于概率分布不完整，不满足概率之和为 1 的基本定义，在数学上是不严谨的，算出的梯度也是扭曲的，所以需要重新计算它们的归一化概率。

我们用 $\hat{\pi}_\theta(v|c_t) = \frac{\pi_\theta(v|c_t)}{\sum_{u \in S_t} \pi_\theta(u|c_t)}$ 对他进行放缩，得到的 $\hat{\pi}_\theta(v|c_t)$ 求和就等于 1 了。
{{< /admonition >}}

### 8.2 special-token masking

对于前面提到的 tokenizer dismatch 问题，解决方案是在 special token 所在的序列位置，将它们的 Loss 权重直接乘以 0。用公式简单表达即为：

$$L_{final}(t) = M_t \times L(t)$$

其中，$M_t = 0$（当位置 $t$ 是特殊 Token 时），否则 $M_t = 1$。

{{< admonition type=question title="为什么只需要看 special token 呢？">}} 
首先我们如果要进行 token-level OPD，那么基本上在同家族模型上进行蒸馏，也就是说它们的 tokenizer 都是一样的。只不过即使是同一个家族的模型，在经过不同的 SFT 微调或加入不同的 RL 框架后，这些系统控制符的 ID 或行为也经常发生细微的变化，所以我只需要控制这些 special token 不影响训练即可。假如我们需要在不同家族模型上进行 OPD，那么一般会选择 sequence-level OPD。
{{< /admonition >}}

## 9. verl 中实现 OPD

1. 整理 DataProto

- 位置：`ray_trainer_multitask.py:1268-1291`
- 作用：每个 training step 从 dataloader 里取出一个 batch，然后包装成 `DataProto`。

```python
for epoch in range(self.config.trainer.total_epochs):
    for batch_dict in self.train_dataloader:
        metrics = {}
        timing_raw = {}
        batch: DataProto = DataProto.from_single_dict(batch_dict)
```

2. Rollout

- 位置：`ray_trainer_multitask.py:1295-1334`
- 作用：走 agent-environment multi-turn loop 进行 rollout 生成 response。

```python
with _timer("gen", timing_raw):
    gen_batch_output = self.traj_collector.multi_turn_loop(
        gen_batch=gen_batch,
        actor_rollout_wg=self.actor_rollout_wg,
        envs=self.envs,
        is_train=True,
    )
```

3. 计算学生模型的 old_logprobs

- 位置：`ray_trainer_multitask.py:1367-1411`
- 作用：rollout 后，需要重新计算 student/actor 对自己刚才生成 token 的 logprob。

```python
with _timer("old_log_prob", timing_raw):
    if use_actor_topk:
        batch.meta_info["kl_topk_k"] = kl_topk_k
        old_log_prob = self.actor_rollout_wg.compute_log_prob_with_logits(batch)
    else:
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

    entropys = old_log_prob.batch["entropys"]
    ...
    old_log_prob.batch.pop("entropys")
    batch = batch.union(old_log_prob)
```

3. 计算教师模型的 ref_logprobs

- 位置：`ray_trainer_multitask.py:1437-1483`
- 作用：计算教师模型的 logprobs

```python
if not self.ref_in_actor:
    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
else:
    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)

batch = batch.union(ref_log_prob)
```

此时 OPD 需要的核心字段已经齐了：

```text
old_log_probs      # student logprob
ref_log_prob       # teacher logprob
responses          # student 生成的 token
response_mask      # 哪些 token 参与训练
```

3. 计算 opd advantages

- 位置：`core_algos.py:757-832`
- 作用：计算 KL 散度得到 advantage

```python
student_log_probs = data.batch["old_log_probs"]
teacher_log_probs = data.batch["ref_log_prob"]

kl_divergence = kl_penalty(
    student_log_probs,
    teacher_log_probs,
    kl_penalty="kl",
)

token_level_rewards = -kl_divergence
# ...
data.batch["token_level_rewards"] = token_level_rewards
advantages = token_level_rewards * response_mask
```

4. 反向更新

- 位置：`ray_trainer_multitask.py:1619-1626`
- 作用：这里用的是 PPO 公式计算 loss，然后反向更新

```python
if self.config.trainer.critic_warmup <= self.global_steps:
    with _timer("update_actor", timing_raw):
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        actor_output = self.actor_rollout_wg.update_actor(batch)

    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    metrics.update(actor_output_metrics)
```

actor worker 侧真正更新在 `dp_actor.py:1170-1389`：

```python
entropy, log_prob = self._forward_micro_batch(
    micro_batch=data,
    temperature=temperature,
    calculate_entropy=calculate_entropy,
)
# ...

pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,
    response_mask=kl_mask,
    cliprange=clip_ratio,
    cliprange_low=clip_ratio_low,
    cliprange_high=clip_ratio_high,
    clip_ratio_c=clip_ratio_c,
    loss_agg_mode=loss_agg_mode,
)

pg_loss.backward()
```