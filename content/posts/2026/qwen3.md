---
title: Qwen3 Technical Report
date: 2026-08-20T14:45:20+08:00
featuredImage: http://img.xilyfe.top/img/20260820144517159.png
authors:
  - Xilyfe
series:
  - 论文阅读
tags: []
categories:
lastmod: 2026-08-20T14:45:20+08:00
---

## 摘要

Qwen3 这个 model family 一共包含八个模型，六个 dense 模型和两个 moe 模型：

![image.png](http://img.xilyfe.top/img/20260820144635699.png)

Qwen3 包含以下几个创新点：
1. 把 thinking 和 no-thinking 两种模式混合在一共模型里面（以往都是分两个模型发布）
2. Qwen3 系列的小模型不是从 0-1 的复杂训练，而是直接在训练出来的旗舰模型上蒸馏

Qwen3 的架构：
- GQA
- SwiGLU
- RoPE
- RMSNorm
- pre-norm
- QKV-bias 变成 QK-Norm

>标准 Transformer 的 Q/K/V 投影通常是无 bias 的线性层，但是 Qwen2 给这三个投影矩阵都加上了偏置项，当年是为了增强模型的外推/长度泛化能力,继承自一些早期实践,但代价是训练不够稳定,尤其在低精度(FP16)下容易数值溢出。QK-Norm 具体做的事情是，在算 attention score（即 $Q*K^T$）之前，先对 Q 和 K 分别做一次 RMSNorm。


## 预训练

- 数据：coding、数学、推理、书籍，用 qwen2.5-vl 解析 PDF 然后优化得到的文本，还有通过 qwen2.5 合成的一些代码数学语料，一共 35T。
- 训练步骤：General Stage (4k 最大长度，30T数据) -> Reasoning Stage (5T reasoning 数据) -> Long Context Stage （32k 最大长度，几百 B 的长文本数据）
- 评估：通用任务、stem 任务、coding 任务、多语言任务的 benchmark

![image.png](http://img.xilyfe.top/img/20260820150750846.png)
## 后训练

![image.png](http://img.xilyfe.top/img/20260820150941670.png)
后训练主要就是两部分：
1. 首先旗舰模型先经过冷启动，然后用 RL 训练推理能力，再把 thinking 和 no-thinking 结合起来，这也是 Qwen3 提出的一共创新点，之后再用 RL 训练一些通用任务比如安全性等等。
2. 非旗舰模型就直接用蒸馏。

### 冷启动

冷启动就是用带 CoT 的数据进行 sft，让模型学会输出 long-CoT 这种推理格式。Qwen3 在冷启动期间的工作量主要是对数据进行筛选，筛选分两方面：
1. query filter：用 Qwen2.5-72B-Instruct 作为 filter，剔除包含多个子问题的题和没有标准答案的开放问题（因为 RL 训练还是 rule-based），其次过滤掉不需要 CoT 就能直接答对，如果简单模型不用推理就能对，那这道题对训练"深度推理能力"没有价值。
2. answer filter：标准过滤回答质量，质量比较低的问题，比如最终答案错误、存在大量重复内容、明显是"蒙对的"，缺乏扎实的推理过程支撑等等

Qwen3 的冷启动阶段筛选掉了大量质量不高的数据，这一阶段的目的只是让模型学会long-CoT的形式，而不是靠SFT把模型的推理能力"堆"到顶。如果冷启动阶段训练得过重、过拟合到这批数据的特定模式，可能会限制模型在后续RL阶段的探索空间和上限。


### 推理强化学习

简单说就是用 GRPO 训练了接近 4k 条高质量数据，原文如下：

>The query-verifier pairs used in the Reasoning RL stage must satisfy the following four criteria: (1) They were not used during the cold-start phase. (2) They are learnable for the cold-start model. (3) They are as challenging as possible. (4) They cover a broad range of sub-domains. We ultimately collect a total of 3,995 query-verifier pairs, and employed GRPO ([Shao et al. 2024](https://arxiv.org/html/2505.09388v1#bib.bib54)) to update the model parameters. We observe that using a large batch size and a high number of rollouts per query, along with off-policy training to improve sample efficiency, is beneficial to the training process. We have also addressed how to balance exploration and exploitation by controlling the model’s entropy to increase steadily or remain stable, which is crucial for maintaining stable training. As a result, we achieve consistent improvements in both training reward and validation performance over the course of a single RL run, without any manual intervention on hyperparameters. For instance, the AIME’24 score of the Qwen3-235B-A22B model increases from 70.1 to 85.1 over a total of 170 RL training steps.

这里提到 Qwen3 扩大了 GRPO 训练的 `batch_size` 还有 `num_group`，另外就是提到**通过控制模型熵值稳定上升或保持稳定来平衡探索与利用**。这应该是在解决 GRPO 训练时候出现的熵坍塌问题，具体来说就是模型没有探索能力了，输出单一固定。技术报告里面没有说解决方案，常见的方案就是在 loss 里面添加熵正则，去掉 KL 散度或者减小权重，大batch size + 高rollout数等等。

### thinking 融合

![image.png](http://img.xilyfe.top/img/20260820162223829.png)


Qwen 是单纯通过 sft 来实现 thinking 和 no-thinking 融合的，他们准备了两份数据，thinking 的数据和 no-thinking 的数据。然后给不同的数据不同的 chat template，拿去 sft。不过是不是思考模式，模型都会输出 think 标签，不过非思考模式是空的而已。

Qwen3 在这部分还提出了一个 **thinking budget** 的概念，具体是说模型在思考超过指定长度时候提前结束做出回答。实现方法是当 thinking 期间 token 超过指定长度，就直接在后面插入 “Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think”，然后模型就会根据前面的思考结果做出回答。

![image.png](http://img.xilyfe.top/img/20260821103845646.png)

>实验对比了 thinking budget 的有效性。横轴是思考的 token 数量，单位 K。


![image.png](http://img.xilyfe.top/img/20260821111121950.png)


### 通用强化学习

general rl 主要是训练模型的指令跟随、格式跟随能力等等，所以用到了各种的任务数据集，这方面 technical report 里面也没具体说。

### 蒸馏小尺寸模型


Qwen3 小尺寸模型的蒸馏分为两个部分：
1. 第一步是是 off-policy 的蒸馏，也就是我们常说的白盒蒸馏，由旗舰大尺寸模型生成 trajectory 和对应的 logits，然后让小模型在这条 trajectory 上面进行 teacher-forcing 得到它的 logits，然后把小模型和大模型的 soft 和 hard prediction 计算 loss，具体在大模型蒸馏那片文章里面提到过。
2. 第二步是 on-policy 蒸馏，小尺寸模型生成 trajectory 然后在大尺寸模型上面跑 teacher-forcing，得到大尺寸模型对每个 token 的 logits，再在小尺寸模型上用最小化 KL 散度来微调。

![image.png](http://img.xilyfe.top/img/20260821111324009.png)
然后千问还对比了小模型蒸馏时候 off-policy 和 on-policy 的效果，结论就是在 RL 训练后的大尺寸模型上 on-policy 蒸馏，效果比 RL 训练小尺寸模型好非常多，而且节省了大量 GPU 时间（因为蒸馏就是 sft）。