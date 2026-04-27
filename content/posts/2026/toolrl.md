---
title: "ToolRL: Reward is All Tool Learning Needs"
date: 2026-04-24T16:26:42+08:00
featuredImage: http://img.xilyfe.top/img/20260426100146239.png
authors:
  - Xilyfe
series:
  - 论文阅读
tags: []
lastmod: 2026-04-26T03:27:53+08:00
---
>**论文**：ToolRL: Reward is All Tool Learning Needs  
**标签**：`Reinforcement Learning` `Tool Use` `LLM Agent` `GRPO`  
**TL;DR**：ToolRL 提出了一套多粒度、可动态调整的奖励体系，系统性地解决了 LLM 工具调用训练中奖励信号过于粗糙的问题，在 TIR（Tool-Integrated Reasoning）场景下显著优于 SFT 和传统 ORM 方法。

## 1. 研究动机

![无标题.jpg](http://img.xilyfe.top/img/20260426101841535.jpg)
1. SFT 存在的局限性：LLM 工具调用的主流训练范式是 SFT——但 SFT 在泛化性、探索能力上存在根本局限。SFT 会模仿 deep-thinking 轨迹中的表面 pattern（如 "but wait"），而非真正学会推理何时调用工具，容易出现 **过度思考和工具滥用**。
2. 粗粒度的奖励信号：Agentic RL 的代表 Search-R1 等论文采用的都是 EM 这样的 orm 奖励，但是在 Tool-Integrated Reasoning（后文统称 TIR）的场景下，多轮对话中每步需调用多个不同工具，每个工具有不同参数结构。单一 binary reward 无法捕捉这种细粒度正确性。

## 2. 创新点

ToolRL 的创新点主要是它设计了**一套通用的奖励范式**。

我们先回顾一下 Search-R1 等 orm 范式的训练过程：
1. 将 search 工具描述注入 system prompt，要求模型用 \<think> / \<search> / \<answer> 特殊 token 组织输出。
2. 每次推理都会进行判断，如果需要进行搜索则将搜索结果插入对话历史，然后重复上述过程继续让模型生成。如果不调用搜索或输出 \<eos> 或者达到长度上限，则结束 rollout。
3. 用 PPO/GRPO 进行优势估计，然后参数更新。

ToolRL 提出的思考是，当场景涉及到 TIR 的时候，对于多个不同的工具调用 outcome-based 的奖励是不是太粗粒度了。假如某几个工具调用正确某几个调用错误，但是结果正确了，模型会不会强化这些错误 tool call 的选择？ToolRL 系统性地分析了工具使用任务中的**奖励设计维度**，包括：
- **尺度**：不同奖励信号之间如何平衡？
- **粒度**：如何拆解奖励信号粒度而非仅是二值选择？
- **动态性**：训练过程中，奖励信号应否随时间变化？

ToolRL 的研究表明，粗粒度、静态、或者仅以最终答案匹配为目标的奖励往往无法最有效地指导模型学习工具推理能力。

### 2.1 多粒度的奖励函数

![image.png](http://img.xilyfe.top/img/20260426111624640.png)

1. 格式奖励：如果模型输出结果满足 `<think></think><tool_call></tool_call><obs></obs>` 就给予奖励，和 Search-R1 的 format reward 没什么区别。

$$
R_{\text{format}} =
\begin{cases}
1, & \text{if all required fields appear and are in the correct order} \\
0, & \text{otherwise}
\end{cases}
$$



2. 正确性奖励：ToolRL 的正确性奖励不是 exact match，而是判断每次 tool call 是不是和预期相同，工具名称、参数名称、参数内容是否匹配。
	1. 工具名称匹配

$$
r_{\text{name}} = \frac{|N_G \cap N_P|}{|N_G \cup N_P|} \in [0,1]
$$

	2. 参数名称匹配

$$
r_{\text{param}} = \sum_{G_j \in G} \frac{|\mathrm{keys}(P_G) \cap \mathrm{keys}(P_P)|}{|\mathrm{keys}(P_G) \cup \mathrm{keys}(P_P)|} \in [0, |G|]
$$

	3. 参数内容匹配

$$
r_{\text{value}} = \sum_{G_j \in G} \sum_{k \in \mathrm{keys}(G_j)} \mathbb{1}[P_G[k] = P_P[k]] \in \left[0, \sum_{G_j \in G} |\mathrm{keys}(G_j)| \right]
$$

最后把三个 correctness reward 合并起来，归一化到 \[-3, 3] 的范围内，加上之前的格式奖励就是 \[3, 4]。

{{< admonition type=question title="一点点疑惑">}} 

我们需要区别一下 ToolRL 和以往 outcome-based 的强化学习。ToolRL 用**事先离线收集好的完整轨迹**，每条 sample 的结构类似：

```
<input></input>
<output>
	<think>
	</think>
	<tool_call>
	</tool_call>
</output>
```

训练过程中，我们把工具信息放在 system prompt 中，然后把 input prompt 和 system prompt 一起喂给模型输出 output，也就是 thinking 部分和 tool call 部分。然后我们对 output 进行格式奖励，对 tool call 部分进行正确性奖励。

{{< /admonition >}}

### 2.2 KL 损失

ToolRL 为了鼓励模型更自由地调整其行为以适应自定义的响应格式和结构化的奖励信号，把损失函数里面 KL Loss 那部分去掉了：

$$
J_{\mathrm{GRPO}}(\theta)
= \mathbb{E}_{Q \sim \mathcal{D},\, s_i \sim \pi_\theta}
\left[
\min \left(
\frac{\pi_\theta(s_i \mid Q)}{\pi_{\theta_{\text{old}}}(s_i \mid Q)} A_i(s_i \mid Q),\;
\operatorname{clip}\!\left(
\frac{\pi_\theta(s_i \mid Q)}{\pi_{\theta_{\text{old}}}(s_i \mid Q)},
1 - \epsilon,\,
1 + \epsilon
\right) A_i(s_i \mid Q)
\right)
\right]
$$

## 3. 实验结果

1. 长度奖励（鼓励更长推理轨迹）可能降低性能，尤其是小模型。
2. 动态调整奖励尺度（从格式优先到正确性优先）能提升训练稳定性。
3. 细粒度奖励分解（工具名、参数名、参数值匹配）比粗粒度设计（仅判断整体是否匹配）效果更好。
4. GRPO 相较于 PPO 训练收敛更快，但是后期不稳定可能出现 collapse 问题，但是两者最终的效果相差不多。
