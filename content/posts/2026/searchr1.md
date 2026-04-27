---
title: Search-R1
date: 2026-04-22T16:26:42+08:00
featuredImage: http://img.xilyfe.top/img/20260426100146239.png
authors:
  - Xilyfe
series:
  - 论文阅读
tags: []
lastmod: 2026-04-27T12:43:44+08:00
---

> **论文**：Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning  
> **标签**：`Reinforcement Learning` `RAG` `Search` `Agentic RL` `GRPO`  
> **TL;DR**：Search-R1 将 DeepSeek-R1 的强化学习范式迁移到搜索增强推理场景，通过 online rollout + ORM 奖励，让模型自主学会何时搜索、搜索什么、如何整合结果——而非依赖人工标注的搜索轨迹。

## 1. 背景与动机

### 1.1 RAG 的固有缺陷

传统 RAG（Retrieval-Augmented Generation）的流程是：**先检索，后生成**。这种解耦设计存在几个系统性问题：

- **检索时机固定**：不管问题难易，总是在第一步就检索，无法根据推理进展动态决定是否需要搜索；
- **查询质量依赖用户**：直接用用户的原始问题去检索，但有时候中间推理产生的子问题才是更好的检索 query；
- **单次检索上限**：复杂多跳问题需要多轮迭代检索，传统 RAG 一次性完成，信息往往不足。

### 1.2 为什么用 RL 而非 SFT

SFT 方案（如 ReAct、ToolFormer）需要大量人工标注的搜索轨迹，标注成本高，且模型学到的是对轨迹的模仿而非搜索策略的本质。

RL 的优势在于：**模型可以通过自我探索发现更优的搜索策略**，不依赖人工设计的搜索时机和查询措辞，只需最终答案正确即可获得奖励。

---

## 2. 方法

### 2.1 整体框架

Search-R1 的训练框架基于 DeepSeek-R1，核心是一个 **online RL + 真实搜索引擎交互** 的循环：

```
问题输入
   ↓
模型生成 <think> ... 决定是否搜索 ...
   ↓ (若决定搜索)
生成 <search>查询词</search>
   ↓
真实搜索引擎返回结果，插入 <information>...</information>
   ↓
模型继续生成，可多次循环
   ↓ (最终输出答案)
生成 <answer>最终答案</answer>
   ↓
与 ground-truth 对比，计算奖励
```

### 2.2 特殊 Token 设计

Search-R1 通过特殊 token 划定推理和工具调用的边界：

|Token|作用|
|---|---|
|`<think>` / `</think>`|包裹内部推理过程（对用户不可见）|
|`<search>` / `</search>`|包裹搜索查询词|
|`<information>` / `</information>`|系统插入的搜索结果（非模型生成）|
|`<answer>` / `</answer>`|最终回答|

模型的训练目标是学会**在 `<think>` 中决策，在 `<search>` 中提问，在 `<answer>` 中作答**。`<information>` 块由系统填充，不参与 loss 计算。

### 2.3 Online Rollout 机制

这是 Search-R1 区别于离线方法的关键设计。训练时的完整流程如下：

1. **生成阶段**：模型根据问题开始生成，遇到 `</search>` token 时暂停；
2. **检索阶段**：系统提取查询词，调用真实搜索 API（论文使用的是内部搜索引擎），获取 Top-K 文档摘要；
3. **注入阶段**：将搜索结果格式化为 `<information>` 块，追加到上下文中；
4. **继续生成**：模型继续推理，可以再次触发搜索，直到生成 `</answer>` 或达到最大步数。

这种 **interleaved 推理-检索** 的方式使模型能够基于已有信息动态调整搜索策略。

### 2.4 奖励函数设计

Search-R1 使用极简的 ORM 奖励，完全基于最终答案质量：

#### 结果奖励（Outcome Reward）

$$ R_{\text{outcome}} = \begin{cases} 1, & \text{if } \text{EM}(\hat{a}, a^*) = 1 \ 0, & \text{otherwise} \end{cases} $$

其中 $\hat{a}$ 为模型生成的答案，$a^*$ 为 ground-truth 答案，EM 表示 Exact Match（先做归一化处理：小写、去标点、去冠词）。

#### 格式奖励（Format Reward）

$$ R_{\text{format}} = \begin{cases} 1, & \text{if output contains valid } \langle\text{answer}\rangle \text{ tags} \ 0, & \text{otherwise} \end{cases} $$

总奖励为两者之和：$R = R_{\text{outcome}} + R_{\text{format}}$

> **极简奖励的合理性**：Search-R1 认为，在单一工具（搜索）场景下，只要最终答案正确，中间的搜索行为就是有效的。相比复杂的过程奖励，简单的结果奖励实现成本低，且不依赖对中间步骤的人工标注。

### 2.5 训练算法：GRPO

Search-R1 采用 **GRPO（Group Relative Policy Optimization）** 而非 PPO，避免了 value function 估计带来的额外参数和计算成本。

GRPO 的核心思路：对同一问题采样 $G$ 个不同的 rollout，用组内平均奖励作为 baseline，通过相对排名估计优势函数：

$$ A_i = \frac{R_i - \text{mean}({R_j}_{j=1}^G)}{\text{std}({R_j}_{j=1}^G)} $$

策略更新目标：

$$ J_{\mathrm{GRPO}}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta) A_i,; \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta , D_{\mathrm{KL}}(\pi_\theta | \pi_{\text{ref}}) \right] $$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ 为重要性采样比率。

**GRPO 相比 PPO 的优势**：

- 无需单独的 critic 网络，显存占用更小；
- 对于 LLM 的长序列生成，value function 估计本身就很困难，GRPO 绕开了这个问题；
- 实现更简洁，训练更稳定。

---

## 3. 实验设置

### 3.1 数据集

Search-R1 在以下知识密集型问答数据集上训练和评测：

|数据集|类型|特点|
|---|---|---|
|**HotpotQA**|多跳推理|需要跨文档推理，天然适合多轮搜索|
|**2WikiMultihopQA**|多跳推理|实体关系链较长|
|**MuSiQue**|多跳推理|干扰项多，难度较高|
|**NQ（Natural Questions）**|开放域 QA|单跳，考察基础检索能力|
|**TriviaQA**|知识问答|事实性问题，搜索价值高|
|**PopQA**|长尾知识|测试对小众知识的检索能力|

### 3.2 基线对比

|方法|搜索机制|训练方式|
|---|---|---|
|Naive RAG|单次检索|无需训练|
|ReAct|多轮交互|SFT|
|IRCoT|迭代检索|无需训练（prompting）|
|**Search-R1**|多轮自适应|RL（GRPO）|

### 3.3 模型

论文在 Qwen2.5-7B 和 Qwen2.5-3B 上均进行了实验，对比 instruct 版本和经过 RL 训练的版本。

---

## 4. 实验结果与分析

### 4.1 主要结论

**Search-R1 在所有测试集上均显著超越 SFT 基线**，尤其在多跳推理数据集（HotpotQA、MuSiQue）上提升幅度最大，验证了 RL 训练在复杂多步推理场景中的独特优势。

与 Naive RAG 相比，Search-R1 的搜索轮次更少但答案质量更高，说明模型学会了**有选择性地搜索**，而非每步都机械触发检索。

### 4.2 涌现出的搜索行为

RL 训练过程中，模型自发涌现出一些有趣的搜索策略，这些策略在 SFT 训练中并不常见：

- **查询分解**：对于复杂问题，模型会先将其拆解成多个子问题，逐个搜索；
- **查询改写**：当第一次搜索结果不理想时，模型会自动调整措辞重新搜索；
- **选择性搜索**：对于模型自身有把握的知识点，会跳过搜索直接回答，减少冗余调用。

### 4.3 训练动态

随着 RL 训练进行，可以观察到：

- 平均搜索轮次先增后减，最终趋于稳定（模型逐渐学会"什么时候不需要搜索"）；
- 推理链长度增加，说明模型倾向于更充分地利用检索到的信息；
- Format reward 快速收敛到满分，outcome reward 持续缓慢提升。

## 5. 局限与未来方向

### 当前局限

- **ORM 的稀疏性**：只有最终答案正确才有奖励，对于需要很多步才能得到答案的问题，梯度信号极其稀疏，训练效率低；
- **搜索引擎依赖**：训练时需要实时调用搜索 API，延迟和成本较高，且搜索结果的质量会直接影响模型学到的策略；
- **封闭式评测**：Exact Match 奖励对于开放式问答不适用，泛化到更复杂任务时需要重新设计奖励；
- **工具单一**：整个框架仅支持搜索，扩展到多工具场景（如 ToolRL 的目标）需要大量额外设计。

### 未来方向

- **过程奖励模型（PRM）**：对搜索行为的中间步骤给予奖励，缓解奖励稀疏问题；
- **多工具协同**：将 Search-R1 的 online 交互机制与 ToolRL 的细粒度奖励结合，构建更通用的 Agentic RL 框架；
- **奖励模型蒸馏**：训练专门的 reward model 来评估搜索策略质量，摆脱对 ground-truth 标注的依赖。

---

## 6. 总结

Search-R1 的最大价值在于**证明了 RL 可以让模型自主习得搜索策略**，不需要人工设计搜索时机和查询模板。它建立了 Agentic RL 的基础范式：online rollout + 结果奖励 + 工具交互。

但 Search-R1 也留下了清晰的局限：奖励信号太粗、工具场景太单一。这些局限正是后续工作（如 ToolRL）的出发点。从 Search-R1 到 ToolRL，是从"能用 RL 训搜索"到"如何设计通用工具调用 RL 奖励"的自然演进。

---

## 参考文献

- Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- ReAct: Synergizing Reasoning and Acting in Language Models
- IRCoT: Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions
- GRPO: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models