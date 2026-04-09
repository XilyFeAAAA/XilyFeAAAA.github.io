---
title: "Q&A"
date: 2026-04-08T00:03:59+08:00
featuredImage: ""
authors:
  - Xilyfe
series: []
tags: []
lastmod: 2026-04-08T12:24:57+08:00
---

## 1. 强化学习

### 1.1 GRPO

#### 1.1.1 怎么判断 GRPO 训练是否收敛
#### 1.1.2 GRPO 不收敛怎么办

1. prompt 有没有设计好，比如选择题需要明确说明 **输出结果为A/B**
2. 注意 reward hacking，现象是越训练 reward 越少
3. 组内优势解决于零，num_generations 太小，奖励方差太小

#### 1.1.3 Qwen3 不输出推理过程，format reward 始终为零



## 2. 监督微调

1. 领域/通用数据集配比
2. 学习率怎么选择