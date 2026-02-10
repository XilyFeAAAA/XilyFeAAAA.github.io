---
title: MiniMind 学习指北(四)：评估
date: 2026-01-22T13:47:19+08:00
featuredImage: http://img.xilyfe.top/img/20260122134824760.png
authors:
  - Xilyfe
series:
  - minimind
tags:
  - 大模型
  - 深度学习
lastmod: 2026-02-10T02:50:00+08:00
---
>这一章我们需要设计一个脚本来验证大模型的对话能力

首先我们预训练让模型学习的是说话的能力，或者说词语接龙的能力，给他一个 prompt 它可以接着说下去，所以我们先测试一下接龙任务：
