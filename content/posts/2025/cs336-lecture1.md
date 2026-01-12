---
title: "Lecture 1: Tokenization"
date: '2025-12-17T13:05:11+08:00'
authors: [Xilyfe]
series: ["CS336"]
tags: ["大模型"]
--- 
 

!!! abstract 
    本节主要讲解常见的几种 Tokenizer 以及它们的优劣，最后介绍了 BPE 的实现思路。
    
## Tokenizer 主要类型

### 1. Character-based

- 优点：不会出现 Out of vocabulary 的情况
- 缺点：词表大小爆炸，需要涵盖每一种语言的每一个字符

### 2. Byte-based

将字符串转码为 UTF-8 bytes，例如：'你' → 'E4 BD A0'，每个 byte 能表示 0-255 就是一个 token。

- 优点：不会出现 OOV、支持所有语言、简单
- 缺点：
	- 压缩比低：一个中文字符需要 3 个 token 表示
	- 序列长度非常长：Transformer 时间复杂度是 O(n²)

### 3. Word-based

- 缺点：词表大小爆炸，容易出现 \[UNK]

### 4. BPE

Byte Pair Encoding 继承了 Byted-base 的思路

1. 首先还是用 UTF-8 编码将字符转为 bytes
2. 统计所有 token pair 出现的次数
3. 对出现最多次的 token 进行合并，直到词表达到目标大小

简单实现：

```python
string = "Hello World"
num_merges = 3
indices = [116, 104, 101, 32, 99, 97, 116, 32, 105, 110, ...]
pair_counts = {
	[116, 104]: 2,
	[104, 110]: 2,
	...
}
# merge [116, 104] into 256
new_indices = [256, 101, 32, 99, 97, 116, 32, 105, 110, ...]
# do again
```

这个实现的问题在于：

1. BPE 可能学习出很多奇怪的组合，例如 'lr'，'ok!'，'ok.' 这种组合，有些意思相仅，有些完全没有语义