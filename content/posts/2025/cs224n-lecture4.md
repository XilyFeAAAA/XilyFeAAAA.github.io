---
title: "Lecture 4: Dependency Parsing"
date: '2025-11-21T11:24:11+08:00'
authors: [Xilyfe]
series: ["CS224N"]
tags: ["深度学习"]
--- 

## 依存语法

传统的传统句法（如短语结构语法）认为句子是由短语组成的层级结构。以`I saw a cat under the table`为例子

- table是名词
- the table 组成名词短语
- under 是介词
- under the table 组成介词短语
- a cat under the table 又组成一个名词短语.....

而**依存语法**的出发点不同，它提出了一个非常重要的假设：
> 句子的结构由词与词之间的依存关系（dependency）决定。

以`I like you`举例:
```
like -> I    (nsubj)
like -> you  (obj)
```

- 一般动词都是一个句子的核心
- I 依赖于 like，依赖关系是主语(nsubj)
- you 依赖于 like， 依赖关系是宾语(obj)

## 依存句法分析

> Dependency Parsing = 给句子中的每个词，确定它依赖于哪个词，也就是预测词之间的依存关系。

当我们预测句子中词与词的依存关系时，模型应该考虑哪些信息来源?

1. Bilexical affinities: 两个具体词之间的亲和度
2. Dependency distance: 依存关系之间的距离
3. Intervening material: 依存关系通常不会跨越动词或标点等强结构边界
3. Valency of heads: 一个中心词一般有规定数量和方向的依存词，例如happy通常只修饰一个名词
---
为了保证结果合法，依存分析通常要满足一些约束：

1. Only one word is dependent of ROOT	整个句子只有一个核心动词（一个 root）
2. No cycles (A→B, B→A)	不允许形成环，否则依存树不成立
3. Dependencies form a tree	所有依存边连通、无环，每个节点只有一个入边
---
箭头是否可以交叉？
- Projective tree（投射句法）：所有依存边都不交叉 → 典型的英语句子结构
- Non-projective tree（非投射句法）：有交叉依存 → 常见于自由语序语言（如德语、俄语、中文）

## 依存分析方法

- 动态规划
- 图算法
- 约束满足法
- 转移系统：维护一个 Stack + Buffer。由于变成一个线性的计算过程，所以时间复杂度也是线性的，相较于其他两种方法低很多。


## 依存关系和深度学习

转移系统（Transition-based Dependency Parsing） 是依存句法分析中最直观、工程上最常用的一类算法。它维护了两个数据结构：Stack和Buffer，并且只进行三个操作：Shift，Left-Arc，Right-Arc。

以句子"I eat apples"为例：
```
1.Initial
Stack = [ROOT]
Buffer = [I, eat, apples]
Arcs = []

2.Shift
Stack = [ROOT, I]
Buffer = [eat, apples]
Arcs = []

3.Shift
Stack = [ROOT, I, eat]
Buffer = [apples]
Arcs = []

4.Left-Arc
Stack = [ROOT, eat]
Buffer = [apples]
Arcs = [(eat, I)]

5.Shift
Stack = [ROOT, eat, apples]
Buffer = []
Arcs = [(eat, I)]

6.Right-Arc
Stack = [ROOT, eat]
Buffer = []
Arcs = [(eat, I), (eat, apples)]

7.Right-Arc
Stack = []
Buffer = []
Arcs = [(eat, I), (eat, apples), (ROOT, eat)]
```

```
依存树:
ROOT
 └── eat
      ├── I
      └── apples
```

从中可以看出，对于深度学习模型它需要做的就是判断每次需要进行哪个操作，三选一。

## 评价指标

在深度学习中训练模型需要有一个损失函数，根据梯度进行优化，也就是说我们需要一个指标来评价预测的依存关系的好与坏。

```
Gold:                  Parsed:
1  2 She     nsubj     1  2 She     nsubj
2  0 saw     root      2  0 saw     root
3  5 the     det       3  4 the     det
4  5 video   nn        4  5 video   nsubj
5  2 lecture obj       5  2 lecture ccomp
```
1. UAS
$$
UAS=\frac{正确预测的head词数}{总词数}=\frac{4}{5}=80\%
$$
2. LAS
$$
LAS=\frac{head和label都预测正式的词数}{总词数}=\frac{2}{5}=40\%
$$


## 神经网络依存分析

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/dependency_parse.png"/>
</div>

在用神经网络对依存分析的每一步操作进行预测的时候，需要当前状态的一个输入值，这个输入的向量应该怎么得到呢？

很显然我们目前有Stack和Buffer还有Arcs三个状态，但是如果把全部的信息一股脑的输入给神经网络就冗余了，它只需要其中的几个关键信息：

1. Stack栈顶和次栈顶的元素: s1, s2
2. Buffer最前面马上要被处理的词: b1
3. 在Arcs中s1, s2, b1的左右孩子

> 需要Arcs中元素对的原因是：假如一个词已经拥有了主语就不可能再有一个主语了

---

知道了模型输入需要的全部信息，接下来就要考虑怎么把它传输给模型。

对于每个被选中的词（如 s1、s2、b1 及其子节点），我们可以取它的词向量（word embedding），同时拼接其他特征的向量，如：
- 词性（POS）嵌入
- 依存关系标签嵌入（对已建立的弧）
- 是否是根节点、是否已有父节点等布尔特征

最终，这些向量会被**拼接**成一个大的状态表示向量：
$$
x=[emb(s1​),emb(s2​),emb(b1​),emb(children of s1​),...]
$$