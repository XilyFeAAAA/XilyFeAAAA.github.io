---
title: CS224N Assignment 2
date: 2025-11-30T11:24:11+08:00
authors:
  - Xilyfe
series:
  - CS224N
tags:
  - 深度学习
featuredImage: http://img.xilyfe.top/img/20260130144509787.png
lastmod: 2026-01-30T02:45:51+08:00
---

## 整体流程

1. 数据预处理：会得到三个数据集以及一个 Parser，在依存分析实验中 Parser 统筹管理转移系统中的全部资源，包括 Stack, Buffer, Arcs 还有一个深度学习的 model。
2. 训练过程：train 函数会进行 n 个 batch 的训练，保存 UAS 最大的一个模型。
3. 使用刚刚保存的最好模型对 test 数据集进行处理

```python
if __name__ == "__main__":
    debug = args.debug

    assert (torch.__version__.split(".") >= ["1", "0", "0"]), "Please install torch version >= 1.0.0"

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()
    model = ParserModel(embeddings)
    parser.model = model
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
```

## 数据预处理

训练数据集 be like:

```conll
1	In	_	ADP	IN	_	5	case	_	_
2	an	_	DET	DT	_	5	det	_	_
3	Oct.	_	PROPN	NNP	_	5	compound	_	_
4	19	_	NUM	CD	_	5	nummod	_	_
5	review	_	NOUN	NN	_	45	nmod	_	_
6	of	_	ADP	IN	_	9	case	_	_
7	``	_	PUNCT	``	_	9	punct	_	_
8	The	_	DET	DT	_	9	det	_	_
```

这个文件是一个 CoNLL-U（或类似 CoNLL 格式） 的依存句法分析标注文件，每一段对应一个句子，“In an Oct. 19 review of ‘The Misanthrope’ at Chicago’s Goodman Theatre …”

- read_conll 函数会负责读取这些文件，并且返回`list[{'word': [], 'pos': [], 'head': [], 'label': []}]`格式的数据，每个字典对应一句话
- 构建 Parser 核心类，里面封装了依存分析的整个系统。
- 读取预训练的词向量
- 构建初始化 [token_num, 50] 大小的词向量矩阵
- 将数据集中的向量变成 one-hot 编码
- create_instances() 从语料中生成“当前状态 → 正确动作”的训练样本

> 已经有预训练的词向量，为什么还要随机生成？ 因为预训练的词向量库不一定能囊括全部的词汇，所以需要初始化一个词向量矩阵，然后用预训练的替换

> 已经有词向量了，为什么还需要把 word 和 pos 等转为 one-hot 编码呢？这样不是丢失了信息吗？ TODO

## 神经网络

PyTorch 框架下，要实现自己的神经网络可以继承 torch.nn.Module 类，它考研自动把 nn.Parameter 和子模块中的参数收集起来，便于优化器访问 model.parameters()。

```python
class CustomizedNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

    def forward(self, x):
        return x
```

---
 
**embedding_lookup**

前面提到在数据预处理的时候，训练集的数据中 word,pos 等向量会转为 one-hot 编码，所以在进入神经网络进行训练时候，需要根据 idx 得到对应的向量表示。

在依存句法分析（Dependency Parsing）中，每一步的输入状态可以用一些“特征单词”来表示，比如：

- 栈顶的词（stack top）
- 缓冲区前几个词（buffer front）
- 它们的子节点（left/right children）

这些单词的索引被拼成一个固定长度的列表，比如：`w = [23, 14, 7, 65, 99, ..., 8]`, 模型输入的 train_set 维度为 [batch_size, n_features]。

```python
    def embedding_lookup(self, w):
        x = self.embeddings[w]
        x = x.view(x.size(0), -1)

        return x
```

这里用到了 PyTorch 中张量索引的技巧，如果用一个张量 a 当另一个张量 b 的索引，那么 a 中的每个元素 i 会被替换为 b 中第一个维度的第 i 个元素，例如：

```python
self.embeddings = torch.tensor([
    [1, 1, 1],   # 词 0
    [2, 2, 2],   # 词 1
    [3, 3, 3],   # 词 2
    [4, 4, 4],   # 词 3
])
w = torch.tensor([[0, 2], [1, 3]])
=>
self.embeddings[w] = 
[
    [[1, 1, 1], [3, 3, 3]],
    [[2, 2, 2], [4, 4, 4]]
]
```

可以计算得到，经过 lookup 查表操作之后，train_set 的维度从 [batch_size, n_features] 变成 [batch_size, n_features, embed_size]。

由于神经网络的输入必须是一维的 feature 向量，所以需要对 train_set 进行展平操作，从 [batch_size, n_features, embed_size] 降维到 [batch_size, n_features * embed_size]

---

**__init__**

```python
    def __init__(self, embeddings, n_features=36, hidden_size=200, n_classes=3,         dropout_prob=0.5):
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embeddings = nn.Parameter(torch.tensor(embeddings))

        # declare
        self.embed_to_hidden_weight = nn.Parameter(torch.empty(self.embed_size * self.n_features))
        self.embed_to_hidden_bias = nn.Parameter(torch.empty(hidden_size))
        
        # initialize
        nn.init.xavier_uniform_(self.embed_to_hidden_weight)
        nn.init.uniform_(self.embed_to_hidden_bias)
        
        # dropout layer
        self.dropout = nn.Dropout(p=self.dropout_prob)
        
        # declare
        self.hidden_to_logits_weight = nn.Parameter(torch.empty(self.hidden_size, self.n_classes))
        self.hidden_to_logits_bias = nn.Parameter(torch.empty(self.n_classes))
```

前馈层是线性计算，我们手动定义的 Weight 和 Bias 与 nn.Linear 有着相同作用：

$$
y=Wx+b
$$

输入 x 的维度为 [batch_size, self.embed_size * self.n_features], 所以 W 维度为 [self.embed_size * self.n_features, hidden_size], 乘积的维度为 [self.batch_size, hidden_size]， 所以 b 的维度是 [hidden_size,]，后续计算同理。

**为什么 b 的维度是 [hidden_size] 或者说 [1, hidden_size]**

首先我们来分析一下$y=xW$得到的矩阵，他的形状是 [batch_size, hidden_size]，这代表有 batch_size 行，每一行宽度是 hidden_size。所以如果我们想给函数加一个偏置项，应该是给每一行加上去，偏置向量的形状应该是 [1, hidden_size]。但是矩阵和向量是如何相加的呢？PyTorch（或 NumPy）的广播规则是：只要两个张量在末尾维度上能匹配，就可以自动扩展前面的维度，也就是说：广播时，b 会被自动扩展为 [1, hidden_size] → [batch_size, hidden_size]。所以在这个语境下，它行为上更像是一个“行向量”。

---

**forward**

```python
    def forward(self, w):
        logits = None
        x = self.embedding_lookup(w)
        logits = nn.ReLU(torch.matmul(x, self.embed_to_hidden_weight) + self.embed_to_hidden_bias)
        logits = self.dropout(logits)
        logits = nn.ReLU(torch.matmul(logits, self.hidden_to_logits_weight) + self.hidden_to_logits_bias)
        return logits

```

**为什么在 forward 前馈计算中是 x 乘以 W 呢？**

这主要是源于 PyTorch 和数学计算的差异：在数学中我们通常规定 x 是一个长度为 n 的列向量，但是在 PyTorch 中输入 x 几乎总是一批行向量。所以在 PyTorch 代码中，一般是$y=xW$或者$y=xW^T$。

> dropout 一定要放在两个线性变换之间吗？隐藏层的输出是模型学习到的特征表征。对这些表征做 dropout，迫使下一层的权重依赖更加广泛、鲁棒的特征组合，从而降低过拟合。 

## 训练

1. Adam Optimizer 需要模型全部神经元当参数
2. 损失函数采用交叉熵损失
3. 训练 n_epochs 个轮次，然后取最高分保存模型。

```python
def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    best_dev_uas = 0
    params = parser.model.parameters()
    optimizer = optim.Adam(params, lr=0.0001)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        print(f"Epoch {epoch} out of {n_epochs}")
        dev_uas = epoch_train(parser, train_data, lr)
        print(f"Epoch {epoch} scored {dev_uas} UAS")
        if dev_uas > best_dev_uas:
            torch.save(parser.model.state_dict(), output_path)
    print(f"train completed with best dev uas of {best_dev_uas}")
```

1. 在训练之前调用`model.train()`，dropout 屏蔽一部分神经元
2. 每次取 batch_size 个数据训练
3. 每次训练的套路都是比较固定了，梯度清零-预测-计算 loss-梯度回传-更新梯度

```python
def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    parser.model.train()
    n_minibatches = math.ceil(len(train_data) / batch_size)
    total_loss = 0
    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            # ---------- 一套连招 -----------
            optimizer.zero_grad()
            logits = parser.model(train_x)
            loss = loss_func(logits, train_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            prog.update(1)
    print(f"Average Train Loss: {total_loss / n_minibatchs}")

    parser.model.eval()
    dev_uas, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS
```

---

上面代码中最重要的部分就是利用 batch data 进行训练的部分，接下来详细分析一下。

1. 首先为什么需要`model.train()`

启用训练模式之后，神经网络会采用 Dropout 以一定概率（例如 p=0.5）随机“屏蔽”一部分神经元输出，这么做是为了让模型不要太依赖某些特征，防止过拟合。
$$
y_i=
\begin{cases}
0\\
\frac{x_i}{1-p}
\end{cases}
$$

2. 为什么需要`optimizer.zero_grad()`手动清空梯度？

PyTorch 中进行反向传播之后计算梯度并不是简单的赋值 (=) 而是选择了梯度累积 (+=)，这里用一个代码举例：

```python
"""
假设 y=w*x, 并且需要训练的 w 为 2
"""

w = torch.tensor([1.0], requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

def loss_func(pred_y, train_y):
    """
    损失函数为 loss = (y1 - y) ^ 2 = (w * x - y) ^ 2
    求导后为 ∂loss/∂w = 2(w * x - y) * x
    """
    return (pred_y - train_y) ** 2

train_x1, train_x2 = torch.tensor([1.0]), torch.tensor([2.0])
train_y1, train_y2 = torch.tensor([2.0]), torch.tensor([4.0])

pred_y1 = w * train_x1
loss1 = loss_func(pred_y1, train_y1)
loss1.backward()

"""
w=1, train_x1=1, pred_y1=1
∂loss/∂w=2*(1-2)*1=-2
所以 w 的梯度为-2
"""

print(f"第一次推导之后 w 的梯度为{w.grad}")

pred_y2 = w * train_x2
loss2 = loss_func(pred_y2, train_y2)
loss2.backward()

"""
w=1, train_x2=2, pred_y1=4
∂loss/∂w=2*(2-4)*2=-8
所以 w 的梯度为-8
"""
print(f"第二次推导之后 w 的梯度为{w.grad}")
```
代码运行后会得到 w 的梯度是-10 而不是-8，因为采用了梯度累积$(-2) + (-8) = -10$，这也就解释了为什么需要在每一次反向传播之前将梯度清零。

**为什么 PyTorch 采用梯度累积呢？**

1. 解决显存不足：梯度累积训练（Gradient Accumulation）

当模型或批次较大（如大模型、高分辨率图像）时，显存可能无法容纳一个完整的大批次数据（例如想使用 atch_size=32，但显存只支持 batch_size=8）。此时可以：

- 将大批次拆分为多个小批次（如 4 个 batch_size=8）；
- 每个小批次计算梯度后不更新参数，而是累积梯度（backward() 自动+=）；
- 累积 4 次后，用总梯度（等价于 batch_size=32 的梯度）更新一次参数。

这样既避免了显存溢出，又等价于使用大批次训练（保持梯度统计特性一致）。

2. 多损失函数场景：合并不同损失的梯度

```python
# 多任务损失
loss_cls = cross_entropy(pred_cls, label_cls)  # 分类损失
loss_reg = mse_loss(pred_reg, label_reg)      # 回归损失

# 分别反向传播，梯度自动累积（+=）
loss_cls.backward(retain_graph=True)  # retain_graph 保留计算图，供下一次 backward
loss_reg.backward()

# 用总梯度（cls 梯度 + reg 梯度）更新参数
optimizer.step()
optimizer.zero_grad()
```
如果 backward() 是 “赋值”，则第二个损失的梯度会覆盖第一个，导致只能用单一损失的梯度更新，无法实现多损失的联合优化。