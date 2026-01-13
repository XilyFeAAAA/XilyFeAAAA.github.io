---
title: "LSTM"
date: '2025-12-19T16:04:11+08:00'
featuredImage: "https://raw.githubusercontent.com/XilyFeAAAA/ImgRepository/main/img/20260113120111595.png"
authors: [Xilyfe]
series: ["DeepLearning"]
tags: ["深度学习"]
--- 

## LSTM 基础概念

LSTM 在 RNN 的基础上很好的解决了长距离详细传递的问题，它引入了 Cell State 和三个门 Forget Gate, Input Gate 和 Output Gate 来传输记忆和决定哪些记忆是需要的，哪些不需要。

- 遗忘门：根据$h^{t-1}$和$x^t$判断 Cell State 哪一些需要遗忘
- 输入门：根据$h^{t-1}$和$x^t$判断需要向 Cell State 传入哪些当前信息
- 输出门：根据$h^{t-1}$和$x^t$判断需要从 Cell State 中输出哪些信息

$$
f^t = \sigma(h^{t-1} @ W_{hf} + x_t @ W_{xf} + b_f) \\
i^t = \sigma(h^{t-1} @ W_{hi} + x_t @ W_{xi} + b_i) \\
o^t = \sigma(h^{t-1} @ W_{ho} + x_t @ W_{xo} + b_o) \\
c^{\~t} = \tanh(h^{t-1} @ W_{hc} + x_t @ W_{xc} + b_c) \\
c^t = f^t * c^{t-1} + i^t * c^{\~t} \\
h^t = o^t * \tanh(c^t)
$$


sigmoid 激活函数会将计算结果隐射到 0-1 的区间，然后与 $c^{t-1}$相乘。值越接近于 1，历史记忆就保留；相反值趋于 0，历史记忆就遗忘。

**为什么 LSTM 相对于 RNN 能够记忆更长的记忆？**

我们回顾一下 RNN 的公式：

$$
h_t=tanh(Wx \cdot x_t + W_h \cdot h_{t-1} + b_h)
$$

由于参数矩阵是固定的，所以进行反向传播时候，梯度要么会非常大要么会非常小。

但是对于 LSTM，它的三个门控机制可以选择每次保留 or 遗忘记忆，使得历史记忆可以长期保存。举一个极端的例子，遗忘门总是为 1，输入门总是为 0，那么历史记忆就能一直在 Cell State 上流通。

实际上，LSTM 不光是解决了长距离依赖的问题，它的各种门，使得模型的学习潜力大大提升，各种门的开闭的组合，让模型可以学习出自然语言中各种复杂的关系。比如遗忘门的使用，可以让模型学习出什么时候该把历史的信息给忘掉，这样就可以让模型在特点的时候排除干扰。


## 基于 LSTM 的 IMDB 文本情感分类项目

### 词嵌入

数据集中基于了 pos 和 neg 的训练集和测试集，每个 txt 文件包含 2w 行的电影评论。我的想法是将评论的句子通过空格进行分割，得到一个个 token，之后可以通过 token2id 的映射将 list[str] 变成 list[int]，这就是输入向量 x 了。

token2id 我通过继承 dict 实现了一个 Vocal 类，更方便我加入填充 token&lt;PAD&gt; 和未知字符\<UNK>：

```python
class Vocal(dict):
    
    def __init__(
        self, 
        train_texts: list,
        test_texts: list,
        unk_token: str = "<UNK>",
        pad_token: str = "&lt;PAD&gt;"
    ):
        super().__init__()
        self.unk_token = unk_token
        self.pad_token = pad_token
        self[self.unk_token] = 0
        self[self.pad_token] = 1
        for sentense in train_texts:
            for token in sentense:
                if token not in self: 
                    self[token] = len(self)
        
        for sentense in test_texts:
            for token in sentense:
                if token not in self: 
                    self[token] = len(self)
    
    def __getitem__(self, key: str) -> int:
        if key not in self:
            key = self.unk_token
        return dict.__getitem__(self, key)
```

x 输入到模型之后通过 lookup 将一个个序号变成 token 对应的词向量：
```python
class Embedding:
    
    def __init__(
        self,
        device: Literal["cuda", "cpu"],
        embed_filepath: str,
        embed_size: int,
        token2id: dict
    ) -> None:
        self.token2id = token2id
        token2vec = {}
        
        if not isinstance(embed_filepath, Path):
            self.embed_fp = Path(embed_filepath)
            
        assert self.embed_fp.exists(), "embed file not found"

        for line in open(self.embed_fp, "r").readlines():
            sp = line.strip().split()
            token2vec[sp[0]] = torch.tensor([float(x) for x in sp[1:]], dtype=torch.float32)

        assert len(sp[1:]) == embed_size, f"预训练向量维度与{embed_size}不符"
        
        self.embeddings = torch.normal(
            mean=0.0, std=0.9, size=(len(self.token2id), embed_size), dtype=torch.float32
        ).to(device)
        
        for token, idx in self.token2id.items():
            if token in token2vec:
                self.embeddings[idx] = token2vec[token].to(device)
            elif token.lower() in token2vec:
                self.embeddings[idx] = token2vec[token.lower()].to(device)
        
    
    def lookup(self, w: np.array) -> np.array:
        # w=[batch_size, seq_len]  o=[batch_size, seq_len, embed_size]
        return self.embeddings[w]
```

词向量我用的是 CS224N Assignment2 里面提供的词向量字典，为了防止词向量不能完全覆盖 imdb 数据集的全部 token，我初始化了一个 len(token) 大小的词向量矩阵，对于词向量字典有的 token，我就直接替换，对于没出现过的 token 的词向量，就直接训练。传入 x 之后应该会输出一个 [batch_size, seq_len, embed_size] 的张量。
### 数据集处理

数据集我使用 torch 自带的 DataSet 和 DataLoader，它提供了 batch 和 shuffle 等功能，我只需要重写__len__和__getitem__两个方法：

```python
class RemarkDataSet(Dataset):
    
    def __init__(
        self,
        texts: list[list[str]],
        labels: list[int],
        token2id: dict
    ) -> None:
        self.texts = torch.tensor([[token2id[token] for token in text] for text in texts])
        self.labels = torch.tensor(labels).float()
        
    
    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        return self.texts[index], self.labels[index]

    
    def __len__(self):
        return len(self.texts)
```
### LSTM 模型

模型的视线难度不大，主要是初始化了三个门控相关的参数，然后实现了前向传播的计算：

```python
class LSTM(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        seq_len: int,
        embed_size: int,
        output_size: int,
        device: Literal["cuda", "cpu"],
        embeddings: Embedding
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.device = device
        self.embeddings = embeddings
    
        # ===== Forget Gate =====
        self.Wxf = nn.Parameter(torch.empty(self.embed_size, self.hidden_size, device=self.device))
        nn.init.xavier_uniform_(self.Wxf)
        
        self.Whf = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size, device=self.device))
        nn.init.xavier_uniform_(self.Whf)
        

        self.bf = nn.Parameter(torch.empty(self.hidden_size, device=self.device))
        nn.init.uniform_(self.bf)
        
        # ===== Input Gate =====
        self.Wxi = nn.Parameter(torch.empty(self.embed_size, self.hidden_size, device=self.device))
        nn.init.xavier_uniform_(self.Wxi)
        
        self.Whi = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size, device=self.device))
        nn.init.xavier_uniform_(self.Whi)
        
        self.bi = nn.Parameter(torch.empty(self.hidden_size, device=self.device))
        nn.init.uniform_(self.bi)
        
        # ===== Output Gate =====
        self.Wxo = nn.Parameter(torch.empty(self.embed_size, self.hidden_size, device=self.device))
        nn.init.xavier_uniform_(self.Wxo)
        
        self.Who = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size, device=self.device))
        nn.init.xavier_uniform_(self.Who)
        
        self.bo = nn.Parameter(torch.empty(self.hidden_size, device=self.device))
        nn.init.uniform_(self.bo)
        
        # ===== Candidate Memory Cell =====
        self.Wxc = nn.Parameter(torch.empty(self.embed_size, self.hidden_size, device=self.device))
        nn.init.xavier_uniform_(self.Wxc)
        
        self.Whc = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size, device=self.device))
        nn.init.xavier_uniform_(self.Whc)
        
        self.bc = nn.Parameter(torch.empty(self.hidden_size, device=self.device))
        nn.init.uniform_(self.bc)
        
        self.fc = nn.Parameter(torch.empty(self.hidden_size, output_size, device=self.device))
        nn.init.xavier_uniform_(self.fc)
        self.sigmoid = nn.Sigmoid()
        
        self.hidden_dropout = nn.Dropout(p=0.3) 
        
    def forward(self, x):
        # x = [batch_size, seq_len, embed_size]
        x = self.embeddings.lookup(x)
        
        batch_size, seq_len, _ = x.shape
        ht = torch.zeros(batch_size, self.hidden_size, device=self.device)
        ct = torch.zeros(batch_size, self.hidden_size, device=self.device)
        for i in range(seq_len):
            # batch_x = [batch_size, embed_size]
            xt = x[:, i, :]
            ht_drop = self.hidden_dropout(ht)
    
            ft = torch.sigmoid(torch.matmul(ht_drop, self.Whf) + torch.matmul(xt, self.Wxf) + self.bf)
            it = torch.sigmoid(torch.matmul(ht_drop, self.Whi) + torch.matmul(xt, self.Wxi) + self.bi)
            ot = torch.sigmoid(torch.matmul(ht_drop, self.Who) + torch.matmul(xt, self.Wxo) + self.bo)
            c_t = torch.tanh(torch.matmul(ht_drop, self.Whc) + torch.matmul(xt, self.Wxc) + self.bc)
            
            ct = ft * ct + it * c_t
            ht = ot * torch.tanh(ct)
        
        return torch.matmul(ht, self.fc).squeeze(1)
```

### 训练&测试

IMDB 情感分析时一个二分类问题，所以损失函数用的是 BCEWithLogitsLoss 而不是交叉损失。

```python
class Classifier:
    
    def __init__(
        self,
        device: Literal["cuda", "cpu"],
        epochs: int,
        learning_rate: int,
        train_dataset: Dataset,
        test_dataset: Dataset,
        save_filepath: str | Path,
        embeddings: Embedding
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.lr = learning_rate
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.save_filepath = save_filepath
        self.embeddings = embeddings
    
    def train(self):
        model = LSTM(
            hidden_size=128,
            seq_len=250,
            embed_size=50,
            output_size=1,
            device=self.device,
            embeddings=self.embeddings
        ).to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        
        train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        
        best_loss = float("inf")
        model.train()
        for epoch in tqdm(range(self.epochs)):
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
                x, y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                prediction = model(x)
                loss = criterion(prediction, y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.cpu().item()
                pred_label = (torch.sigmoid(prediction) > 0.5).long()
                epoch_acc += (pred_label == y).sum().item()

            epoch_loss /= len(train_loader)
            epoch_acc /= len(train_loader.dataset)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), self.save_filepath)
            
            print(f"第{epoch+1}/{self.epochs}轮训练，acc={epoch_acc:.4f}, loss={epoch_loss:.4f}")

    def test(self):
        model = LSTM(
            hidden_size=128,
            seq_len=250,
            embed_size=50,
            output_size=1,
            device=self.device,
            embeddings=self.embeddings
        ).to(self.device)
        model.load_state_dict(torch.load(self.save_filepath))
        
        model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=True)
        
        total_acc = 0.0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(test_loader):
                x, y = batch_x.to(self.device), batch_y.to(self.device)
                pred = model(x)
                pred_label = (torch.sigmoid(pred) > 0.5).long()
                total_acc += (pred_label == y.long()).sum().item()
        print(f"测试准确率：{total_acc / len(test_loader.dataset):.4f}")
```

### 试验收获

#### numpy 和 torch 别混用

训练模型继承的是 torch.nn.Module，前馈计算时候基本用的都是 PyTorch 的张量，所以最好声明的矩阵啥的全用 tensor，避免后面报错。

#### 二分类问题的损失函数

之前 CS224N 的 Assignment2 和 RNN 实验预测字母都是一个多分类问题，用的是 CrossEntropy 这个损失函数。这次情感判断是一个二分类问题，GPT 告诉我需要用 BCEWithLogitsLoss 这个损失函数。

最初我的代码 output_size 是 2，我的想法与多分类问题相同，model 输出的 y 形状为 [batch_size, 2]，哪一边概率大预测的就是哪一个，但是输入到损失函数中报错了，提示我 BCEWithLogitsLoss 的输入应该是一维的张量。BCEWithLogitsLoss 函数会把输入的向量进行一个 sigmoid 激活，映射到 0-1 之间，所以输入张量的形状应该是 [batch_size]，这意味着我设置 output_size=1 后，还需要进行一步 squeeze，因为 [batch_size, 1] 和 [batch] 形状不同。

```python
def forward(self, x):
    // ...
    return torch.matmul(ht, self.fc).squeeze(1)
```

如果在分类层输出 [batch_size,2] 然后经过 softmax 激活送入 CrossEntropy 虽然有一样效果但是不完全等价，二分类问题还是建议 output_size=1+二元交叉熵。

#### 过拟合

第一个版本的代码训练后出现 train 上 100%准确率，test 上 50%正确率，明显是过拟合了。我进行了以下调整：

- batch_size 从 16 → 32，batch_size 越大训练越稳定，但是相对训练速度会下降。
- hidden_size 从 520 → 250，模型容量太大也有可能导致模型学习到过多的训练集特征，导致过拟合。
- dropout 可以在训练期间随机丢弃一些参数，增强模型的学习能力，避免过拟合。
- 优化器从 Adam 变成 AdamW，为了提高训练速度，我没有用 IMDB 全部的训练集，Adam 在小数据集上容易过拟合。
- 数据集从 5000 提高到了 20000

### 反思

#### &lt;PAD&gt; 的词向量

深度学习的模型要求输入的向量有统一的维度，但是 IMDB 评论的长度不固定，所以需要我们进行截长或者补短。在代码中我设定 SEQ_LEN=250，如果句子的 token 不足 SEQ_LEN，会补充&lt;PAD&gt;，同时我在 embeddings 中也为&lt;PAD&gt; 初始化了一个可训练的张量，也就是说 **&lt;PAD&gt; 会对训练产生影响** 。

解决方法：

**1. torch.nn.Embedding**

目前我们的 embedding 是一个张量而不是 nn.Parameter，也就是说它不能训练。我们可以使用 PyTorch 自带的 Embedding，他除了能自训练词向量，还可以设置&lt;PAD&gt; 对应词向量不参与训练。

```python
class Embedding:
    
    def __init__(
        self,
        device: Literal["cuda", "cpu"],
        embed_filepath: str,
        embed_size: int,
        token2id: dict
    ) -> None:
        self.token2id = token2id
        token2vec = {}
        
        if not isinstance(embed_filepath, Path):
            self.embed_fp = Path(embed_filepath)
            
        assert self.embed_fp.exists(), "embed file not found"

        for line in open(self.embed_fp, "r").readlines():
            sp = line.strip().split()
            token2vec[sp[0]] = torch.tensor([float(x) for x in sp[1:]], dtype=torch.float32)

        assert len(sp[1:]) == embed_size, f"预训练向量维度与{embed_size}不符"
        
        weight = torch.normal(
            mean=0.0, std=0.9, size=(len(self.token2id), embed_size), dtype=torch.float32
        ).to(device)
        
        for token, idx in self.token2id.items():
            if token in token2vec:
                weight[idx] = token2vec[token].to(device)
            elif token.lower() in token2vec:
                weight[idx] = token2vec[token.lower()].to(device)
        
        weight[pad_id].zero_()

        self.embedding = nn.Embedding(len(token2id), embed_size, padding_idx=pad_id)
        self.embedding.weight.data.copy_(weight)
```

nn.Embedding 可以设置 padding_idx，这样 &lt;PAD&gt; 对应的词向量就会不参与计算和训练。

**2. pack_padded_sequence**

它让 LSTM 在计算时 跳过 句子中 &lt;PAD&gt; 位置，不做无意义的时间步计算。

```python
class Model(nn.Module):
    def __init__(...):
        super().__init__()
        self.embedding = ...
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        h = h[-1]  
        return self.fc(h)
```

#### 分词器

在调试的过程我发现，单单通过空格也就是 split 来对句子进行分词是不足的，会出现 me.，ok! 这样奇怪的 token，甚至有些附带了换行符。

优化方向包括：

1. 去掉 HTML 换行标签等
2. 统一引号（避免奇怪的 Unicode 引号）
3. 通过 torchtext 库来进行 tokenizer

```python
from torchtext.data.utils import get_tokenizer
tok = get_tokenizer("basic_english")
tok("I don't like this movie!!! it's bad")  
# -> ['i', "don't", 'like', 'this', 'movie', '!', '!', '!', "it's", 'bad']
```

#### 正则化不足

现在仅对 $h_t$ 做了 dropout，但更通用的是对输入嵌入和 LSTM 层间做 dropout，建议在 $xt$ 或 $c_t$ 上做 dropout，而不是只在 $h_t$ 上。

model

#### 其他

1. 数据泄露问题：Embedding 应该仅用训练集构建词汇表，测试集的未知词统一用 &lt;UNK&gt; 表示
2. 测试集 DataLoader 设为 shuffle=True，测试时无需打乱数据，反而增加计算开销
