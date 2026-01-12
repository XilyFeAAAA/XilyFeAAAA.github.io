---
title: "CS224N Assignment 4"
date: '2025-11-30T11:24:11+08:00'
authors: [Xilyfe]
series: ["CS224N"]
tags: ["深度学习"]
--- 

## (a) MinGPT

任务(a) 要求阅读 `mingpt-demo/play_char.ipynb` 代码

### 1. 位置编码

GPT 用的位置编码不是正余弦函数，而是自训练的参数矩阵：

```python
def __init__(self, ...):
	self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))


def forward(self, idx, targets):
	position_embeddings = self.pos_emb[:, :t, :]
	x = self.drop(token_embeddings + position_embeddings)
```

之前提到过这个的好处是表达能力比单纯的正余弦更强。

> 每一个文本输入都是用同一个 Position Embedding，所以矩阵的 size(0)=1，然后通过张量广播就好了。

### 2. 参数初始化

之前我们通过 nn.Linear 或者 nn.Embedding 定义的模块，PyTorch 会对他们的参数进行默认 Kaiming 初始化。这里 GPT 手动对参数进行了初始化如下：

```python
model.apply(init_function)
```

`apply` 方法要求传入一个形参为 module 的函数，它会遍历模型的所有子模块，然后用 init_function 方法对它进行处理。

```python
def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

对于 nn.Linear 和 nn.Embedding，GPT 将参数初始化为均值 0，标准差 0.02 的正态分布，线性层的偏置初始化为 0，LayerNorm 的缩放因子设为 1。

### 3. 优化器

GPT 的优化器用的是 AdamW，它采用了 Weight Decay 来防止模型过拟合，这其实也就是 L2 正则化。

$$
\text{loss}_{total} = \text{loss} + \lambda \|W\|^2
$$

- $\lambda$ 是超参数，Weight Decay 的系数
- $|W\|^2$ 是全部参数的平方和

在损失函数中加入参数的平方和，就可以在反向传播的时候让模型趋向于减小参数，从而降低模型复杂度，增强泛化能力。

| 参数类型           | 为什么不加 / 要加 weight decay                                                                                     | 实际影响  |
| -------------- | ----------------------------------------------------------------------------------------------------------- | ----- |
| Linear weight  | 参数量巨大，是模型表达能力的主要来源，容易过拟合 → 必须加 weight decay 正则化                                                             | 强烈推荐加 |
| bias           | 只有一维（每个神经元一个），参数量极少，过拟合风险几乎为 0；加了反而会干扰梯度信号（bias 只需要平移）                                                      | 坚决不加  |
| LayerNorm γ, β | LayerNorm 参数本身就是做归一化的，强制让它们变小会破坏归一化效果（尤其是 γ 是 scale 参数，加 decay 会让模型倾向于输出更小的方差，效果变差）                         | 坚决不加  |
| Embedding      | 争议最大。早期 GPT-2 不加，后来 LLaMA1/2、Qwen、DeepSeek 等很多模型也给 Embedding 加了小的 weight decay（0.01~0.1），发现能稍微提升泛化。目前两种做法都有 | 可加可不加 |

```python
optim_groups = [
    {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
    {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
```


### 4. 多层感知机

可以用 `nn.Sequential` 直接组合多个模块，不用自己再继承 `nn.Module` 然后写 forward 。

```python
self.mlp = nn.Sequential(
    nn.Linear(config.n_embd, 4 * config.n_embd),
    nn.GELU(),
    nn.Linear(4 * config.n_embd, config.n_embd),
    nn.Dropout(config.resid_pdrop),
)
```

## (b) 阅读 `src/dataset.py`

Assignment 4 的最终目的是完成一个模型，可以读取一系列的 Q&A(某人的出生地点) 然后回答问题，例如：

- input: Where was Bryan Dubreuiel born?
- output: Atlanta

模型的训练方式很有意思，它并非是将 Question 输入然后输出 Birthplace，而是**将整个 Q&A 输出进去，然后输出答案的定位**。也就是说这个任务不是生成答案，而是 “定位答案”。

```
x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□
y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□
```

如果 y 不加 PAD，而变成 `Lebanon` ，那模型会被训练成生成问题后直接生成答案，模型就会学到：

> “Where was... born?” → Lebanon

它就成了文本生成模型，而不是抽取模型。但这个任务是抽取式 QA，希望模型学会：在 x 的上下文里定位答案的位置、不是生成答案，因此 y 前面的位置全部用 PAD。


## \(c) Finetune without pretraining

```python
if args.variant == 'vanilla':
    # TODO: [part c] Make some model here
    from models import GPT
    model = GPT(mconf).to(device)
    
    
elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    # [part c]
    ### YOUR CODE HERE ###
    from trainer import Trainer, TrainerConfig
    tconf = TrainerConfig(
        max_epochs=75,
        batch_size=256,
        learning_rate=args.finetune_lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=200*len(pretrain_dataset)*block_size,
        num_workers=4,
        writer=writer
    )
    text = open(args.finetune_corpus_path, 'r').read()
    train_dataset = dataset.NameDataset(pretrain_dataset, text)
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)
```

## (d) 预测

运行 `scripts/run_vanilla_no_pretraining.bat` 就能看到刚刚微调但没有预训练的模型，在数据集上面的表现了。最终的结果是正确率 1.8%。

实验还让我们每个问题都预测 London 将结果和之前的对比：

```python
def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    predictions = ["London" for _ in range(2000)]
    total, correct = utils.evaluate_places(r"/root/cs224n/birth_places_train.tsv", predictions)
    accuracy = correct / total

    return accuracy
```

输出的准确率是 4.55 %。


## (e) Span Corruption

Span Corrutpion 跨度破坏，是一种预训练的方法，之前在 Lecture 9 介绍过。但是之前介绍的 Span Corruption 是 Encoder-Decoder 架构的，并且模型参数/训练数据量大，所以这次的 Span Corruption 有所不同。

简单来说，以 "I want to be an engineer" 这个句子为例：

1. 截取 document 前面一个部分，平均长度为 block_size 的 7/8，最短为 4。
2. 将 document 分为 prefix + masked + suffix 三部分，prefix 和 masked 两部分平均长度为 document 的 1/4。
3. 将 document 重新组装为 prefix + mask_char + suffix + mask_char + masked + pad_char，长度为 block_size + 1。

最后得到形如 "I want??an engineer?? to be □□□□□□□□□□□□" 的句子。由于 GPT 是一个 Decoder Only 的结构，所以是自回归训练，还需要调整为 `x, y = document[:-1], document[1:]`

---

我在看 CharCorruption 的时候很奇怪，为什么处理文本处理的这么奇怪。我们先回顾一下 T5 Span Corruption 是怎么处理的：

假设有文本 "What are you doing Jennie?"，会随机选择几个 span 进行 mask，得到

- input: What "masked" doing "masked"
- output: "masked" are you "masked" Jennie?

思考之后我才发现这是因为 Decoder-Only 架构的局限性，它没法像 Encoder-Decoder 结构一样可以把 Mask 里面的信息在 Encoder 中告诉模型，只能把 Mask 的部分拼接在后面。T5（encoder-decoder）可以肆无忌惮地把一段文字直接删掉，替换成 Mask ，因为 Encoder 已经把原文全看过了，上下文信息已经编码进隐藏状态了，Decoder 只管从 sentinel token 开始生成被删掉的内容就行。但 GPT(decoder-only) 模型是因果自回归，它每一步只能看到之前的 token。如果你在中间直接把一段删掉换成 ⁇，模型在预测时根本不知道被删掉的内容长什么样，等到后面想让它复原时，信息已经彻底丢失了。

```python
def __getitem__(self, idx):
        # TODO [part e]: see spec above
        document = self.data[idx]
        document = document[:(tsize := random.randint(4, int(self.block_size*7/8)))]
        prefix = document[:(psize := random.randint(1, int(tsize/4)))]
        masked_document = document[psize:psize+(msize := random.randint(1, int(tsize/4)))]
        suffix = document[psize + msize:]
        output = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_document
        output += self.PAD_CHAR * (self.block_size + 1 - len(output))
        x, y = output[:-1], output[1:]
        return torch.tensor([self.stoi[c] for c in x], dtype=torch.long), torch.tensor([self.stoi[c] for c in y], dtype=torch.long)

```

## (f) Finetune with pretraining

之前的实验可以看到，模型不进行预训练直接微调后的结果非常糟糕，所以 (f) 任务就要求先进行预训练。

在这里再明确一下: 预训练和微调都是 train，只是**数据、任务目标、训练方式不同**。

- 预训练：预训练任务是 CharCorruption，训练方法是自回归
- 微调：预训练任务是给出完整 Q&A 然后定位到答案，训练方法也是自回归

所以预训练的代码就很简单了：

```python
if args.function == 'pretrain':
    assert args.writing_params_path is not None
    # TODO [part f]:
    from trainer import Trainer, TrainerConfig
    tconf = TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=args.pretrain_lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=650*len(pretrain_dataset)*block_size,
        num_workers=4,
        writer=writer,
    )
    trainer = Trainer(model, train_dataset, test_dataset)
    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)
```

实验结束在测试集上正确率是 5%，没有达到作业要求。把 CharCorruption 里面 prefix 和 mask 长度改为句子的 1/2 之后正确率达到 15% 了。

## (g) RoPE


RoPE 的具体实现参考文章 [RoPE](rope.md)

```python
def precompute_rotary_emb(dim, max_positions):
    rope_cache = None
    # TODO: [part g]
    # theta.shape = [dim/2, ]
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(max_positions)
    freqs = torch.outer(positions, theta)
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    rope_cache = torch.stack([cos, sin], dim=-1)
    return rope_cache
```

```python
def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x."""
    rotated_x = None
    ### YOUR CODE HERE ###
    _, _, seq_len, dim = x.shape
    rot = torch.view_as_complex(rope_cache[:seq_len, ...]) 
    
    rotated_x = x.view(*x.shape[:-1], dim // 2, 2)
    rotated_x = torch.view_as_complex(rotated_x)
    rotated_x = rotated_x * rot
    rotated_x = torch.view_as_real(rotated_x).reshape_as(x)
    
    ### END YOUR CODE ###
    return rotated_x
```

最后 MHA 里面把 Query 和 Key 进行旋转就好

```python
def forward(self, x):
	B, T, C = x.size()
	# calculate query, key, values for all heads in batch and move head forward to be the batch dim
    k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    if self.rope:
	    q = apply_rotary_emb(q, self.rope_cache)
        k = apply_rotary_emb(k, self.rope_cache)
    # ...   
```
