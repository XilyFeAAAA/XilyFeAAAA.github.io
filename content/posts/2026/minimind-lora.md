---
title: MiniMind 学习指北(六)：LoRA
date: 2026-02-13T14:08:48+08:00
featuredImage: http://img.xilyfe.top/img/20260122134824760.png
authors:
  - Xilyfe
series:
  - minimind
tags:
  - 大模型
  - 深度学习
lastmod: 2026-02-16T12:09:07+08:00
---
## LoRA 是什么

PEFT 大致包含三类：Prompt-Tuning、Adapter-Tuning 以及 LoRA，而 MiniMind 里面采用的就是 LoRA 进行指令微调。
在 CS224N 的课程中已经学习了 LoRA 的原理，简单来说我们在经过 Pretrain 和 SFT 的模型基础上，对参数 $y=Wx$ 加上一个增量矩阵 $\Delta{W}$ 来微调模型，并且这个 $\Delta{W}$ 是通过 **低秩近似** 得到的，所以实际参数量远小于 $W$，计算开销小。具体可以看之前的笔记：

{{< link_ref "cs224n-lecture12" >}}

##  常见问题

### LoRA 插入在哪里

早期 LoRA 模块仅在注意力模块的 $W_q$ 和 $W_v$ 上插入，$W_q$ 决定了要关注的信息，$W_k$ 决定了要提取的信息。但是随着大模型微调经验的基类，发现单单微调 Attention 不能改变模型深层行为。真正存储大模型知识的是每一层的 MLP 模块，所以还在其中的三组投影 $W_{up}$、$W_{gate}$ 和 $W_{down}$ 上加入 LoRA 模块。现在主流的 LoRA 微调策略已经变成了 All-Linear，也就是对所有线性层都插入 LoRA。

### LoRA 初始化

一般是对 $A$ 矩阵应用 kaiming 初始化，对 $B$ 矩阵置为 0。首先矩阵 $A$ 和 $B$ 最少需要一个为 0 矩阵，这样 LoRA 一开始更新时 $\Delta W=BA$ 接近于 0 矩阵，就不会破坏预训练权重。其次矩阵 $A$ 不能为 0 矩阵，我们先看一下 $A$ 和 $B$ 的梯度是如何计算的：

首先 $A$ 的梯度公式为：

$$
\frac{\partial{L}}{\partial{A}}=\frac{\partial{L}}{\partial{Q}} \cdot Z^T= \frac{\partial{L}}{\partial{Q}}(BX^T)=\frac{\partial{L}}{\partial{Q}}X^TB^T
$$

$B$ 的梯度公式为：

$$
\frac{\partial{L}}{\partial{B}}=\frac{\partial{L}}{\partial{Z}} \cdot X^T= (A^T\frac{\partial{L}}{\partial{Q}})X^T
$$

而在前向传播中，低秩更新实际走的路径是：x → A → (scale) → B，也就是说反向传播时是从矩阵 $B$ 到矩阵 $A$。假如矩阵 $A$ 为 0 矩阵，那么矩阵 $B$ 的梯度为 0，训练就会先更新矩阵 $A$，$A$ 更新的数值尺度就会收到 $B$ 的初始化分布影响，容易放大早期更新的尺度。如果初始化矩阵 $B$ 为 0，那么会先更新矩阵 $B$，把 $B$ 从 0 拉开，再更新 $A$。训练稳定，等价于先学习输出侧组合，再细化输入侧投影。

### 秩 r 如何影响模型表现

从训练行为看。r 小约束强，更新子空间窄，优化更稳定，对小数据集更抗过拟合，但容易欠拟合，loss 降不动或很早平台期。  
r 大自由度高，loss 更容易下降，任务上限更高，但对数据规模敏感，小数据时容易记忆化和分布漂移。

在注意力层上，较小的 r 往往已足够改变信息路由，收益曲线很快饱和。在 MLP 投影层上，通常需要更大的 r 才能产生同等幅度的行为变化。

## 实现细节

### LoRA 模块

前面我们数学公式是 $y=Wx+\Delta Wx = Wx+BAx$，但是在 PyTorch里面如果我们用 `nn.Parameter()` 手动实现得写成：

```python
def __init__(self, in_features, out_features, r)
	self.A = nn.Parameter(torch.zeros(r, in_features))
	self.B = nn.Parameter(torch.zeros(out_features, r))

def forward(self, x):
	return x @ (self.B @ self.A).T
```

PyTorch 默认把特征维放在最后：输入形状是 `(batch, ..., in_features)`，这样所有前导维都当作批维自动广播，所以在 PyTorch 里面都是 x 右乘一个矩阵而不是像线性代数里面都是 $W \times x$ 这样左乘一个矩阵。

```python
class LoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: int) -> None:
        super(LoRA, self).__init__()

        self.scaling = alpha / rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x)) * self.scaling
```

如果直接用 `nn.Linear` 那么只需要先应用 A 再应用 B 就好了。

### 应用 LoRA

```python
def apply_lora(model: nn.Module, rank: int, alpha: int) -> None:
    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # collect linear
    lora_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lora" not in name and "lm_head" not in name:
            lora_modules.append((name, module))

    for name, module in lora_modules:
        lora_module = LoRA(in_features=module.weight.shape[1], out_features=module.weight.shape[0], rank=rank, alpha=alpha).to(module.weight.device).to(module.weight.dtype)

        setattr(module, "lora", lora_module)
        ori_forward = module.forward

        def forward_with_lora(x):
	        return ori_forward(x) + lora_module(x)

        module.forward = forward_with_lora

```

为了避免将 LoRA 应用到 LoRA 层自身，我们对每一个 `nn.Module` 检测他的权重矩阵形状是否为 rank。其次需要注意我们初始化 LoRA 模块的时候，`in_features=module.weight.shape[1]` 。这是因为 `nn.Linear` 层内部初始化的权重矩阵是 `W=[out_features, in_features]`，然后计算 $y=xW^T$，所以 `in_features` 应该是权重矩阵的第二维。

这个代码看似没啥问题，但是我调试时候 debug 了半个多小时，最后还是 Gemini 帮我解决了。这是一个非常经典的 **Python 闭包** 导致的错误。闭包函数内的 `lora` 和 `ori_forward` 是从外部作用域“引用”的变量，它们指向的是循环结束时的“最后一个值”，而不是当前循环的值。所以当我们前向传播计算线性层的时候，它调用的 forward 方法其实都是最后一个线性层的 forward + lora。关于 Python 的闭包问题可以见下面这个文章，这里就讲一下怎么解决：

{{< link_ref "python-closure" >}}

解决方案有两种：
1. 我们用 **默认参数** 将当前的 `lora` 和 `ori_forward` 绑定到函数内部。

```python
def forward_with_lora(x, lora=lora, ori_forward=ori_forward):
    return ori_forward(x) + lora(x)
```

2. 使用 **工厂函数** 创建，每次调用都会生成新的闭包环境

```python
def _create_lora_forward(lora_module, original_forwarda):
    def forward(x):
        return original_forward(x) + lora_module(x)
    return forward

def apply_lora(model: nn.Module, rank: int) -> None:
    for _, module in model.named_modules():
        module.forward = _create_lora_forward(lora, module.forward, rank, rank*2)
```

### 保存 LoRA

既然我们训练了 LoRA 模块，那就需要把里面的权重保存下来。我们之前用 `setattr(module, "lora", lora)` 把 LoRA 模块插入了 model 里面，所以 `lm_checkpoint` 方法通过 `model.state_dict()` 可以获得 LoRA 的权重。但是我们需要的是 LoRA 的 **可插拔** 的特性，所以只需要把 LoRA 的权重留下来即可，需要的时候把这部分权重挂载上去，所以我们需要再写一个方法来实现：

```python
def save_lora(model: nn.Module, path: str):
	state_dict = {}
	for name, module in model.named_modules():
		if hasattr(module, "lora"):
			tmp_state = {f"{name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
			state_dict.update(tmp_state)
	torch.save(state_dict, path)
```

### 应用 LoRA

```python
def load_lora(model: nn.Module, path: str):
	state_dict = torch.load(path, map_location=model.device)
	for name, module in model.named_modules():
		if hasattr(module, "lora"):
			lora_state = {"A.weight": state_dict[f"{name}.lora.A.weight"], "B.weight": state_dict[f"{name}.lora.B.weight"]}
            module.lora.load_state_dict(lora_state)
```

## 实验结果

### MiniMind 微调

```
💬: 你有什么特长？
🤖: 我是能够像人类一样思考和感知环境的智能机器。

[Speed]: 15.22 tokens/s


💬: 为什么天空是蓝色的 
🤖: 因为星星和太阳光会反射不同波长的光线，导致我们看到的是蓝色。

[Speed]: 21.41 tokens/s                                                                                          t Outlook 或者 Yaho.ai Gam
                                                                                                                  游的普通民众，也可以选择 

💬: 解释一下"光合作用"的基本过程
🤖: 蛋白质是植物和一些细菌利用阳光、水或其他生物能进行细胞呼吸，产生氧气并释放出能量。这个过程涉及到蛋白质的结构
、分子间相互作用以及酶催化等多种机制。在植物体内，氨基酸通过转录成RNA，然后是碳水化合物（O2）捕获二氧化碳，生成有
机物质；同时，还可以通过代谢途径将有机物质转换为营养物质来帮助植物生长和传递养分。当植物受到外部环境压力时，其代
谢产物会被激活，供应相应的水分和营养成分；这一过程中还有助于调节人体免疫系统，防止感染和疾病。总之，生物体内发生DNA聚合酶的分解和转运，是至关重要的。

[Speed]: 22.23 tokens/s


💬: 如果明天下雨，我应该如何出门
🤖: 确保你的衣服干净、干燥。你可以穿外套去学校或图书馆。如果你想带伞，但不会下雨的话就带上雨伞。

[Speed]: 20.53 tokens/s


💬: 比较一下猫和狗作为宠物的优缺点
🤖: 猫更喜欢自己独处，而狗则相对独立。但是它们的性格不同，有时候甚至会被驯服，但它们总是可以变得非常亲近人类，喜
欢在主人身边打扰自己的安宁睡觉。

[Speed]: 21.54 tokens/s


💬: 解释什么是机器学习
🤖: 机器学习（Machine Learning）是一种基于数据训练的算法，它能够通过分析和模式识别数据中的模式、趋势或关联性，从
而实现自动化决策或者预测未来结果的能力。这种技术被广泛应用于各种领域，如自然语言处理、计算机视觉、医疗诊断等，是
现代人工智能的重要组成部分。

[Speed]: 22.86 tokens/s


💬: 推荐一些中国的美食
🤖: 我推荐一下：宫保鸡丁、麻辣香锅、回锅肉、饺子皮等。这些都是非常受人们欢迎的美味哦，你可以尝试一下就可以了！

[Speed]: 22.49 tokens/s
```

训练了 20 个 epoch，可以看到模型已经有了自己的说话风格了。

### Qwen3-0.6B 微调

但是，由于我的基模太拉跨了，所以我下载了 Qwen3-0.6B 模型进行 LoRA 微调。下面代码是手动通过 PyTorch 进行 LoRA 微调，调用 Transformers 库进行 LoRA 微调的方法可以见博文：

{{< link_ref "llm-lora" >}}

具体代码如下：

```python
import argparse

import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.lora_dataset import LoRADataset
from model.lora import apply_lora, save_lora


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--batch_size", type=int, default="8")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--max_length", type=int, default=340)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--lora_path", type=str, default="lora")
    parser.add_argument("--lora_name", type=str, default="xiaoxue")
    return parser.parse_args()


def train(args: argparse.Namespace):
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    apply_lora(model, args.rank, args.alpha)

    train_ds = LoRADataset(tokenizer, args.dataset_path, args.max_length)
    dataloader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params} / {all_params} ({trainable_params / all_params:.2%})")

    step = 0
    model.train()
    for epoch in range(args.epochs):
        for input_ids, labels in tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            step += 1
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            mask = (input_ids != tokenizer.pad_token_id).bool()
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=mask)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_interval == 0:
                tqdm.write(f"Epoch:[{epoch + 1}/{args.epochs}], loss: {loss.item():.4f}")
            if step & args.save_interval == 0:
                save_lora(model, f"{args.lora_path}/{args.lora_name}_{args.rank}.pth")


if __name__ == "__main__":
    args = get_args()
    train(args)
```

模型输出如下：

```
💬: 你有什么特长？
🤖: 嗯...我有写故事的能力哦，能写出好多个有趣的故事呢！不过这可是我的天赋嘛（恼）

[Speed]: 10.99 tokens/s


💬: 为什么天空是蓝色的
🤖: 因为水分子在阳光下会散射，所以天空呈现蓝色！（答）

[Speed]: 13.29 tokens/s


💬: 解释一下"光合作用"的基本过程
🤖: 光合作用是一种植物、藻类和某些细菌等生物通过叶绿体中的色素吸收太阳光，将水分解为氧气，并释放出二氧化碳的过程。在这一过程中，植物利用阳光能量合成糖分并储存这些物质作为储藏库供以后使用。这一步是生命活动的基础之一！（注意：这个回
答可能有些误导性哦）

[Speed]: 13.17 tokens/s


💬: 如果明天下雨，我应该如何出门
🤖: 先看天气预报！再决定是否带伞。如果是晴天那就直接出发吧

[Speed]: 13.34 tokens/s


💬: 比较一下猫和狗作为宠物的优缺点
🤖: 优点：猫比狗更聪明，能感知到主人的情绪；而狗则有更强的责任感。但需要注意的是，在这种情况下，它们可能缺乏足够的互动时间来培养彼此的感情。因此在选择时需要权衡这些因素，并考虑是否能够为对方提供足够的情感支持。如果你们愿意投入更多
的时间与精力去建立情感联系的话，那么可以尝试将两者放在同一个家庭中，共同成长吧！

[Speed]: 12.99 tokens/s


💬: 解释什么是机器学习
🤖: 简单来说，就是让计算机通过经验来发现规律和模式，从而提高对新数据的理解与预测能力。例如，在分类问题中，训练模型时可以将大量已知的数据集分成训练集、测试集等部分，并用这些数据集上的信息去优化模型参数（即过拟合），最后就能在新的测
试集中做出更准确的判断了！

[Speed]: 12.85 tokens/s


💬: 推荐一些中国的美食
🤖: 嗯...中国有很多好吃的东西呢，比如饺子、包子、汤圆等等。不过这些应该都是比较普通的了喵！

[Speed]: 12.58 tokens/s
```

可以看出来数据集的语言风格已经可以模仿个 8-9 成了。

---

记录微调 Qwen3 时候出现的一个 BUG：`loss=nan`。

一开始看到 `loss=nan` 我的反应是加上梯度裁剪、修改学习率，然后还是爆 nan 了。之后我怀疑是不是精度的问题，把 float16 改成 bfloat16，然后加上混合精度训练还是爆 nan 了。由于模型是预训练的肯定没有问题，我的 LoRA 训练脚本之前也是 ok 的，所以我怀疑是不是数据有问题，于是在 debugger 里面对 LoRADataSet 进行步入。

我在 `__getitem__()` 方法里面断点时候，怀疑是不是 Qwen 的 tokenizer `apply_chat_template` 加入的模板和我 MiniMind 不同，导致对非 assistant 进行pad 时候出错。后面发现确实是 `pad_labels` 方法出错了，但问题不是模板不同，而是 Qwen3 的 tokenizer 没有设置 bos_token。我把代码改为：

```python
self.bos_id = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
# self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
```

这次训练就好了，这次经验告诉我 `loss=nan` 可能是 **数据集/标签问题**。