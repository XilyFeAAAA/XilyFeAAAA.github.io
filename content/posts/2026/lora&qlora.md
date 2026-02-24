---
title: LoRA&QLoRA
date: 2026-02-16T20:21:20+08:00
featuredImage: http://img.xilyfe.top/img/20260216202310375.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 大模型
lastmod: 2026-02-24T12:35:13+08:00
---
## LoRA

### 为什么选择旁路而不是堆叠

LoRA 现在的方案是 $y=Wx+\Delta{W}x$ 这样在旁路加一个矩阵进行微调，那为什么不选择 $y=W_2(W_1x)$ 这样的堆叠方案呢？

首先这种旁路设计可以保持原权重函数不变。LoRA 只提供一个低秩修正项。初始化时 A、B 接近零，模型行为≈原模型。训练是“微调偏移量”，不是“重建映射”。如果改成堆叠新层，前向函数直接改变，初始输出就漂移，大模型容易不稳定。

其次梯度隔离更干净。 旁路结构只训练 A、B，主干 W 冻结。梯度只流经低秩分支，等价于在参数空间做低维子空间更新。  
若采用堆叠层，新层会改变中间表示分布，等于对后续所有层输入分布做扰动，冻结主干时适配效率反而下降。

最后旁路方式使得计算与并行实现简单。可以拆成两个小矩阵乘法并与主路径并行执行，GPU 上容易融合。堆叠层则是严格串行，多一次完整层延迟。

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


## QLoRA

随着 LLM 参数量不断攀升，全量微调所需的显存越来越大。所以出现了 LoRA，它的思路是：冻结主模型权重，只训练少量的低秩适配器。这样虽然需要加载整个模型到显卡，但是由于只需要训练 LoRA 的 $A$ 和 $B$ 两个权重矩阵，所以优化器的参数非常少，并且中间激活值的占用也大幅度减小，所以显存需求大幅降低。但加载模型本身仍需高精度，显存占用依然较大。QLoRA 的思路就是在 LoRA 的基础上，将预训练模型量化为 4-bit，进一步压缩显存占用。

QLoRA 采用的 NF4 量化知识可见：

{{< link_ref "nf4-quantization" >}}

### Qwen3 QLoRA 实践

```python
import argparse

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

if __name__ == "__main__":
    args = get_args()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        r"D:\dev\github\minimind\model\qwen3-0.6b",
        quantization_config=bnb_config,
        dtype=args.dtype,
        trust_remote_code=True,
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrainer(
        r"D:\dev\github\minimind\model\qwen3-0.6b"
    )

    model = prepare_model_for_kbit_training(model)

    custom_dataset = Dataset()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    sft_config = SFTConfig(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        logging_steps=args.log_interval,
        fp16=(args.dtype == "float16"),
        bf16=(args.dtype == "bfloat16"),
        num_train_epochs=args.epochs,
        save_steps=args.save_interval,
        optim="paged_adamw_32bit",
        max_length=args.max_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=custom_dataset,
        peft_config=peft_config,
    )

    trainer.train()
```

在 SFTTrainer 里面传入 `peft_config` 它内部就会在 trainer 初始化时自动调用 `get_peft_model`，注入 LoRA 层，冻结 base 权重，并且返回一个 PeftModel 包装对象。

另外我们需要注意：**如果用 NF4 量化模型进行训练，那么必须要用 `prepare_model_for_kbit_training` 这个方法对模型进行**。如果只是进行推理的话，`bitsandbytes` 会在计算每一层矩阵乘法时，动态地将 4-bit 权重反量化为 float16 或 bfloat16。计算完 $y = Wx + b$ 后，中间结果就丢弃了。这个过程对精度的波动不敏感，只要模型能输出合理的概率分布即可。但是推理涉及到梯度的反向传播，很可能出现精度溢出的问题。`prepare_model_for_kbit_training` 将 **LayerNorm/RMSNorm** 转换为 float32。

>当 SFTTrainer 传入 `peft_config` 进行 LoRA 微调后，Trainer 内部会把 model 转为 PeftModel。对 PeftModel 调用 `save_model` 进行保存，它只会保存 LoRA 部分不会保存基模。我们打开 `adapter_config.json` 也能观察到，配置文件内部指向了原模型的地址：
>![image.png](http://img.xilyfe.top/img/20260224003451288.png)

