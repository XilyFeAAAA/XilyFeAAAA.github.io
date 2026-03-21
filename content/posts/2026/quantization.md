---
title: 大模型量化
date: 2026-02-17T14:11:42+08:00
featuredImage: http://img.xilyfe.top/img/20260319143815942.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 大模型
lastmod: 2026-03-21T04:56:03+08:00
---
## 数据类型

![image.png](http://img.xilyfe.top/img/20260319145306223.png)

计算机中浮点数按 IEEE 754 标准存储，用 “指数+尾数” 的方式来表示一个实数的“科学计数法”形式。任何浮点数都可以写成：

$$
(-1)^S \times M \times 2^E
$$

如上图：
1. FP32 存储了 8 个指数位和 23 个尾数位，表示范围在 -1.18e38 到 3.4e38
2. FP16 存储了 5 个指数位和 10 个尾数位，表示范围在 -65k 到 65k
3. BF16 的优化是减小了精度提高了表示范围，它存储了 8 个指数位和 7 个尾数位，表示范围在 -3.39e38 到 3.39e38

![image.png](http://img.xilyfe.top/img/20260319150405954.png)

整数在计算机中就按二进制存储，例如 INT8 就是存储了 8 位二进制数，1 个符号位 7 个数值位，表示范围是 -128 到 127。

## 量化介绍

![image.png](http://img.xilyfe.top/img/20260319150900214.png)

量化旨在将模型参数的精度从较高的位宽(如 32 位浮点)降低到较低的位宽(如 8 位整数)。最小化表示模型参数的位数(以及在训练期间)是非常引人注目的。但是,随着精度的降低, 模型的精度通常也会降低。如 INT4 量化，就是把所有浮点数映射到 $2^4$ 个整数也就是 \[-8, 7] 的区间内，这样子原先一个 FP32 类型的参数需要存储 4B，现在就只需要存储这 4个 bit 也就是 0.5B，缩小了 8 倍。


### 对称量化

量化分为 **对称量化** 和 **非对称量化** 两种。在对称量化中, 原始浮点值的范围映射到量化空间中围绕零的对称范围。在前面的示例中, 请注意量化前后的范围如何保持以零为中心。 这意味着浮点空间中零的量化值在量化空间中也为零。

![image.png](http://img.xilyfe.top/img/20260319152011114.png)

我们把里面**最大的绝对值**作为映射的范围，假设 $\alpha$ 为最大绝对值 $b$ 为量化后的位宽，那么我们有：
- $s=\frac{2^{b-1}-1}{\alpha}$ 为比例因子
- $x_{\text{int}}=\text{round}(s\cdot x)$ 为量化后的值

应用量化, 然后应用反量化过程来检索原始数据, 如下所示:

![image.png](http://img.xilyfe.top/img/20260319152400110.png)

当您对值进行反量化以返回到 FP32 时, 它们会失去一些精度并且不再可区分。这通常被称为量化误差, 我们可以通过找到原始值和反量化值之间的差异来计算。

### 非对称量化

非对称量化在零附近不对称, 它将浮点数范围的最小值 $β$ 和最大值 $α$ 映射到量化范围的最小值和最大值。
 
![image.png](http://img.xilyfe.top/img/20260319153057000.png)

1. 同样计算映射的比例因子：$s=\frac{127-(-128)}{\alpha-\beta}$
2. 由于不是以零为中心，所以需要计算偏移量： $z = \text{round}(-s\cdot \beta) - 2^{b-1}$
3. 量化值：$x_{int}=\text{round}(s\cdot x+z)$

{{< admonition type=info title="量化是什么">}} 
可以看出来，量化的本质是一个映射公式，最常用的线性量化公式就是 $r = S \times (q - Z)$。

其中：
- r ：原始的浮点数。
- q ：量化后的整数。
- S ：缩放因子。
- Z ：偏移量（对应浮点数 0 的整数值）。
{{< /admonition >}}


### 量化粒度

![image.png](http://img.xilyfe.top/img/20260319190742463.png)


量化粒度是量化技术中的一个重要概念，它决定了量化操作的精细程度。量化粒度影响着量化参数的共享方式，即量化的规模和范围。不同的量化粒度可以带来不同的精度和效率的权衡：
- Per-tensor：整个张量或整个层级共享相同的量化参数（scale和zero-point）。这种方式的优点是存储和计算效率较高，但可能会导致精度损失，因为一个固定的量化参数难以覆盖所有数据的动态范围;
- Per-channel：每个通道或每个轴都有自己的量化参数。这种方式可以更准确地量化数据，因为每个通道可以有自己的动态范围，但会增加存储需求和计算复杂度;
- Per-group：在量化过程中，将数据分成块或组，每块或每组有自己的量化参数。

## 量化时机

### 量化感知训练 QAT

QAT 量化感知训练：首先正常预训练模型，然后在模型中插入“伪量化节点”，继续微调。所谓“伪量化节点”，就是对权重和激活先量化，再反量化，这样引入了量化误差让模型在训练过程中感知到量化操作，在优化 training loss 的同时兼顾 quantization error。

$$
\hat{x} = s[\text{clamp}(\lfloor{\frac{x}{s}}\rceil + z, 0, 2^b-1)-z]
$$

注意到，反向传播时四舍五入算子 $\lfloor \rceil$ 的梯度几乎处处为零，无法进行反向传播。一般是通过 straight-through estimator (STE) 解决，它近似地认为 $\lfloor  \rceil$ 的梯度始终为 1：$\frac{\partial  \lfloor x \rceil }{\partial x} = 1$。通过 QAT，可以减小量化误差，尝试用更低的位宽去量化模型。

### 训练后量化 PTQ

QAT 虽好，但插入“伪量化节点”后微调大大增加了计算成本，尤其是面对超大规模的 LLM。目前针对 LLM 的量化研究都集中在 Post-training quantization ——训练后量化。PTQ 就像是事后补救。它是指在一个已经训练好的浮点模型（FP32）上，直接通过一些统计手段将其转换为定点模型。

- **权重量化**：在 LLM 时代早期，**权重量化**的做法非常流行（比如 `bitsandbytes` 的 NF4 量化），因为大模型的瓶颈往往不在于计算速度，出于 **显存带宽**  和 **显存容量** 的考量。权重量化就是单纯想节省显存代码，它把权重存成 INT4/INT8，40GB 的模型变成 10GB。在推理时 GPU 从显存里读出 INT8 权重，反量化回 FP16，然后和 FP16 的激活值做矩阵乘法。这样显存占用大幅下降，但计算速度提升有限，甚至因为多了反量化的步骤，反而可能变慢一点点。
- **全量化**：目标是不仅省显存，还要追求加快计算速度。权重仍然以是以 INT4/INT8 存储，但同时**激活值**也被量化成了 INT8。这种方式下，GPU 不再进行繁琐的反量化过程，而是直接调用底层的 **INT8 Tensor Core** 硬件单元，全程都是用 INT8 进行计算，原本硬件一次只能算 1 组 FP32 的乘法，现在同样的硬件面积，一次能算 4 组甚至更多 INT8 的乘法。

{{< admonition type=info title="量化对象">}} 
在量化过程中，我们除了对 **权重** 和 **激活** 进行量化，还会对 **梯度** 进行量化，这样在 DP 或者 DDP 这种分布式计算中就可以**减少通信量**。
{{< /admonition >}}

{{< admonition type=question title="INT8 这么小的范围不会导致溢出吗？">}} 
全量化的关键在于 INT32 累加器。真正的量化计算过程是这样的：
1. 输入： 权重 $W$ 是 INT8，激活值 $X$ 也是 INT8。
2. 相乘： $W \times X$ 的中间结果会被存放在一个更宽的寄存器里，比如 INT32。
3. 累加： 所有的乘积项会在 **INT32** 的空间里进行累加。因为 $31$ 位符号位能承载巨大的数值，所以即便几千个数字相加也不会溢出。
4. 再量化： 在这一层的计算结束，准备传给下一层之前，我们会把这个 INT32 的巨量数值，通过缩放重新“挤压”回 INT8。
{{< /admonition >}}

#### 动态量化

对于权重而言，我们可以在推理前事先计算好量化系数，完成量化。但是对于激活（即各层的输入），它们事先是未知的，取决于具体的推理输入，会更加棘手。根据对激活的量化，分为动态与静态量化。动态量化就是在数据通过隐藏层后, 收集激活值，然后, 使用这种激活分布来计算量化输出所需的零点和比例因子值：

![image.png](http://img.xilyfe.top/img/20260319233916691.png)

动态量化的优点在于往往更准确一些, 因为它只尝试计算每个隐藏层的 s 和 z 值，但是 它可能会增加计算时间。

#### 静态量化

静态量化使用一小部分代表性数据集（通常只需几十到几百个样本）跑一遍前向推理，目的是统计激活值的分布范围。收集这些值后, 我们可以计算出在 推理过程中执行量化所需的 s 和 z 值。当执行实际推理时, s 和 z 值不会重新计算, 而是在所有激活上应用之前计算的 s 和 z 值来量化它们。

![image.png](http://img.xilyfe.top/img/20260319234032488.png)

同样，静态量化的准确性较低，但速度更快，因为它已经知道用于量化的 s 和 z 值。

## 量化方案

### GPTQ

GPTQ 属于 PTQ + weight-only + uniform+ per-channel，它的 motivation 来自于传统的 PTQ 方法例如 INT8 ZeroQuant 在 3-4bit 时精度下降过大，而更复杂的方法复杂度太高，耗时长。GPTQ 就是希望在不 retraining 的情况下让 3-4bit 仍然保持高精度。

![image.png](http://img.xilyfe.top/img/20260320115443901.png)

GPQT 的思想是按照行(channel) 去做量化，每次量化完在乘法时候又反量化，与原来不量化相比肯定有误差，那么就可以对这一行的其他元素做一些调整，使得误差尽可能减小。我们思考一个问题，假如我们知道原始输出为：

$$
y=x_1\cdot a+x_2\cdot b
$$

对 $x_1$ 进行量化后输出为：

$$
y = x_1'\cdot a + x_2\cdot b
$$

那么输入的误差就是：

$$
(x_1-x_1')\cdot a = q\cdot a
$$

假如我们把 $x_2$ 调整为 $x_2'$ 那么输出变化就是 $(x_2-x_2')\cdot b$，只要我们能让 $(x_2-x_2')\cdot b=-q\cdot a$，$x_1$ 量化导致的输出误差就被部分抵消了，所以可以得到新的 $x_2'=x_2+\frac{q\cdot a}{b}$，GPTQ 的思路就是如此。

![image.png](http://img.xilyfe.top/img/20260320123039485.png)

  
在这个逐层量化过程中, GPTQ 首先将层的权重计算得到反黑塞矩阵，它是模型损失函数的二阶导数, 它告诉了层中每个权重的反向重要性。接下来, 我们对权重矩阵中第一行的权重 $0.5$ 进行量化（因为它的重要性最大），然后对权重矩阵进行反量化:

![image.png](http://img.xilyfe.top/img/20260320123236845.png)

这样我们就能计算得到量化 $0.5$ 导致的误差了 $q=\frac{x_1-x_1'}{h_1}$(由于 $h_1$ 是反向重要性，所以我们需要除去它)，然后我们就可以分别对这一行的其它权重进行处理 $x_2'=x_2+q*h_2$。

![image.png](http://img.xilyfe.top/img/20260320123501523.png)

### GGUF系列

介绍 GGUF 之前先考虑一个问题，假设某层有一行权重，数值范围差异极大 $[0.0002, 0.0003, 0.0001, 48.5, 51.2, 49.7]$。假如我们采用 per-channel 的方式计算 scale 就会得到：

```
max = 51.2 → scale = 51.2 / 7 ≈ 7.31  
  
小权重 0.002 / 7.31 = 0.00027 → round → 0  
大权重 48.5 / 7.31 = 6.63 → round → 7 
```

可以看到小权重的信息被完全抹掉了。GGUF 的解决方案是把权重切成一个个 Block，每块单独算 scale，局部范围内数值差异小，量化误差就小得多。

接下来模拟 GGUF Q4_K 的量化过程，假设总权重为 $[2.4,1.8,3.2,2.1,8.1,7.5,9.0,8.8,0.3,0.5,0.2,0.4]$。
1. Q4_K 中规定：每个 super block 包含 8 个 sub block，每个 sub block 包含 32 个权重（这里简化为 4 个）：$[2.4,1.8,3.2,2.1]$、$[8.1,7.5,9.0,8.8]$、$[0.3,0.5,0.2,0.4]$。
2. 通过 absmax 法求每块的 s_sub：
	1. Sub Block 1：max = 3.2 → s_sub_1 = 3.2 / 7 = 0.457  
	2. Sub Block 2：max = 9.0 → s_sub_2 = 9.0 / 7 = 1.286  
	3. Sub Block 3：max = 0.5 → s_sub_3 = 0.5 / 7 = 0.071
3. GGUF 采用对称量化，所以除去 s_sub 就可以把权重量化到 INT4
	1. $[2.4,1.8,3.2,2.1]$ → $[5,4,7,5]$
	2. $[8.1,7.5,9.0,8.8]$ → $[6,6,7,7]$
	3. $[0.3,0.5,0.2,0.4]$ → $[4,7,3,4]$
4. Q4_K 中一个 super block = 8 个 sub block，每个 sub block 都有一个 scale。如果用 FP32 存所有 scale，开销很大共需要 $8\times32=256 \text{bit}$。GGUF 的做法是把 s_sub 再量化当作另一组权重处理，对 8 个 s_sub 做 absmax：$max(s_sub) = 1.286$ 得到 $s_{super} = 1.286 / 15 = 0.0857$
5. 所以以 Q4_K 为例，一个 super block 的完整存储布局为：
	1. s_super（1个，6 bit）
	2. s_sub × 8（INT4）
	3. 权重 × 256（INT4）

### BitsAndBytes

BitsAndBytes 是一个第三方库，它采用的是 NF4 量化。传统的 INT4/FP4 量化通常是均匀分布的量化，它们在数值轴上均匀分配了 16 个值。但是根据信息论，权重一般都呈现正态分布：

![image.png](http://img.xilyfe.top/img/20260216230212625.png)

也就是说在值在 0 附近的权重最多，绝对值越大权重分布的就越少。那如果按照传统的均匀分布量化就显得很不合理了，\[-2, -0.15, -0.10, -0.9, -0.2, 0.1, 0.2, 0.3, 0.5, 0.11, 1.5, 2.1] 按照均匀分布，就会导致 0 附近 **大量数值相近** 的权重被映射到一个整数上，而两侧少量的权重能够单独映射到一个整数。

NF4 的思路就是，在 0 的附近让 scale 更小 —— 整数单位代表更少的浮点数值，绝对值越大 scale 越大。在相同的 4-bit 存储空间下，NF4 能比 INT4 保留更多的权重信息，尤其是对于接近 0 的关键权重。

额外优化：
1. 双层量化：在量化过程中，我们需要存储一些统计量比如缩放因子 scale 和零点 zero_point，来将 4-bit 数据还原为高精度浮点数。普通量化通常使用 FP32 存储这些统计量，双重量化就是对这些统计量**再次进行量化**，通常量化为 8-bit。这样平均每参数额外节省约 0.37 bit，对于大模型这能节省数百 MB 到数 GB 的显存。
2. 分页优化器：在前向计算的时候，由于我们需要把基模中 NF4 存储的的权重进行反量化得到 FP16 的权重，然后与输入数据做运算，这会导致偶尔出现内存峰值，导致显存溢出。分页优化器它利用了 NVIDIA 的统一内存特性。当 GPU 显存由于梯度峰值不够用时，它会自动把优化器状态暂时“换页”移到 CPU 的内存（RAM）里去，等需要更新时再取回来。这虽然会稍微慢一点点，但能确保证训练不会因为显存爆了而崩溃。


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. 配置量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 开启 4-bit 量化
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_use_double_quant=True, # 开启双重量化
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算时的精度
)

# 2. 加载模型
model_id = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,       # 与 compute_dtype 保持一致
)
```

## 手写 Quantization

```python
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@dataclass
class QuantConfig:
    bit_width: int = 8
    eps: float = 1e-8


class QuantLinear(nn.Module):
    def __init__(
        self,
        weight: torch.IntTensor,
        scale: torch.Tensor,
        bias: Optional[torch.tensor] = None,
    ):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp = self.weight.float() * self.scale
        return F.linear(x, weight_fp, self.bias)


class QuantInt8:
    def __init__(self, config: Optional[QuantConfig] = None):
        self.cfg = config or QuantConfig()

    def from_pretrained(self, model_id: str) -> nn.Module:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        return self.quantize(model)

    def quantize(self, model: nn.Module) -> nn.Module:
        def replace(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    qweight, scale = self._quantize(child.weight)
                    setattr(module, name, QuantLinear(qweight, scale, child.bias))
                else:
                    replace(child)

        with torch.no_grad():
            replace(model)

        return model

    def _quantize(self, weight: torch.Tensor) -> tuple[torch.IntTensor, torch.Tensor]:
        alpha = weight.abs().max()
        qmax = 2 ** (self.cfg.bit_width - 1) - 1
        scale = (alpha / qmax).clamp(min=self.cfg.eps)
        qweight = torch.round(weight / scale).clamp(-qmax, qmax).to(torch.int8)
        return qweight, scale
```

## llm-compressor

llm-compressor 是 vLLM 出的一个量化框架，可以很方便的帮助我们对模型进行 AWQ/GPTQ 等量化。它提供了多种 compression scheme 和 compression algorithm。scheme 定义把模型量化成什么样：
1. **W4A16**：权重 int4，激活 fp16，显存最省
2. **W8A8 INT8**：权重+激活均 int8，吞吐最高
3. **FP8**：浮点 8 位，H100 原生硬件支持

algorithm 定义怎么找到最优的量化参数：
1. **RTN**：直接四舍五入，就是我手写的量化代码。
2. **GPTQ**：Hessian 误差补偿，精度提高了，但是需要校准。
3. **SmoothQuant**：专门解决 W8A8 的问题：激活里有少数极大值（离群值），直接量化会损失精度。它把这些离群值"平滑"到权重侧去承担，让激活变得容易量化。通常和 GPTQ 一起用。
4. **AWQ**：它发现只有少数权重通道对精度影响巨大，保护这些通道不被截断，其他通道正常量化。W4A16 场景下效果经常比 GPTQ 更好。


```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]"
).shuffle(seed=42)

# Preprocess the data into the format the model is trained with.
def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }
ds = ds.map(preprocess)
# Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )
ds = ds.map(tokenize, remove_columns=ds.column_names)
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True)
```
