---
title: NF4 量化大模型
date: 2026-02-17T14:11:42+08:00
featuredImage: http://img.xilyfe.top/img/20260217141421109.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 大模型
lastmod: 2026-02-17T03:18:47+08:00
---
## NF4 量化

介绍 NF4 量化之前我们先说说量化是什么，量化本质是把 **浮点数张量压缩到有限的整数集合里**。例如 INT4 量化，就是把所有浮点数映射到 $2^4$ 个整数也就是 \[-8, 7] 的区间内，这样子就只需要存储这 4个 bit 也就是 0.5B。具体的公式是：

$$
x_{int} \approx \text{round}(x_{float} \div \text{scale}) \approx \text{round}(x_f \div \frac{max(abs(x_f))}{7})
$$

为什么这里需要除以 scale 呢？scale 通常表示 **"1 个整数单位代表多少浮点数值”**。我们设想一个例子，假如向量 `x=[-0.7, -0.5, 0.1, 0.9, 1.4]`，我们需要把它映射到 INT4 也就是 \[-8, 7] 的整数区间内。那我们计算得到 $scale = \frac{1.4}{7}=0.2$ 也就是一个整数区间表示 0.2 个浮点数值。那么我们就可以把 1.4 映射到 7，-0.7 映射到 -4，看起来就很合理了对吧？

但是在模型的参数中难免出现极端的分布，例如 `[-0.7, 0.2, 1.2, 1000]`。这样 1000 就会严重干扰浮点数和整数的映射，scale 变得很大就会导致模型精度损失很严重，所以 LLM 一般会每 32 或 64 个权重一组 scale。

这时候又有问题了：经过量化权重的数值完全变样了，那么模型还能正常输出吗？实际上推理时不会直接拿量化的整数去当真实权重用，算子内部做反量化，通过 $x_{float} = x_{int} \times scale$ 得到真实权重。但是那些被映射到相同整数的不同浮点数，它们被反量化之后的值就相同了，这就是量化导致的精度损失。

---

了解完基础的量化知识就可以说说 NF4 量化了。传统的 INT4/FP4 量化通常是均匀分布的量化，它们在数值轴上均匀分配了 16 个值。但是根据信息论，权重一般都呈现正态分布：

![image.png](http://img.xilyfe.top/img/20260216230212625.png)

也就是说在值在 0 附近的权重最多，绝对值越大权重分布的就越少。那如果按照传统的均匀分布量化就显得很不合理了，\[-2, -0.15, -0.10, -0.9, -0.2, 0.1, 0.2, 0.3, 0.5, 0.11, 1.5, 2.1] 按照均匀分布，就会导致 0 附近 **大量数值相近** 的权重被映射到一个整数上，而两侧少量的权重能够单独映射到一个整数。

NF4 的思路就是，在 0 的附近让 scale 更小 —— 整数单位代表更少的浮点数值，绝对值越大 scale 越大。在相同的 4-bit 存储空间下，NF4 能比 INT4 保留更多的权重信息，尤其是对于接近 0 的关键权重。

## 双层量化

在量化过程中，我们需要存储一些统计量比如缩放因子 scale 和零点 zero_point，来将 4-bit 数据还原为高精度浮点数。普通量化通常使用 FP32 存储这些统计量，双重量化就是对这些统计量**再次进行量化**，通常量化为 8-bit。这样平均每参数额外节省约 0.37 bit，对于大模型这能节省数百 MB 到数 GB 的显存。

## 分页优化器

在前向计算的时候，由于我们需要把基模中 NF4 存储的的权重进行反量化得到 FP16 的权重，然后与输入数据做运算，这会导致偶尔出现内存峰值，导致显存溢出。

分页优化器它利用了 NVIDIA 的统一内存特性。当 GPU 显存由于梯度峰值不够用时，它会自动把优化器状态暂时“换页”移到 CPU 的内存（RAM）里去，等需要更新时再取回来。这虽然会稍微慢一点点，但能确保证训练不会因为显存爆了而崩溃。

## 具体应用

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

## 对比数据

由于手头只有一个 RTX 3050Ti 跑模型（~~没钱去 autodl 租了~~），所以这次就在笔记本上跑 Qwen3 0.6B 和 Qwen3 8B 两个模型。

1. 对比内存占用

通过 transformers 库提供的 `get_memory_footprint` 方法，可以让我们清晰地看到模型占用的显存：

```python
tokenizer_fp16 = AutoTokenizer.from_pretrained("")
model_fp16 = AutoModelForCausalLM.from_pretrained("", quantization_config=bnb_config, trust_remote_code=True, torch_dtype=torch.float16)
print(f"FP16 模型大小：{model_fp16.get_memory_footprint() / 1024**3:0.3f}G")
del tokenizer_fp16
del model_fp16

tokenizer_nf4 = AutoTokenizer.from_pretrained("")
model_nf4 = AutoModelForCausalLM.from_pretrained("", torch_dtype=torch.float16)
print(f"NF4 量化后模型大小：{model_nf4.get_memory_footprint() / 1024**3:0.3f}G")
del tokenizer_nf4
del model_nf4
```

| 模型\精度      | FP16    | NF4    |
| ---------- | ------- | ------ |
| Qwen3-0.6B | 1.400G  | 0.785G |
| Qwen3-8B   | 15.256G | 5.553G |

很显然随着模型大小增加，NF4 量化减小的内存占用越来越明显。

2. 推理速度对比

因为显卡放不下 Qwen3-8B，所以就对 0.6B 的模型进行测试：

```python
prompt = "中国的首都是哪里？请用中文回答。"
tokenizer_fp16 = AutoTokenizer.from_pretrained("")
model_fp16 = AutoModelForCausalLM.from_pretrained("", torch_dtype=torch.float16).to("cuda")
print(f"FP16 模型大小：{model_fp16.get_memory_footprint() / 1024**3:0.3f}G")
start_time = time.time()
inputs = tokenizer_fp16(prompt, return_tensors="pt").to("cuda")
generated_ids_fp16 = model_fp16.generate(**inputs, max_new_tokens=50)
end_time = time.time()
fp16_time = end_time - start_time
print(f"FP16模型生成50个词元耗时: {fp16_time:.4f} 秒")
del tokenizer_fp16
del model_fp16
```

这里我测试了生成 50 个 token 的耗时，实验结果如下：

| 模型\推理耗时    | FP16    | NF4     |
| ---------- | ------- | ------- |
| Qwen3-0.6B | 3.2024s | 4.7461s |

实验结果比较符合我的猜测：在显存足够容纳模型的前提下，FP16 的纯推理速度通常略快于 NF4。因为 FP16 有原生硬件支持（Tensor Core），并且 NF4 需要反量化计算，会引入额外的计算延迟。但是在显存不足的情况下，FP16 可能会频繁触发内存交换，那么 NF4 也许就会更快了。

3. 性能对比

这里我直接跑 Gemini 写的困惑度计算的代码了：

```python
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# --- 1. 定义一个能够接受模型对象的 PPL 计算函数 ---
def calculate_perplexity(model, tokenizer, text_list, device="cuda", max_length=2048):
    model.eval()
    # 将所有文本拼接成一个长字符串（WikiText 标准评测方式）
    # 过滤空行，避免噪音
    text = "\n\n".join([t for t in text_list if t.strip()])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    nlls = []
    stride = 512  # 滑动窗口步长
    seq_len = input_ids.size(1)

    # 使用滑动窗口计算 Loss，这是标准的 PPL 计算逻辑
    pbar = tqdm(range(0, seq_len, stride), desc="Calculating PPL")
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - begin_loc  # 这里的逻辑可能有细微差异，通常以 stride 为步进

        # 这里的处理主要是为了处理长文本切片
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_chunk.clone()

        # 如果不是第一段，忽略前面的上下文对 loss 的贡献（只计算当前窗口的 loss）
        # 但对于简单评测，直接计算整段 loss 也是常见的近似
        target_ids[:, :-trg_len] = -100

        if input_ids_chunk.size(1) == 0:
            break

        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            # outputs.loss 是 cross entropy loss
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

    if not nlls:
        return float("inf")

    # PPL = exp(平均 NLL)
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


# --- 主流程 ---

# 1. 配置量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 开启 4-bit 量化
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # 开启双重量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时的精度
)

# 1. 加载数据

data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:5%]")
print(f"数据加载完毕，样本数: {len(data['text'])}")

# 2. 加载模型和分词器
model_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda",  # 直接加载到 CUDA
)

# 3. 计算困惑度 (直接传入模型对象)
ppl = calculate_perplexity(model, tokenizer, data["text"])
print(f"NF4 模型的困惑度: {ppl:.4f}")

# 4. 清理
del model_fp16
torch.cuda.empty_cache()
```

实验结果如下：

| 模型\困惑度     | FP16    | NF4     |
| ---------- | ------- | ------- |
| Qwen3-0.6B | 18.8580 | 21.2641 |

可以看到 NF4 损失了精度在性能上确实下降比较厉害，可能和模型大小也有关系。