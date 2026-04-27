---
title: MedicalGPT 学习指北
date: 2026-03-27T11:09:35+08:00
featuredImage: http://img.xilyfe.top/img/20260327113828861.png
authors:
  - Xilyfe
series:
  - 项目笔记
tags: []
lastmod: 2026-04-09T04:23:24+08:00
---

{{< admonition type=info title="Summary">}} 
Minimind 和强化学习暂时告一段落了，现在准备开始一个新的项目 “MedicalGPT”。这个项目也是 Github 上的一个开源项目，实现了包括增量预训练、有监督微调、RLHF 和 DPO。这个项目中我主要会学习其中的一些 trick、数据构造思路、训练评估的完整流程，总体如下：

1. 增量预训练：对比不同超参、数据配比的效果
2. 有监督微调：构造不同的数据集，对比在 cpt 模型和基模上的效果
3. DPO、GRPO

其次 MedicalGPT 项目的学习思路是，先构造数据集、修改训练脚本跑对比实验，然后自己再实现简易版的训练代码，从而提高学习效率。
{{< /admonition >}}

{{< admonition type=warning title="提示">}} 
实验最初用的是 Qwen3.5，但是它会自动附带推理标签，导致 GRPO 训练出现问题，所以从 GRPO 部分开始改用 Qwen2.5-3B-Instruct 模型。
{{< /admonition >}}

## 1. 评估框架

### 1.1 Ceval 数据集

我们采用 lm-evaluation-harness 框架 + ceval 的医疗数据集对模型进行评估。评估流程还是比较简单的，我们用测试 baseline 来举例：

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /root/autodl-tmp/hf_cache
export HF_HOME=/root/autodl-tmp/hf_cache

git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness

python -m lm_eval --model hf --model_args pretrained=dir --tasks ceval-valid_basic_medicine,ceval-valid_clinical_medicine,ceval-valid_physician,ceval-valid_veterinary_medicine --num_fewshot 5 --batch_size auto --device cuda
```

这样就可以看到模型在 4 个数据集下面的 fewshot 得分：

![d98322d1d65dfdadadd28720f03cadfd.png](http://img.xilyfe.top/img/20260327131205690.png)

### 1.2 困惑度

Perplexity 困惑度是衡量语言模型对测试文本的"预测难度"，值越低说明模型对该语料分布拟合越好，对输出越自信：
1. PT/CPT 阶段：最直接反映模型是否真的学到医学语言分布的信号，可以在医学教材/病历测试集上测 PPL 来验证训练有没有跑偏。
2. SFT/RLHF 阶段：指令微调后 PPL 的解释性大幅下降，模型可能 PPL 升高但实际回答更好，因为 SFT 改变了输出分布。其次在 SFT/RLHF 阶段用户关心的是答得对不对、安不安全，而非 PPL。

```python
@torch.no_grad()
def calc_ppl_sliding_window(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 2048,
    stride: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    model.eval()
    loss_fn = CrossEntropyLoss(reduction="sum")

    total_nll = 0.0
    total_tokens = 0
    per_text_ppls = []

    iterator = tqdm(texts, desc="Computing PPL") if verbose else texts

    for text in iterator:
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids.to(device)  # [1, seq_len]
        seq_len = input_ids.size(1)

        text_nll = 0.0
        text_tokens = 0
        prev_end = 0

        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            # 当前窗口的 input_ids
            chunk = input_ids[:, begin:end]
            # 只计算新 token 的 loss（去掉与上一个窗口重叠的部分）
            target_len = end - prev_end

            with torch.no_grad():
                outputs = model(chunk)
                logits = outputs.logits  # [1, chunk_len, vocab_size]

            # 标准 next-token shift：logits[i] 预测 token[i+1]
            # shift 后长度为 chunk_len - 1
            shift_logits = logits[0, :-1, :]  # [chunk_len-1, vocab]
            shift_labels = chunk[0, 1:]  # [chunk_len-1]

            # 只对新 token 部分（最后 target_len 个位置）计算 loss
            # 同时 clamp 防止 target_len > shift 后的长度（极短文本边界情况）
            actual_target = min(target_len, shift_labels.size(0))
            shift_logits = shift_logits[-actual_target:].contiguous()
            shift_labels = shift_labels[-actual_target:].contiguous()

            nll = loss_fn(shift_logits, shift_labels)
            text_nll += nll.item()
            text_tokens += target_len

            prev_end = end
            if end == seq_len:
                break

        if text_tokens > 0:
            text_ppl = math.exp(text_nll / text_tokens)
            per_text_ppls.append(text_ppl)
            total_nll += text_nll
            total_tokens += text_tokens

    overall_ppl = (
        math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    )

    return {
        "ppl": round(overall_ppl, 4),
        "total_tokens": total_tokens,
        "num_texts": len(per_text_ppls),
        "per_text_ppls": [round(p, 4) for p in per_text_ppls],
        "avg_text_ppl": round(sum(per_text_ppls) / len(per_text_ppls), 4)
        if per_text_ppls
        else None,
    }
```

## 2. CPT
### 2.1 构造数据集

测试完 baseline 我就开始着手 Continues Pretrain 了，PT 阶段我准备进行如下实验：

|      | 基模         | 阶段  | 训练方式 | 超参数                      | 数据集                                              | 简述            |
| ---- | ---------- | --- | ---- | ------------------------ | ------------------------------------------------ | ------------- |
| 实验 1 | Qwen3.5-2b |     |      |                          |                                                  | baseline      |
| 实验 2 | Qwen3.5-2b | pt  | lora | lr:2e-4 ranl:16 alpha:32 | medical_book_zh.json                             | lora pt       |
| 实验 3 | Qwen3.5-2b | pt  | lora | lr:5e-5 rank:8 alpha:16  | medical_book_zh.json                             | lora pt       |
| 实验 4 | Qwen3.5-2b | pt  | lora | lr:5e-5 rank:8 alpha:16  | medical_book_zh.json + wikipedia-cn.json(:60000) | 混合数据集 lora pt |

当对一个垂直领域进行继续预训练完后，大模型的通用能力会被遗忘，为此惯用的技巧是混一些通用能力+领域内的数据一起训练，这样的话既能学习新领域知识又能维持住通用能力。把这个数据的比例应该是多少呢？参考刘聪 NLP 的文章，我按照医疗:通用=1:5 来进行配比。

原文如下：
>由于没有大量的资源做 from scratch 的通用数据和领域数据配比的实验，个人的经验完全来自于 continue pretraining 和 sft。对 continue pretraining 来说，如果要让模型不丢失通用能力，比如 summarization，qa 等，**「领域数据的比例要在15%以下」** ，一旦超过这个阈值，模型的通用能力会下降很明显。所以在做领域数据 continue pretraining 的时候，一定更要混大量的通用数据。而且，这个阈值和不同的预训练模型是相关的，有些模型比如 llama 需要控制的阈值更低。这个阈值其实是个经验主义的结论，我也不能保证是适用于所有人的，但和不少同行交流下来，感觉大家的范围都在 10%-15% 左右。

### 2.2 训练脚本

```bash
python pretraining.py \
    --model_name_or_path /root/autodl-tmp/models/Qwen/Qwen3___5-2B \
    --train_file_dir /root/MedicalGPT/data/pretrain/train \
    --validation_file_dir /root/MedicalGPT/data/pretrain/validate \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --num_train_epochs 0.5 \
    --learning_rate 5e-5 \
    --warmup_steps 5 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 10 \
    --block_size 512 \
    --packing True \
    --output_dir /root/autodl-tmp/models/medicalGPT/pt_lora \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir ./cache
```

### 2.3 参数合并

由于 MedicalGPT 的代码中保存的是 LoRA 参数而不是完整的模型，我们需要手动进行合并，把参数 merge 到基模上再保存：

```python
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser


@dataclass
class ModelArguments:
    model_dir: str = field()
    lora_dir: str = field()
    output_dir: str = field()
    tok_dir: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.tok_dir is None:
            self.tok_dir = self.model_dir


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments))
    (args,) = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.lora_dir)
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tok_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"model saved in {args.output_dir}")
```

```bash
python merge_peft.py \
	--model_dir /root/autodl-tmp/models/Qwen/Qwen3___5-2B \
	--lora_dir /root/autodl-tmp/models/medicalGPT/pt_lora
	--output_dir /root/autodl-tmp/models/medicalGPT/pt
```

### 2.4 实验结果

#### 2.4.1 ceval 评分

|      | basic           | clinical        | physician       | veterinary      | average |
| ---- | --------------- | --------------- | --------------- | --------------- | ------- |
| 实验 1 | 0.5263 ± 0.1177 | 0.3182 ± 0.1016 | 0.5918 ± 0.0709 | 0.5217 ± 0.1065 | 0.4895  |
| 实验 2 | 0.4737 ± 0.1177 | 0.5000 ± 0.1091 | 0.6735 ± 0.0677 | 0.6957 ± 0.0981 | 0.5855  |
| 实验 3 | 0.6316 ± 0.1137 | 0.5000 ± 0.1091 | 0.6122 ± 0.0703 | 0.7826 ± 0.0879 | 0.6316  |
| 实验 4 | 0.5789 ± 0.1164 | 0.4091 ± 0.1073 | 0.6531 ± 0.0687 | 0.7826 ± 0.0879 | 0.6059  |
#### 2.4.2 ppl 困惑度

|      | 全文 PPL  | 平均文本 PPL |
| ---- | ------- | -------- |
| 实验 1 | 23.6631 | 31.4767  |
| 实验 2 | 22.4226 | 30.5782  |
| 实验 3 | 21.7006 | 28.2931  |
| 实验 4 | 21.3896 | 27.865   |

#### 2.4.3 实验分析

1. 实验 2 第一次 cpt 跑出来的结果不尽人意，basic-medical 数据集的评分甚至低于 baseline。一方面是 LoRA 的 rank 设置太大了，覆盖了原有的知识分布，改写 attention pattern。其次学习率设置太大了，cpt 的本质是 **在已有分布上微调 token 概率**，原本模型已经有能力了，学习率过大可能导致原有知识被覆盖，出现**灾难性遗忘**和**推理能力下降**。
2. 实验 3 中我减小了学习率，得到了最高的评分。
3. 实验 4 我混合了通用数据集和领域数据集，从而避免通用能力下降，结果是虽然 ppl 降低了，但是医疗数据集的评分下降。一方面混合 CPT 里通用语料占比大，模型确实更好地"预测医疗文本的下一个 token"（PPL 降低），但这可能只是因为语言流畅性、标点、常见词预测变好了，而不是真正学到更多医学知识。另一方面，在总数据量不大的情况下，通用数据集可能稀释了模型对医疗特定知识的注意力，或者在 LoRA 有限的秩空间内造成了参数更新的冲突。

### 2.5 代码分析

```python
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
)
from transformers.utils.versions import require_version

from datasets import DatasetDict, load_dataset


@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    tokenizer_name_or_path: Optional[str] = field(default=None)
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    trust_remote_code: Optional[bool] = field(default=False)
    use_fast_tokenizer: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)

    def __post_init__(self):

        assert not (self.load_in_4bit and self.load_in_8bit)

        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_dir: Optional[str] = field(default=None)
    validate_dir: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_validate_samples: Optional[int] = field(default=None)
    block_size: Optional[int] = field(default=512)
    validation_split_percentage: Optional[float] = field(default=0.1)
    packing: bool = field(default=True)
    streaming: bool = field(default=False)

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`"
            )


@dataclass
class PeftArguments:
    use_peft: bool = field(default=True)
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.0)
    lora_alpha: Optional[float] = field(default=16.0)
    qlora: bool = field(default=False)


@dataclass
class TrainingArguments:
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    output_dir: str = field()


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_pred
    preds = torch.argmax(logits, dim=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def main():
    logger = logging.getLogger(__name__)
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, PeftArguments, TrainingArguments)
    )
    model_args, data_args, peft_args, train_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )[:4]

    model_dtype = (
        model_args.dtype
        if model_args in ["auto", None]
        else getattr(torch, model_args.dtype)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.add_eos_token = True

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                "block_size should be less then or equal to tokenizer.model_max_length"
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def tokenize_padding_fn(examples: dict[str, str]):
        inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=block_size,
        )
        inputs["labels"] = [
            [
                input_id if input_id != tokenizer.pad_token_id else -100
                for input_id in sample
            ]
            for sample in inputs["input_ids"]
        ]
        return inputs

    def tokenize_packing_fn(examples: dict[str, str]):
        inputs = tokenizer(examples["text"], add_special_token=True)
        all_input_ids = []
        all_attn_masks = []
        for ids, masks in zip(inputs["input_ids"], inputs["attention_mask"]):
            all_input_ids.extend(ids)
            all_attn_masks.extend(masks)

        num_blocks = len(all_input_ids) // block_size

        output = {
            "input_ids": [
                all_input_ids[i * block_size : (i + 1) * block_size]
                for i in range(num_blocks)
            ],
            "attention_mask": [
                all_attn_masks[i * block_size : (i + 1) * block_size]
                for i in range(num_blocks)
            ],
        }
        output["labels"] = output["input_ids"].copy()
        return output

    model_kwargs = {}
    if model_args.load_in_4bit:
        if peft_args.use_qlora:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=model_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_dtype,
            )
    elif model_args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=model_dtype,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    if model_args.load_in_4bit or model_args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    if peft_args.use_peft:
        model = get_peft_model(
            model,
            LoraConfig(
                r=peft_args.lora_rank,
                lora_alpha=peft_args.lora_alpha,
                lora_dropout=peft_args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules="all-linear",
            ),
        )
        logger.info("use lora training")
    else:
        logger.info("use full-parameter training")

    dataset = DatasetDict()
    if data_args.dataset_name:
        logger.info("begin downloading datasets")
        raw_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
        )
    else:
        data_files = {}
        if data_args.train_dir and (folder := Path(data_args.train_dir)).exists():
            data_files["train"] = [
                str(p) for p in folder.rglob("*") if p.suffix in {".json", ".jsonl"}
            ]
        if data_args.validate_dir and (folder := Path(data_args.validate_dir)).exists():
            data_files["validation"] = [
                str(p) for p in folder.rglob("*") if p.suffix in {".json", ".jsonl"}
            ]

        raw_dataset = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
        )

    if train_args.do_eval and "validation" not in raw_dataset.keys():
        dataset["train"] = raw_dataset["train"].train_test_split(
            test_size=data_args.validation_split_percentage
        )
        dataset["validation"] = dataset.pop("test")
    else:
        dataset = raw_dataset

    logger.info(f"dataset: {dataset}")

    lm_datasets = dataset.map(
        tokenize_packing_fn if data_args.packing else tokenize_padding_fn,
        batched=True,
        num_proc=data_args.preprocessing_num_workers if data_args.streaming else 1,
        remove_columns=list(dataset["train"].features()),
        desc="Running tokenizer on dataset",
    )

    train_dataset = lm_datasets["train"]
    if data_args.max_train_samples and data_args.max_train_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    val_dataset = lm_datasets["validation"]
    if data_args.max_validate_sampels and data_args.max_validate_sampels < len(
        val_dataset
    ):
        val_dataset = val_dataset.select(range(data_args.max_validate_sampels))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset if train_args.do_train else None,
        eval_dataset=val_dataset if train_args.do_eval else None,
        processing_class=tokenizer,
        compute_metrics=compute_metrics if train_args.do_eval else None,
    )

    trainer.train()
    model.save_pretrained(train_args.output_dir)


if __name__ == "__main__":
    main()
```

我自己重写的简化版代码去掉的分布式训练的部分，简单讲一下代码里面有什么可以学习的地方：
1. 用 `dataclass` + `HfArgumentParser` 可以简化 parser 编写。
2. 如果用 padding 来对齐，需要给 labels 设置 ignore_index，MedicalGPT 的代码里面忽略了这一点。
3. 如果用流式读取数据集，那么 map 的时候需要固定 `num_proc=1`，不能用多进程处理了。

## 3. SFT

### 3.1 构造数据集

SFT 我预想做以下实验：

|      | 基模             | 阶段  | 训练方式 | 数据集                                       |
| ---- | -------------- | --- | ---- | ----------------------------------------- |
| 实验 1 | Qwen3.5-2b     | sft | lora | HuatuoGPT_sft_data_v1.jsonl 350mb         |
| 实验 2 | Qwen3.5-2b-cpt | sft | lora | HuatuoGPT_sft_data_v1.jsonl 350mb         |
| 实验 3 | Qwen3.5-2b     | sft | lora | HuatuoGPT_sft_data_v1.jsonl 召回筛选 350mb 数据 |

1. 对比在 baseline 上进行 sft 和在 cpt 后的模型上 sft，哪一个效果更好。
2. 其次希望对比在筛选过的数据集上进行 sft 能不能得到更好的效果。

#### 3.1.1 格式化数据集

HuatuoGPT_sft_data_v1.jsonl 数据集是 alpaca 格式，包含 instruction、input 和 output 三个字段。但 MedicalGPT 的 sft 训练脚本针对的是 sharegpt 格式，所以我们需要处理一下对齐格式：

```python
import json
import argparse


def alpaca_to_sharegpt(in_file, out_file):
    with open(in_file, "r", encoding="utf-8") as f_in, \
         open(out_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line.strip())

            instruction = data.get("instruction", "").strip()
            output = data.get("output", "").strip()

            sharegpt_format = {
                "conversations": [
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": output}
                ]
            }

            f_out.write(json.dumps(sharegpt_format, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="alpaca jsonl 文件")
    parser.add_argument("--output", type=str, required=True, help="输出 sharegpt jsonl 文件")
    args = parser.parse_args()
    alpaca_to_sharegpt(args.input, args.output)
```

#### 3.1.2 整体思路

![image.png](http://img.xilyfe.top/img/20260331112639827.png)

- **阶段 1 语言+PPL 过滤**：用一个轻量语言模型计算每条文本的困惑度，PPL 异常高的样本往往是乱码、拼音混杂或语法崩坏的文本，规则过滤发现不了这类问题，PPL 能精准识别。
- **阶段 2 MinHash + LSH**：两者配合是大规模去重的标准方案。MinHash 把文本压缩成签名，LSH 把相似签名的文本归到同一个桶里，只需在桶内做精确比较，避免了两两对比的 O(n²) 复杂度。
- **阶段 4 测试集召回**：本质就是以测试集（Anchor）为查询，在训练语料库里做检索，只保留测试集感兴趣的数据。
- **阶段 5 LLM 打分**：用较小的模型批量打分，Prompt 里明确要求输出 1-5 分并给出理由，分数低于阈值直接丢弃。

#### 3.1.3 源数据预清洗

首先是长度过滤，最初的想法是对长度极端分布的样本过滤掉，例如：

![image.png](http://img.xilyfe.top/img/20260328130800014.png)

把图中 q 和 a 长度在底部 2% 和顶部 1% 的红色样本过滤掉。我们统计了任意 n 个样本得到长度分布以及长度的阈值，然后通过遍历把长度在阈值外的样本过滤掉。

```python
def infer_threshold(self, samples: list[dict]) -> dict:
    tqdm.write("计算长度阈值中...")
    random.shuffle(samples)
    probe = samples[: len(samples)
    q_lens = [len(s["instruction"]) for s in probe]
    a_lens = [len(s["output"]) for s in probe
    threshold = dict(
        q_min=int(np.percentile(q_lens, 2)),
        q_max=int(np.percentile(q_lens, 98)),
        a_min=int(np.percentile(a_lens, 2)),
        a_max=int(np.percentile(a_lens, 98)),
    
    threshold["q_min"] = max(threshold["q_min"], self.args.q_min)
    threshold["q_max"] = max(threshold["q_max"], self.args.q_max)
    threshold["a_min"] = max(threshold["a_min"], self.args.a_min)
    threshold["a_max"] = max(threshold["a_max"], self.args.a_max
    tqdm.write(f"question 长度 P5={threshold['q_min']}, P95={threshold['q_max']}")
    tqdm.write(f"answer 长度 P5={threshold['a_min']}, P95={threshold['a_max']}"
    return threshol
@staticmethod
def length_filter(sample: dict, thresh: dict) -> bool:
    q_len, a_len = len(sample["instruction"]), len(sample["output"])
    return (
        thresh["q_min"] <= q_len <= thresh["q_max"]
        and thresh["a_min"] <= a_len <= thresh["a_max"]
    )
```

但是实际操作中发现这种筛选方法存在问题，例如：

```json
{"instruction": "肾血尿的辅助治疗有些什么？", "input": "", "output": "补虚固本"}
```

这个样本虽然长度很短，但是其中的信息密度是很大的，这是一个优质的 Q/A 数据，我们不能单纯因为它的回答短而过滤掉，所以我在预处理中不采用长度过滤。

---

语言过滤的实现比较简单，我们可以直接利用现成的第三方库，比如 fasttext。

```python
model = fasttext.load_model("lid.176.bin")

def is_zh(text):
    label, prob = model.predict(text)
    return label[0] == "__label__zh" and prob[0] > 0.8
```

---

语言识别只能确保文档是目标语言，但无法判断内容质量。一段语法正确的垃圾广告和一篇优质的技术文章，在语言识别上可能得到相同的分数。这就需要更精细的质量评估机制。这里我们就可以采用 **困惑度（Perplexity）过滤**，如果一段文本与模型训练数据的分布相似，困惑度就低；如果文本包含大量噪声、乱码或不自然的表达，困惑度就高，我们就可以根据困惑度来筛选出低质量的文本：

```python
@torch.no_grad()
def calc_ppl_batch(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 2048,
    stride: int = 512,
    batch_size: int = 2,
    device: str = "cuda",
):
    model.eval()
    loss_fn = CrossEntropyLoss(reduction="none")

    all_ppls = []

    for i in tqdm(range(0, len(texts), batch_size), desc="PPL"):
        batch_texts = texts[i : i + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        batch_size_, seq_len = input_ids.shape

        # 每条样本独立统计
        nll_sum = torch.zeros(batch_size_, device=device)
        token_count = torch.zeros(batch_size_, device=device)

        prev_end = 0

        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)

            chunk_ids = input_ids[:, begin:end]
            chunk_mask = attention_mask[:, begin:end]

            outputs = model(chunk_ids)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = chunk_ids[:, 1:]
            shift_mask = chunk_mask[:, 1:]

            # 计算逐 token loss
            loss = loss_fn(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            loss = loss.view(batch_size_, -1)

            # mask padding
            loss = loss * shift_mask

            # sliding window 只统计新 token
            target_len = end - prev_end
            loss = loss[:, -target_len:]
            mask = shift_mask[:, -target_len:]

            nll_sum += loss.sum(dim=1)
            token_count += mask.sum(dim=1)

            prev_end = end
            if end == seq_len:
                break

        ppl = torch.exp(nll_sum / token_count)
        all_ppls.extend(ppl.detach().cpu().tolist())

    return all_ppls
```

#### 3.1.4 数据去重

语义相似度去重又是一个比较麻烦的环节，一开始我的想法是 **n-gram + MinHash + LSH 做近似去重**，这也是一个比较主流的去重方法，具体原理可见文章：

{{< link_ref "data_clean" >}}

```python
@staticmethod
def build_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for ch in text:
        m.update(ch.encode("utf-8"))
    return m

def minhash_dedup(self, samples: list[dict]) -> list[dict]:
    tqdm.write("MinHash 去重中...")
    lsh = MinHashLSH(
        threshold=self.args.minhash_threshold, num_perm=self.args.minhash_num_perm
    )
    kept = []
    for idx, s in enumerate(tqdm(samples, desc="MinHash")):
        q = s.get("instruction", "")
        m = self.build_minhash(q, self.args.minhash_num_perm)
        key = f"doc_{idx}"
        result = lsh.query(m)
        if not result:
            lsh.insert(key, m)
            kept.append(s)
    return kept
```

测试中出现了这样的问题，医疗数据集中可能出现大量重复的词，例如："怎么办"、"是什么"、"如何治疗"、"怎么做"。这种词就会导致算法误判两个句子的相似度，例如 "糖尿病怎么治疗" 和 "低血糖怎么治疗" 两个完全不同意思的句子可能判断为相似。这个问题的本质是：**MinHash 的输入信号被模板词污染了**。MinHash 把 "怎么治疗" 这个模板当成核心特征，两个问题的交集大，Jaccard 自然高。我的解决方案是过滤掉 stopwords，让 MinHash 只关注实体词。

那我们应该如何知道哪些词是 stopwords，哪些词很重要呢？这里引入 **逆文档频率 (Inverse Document Frequency)** 的概念，它评估一个词在语料库中的“稀有程度”，假如一个词 "心脏" 出现的频率很低，说明它比较稀有，那么我们就可以且认为它是比较重要的词。同样 "怎么样" 这个词出现频率非常高，那么就不怎么重要了，这个 stopwords 我们就不应该考虑。

```python
def collect(self, docs: list[str]):
    self.n_docs = len(docs)
    for doc in tqdm(docs, desc="统计词频"):
        for w in set(self.tokenize(doc)):
            self.freqs[w] += 1
    for w, df in tqdm(self.freqs.items(), desc="计算IDF"):
        idf = math.log((self.n_docs + 1) / df) + 1.0
        self.idf[w] = idf
        if idf <= self.df_threshold:
            self.stopwords.add(w)
    print(f"stopwords size: {len(self.stopwords)}")
    return self

def build_features(self, text: str):
    tokens = self.tokenize(text)
    word_ngrams = [" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    word_ngrams = [
        g
        for g in word_ngrams
        if any(self.idf.get(w, 0) > self.df_threshold for w in g.split())
    ]
    char_ngrams = self.char_ngrams(text)
    return set(word_ngrams) | set(char_ngrams)
```

>这里我试过能不能把 char ngrams 去掉只使用 word ngrams，结果是效果更差了。因为 word ngram 很受 df_threshold 的影响，例如 `{"kept": "肢端肥大症性心肌病的发病原因？", "duplicates": [{"text": "肢端肥大症性心肌病的辅助治疗有些什么？", "sim": 1.0}]}` 这个例子中大概率把 "发病原因" 和 "辅助治疗" 过滤掉了，导致两个句子判断为完全相似。

实验中还存在一个问题 ：

```
{
"kept": "肾肿瘤的患病比例是多少？", 
"duplicates": [
		{"text": "喉肿瘤的患病比例是多少？", "sim": 0.8594}, 
		{"text": "枕叶肿瘤的患病比例是多少？", "sim": 0.7812}
	]
}
```

这个例子里面，"肿瘤"、"患病"、"比例" 都是低频词，导致这两个句子被识别为相似的句子，但实际上 "肾"、"喉"、"枕叶" 才是决定相不相同的关键。我的想法是在最后用一个额外的函数来判断这些潜在的句子，是不是真的相同：

```python
def is_real_duplicate(self, text_a: str, text_b: str) -> bool:
    tokens_a = set(self.tokenize(text_a)) - self.stopwords
    tokens_b = set(self.tokenize(text_b)) - self.stopwords

    # 互相独有的词
    diff_tokens = (tokens_a - tokens_b) | (tokens_b - tokens_a)

    if not diff_tokens:
        return True

    # 差异词中最大 IDF
    max_diff_idf = max(self.idf.get(w, 0) for w in diff_tokens)

    # 如果差异词有高 IDF（语义重要），则不是重复
    return max_diff_idf <= self.df_threshold
```

我们在两个句子独有的词中找最大的 idf，如果 idf 大于阈值，说明他们独有的词是关键词，就应该保留。

#### 3.1.5 向量化

向量化主要用的是 `sentence_transformers` 库，我们用 Qwen3-Embedding 模型得到数据的向量表示：

```python
def embed(
    self,
    texts: list[str],
    prompt_name: Optional[str] = None,
    batch_size: int = 128,
    desc: str = "Embedding",
):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if any(not isinstance(x, str) for x in texts):
        bad = next(type(x).__name__ for x in texts if not isinstance(x, str))
        raise TypeError(f"embed expects list[str], but got element type: {bad}")
    if not texts:
        return np.empty((0, self.dim), dtype=np.float32

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
        embs = self.model.encode(
            batch,
            prompt_name=prompt_name,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        all_embs.append(embs)
    return np.vstack(all_embs)
```

- 为了方便存储计算，向量保存为 numpy 格式
- 为了下一步用余弦相似度进行匹配，这里需要进行一下 normalize

```python
def vectorize_source(
    self,
    samples: list[dict],
    shard_size: int = 50000,
    shard_dir: str = ".cache/shard_cache",
    ans_truncate: int = 200,
    shard_info_path: str | None = None,
):
    shard_cache = Path(shard_dir)
    shard_cache.mkdir(parents=True, exist_ok=True)
    n_samples = len(samples)
    shard_info = []
    for shard_id, i in enumerate(range(0, n_samples, shard_size)):
        emb_dir = shard_cache / f"shard_{shard_id}.npy"
        meta_dir = shard_cache / f"shard_{shard_id}.jsonl"
        batch = samples[i : i + shard_size]
        if emb_dir.exists() and meta_dir.exists():
            tqdm.write(f"Shard {shard_id} 已缓存")
        else:
            texts = [
                item.get("instruction", "") + item.get("output", "")[:ans_truncate]
                for item in batch
            ]
            embs = self.embed(texts, desc=f"Shard {shard_id} Embedding")
            np.save(str(emb_dir), embs)
            with open(meta_dir, "w", encoding="utf-8") as f:
                for sample in batch:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        shard_info.append(
            {
                "emb_dir": str(emb_dir),
                "meta_dir": str(meta_dir),
                "n_shard": len(batch),
                "offset": i,
            }
        )
    final_shard_info_path = (
        Path(shard_info_path)
        if shard_info_path is not None
        else shard_cache / "shard_info.json"
    )
    final_shard_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_shard_info_path, "w", encoding="utf-8") as f:
        json.dump(shard_info, f, ensure_ascii=False, indent=2)
    tqdm.write(f"Shard info saved: {final_shard_info_path}")
    return shard_info
```

这里采用了一些工程化处理：
1. 由于数据量较大，所以分批次进行向量化，向量化之后存到文件里面，从内存中删除。
2. 为了提高稳定性，我把每个分片的结果都进行了缓存，每次会先进行检查是否有缓存。

#### 3.1.6 相似度匹配

![image.png](http://img.xilyfe.top/img/20260331124315121.png)

参考美团黑子的做法，我将 ceval 的测试集作为目标数据分布，然后把测试集和数据集都进行向量化做相似度匹配，选择最相似的 topK 条数据，其实也就是我们把评测集当做目标进行召回，选择更贴合评测集的数据。

```python
def _search_one_shard(self, shard: dict):
    emb_path = Path(shard["emb_dir"])
    offset = int(shard["offset"])
    src_vec = np.load(emb_path)
    if src_vec.ndim != 2:
        raise ValueError(
            f"source shard must be 2D, got {emb_path}: {src_vec.shape}"
        )
    src_vec = src_vec.astype("float32")
    if src_vec.shape[0] == 0:
        return None
    # 维度一致性检查
    if src_vec.shape[1] != self.anc_vec.shape[1]:
        raise ValueError(
            f"dim mismatch: anchor_dim={self.anc_vec.shape[1]}, "
            f"source_dim={src_vec.shape[1]}, file={emb_path}"
        )
    if self.normalize:
        src_vec = self.normalize_vectors(src_vec)
    dim = src_vec.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(src_vec)
    k_eff = min(self.top_k, src_vec.shape[0])
    if k_eff <= 0:
        return None
    scores, local_ids = index.search(self.anc_vec, k_eff)
    return scores, local_ids, offset

def query_topk(self) -> list[list[tuple[float, int]]]:
    n_anchor = self.anc_vec.shape[0]
    # 小顶堆，存 (score, source_global_id)
    heaps: list[list[tuple[float, int]]] = [[] for _ in range(n_anchor)]
    for shard in tqdm(self.shard_info, desc="Searching shards"):
        out = self._search_one_shard(shard)
        if out is None:
            continue
        scores, local_ids, offset = out
        # 遍历每个 anchor 的局部 topK
        for a_id in range(n_anchor):
            for j in range(local_ids.shape[1]):
                lid = int(local_ids[a_id, j])
                if lid < 0:
                    continue
                gid = offset + lid
                sc = float(scores[a_id, j])
                heap = heaps[a_id]
                if len(heap) < self.top_k:
                    heapq.heappush(heap, (sc, gid))
                else:
                    if sc > heap[0][0]:
                        heapq.heapreplace(heap, (sc, gid))
```

美团黑子的做法是通过限定相似度的阈值来进行筛选，但我认为这个阈值不太好设定，所以采用 topk 的方式进行筛选，这种方式也能控制最后筛选出来的数据数量。

#### 3.1.8 打分筛选

阶段五我希望通过 api 让大模型对前面筛选后的数据集打分。我们通过 prompt engineering 设计好提示词，然后让大模型对数据的 Q/A 从多个维度进行打分，我们就把在某个分数阈值下的数据去除掉，留下大模型认为高质量的数据。这里分享两个可以提高效率的 trick：
1. 由于数据之间没有关联，所以我们可以通过多线程并发的发送 api 请求，提高打分效率
2. 可以在 prompt 里面插入多条数据，让大模型返回多条评分，这样除了提高效率还能降低 api 的开销。

### 3.2 训练脚本

```bash
python supervised_finetuning.py \
    --model_name_or_path /root/autodl-tmp/models/Qwen/Qwen3___5-2B \
    --train_file_dir ./data/finetune \
    --per_device_train_batch_size 12 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --model_max_length 512 \
    --max_eval_samples 10 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 5 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 2 \
    --preprocessing_num_workers 4 \
    --output_dir /root/autodl-tmp/models/medicalGPT/sft_on_base_better_dataset \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir ./cache --flash_attn True
```

没什么好说的这里用的还是 LoRA 来训练，三个对比实验我们改一下模型和数据集就好了。

### 3.3 实验结果

|      | basic           | clinical        | physician       | veterinary      | average |     |
| ---- | --------------- | --------------- | --------------- | --------------- | ------- | --- |
| 实验 5 | 0.5263 ± 0.1177 | 0.5000 ± 0.1091 | 0.6122 ± 0.0703 | 0.7391 ± 0.0936 | 0.5944  |     |
| 实验 6 | 0.5263 ± 0.1177 | 0.5000 ± 0.1091 | 0.6327 ± 0.0696 | 0.6957 ± 0.0981 | 0.5887  |     |
| 实验 7 | 0.5263 ± 0.1177 | 0.5000 ± 0.1091 | 0.6735 ± 0.0677 | 0.6522 ± 0.1015 | 0.5880  |     |

结果很难受，我精挑细选召回了 5w 条数据进行训练，效果还不如直接 sft 的模型。一方面可能是 Ceval 数据集进行评估本来就不是很好，因为 Ceval 是 A/B/C/D 四选一，而 2b 模型的指令遵循 不是很好；另一方便 Ceval 数据集只要小几十条数据，所以评分都大差不差。不过我已经完整过了一遍洗数据的流程，所以简历上数据就编一编得了，SFT 到此结束。

## 4. GRPO

### 4.1 概要

GRPO 和 PPO 以及 DPO 不同，我们不用训练 Reward 模型，他对显存的需求也最低，所以 RLHF 部分我们先进行 GRPO 的实验。GRPO 最重要就两个部分，**奖励函数的设计** 和 **数据集**。MedicalGPT 写好的 GRPO 脚本是通过 trl 来进行训练的，我们就没必要自己修改了，具体代码可能留在后面在分析。这一章节的具体流程如下：
1. 制作一个 CoT 数据集用于冷启动
2. 修改奖励函数

### 4.2 奖励函数

MedicalGPT 提供的 `grpo_training.py` 里面自带两个奖励函数，格式奖励和正确性奖励。但很奇怪的是它的准确性奖励是针对于数学数据集的，我觉得有两个可能：一个是用数学数据集也能训练处模型的推理能力；第二个可能就是单纯是用于演示，我们还需要自己更改？不过无论如何我还是自己改写了奖励函数。小红书上一个帖子里提到他在 MedicalGPT GRPO 中用了四个奖励函数：
1. 格式奖励
2. 相似度奖励
3. 正确性奖励
4. 困惑度惩罚

我认为 PPL 惩罚其实用处不是很大，因为现在的基模都很牛逼了，不会生成 ppl 很大的文本，其次 PPL 惩罚会**抑制模型学习新的推理模式**。然后相似度奖励需要额外用 Embedding 模型计算向量，开销太大了，不如直接用 LLM 对正确性进行评分。

#### 4.2.1 格式奖励

`grpo_training.py` 自带的格式奖励就是单纯验证，输出的文本是否为 `<think></think><answer></answer>` 格式。但是在 LLM Reasong 那篇文章我们也提示过，这种奖励有点太过苛刻了，模型从头到尾输出格式都不满足奖励函数，根本学习不到。所以在 `format_reward` 之外，还可以给正确的 tag 给予奖励。

```python
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]

    rewards = [1.0 if match else -1.0 for match in matches]
    logger.debug(f'format rewards: {rewards}')
    return rewards

def tag_reward(completions, **kwargs):
    def tag_num(text):
        reward = 0.0
        for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
            if text.count(tag) == 1:
                reward += 0.125
        return reward

    responses = [completion[0]["content"] for completion in completions]
    return [tag_num(response) for response in responses]
```

#### 4.2.2 准确性奖励

准确度奖励或者说正确性奖励，本质就是 LLM as Judge，我们让 judge 模型对 grpo 生成的回复进行打分。最初我是想在本地跑一个 Qwen-4B 当评分模型，我提供 Question/Reference Answer 想着应该可以 work。

```python
import json

SYSTEM_PROMPT = """You are a medical QA evaluator. Score each answer independently based on the reference answer.

## Scoring Criteria (0.0 - 1.0 continuous scale)
- 1.0: Fully correct and complete. Key medical facts match the reference precisely.
- 0.8-0.9: Mostly correct with very minor omissions or slight inaccuracies.
- 0.6-0.7: Generally correct but missing important details or contains some inaccuracies.
- 0.4-0.5: Partially correct. Core idea is present but significant details are missing or some errors exist.
- 0.2-0.3: Mostly incorrect. Only small parts are relevant or correct.
- 0.0-0.1: Completely incorrect, irrelevant, or refusal to answer.

## Important Rules
- Judge based on medical accuracy, not writing style or length
- If the answer would be harmful to a patient, always score 0.0
- Use the full range of scores (not only round numbers like 0.0, 0.5, 1.0)

## Output Rules (STRICT)
- Output MUST be a valid JSON array of floats
- DO NOT include any explanation, text, or markdown
- DO NOT include trailing commas
- The length of the array MUST equal the number of input items
- Each score MUST be a float between 0.0 and 1.0
- Keep at most 2 decimal places

## Enforcement
- You must strictly follow the scoring rubric
- Your output will be automatically parsed; any format error will be considered a failure
- Do not output anything other than the JSON array
"""


def build_input(prompts, references, answers) -> str:
    tmp_lines = []
    for idx, (prompt, refer, ans) in enumerate(zip(prompts, references, answers)):
        tmp_lines.append(
            f"QA {idx + 1}\n"
            + f"Question: {prompt}\n"
            + f"Reference Answer: {refer}\n"
            + f"Answer: {ans}\n"
        )
    input_text = "".join(tmp_lines)
    return f"""Score the following QA pairs based on the reference answers.
## Input
{input_text}
"""


def wrapper(judge_model, tokenizer):
    def accuracy_reward(prompts, completions, **kwargs):
        references = kwargs.get("reference")
        answers = [completion[0]["content"] for completion in completions]
        assert len(prompts) == len(references) == len(answers)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_input(prompts, references, answers)},
        ]
        print(messages)
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
            use_thinking=False,
        ).to(device)

        generated_ids = judge_model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
            max_new_tokens=8 * len(prompts) + 16,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
        try:
            scores = json.loads(response)
        except json.JSONDecodeError:
            scores = [0.0] * len(prompts)

        assert isinstance(scores, list)
        return scores

    return accuracy_reward
```

想法很美满现实很残酷，在 4B 模型上换了很多个提示词，输出的奖励都是 0，我把相同提示词喂给网页版的 Qwen 和 ChatGPT 都能回复合理的 reward，说明参数还是太少了。第二个想法是调用大模型的 API 来进行打分，在 AutoDL 上实际测试了下，每个请求的耗时差不多在 2~3 秒，在线训练中这么长的耗时太不切实际了。最重要的是这种方法进行训练，max_completions_length 至少需要 1024 以上，真的炼不起。

#### 4.2.3 真·准确性奖励

苦思冥想之后我蹦出了一个 idea：DeepSeek-R1 的 GRPO 训练是用数学数据集，这样就可以用输出的答案来评价是否正确。那我们是不是可以把任务改成医疗选择题，然后让大模型从中选择 A or B，这样也有一个标准答案进行打分了。

1. 首先我们对数据集进行 map 把它变成选择题的格式：
```python
def build_user_content(question, chosen, rejected):
    if random.random() < 0.5:
        a, b = chosen, rejected
        correct = "A"
    else:
        a, b = rejected, chosen
        correct = "B"

    content = f"""问题: {question}
答案A. {a}
答案B. {b}
请从A、B中选择正确答案，在<answer>标签中只填写字母A或B，不要包含任何其他内容。"""
    return correct, content

def process_sample(x):
    correct, content = build_user_content(x["question"], x["chosen"], x["rejected"])

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "label": correct,
    }
dataset = dataset.map(
    process_sample,
    num_proc=script_args.preprocessing_num_workers,
    desc="Processing dataset" if is_main_process else None,
)
```

2. 其次就是奖励函数了，除了正确性奖励我还加了一个 `letter_reward` 加快模型收敛，类似 tag reward 之于 format reward：
```python
def extract_content(completions) -> list[str]:
    return [completion[0]["content"] for completion in completions]

def extract_answer(completions):
    contents = extract_content(completions)
    return [content.split("<answer>")[-1].split("</answer>")[0] for content in contents]

def acc_reward(labels, completions, **kwargs):
    answers = extract_answer(completions)
    return [2.0 if ans == label else 0.0 for ans, label in zip(answers, labels)]

def letter_reward(completions, **kwargs):
    answers = extract_answer(completions)
    return [answer in ["A", "B"] for answer in answers]
```

下一步就可以开始训练了。

### 4.3 实验结果

GRPO 训练的显存占用也是非常之离谱啊，开启了 `gradient_checkpointing` 之后，`batch_size=4;num_generations=4;max_completion_len=512` 还是用了 80G 的显存，只好在 AutoDL 上租了一个 96G 的 RTX 9000 Pro 来用。为了不浪费宝贵的算力，先跑 220 个 step 看一下训练情况，swanlab 如图所示：

![image.png](http://img.xilyfe.top/img/20260407144611654.png)

各个 reward 只有 tag reward 呈上升趋势，其他 reward 可以说基本为 0，说明输出没有按照我们要求的 CoT 的格式。如果输出的答案没有用 `<answer></answer>` 包裹，那 acc reward 和 letter reward 为 0 也可以理解了。所以我准备先用 CoT 数据集跑一轮 sft 进行冷启动，如图在推理时候能输出 think 和 answer 标签了：

![image.png](http://img.xilyfe.top/img/20260407230753777.png)

OK，接下来就可以跑 GRPO 了，训练脚本如下：

```bash
python grpo_training.py \
    --model_name_or_path /root/autodl-tmp/models/Qwen2.5-sft \
    --train_file_dir data/grpo \
    --num_train_epochs 1 \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 13 \
    --output_dir /root/autodl-tmp/models/Qwen2.5-grpo-on-sft-lora \
    --dtype bfloat16 \
    --bf16 True \
    --report_to swanlab \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --beta 0.001 \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --use_vllm True \
    --logging_steps 10 \
    \
    `# QLoRA配置` \
    --use_peft True \
    --qlora False \
    --load_in_4bit False \
    --lora_target_modules all \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    \
    `# 显存优化配置` \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --num_generations 2 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 1024
```

跑了 700 个 step 训练曲线如图：

![image.png](http://img.xilyfe.top/img/20260407231247337.png)

tag reward 和 format reward 已经不是零了，说明 sft 冷启动有效果，模型可以输出正确格式了。但是 700 个 step 之后奖励还是没法收敛，震荡的很厉害，明显是 reward model 设计的有问题。进一步观察了一下模型的输出：

```
--- completion sample ---
COMPLETION: A
-------------------------                                                                                                --- completion sample ---
COMPLETION: B
-------------------------
```

训练到后面，模型逐渐倾向于不输出思维过程和标签，单独输出答案 A/B。这明显是出现 reward hacking 了：模型发现了一个捷径：直接输出 A/B → letter reward 可以得到 1，acc_reward 有机会得 2。是 reward model 设计的问题，假如答案 A/B 不用 `<answer></answer>` 包裹也能提取出来。

```python
def extract_think(completions):
    contents = extract_content(completions)
    results = []
    for content in contents:
        if "<think>" in content and "</think>" in content:
            results.append(content.split("<think>")[-1].split("</think>")[0].strip())
        else:
            results.append("")
    return results


def extract_answer(completions):
    contents = extract_content(completions)
    results = []
    for content in contents:
        if "<answer>" in content and "</answer>" in content:
            results.append(content.split("<answer>")[-1].split("</answer>")[0].strip())
        else:
            results.append("")
    return results
```

其次由于显存不够用的原因，`num_generation` 设置的很小，每组里面只有 2 个样本。这导致 reward std 奖励方差很小，一组里所有生成的 completion 得分一样，advantage=0，完全没有学习信号。

所以我再进行以下更改后又跑了一轮：
1. 更新 `extract_think` 和 `extract_answer`，确保不出现 reward hacking。
2. 提高 `num_generations` 避免一组 completion 的得分相同导致没有组内相对优势。

![image.png](http://img.xilyfe.top/img/20260408104017917.png)

这次 reward 曲线正常了，但是 acc reward 中 0.7 左右震荡，说明模型没学会回答问题。要么是 sft 训练不充分，要么是 GRPO 训练步数不够，要么就是对 GRPO 来说二分类的选择题太难收敛了，任务设计的不合理。


## 5. DPO

### 5.1 代码修改

DPO 的数据集是 MedicalGPT 提供的 reward dataset，其中包含问题以及正反例，结构如下：

```json
[
{"question": "", "chosen": "", "rejected": ""}
]
```

我们需要对 MedicalGPT 的 `dpo_training.py` 修改一下，让 `mapping_fn` 适应这个数据集格式：

```python
def return_prompt_and_responses(examples) -> dict[str, str]:
  prompts = []
  for question in examples["question"]:
      prompts.append(
          tokenizer.apply_chat_template(
              [{"role": "user", "content": question}],
              tokenize=False,
              add_generation_prompt=True,
          )
      )
  return {
      "prompt": prompts,
      "chosen": examples["chosen"],
      "rejected": examples["rejected"],
  }
```

### 5.2 实验一

第一版实验基本没有对超参数进行修改：
- $\beta=1$
- $\text{learning\_rate}=5e-4$

实验结果如下：

![image.png](http://img.xilyfe.top/img/20260409130557231.png)

图表中可以看出，reward margin 随着训练逐渐增加，这是一个很好的信号，其次 reward accuracy 也逐渐基本接近 1 了，所以在 chosen 和 rejected 里面能够方便出好答案和坏答案。但仔细看看 reward chosen 和 reward rejected 我们会发现，两条曲线都呈下降趋势，reward margin 能够提高只是因为 <mark>reward rejected 比 reward chosen 下降的更快</mark>。发现网上有两种声音：
1. DPO 采用的是 Bradley–Terry 模型，其本身就存在这种优化不确定性，而 DPO 刚好倾向同时降低概率来做优化。也就是这是正常的，只需要 reward margin 下降就好了。
2. 另一种声音是，正样本的质量不够/学习率太高，$\beta$ 太小，导致偏离 sft 的分布了。

### 5.3 实验二

因此我准备开始第二版实验，参考了网上的一些做法，准备在 DPO 的 loss 里面加入低权重的 chosen sft loss，其次把 $\beta$ 调大学习率降低一点。

DPO 支持多种 loss 组合，在 `DPOConfig` 中将 `loss_type` 和 `loss_weights` 设置为列表即可：

```python
from trl import DPOConfig, DPOTrainer  
  
training_args = DPOConfig(  
    loss_type=["sigmoid", "sft"],  
    loss_weights=[1.0, 0.1],
)
```

实验结果如下：

![image.png](http://img.xilyfe.top/img/20260409153210781.png)

这次实验就成功非常多了，reward chosen 和 reward rejected 都如预期的上升和下降。