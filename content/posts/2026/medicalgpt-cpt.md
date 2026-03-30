---
title: MedicalGPT 学习指北
date: 2026-03-27T11:09:35+08:00
featuredImage: http://img.xilyfe.top/img/20260327113828861.png
authors:
  - Xilyfe
series:
tags: []
lastmod: 2026-03-30T09:06:45+08:00
---
{{< admonition type=info title="Summary">}} 
Minimind 和强化学习暂时告一段落了，现在准备开始一个新的项目 “MedicalGPT”。这个项目也是 Github 上的一个开源项目，实现了包括增量预训练、有监督微调、RLHF 和 DPO。这个项目中我主要会学习其中的一些 trick、数据构造思路、训练评估的完整流程，总体如下：

1. 增量预训练：对比不同超参、数据配比的效果
2. 有监督微调：构造不同的数据集，对比在 cpt 模型和基模上的效果
3. GRPO：构造不同数据集对比相关

其次 MedicalGPT 项目的学习思路是，先构造数据集、修改训练脚本跑对比实验，然后自己再实现简易版的训练代码，从而提高学习效率。
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

|      | 基模             | 阶段  | 训练方式 | 数据集                               |
| ---- | -------------- | --- | ---- | --------------------------------- |
| 实验 1 | Qwen3.5-2b     | sft | lora | HuatuoGPT_sft_data_v1.jsonl 350mb |
| 实验 2 | Qwen3.5-2b-cpt | sft | lora | HuatuoGPT_sft_data_v1.jsonl 350mb |
| 实验 3 | Qwen3.5-2b     | sft | lora | train_zh_0.json 1gb 召回筛选 350mb 数据 |

1. 对比在 baseline 上进行 sft 和在 cpt 后的模型上 sft，哪一个效果更好。
2. 其次希望对比在筛选过的数据集上进行 sft 能不能得到更好的效果。

#### 3.1.1 格式化数据集

train_zh_0.json 数据集是 alpaca 格式，包含 instruction、input 和 output 三个字段。但 MedicalGPT 的 sft 训练脚本针对的是 sharegpt 格式，所以我们需要处理一下对齐格式：

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

数据筛选的 pipeline 包括多锚点目标分布 + 粗筛 + 质量过滤 + 多样性去冗：

![image.png](http://img.xilyfe.top/img/20260327230600346.png)

#### 3.1.3 构建目标锚点

#### 3.1.4 源数据预清洗

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

#### 3.1.6 相似度匹配

#### 3.1.7 多样性过滤

#### 3.1.8 打分筛选


### 3.2 训练脚本


## 4. RLHF


## 5. GRPO