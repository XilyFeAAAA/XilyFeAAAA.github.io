---
title: LLM reasoning & Chain of Thoughts
date: 2026-03-07T12:44:58+08:00
featuredImage: http://img.xilyfe.top/img/20260307124624416.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 大模型
lastmod: 2026-03-11T12:29:00+08:00
---
>本文系统介绍了当前大语言模型推理（Reasoning）能力的核心技术路线，包括推理范式、推理蒸馏以及基于强化学习的推理训练方法，并实现了一套完整的 Reasoning Pipeline。在推理蒸馏阶段，利用 open-thoughts 数据集（由 DeepSeek-R1 生成的高质量推理轨迹）对 Qwen3.5 模型进行监督微调，使模型能够学习显式的 Chain-of-Thought 推理过程，从而获得基础推理能力。在此基础上，通过设计多种奖励函数对模型进行强化学习训练，进一步提升模型的推理能力与输出质量。本文通过完整的工程实践展示了从推理数据构建、蒸馏训练到强化学习优化的全过程，为构建具备推理能力的中小规模语言模型提供了一套可复现的实现方案。

## Reasoning Paradigm

CoT 的开山之作 **Chain of Thought Prompting Elicits Reasoning in Large Language Models** 里面提到了 System-1 和 System-2 的概率，是心理学家 Daniel Kahneman 在 《Thinking, Fast and Slow》 中提出的认知模型。它提出了，System-1 任务和 System-2 任务分别代表两种不同的思维方式所处理的任务类型。System-1 代表了那些<mark>几乎不需要思考，本能就能做出回答</mark>的任务，比如看到 “2 + 2 = ?” 我们就立刻能反应过来答案是 4；System-2 代表了<mark>需要主动投入注意力、消耗认知资源，需要逻辑的步步推导</mark>的任务，比如数学证明题或者逻辑推导。

过往的大模型随着大模型参数量、算力开销、数据量协同增长，在标准提示下，其在 System-1 任务上性能显著增强。然而在 System-2 任务上，大模型表现出了“Flat Scaling Curves"现象一即模型规模增长未带来预期性能提升。
- 面对 System-1 问题，如常识问答、情感分类、意图识别等，随规模变大，大模型性能显著提升
- 面对 System-2 问题，如复杂数学计算、逻辑推理等，大模型性能提升缓慢甚至停滞不前。

![image.png](http://img.xilyfe.top/img/20260307155810862.png)

仅靠模型规模的扩大不足以解决所有问题，我们需要探索新的方法以提升模型的推理能力和智能水平。人类在解决复杂问题时，通常会逐步构建推理路径以导出最终答案。基于这一理念，一种创新的 Prompt 范式——Chain-of-Thought。CoT通过在提示中嵌入一系列中间推理步骤，引导大语言模型模拟人类解决问题时的思考过程，可以显著提升大语言模型处理复杂任务中的表现，从而突破“Flat Scaling Curves”的限制，提升模型处理System2任务的能力。

>CoT 的原理我们可以直观的理解，如果我们让大模型的 response 中输出它的思考过程，它会倾向于输出更多 token 会让它消耗更多的算力来思考，类似我们人类在思考复杂问题耗费更多脑细胞。

---

思维链可以按照推理方式分为三种，第一种是 **按部就班** 类型。在这种模式下，模型一步接着一步的进行推理，最终得到结论。其确保了推理过程的清晰和有序，使得模型的决策过程更加透明和可预测。论文中提出的方法是 Few-Shot-CoT，也就是标准 CoT。如上图，我们在 prompt 中加入类似 QA 的推理链示例，这就可以引导模型自行生成一条推理链。

![image.png](http://img.xilyfe.top/img/20260307162433240.png)

标准的 CoT 方法在提升模型推理能力方面取得了一定的成功，但是需要费时费力地手工编写大量 CoT 示例，并且过度依赖于 CoT 的编写质量，由此引出了 Zero-Shot-CoT。Zero-Shot-CoT 的本质就是在输出的开始加上 **let's think step by step**，无需手工标注的 CoT 示例，依然展现出了与原始少样本 Few-Shot-CoT 相媲美甚至更优的性能。

![image.png](http://img.xilyfe.top/img/20260307163013625.png)

在 Zero-Shot-CoT 的基础之上，Auto-CoT 引入与待解决问题相关的问题及其推理链作为示例，以继续提升 CoT 的效果。Auto-CoT 无需人工标注成本，但是性能超过需要了手工标注的 CoT 和无需手工标注的 Zero-Shot-CoT。它先用 K-Means 算法筛选出和问题相关的样本，然后在这些样本上利用  Zero-Shot-CoT 生成思维链内容，得到包含思维链的 QA 示例。然后把这些示例作为少样本示例，引导大模型生成 CoT 和答案。

![image.png](http://img.xilyfe.top/img/20260307163807705.png)

人类在解决 System-2 类问题时，会有一个反复选择以及回溯的过程。已有的 CoT 提示方法无法模拟这种过程，从而导致其能力受限。即无法解决复杂的 System-2 的问题，故已有的 CoT 提示存在问题：大语言模型顺序链式输出，不存在规划、前瞻性思考、自我评估和回溯的过程。

---

第二种思维链是 **三思而后行**，我们以 Tree of Thoughts，ToT 举例。

![image.png](http://img.xilyfe.top/img/20260307170831426.png)

ToT 将推理过程视为一棵思维树，从**拆解、衍生、评估、搜索**四个方面进行构造。首先它会用下面 prompt 来生成每一层的 thought 分支：

```
You are solving a problem step by step using Tree-of-Thought reasoning.

Problem:
{problem}

Current reasoning state:
{current_state}

Generate {k} possible next reasoning steps.

Each step should:
- be short
- be a single logical operation
- move closer to solving the problem

Return them in the format:

1. thought: ...
2. thought: ...
3. thought: ...
```

例如对于 24 点问题，生成的可能为：

```
1. thought: 8 × 4 = 32 → numbers [32,7,8]
2. thought: 7 + 8 = 15 → numbers [4,15,8]
3. thought: 8 − 4 = 4 → numbers [4,7,8]
```

这一步相当于树的展开：

```
state
 ├ thought1
 ├ thought2
 └ thought3
```

之后 ToT 会再利用 LLM 对这几个 thought 打分，只保留分数最高的 k 个 thought，然后再用这 k 个 thought 更新 prompt，再进行下一层的思考。当找到正确解/达到最大深度/ score 太低的情况下，搜索结束，然后选择分数最高的那个思维链。

---

最后一种思维链方式是 **集思广益**。集思广益模式强调的是通过汇集多种不同的观点和方法来优化决策过程。在这种模式下，模型不仅仅依赖于单一的推理路径，而是通过探索多种可能的解决方案从中选择最优的答案。Self-Consistency 引入多样性的推理路径，通过 Zero-Shot-CoT 一次性生成多个带思维链的回复，然后从中选择出现频率最高的答案作为最终的、最一致的答案。

![image.png](http://img.xilyfe.top/img/20260307172953114.png)

对于数学题，我们可以直接从答案的值来判断生成的 response 是不是一致的，但是对于开放性的问题 self-consistency 就不适用了。Universal Self-Consistency 利用 LLMs 自身选择最一致答案，它把生成的 response 再送入 prompt 到 LLM，有 LLM 决定哪一个答案最后。在多任务中性能良好且输出匹配度高、对顺序鲁棒。

![image.png](http://img.xilyfe.top/img/20260307205917441.png)

至此为止，前面我们提到的 CoT 都是 <mark>推理侧的 Chain of Thoughts</mark>，直接对 prompt 进行加工，是一种 Prompt Engineeing。但是这种 CoT 都模型是有要求的。最早发现 CoT 时，研究者就发现一个现象：**CoT 是一种“涌现能力”**。论文的结论是，对小模型使用 CoT prompt 基本没有效果，在 70B 以上规模的模型上才可以实现简单的推理。其次，如果模型训练时没有 reasoning trace，CoT 也很难发挥作用。

## Reasoning Distill

训练侧 CoT 相较于推理侧 CoT 是一种内化式能力的升级：
- 推理侧受上下文窗口和 prompt 引导能力限制，通常只有几百 token。训练侧可轻松生成几千甚至上万 token 的长思考链。
- 推理侧 CoT 在 7B-32B 模型上几乎没用；训练侧通过数据+RL 可以把 7B 模型逼出接近 70B 的 reasoning 能力。
- 推理侧需要每次都多采样 8-64 次，latency 和成本极高。训练侧模型单次 forward 即可输出高质量思考，推理成本更低。

目前训练侧的 CoT 分为两种，蒸馏 CoT 和 RL 强化 CoT。蒸馏 CoT 让用强教师模型（DeepSeek-R1、o1-preview、Qwen-max-thinking）对海量问题生成长 CoT，得到格式如 `{"prompt": "...", "thinking": "一步一步思考过程...", "answer": "最终答案"}` 的数据集，让学生模型 SFT 来学习。

```
💬: 101里面有几个1
🤖: 在数字 101 中，**只有一个“1”**。  
所以答案是：**1个**。

[Speed]: 17.60 tokens/s


💬: 101里面有几个1，让我们一步一步思考
🤖: 我们来一步步分析：
**题目：**
“101里面有几个‘1’？”
---
### 第一步：理解题意  
- 问题是在问 **数字中的各个数位上有哪些是‘1’**。
---
### 第二步：分解数字  
数字 `101` 可以表示为：
1 × 10² + 0 × 10¹ + 1 × 1⁰ = 100 + 0 + 1 = 101
也就是说，在这个数字中，有三个位置上的数字分别是：  
- 十位（第2位）：**0**
- 百位（第1位）：**1**
- 个位（第0位）：**1**
因此，“101”中有两个 `'1'`。  
---
✅ 答案：**有两个**。

[Speed]: 20.24 tokens/s
```

可以看到，我在 Qwen3-0.6B 的蒸馏模型上通过 CoT 让模型成功解决了正常提问无法解决的问题，提高了思维能力。而 Qwen3-0.6B 模型能正常推理就是蒸馏了完整的 Qwen3 模型，得到了教师模型的推理能力。

---

这里我复现哔哩哔哩 Up [偷星九月33](https://space.bilibili.com/349950942) 的视频 [使用DeepSeek-R1蒸馏训练自己的本地小模型](https://www.bilibili.com/video/BV1ekN2ebE68/)。视频里面用的数据集来自 [open-thoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) 这个项目，它使用 DeepSeek-R1 生成数据来微调模型，具体生成方式如下：
1. 将不同领域的数据（代码、数学、科学、谜题）输入到DeepSeek-R1模型中，生成带思考过程的答案。
2. 对生成的结果进行筛选（过滤掉一些不正确的答案），对于科学类的问题（开放性问答），无需验证，对于数学和谜题类的问题，通过大模型对生成结果和标准答案进行评判，对于代码类的问题，通过执行代码通过与否进行判断。
3. 将所有数据进行混合，得到最终的数据集。

![image.png](http://img.xilyfe.top/img/20260309134231270.png)

这里我用 Llamafactory 框架对 Qwen-3.5-2b-base 进行微调，具体方法可见：

{{< link_ref "llamafactory" >}}

微调配置如下：

```yaml
### model
model_name_or_path: /root/llm/models/qwen3.5-2b-base
quantization_bit: 4
quantization_method: bnb 
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: open_thoughts
template: qwen
cutoff_len: 2048
packing: true
train_on_prompt: false
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen3.5-2b/lora/sft
logging_steps: 100
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 1.0e-4
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
flash_attn: fa2
```

然后，实验失败了。训练 500 steps 的模型能够输出思考过程，但是不能完全按照制定的格式输出，其次输出的内容非常冗长，模型会反复推翻自己的决策。我们认为主要存在两个原因：
1. 模型没有经过冷启动，如果像 Deepseek 一样在 SFT 阶段，就给他训练带有 `<think>` 标签的数据，那么输出的格式应该会规范很多，或者说一个用 Instruct 版本而不是 Base 版本。
2. 配置里面 `cutoff_len` 设置太小了，数据集不考虑 System Prompt 和 prompt，单单 response 长度就在 8000 token 上下了，但是由于设备限制，训练时候截取了 2048 的最大长度，有可能思考部分都没结束。

## Reasoning RL


![image.png](http://img.xilyfe.top/img/20260309234001941.png)


在 o1 和 R1 出现之前，我们习惯用 SFT 来训练模型。但SFT 仅仅是给模型看“问题 + 答案”，让它模仿。模型学会了“看起来像推理”的格式，但没学会“如何推理”。一旦遇到没见过的问题，它容易幻觉或放弃。而 RL 的逻辑是：
- 让模型自己尝试生成多个答案。
- 告诉它哪个对了，就给奖励新号，哪个错了就给惩罚的信号。
- 模型为了拿高分，会自发进化出思维链（Chain of Thought），甚至自我反思。

传统方法通过大量人类偏好数据训练一个奖励模型，奖励模型预测的 rewards 是定义在连续实数域 $\mathbb{R}$ 上的一个数。虽然由于训练时候 sigmoid 函数使得实际中大部分 rewards 都分布在 $[-4,+4]$ 之间，**但还是连续且平滑的**。而 Deepseek-R1 使用的 Rule-Based Reward 其分布是<mark>离散的</mark>，比如我们规定答案正确就 +1，答案错误 -1，或者格式正确就基于对应分数。但是由于我们无法定义大量的规则来穷举 -reward 到+reward 之间所有的情况，所以奖励的分布是 **不连续不平滑的**，在梯度下降里，我们需要的信号是"这个回答比上一个**好多少**"，但离散 reward 只能告诉你"好"或者"不好"，没有**程度**的概念。比如两个答案都得了 -0.5，但一个其实差一点点就答对了，另一个推理完全错误——梯度信号看不出区别，全是 -0.5。这种离散的 reward 作为监督信号可能导致模型快速收敛到某些鞍点，最终导致模型训练的失效。

>**为什么不用 Reward Model？**
> 规则奖励是客观的，对就是对，错就是错，没有噪声。而 Reward Model 容易让模型找到捷径，比如反复输出关键词或者输出长串无意义文本，出现 Reward Hacking。

OpenAI 的解决方法是：训练一个模型来得到平滑的 reward。OpenAI 的《Rule Based Rewards for Language Model Safety》这个文章中，他们首先针对每条 rule 都构造了正负样本数据集，然后用这个标注数据集去训练一个分类模型。他们是通过训练一个连续的模型来拟合离散的 reward 分布以产生平滑的reward。但这是个 supervised learning 的方法，需要我们对于每个 rule 都有标注数据，限制了该方法的应用场景。比如采集 math/coding 这种任务带 reasoning 过程的负样本其成本是极高的。

Deepseek-R1 提出的 GRPO 给出了一个 unsupervised 的方法解决了离散 reward 平滑性问题：normalization。GRPO 算法对每组采样的 **大量** 答案做 normalization，就把离散问题变成了平滑、自适应的连续信号。当样本量足够大的话，normalization 就是最简单也最有效的平滑方法了。

![image.png](http://img.xilyfe.top/img/20260310135641339.png)
>可以看到右边四张图（也就是 normalization 之后）出现了更多不同的数值，而且最大/最小值会自动随训练阶段变化。

## Reward 设计

R1 的 Rule-Based Reward 分为三种主要类型：结果奖励、格式奖励和语言一致性奖励。

### 结果奖励

Outcome Reward 是只看最终答案对不对，不管推理过程。这种奖励方式实现简单，信号明确，不需要人工标注推理过程。但是缺点是它完全不管过程，模型可能推理过程一塌糊涂，但凑巧答对了。其次它对答案的提取非常讲究，比如正确答案是 20 我们需要处理各种格式："答案是 42"，"所以 x = 42" 等等。

```python
def correct_reward(answer, completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    responses = [response.split("<answer>")[-1].split("</answer>")[0] for response in responses]
    return [CORRECT_REWARD if resp == ans else 0.0 for resp, ans in zip(responses, answer)]
```

对于代码题 Deepseek-R1 直接在 sandbox 里面执行生成的代码，这样来判断答案的准确性：

```python
def code_reward(generated_code, test_cases):
    passed = 0
    for test in test_cases:
        try:
            result = run_code(generated_code, test.input)
            if result == test.expected_output:
                passed += 1
        except Exception:
            pass
    return passed / len(test_cases)
```

>trl 库的 GRPOTrainer 会自动把生成的 completions 加上数据集传入 reward function，所以这里我们显示定义了 dataset 里面的 answer 和 completions 两个参数。

### 格式奖励

格式奖励包括两部分，首先是输出的回复需要按照 `<think> ... </think> <answer> ... </answer>` 的格式来输出。其次为了让训练能快速拟合，加入了 Tag Reward，如果 `<think>`、`</think>` 等 tag 正好出现一次那么也给予奖励。

```python
def soft_format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]

def tag_reward(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [tag_num(response) for response in responses]
```

>这里的代码还存在问题，后面会具体说

### 语言一致性奖励

早期 R1 训练时发现，模型在 think 阶段会突然切换语言——比如你用中文问问题，它在 think 里混入了英文，或者在一段中文推理中突然冒出韩文。这会严重影响用户体验。

```python
def lan_consistency_reward(response: str, input_language: str = "zh") -> float:
    from langdetect import detect
    
    think_content = extract_think(response)
    if not think_content:
        return 0.0
    
    try:
        detected_lang = detect(think_content)
    except Exception:
        return 0.0
    
    # 语言代码映射
    lang_map = {"zh": ["zh-cn", "zh-tw", "zh"], "en": ["en"]}
    expected_langs = lang_map.get(input_language, [input_language])
    
    if detected_lang in expected_langs:
        return 0.1
    else:
        return -0.2 
```

### LLM as Judge

对于没有标准答案的开放性推理问题，比如分析类、写作类，这时候需要用另一个语言模型来当裁判。我们们的方法是把模型生成的 completion 和标准答案同时放进一个 prompt，比如：

```python
prompt = """
你是一个严格的评分裁判。请评估以下回答的推理质量。

问题：{question}

模型回答：{model_response}

请从以下维度打分（每项 0-10 分）：
1. 推理逻辑性：推理步骤是否清晰、合理、无跳跃
2. 答案准确性：最终答案是否正确或合理
3. 推理完整性：是否覆盖了问题的关键点
4. 自我纠错：是否有发现并纠正自己错误的迹象

只输出 JSON，格式：
{"logic": 8, "accuracy": 9, "completeness": 7, "self_correction": 5}
"""
```

我们对输出的逻辑性、准确度、完整性和自我纠错能力进行打分，然后赋予不同的权重得到最终评分。

## RL-Reasoning 实践

### 前情提要

这里我们采用 gsm8k-chinese 这个数据集来训练 qwen3-0.6b：

![image.png](http://img.xilyfe.top/img/20260310170648317.png)

### 训练代码

由于我们的 Outcome Reward 是判断输出是否等于答案，所以为了加快训练收敛加了一个 Digit Reward，如果输出是纯数字那么也给予奖励。

```python
import unsloth
import re

import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel


MODEL_DIR = r"D:\dev\llm\models\qwen3-0.6b"
DATASET_DIR = r"D:\dev\llm\datasets\GSM8K_zh.json"
OUTPUT_DIR = "./output/qwen3"


TAGS = ["<think>", "</think>", "<answer>", "/<answer>"]
SYSTEM_PROMPT = ""
CORRECT_REWARD = 2.0
DIGIT_REWARD = 0.5
TAG_REWARD = 0.125
HARD_FORMAT_REWARD = 2
SOFT_FORMAT_REWARD = 1


def preprocess_func(tokenizer):
    def inner(item):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question_zh"]}
            ],
            "answer": item["answer_only"]
        }

    return inner


def correct_reward(answer, completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    responses = [response.split("<answer>")[-1].split("</answer>")[0] for response in responses]
    return [CORRECT_REWARD if resp == ans else 0.0 for resp, ans in zip(responses, answer)]


def digit_reward(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    responses = [response.split("<answer>")[-1].split("</answer>")[0] for response in responses]
    return [DIGIT_REWARD if resp.isdigit() else 0.0 for resp in responses]


def hard_format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [HARD_FORMAT_REWARD if match else 0.0 for match in matches]


def soft_format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [SOFT_FORMAT_REWARD if match else 0.0 for match in matches]


def tag_reward(completions, **kwargs):
    def tag_num(text):
        reward = 0.0
        for tag in TAGS:
            if text.count(tag) == 1: reward += TAG_REWARD
        return reward

    responses = [completion[0]["content"] for completion in completions]
    return [tag_num(response) for response in responses]


if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files=DATASET_DIR, split="train")
    dataset = dataset.map(preprocess_func(tokenizer), remove_columns=dataset.column_names)

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=50,
        bf16=False,
        fp16=True,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=256,
        num_train_epochs=1,
        save_steps=200,
        max_grad_norm=0.1,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[
            correct_reward,
            digit_reward,
            hard_format_reward,
            soft_format_reward,
            tag_reward
        ]
    )
    trainer.train()

```

### 训练结果

![image.png](http://img.xilyfe.top/img/20260310172211204.png)

### 思考

#### 为什么 loss 初试为零，然后不降反升？

![image.png](http://img.xilyfe.top/img/20260310224319018.png)

这条 reply 很直接的指出了，由于 GRPO 的训练是 <mark>one exploration step per iteration</mark>，这导致新旧策略的 ratio 恒等于 1，GRPO 的公式就可以简化为：

$$
\mathcal{L}^{GRPO}(\theta) = - \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \hat{A}_{i,t} - \beta\, \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{ref}\right] \right)
$$

回忆 GRPO 里 Advantage 的定义 $\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}$，把 G 个回答的 Advantage 加起来求均值：

$$
\frac{1}{G}\sum_{i=1}^{G} \hat{A}_i = \frac{1}{G}\sum_{i=1}^{G} \frac{r_i - \mu_r}{\sigma_r} = \frac{1}{\sigma_r} \cdot \frac{1}{G}\sum_{i=1}^{G}(r_i - \mu_r)
$$

而 $\frac{1}{G}\sum(r_i - \mu_r)$ 就是每个值减去均值再求平均，这个结果永远等于 0，最终 GRPO 的损失就化为：

$$
\mathcal{L}^{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \beta\, \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{ref}\right] \right)
$$

在训练开始模型和 ref model 是一样的，所以我们得到的损失就是从 0 开始，并且随着训练，模型慢慢偏离 ref model 导致 KL 散度上升。或者换句话说：<mark>GRPO 的 loss 仅仅体现了 KL 散度，不能体现模型的训练效果</mark>。那这里又引出了一个问题：如果损失函数只和 KL 散度有关系，那么 GRPO 到底在干啥？

#### GRPO 的损失只与 KL 散度有关，那么它在干什么？

问为什么之前我们先看一下是不是，我们看一下 trl 库里面 GRPOTrainer 的源码：

![image.png|104](http://img.xilyfe.top/img/20260310234326912.png)

注释里面提到，如果我们按照 GRPO 原文中的方法采用 `num_iterations = 1`，那么 `old_per_token_logps = per_token_logps.detach()`，计算出来的 ratio 为 `torch.exp(per_token_logps - old_per_token_logps)` 就是恒等于 1。也就是说前面 issue 说的没错，在 GRPO 前向计算中损失函数确实只和 KL 散度有关。但是这里我们忽略了一个问题，<mark>损失函数的值只和 KL 散度有关，但是损失函数的梯度和 ratio 有关</mark>。

在 $\theta = \theta_{old}$ 处求梯度：

$$
= \frac{1}{G} \sum_{i = 1}^{G} \frac{1}{\left|\right. o_{i} \left|\right.} \sum_{t = 1}^{\left|\right. o_{i} \left|\right.} \left[\right. \hat{A}_{i , t} + \beta \left(\right. \frac{\pi_{r e f} \left(\right. o_{i , t} \left|\right. q , o_{i , < t} \left.\right)}{\pi_{\theta} \left(\right. o_{i , t} \left|\right. q , o_{i , < t} \left.\right)} - 1 \left.\right) \left]\right. \nabla_{\theta} log ⁡ \pi_{\theta} \left(\right. o_{i , t} \left|\right. q , o_{i , < t} \left.\right)
$$

我们先看看初始阶段的梯度情况，也就是当 $\pi_{\theta} = \pi_{ref}$ KL散度的梯度部分变为零。

$$
\nabla_{\theta} \mathcal{J}_{G R P O} \left(\right. \theta \left.\right) = \frac{1}{G} \sum_{i = 1}^{G} \frac{1}{\left|\right. o_{i} \left|\right.} \sum_{t = 1}^{\left|\right. o_{i} \left|\right.} \hat{A}_{i , t} \nabla_{\theta} log ⁡ \pi_{\theta} \left(\right. o_{i , t} \left|\right. q , o_{i , < t} \left.\right)
$$

虽然对 $\hat{A}_{i , t}$ 求和为零，但是 $r_{i,t}$ 的梯度为 $\nabla_\theta \log \pi_\theta$ 这是非零的，而且每个 token 的 $\nabla_\theta \log \pi_\theta$​ 都不一样，所以乘上 $\hat{A}_i$ 之后加权求和整体梯度不为 0。

>做个总结：GRPO 中采用 on-policy，新旧策略完全相同加上 Advantage 的组内均值恒为 0，导致 loss 初始值为 0。但是损失函数的数值只体现 KL，但优化靠的是梯度。即使 loss 数值只剩 KL 项，梯度依然能提供有效的策略更新信号。更新一步后，模型偏离 ref model → KL 上升 → loss 数值变大，但这正是预期的行为。

#### entropy 一直降低怎么解决？


#### format_reward 为什么一直为 0？