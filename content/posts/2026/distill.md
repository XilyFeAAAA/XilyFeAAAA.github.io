---
title: 大模型知识蒸馏
date: 2026-03-05T19:44:00+08:00
featuredImage: http://img.xilyfe.top/img/20260305194531119.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 大模型
lastmod: 2026-03-10T04:42:54+08:00
---
>模型蒸馏即知识蒸馏，是一种模型压缩和加速技术。在深度学习中，大型深度神经网络虽性能优异，但因计算复杂度高、存储需求大，难以部署在资源受限设备上。模型蒸馏通过构建师生架构，让小的学生模型学习大的教师模型的知识，使学生模型在保持较小规模的同时，尽可能接近教师模型的性能。

## 介绍

　大模型本质是大量的矩阵运算，想要提高效率，就要想办法提升矩阵运算的效率，大致的思路如下：
- 知识蒸馏：大模型去掉“水分”，保留“精华”后得到小模型
- 模型剪枝：矩阵中某些元素毫无卵用，留着纯属“占着茅坑不拉屎”
- 模型量化：FP32、FP16用INT8、INT4替代，减少存储和计算
- 参数共享：部分层级之间共享参数，减少存储空间，提升计算效率
- 低秩分解：把大矩阵分解成low -rank 小矩阵，减少存储空间，提升计算效率
- 参数搜索：使用算法或启发式方法来确定最佳的参数配置

 这么多方法，相比之下知识蒸馏是比较流行的，效果也是比较好的。模型的知识蒸馏分为两种：白盒蒸馏和黑盒蒸馏，两种蒸馏方式的区别是 <mark>对教师模型的访问权限不同</mark>。

白盒蒸馏中学生模型可以访问教师模型**内部信息**，包括 logits、hidden states 等，蒸馏方式就是让学生的 logits 尽量拟合教师的 logits 分布。白盒蒸馏的信息量最大，蒸馏效果最好，收敛更稳定，但缺点是教师模型必须是开源的模型，我们才访问教师模型内部的 logits 等信息。白盒蒸馏的实质是 <mark>用 KL 散度尽可能让学生模型贴近教师模型</mark>

黑盒蒸馏中学生模型只能访问 **教师模型的输出**，我们通常是准备好一批 prompt 传给大模型产商的 API，让学生模型的输出标签拟合教师模型的输出。黑盒蒸馏的实质是 <mark>通过教授模型得到高质量的数据集，然后通过 SFT 或者 RLHF 的方式进行微调</mark>，本质是 <mark>数据蒸馏+模仿学习</mark>。


## 白盒蒸馏

传统的模型训练过程中，训练语料中的目标 token 会被表示为 one-hot 向量 作为 ground truth。模型经过 softmax 得到预测概率分布后，与 ground truth 计算交叉熵，再通过反向传播更新模型参数。由于 ground truth 是 one-hot 形式，即正确 token 的概率为 1，其余 token 的概率为 0，这种训练目标被称为 <mark>hard target</mark>。目前主流的大模型本身也是通过这种 hard target 的方式训练得到的。

然而，hard target 也存在一定的问题。由于 one-hot 分布非常极端，它只强调“正确类别”，完全忽略了不同 token 之间的相似关系，因此小模型在学习时往往难以捕捉到更丰富的类别信息。相比之下，teacher 模型在 decoder 输出的是一个完整的概率分布，例如 `[0.1, 0.6, 0.05, 0.15, 0.04, 0.06]`。这种概率分布被称为 <mark>soft target</mark>。与 one-hot 的 hard target 不同，soft target 能够反映不同 token 之间的相对关系，使 student 模型不仅学习正确答案，还能学习其他候选 token 的概率结构，从而获得更丰富的特征信息。因此，利用 soft target 进行训练通常可以提升模型的泛化能力和鲁棒性。

![image.png](http://img.xilyfe.top/img/20260310163242529.png)

白盒知识蒸馏的全流程如下：
- teacher 模型对 input 做 feed forward 计算，得到的结果经过 softmax(t) 后得到 soft target；
- student 模型同样对 input 做 feed forward 计算，然后分叉：
    - 和 teacher 一样，得到的结果经过 softmax(t) 后得到 soft predictions；
    - 设置 `T=1`，和原来的 softmax 效果一样，得到 hard predictions；
- soft target 和 soft predictions，**KL 散度** 用于衡量 teacher 和 student 之间的差异
-  hard target 和 hard prediction，**交叉熵损失** 用于衡量 student 和 ground truth 之间的差异

>如果仅仅利用 KL 散度进行训练，要是教师模型在某个样本上预测错了且概率分布非常自信，那么仅靠 soft targets 学生模型会完美地继承这个错误。hard targets 确保即使老师带歪了路，学生依然能通过真实标签发现错误。其次 hard targets 是 one-hot 分布，它是极度陡峭能提供非常明确、强烈的梯度信号，迫使模型快速向正确方向靠拢。

其次我们聊一下白盒蒸馏中的 KL 散度，它衡量两个概率分布 $q$ 和 $p$ 的差异：

$$
\text{KL}(p \parallel q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim q}\left[ \log \frac{p(x)}{q(x)} \right]
$$

在 RLHF 中我们学习的 KL 散度是前向 KL 散度（Forward KL Divergence），也就是上面的公式，我们代入 LLM 的语境下是：

$$
\mathcal{L}_{\text{FKL}} = \text{KL}(p_{\text{teacher}} \parallel p_{\text{student}}) = \mathbb{E}_{y \sim p_{\text{teacher}}}\left[ \log \frac{p_{\text{teacher}}(y|x)}{p_{\text{student}}(y|x)} \right]
$$

![image.png](http://img.xilyfe.top/img/20260306123647680.png)

minillm 论文中提出了这么一个问题：前向 KL 散度可能会使学生模型高估教师模型中概率比较低的位置。结合公式来看，当 $p$ 增大时，为了使得 KL 散度减小，则 $q$ 也需要增大。但是当 $p$ 趋于 0 时，无论 $q$ 取任何值，KL 散度都比较小。也就是说当 $p$ 趋于 0 的时候，$p$ 主导了 KL 散度的大小，<mark>这样就起不到优化 q 分布的效果</mark>，可能会使 $q$ 分布高估概率低的位置，对应图片里面橙色虚线的部分。

$$
\mathcal{L}_{\text{RKL}} = \text{KL}(p_{\text{student}} \parallel p_{\text{teacher}}) = \mathbb{E}_{y \sim p_{\text{student}}}\left[ \log \frac{p_{\text{student}}(y|x)}{p_{\text{teacher}}(y|x)} \right]
$$

相对于前向 KL 散度这种 mean-seeking，反向 KL 散度是一种 mode-seeking，学生只关注教师高概率的主模态，把低概率区域压得很低，因此可以减少幻觉，提高校准度和长文本连贯性。

>这里需要注意，PPO 中的 KL 散度是策略约束项，所以它的 KL 散度不需要整个 vocab 的概率分，只在 labels 上 gather 一下 log_prob 就够了。白盒蒸馏的 KL 散度 <mark>必须用整个 vocab 的概率分布</mark>，它的目的是让学生模仿老师。

```python
class DistillTrainer:
    def __init__(
            self,
            student_model: nn.Module,
            teacher_model: nn.Module,
            tokenizer: TokenizerType,
            config: DistillConfig,
            train_dataset: Dataset
    ):
        self.student_model = student_model.to(config.device)
        self.teacher_model = teacher_model.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = AdamW(params=student_model.parameters(), lr=config.lr)

        self.dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=config.batch_size
        )

    def compute_loss(self, exp: dict) -> torch.Tensor:
        teacher_logits = exp["teacher_logits"]
        student_logits = exp["student_logits"]
        labels = exp["labels"]
        mask = exp["mask"].long()
        T = self.config.T

        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_probs = F.softmax(student_logits / T, dim=-1)

        # [b, l, v]
        kl = student_probs * (student_probs.log() - teacher_probs.log()).sum(dim=-1)
        distill_loss = (kl * mask).sum() / (mask.sum() + 1e-8)
        distill_loss = distill_loss * (T ** 2)

        # ce loss
        ce_loss = F.cross_entropy(
            input=student_logits.view(-1, student_logits.size(-1)),
            target=labels.view(-1),
            ignore_index=-100
        )

        return self.config.alpha * ce_loss + (1 - self.config.alpha) * distill_loss

    def get_experience(self, batch):
        self.tokenizer.padding_side = "right"
        prompt_inputs = self.tokenizer.apply_chat_template(
            [
                [{"role": "user", "content": prompt}]
                for prompt in batch["prompts"]
            ],
            add_generation_prompt=True,
            tokenizer=True,
            return_tensors="pt",
            padding=True,
            max_length=self.config.max_length
        ).input_ids

        # [bs]
        prompt_len = prompt_inputs["attention_mask"].sum(dim=1)

        seq_inputs = self.tokenizer.apply_chat_template(
            [
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": label}]
                for prompt, label in zip(batch["prompts"], batch["labels"])
            ],
            add_generation_prompt=False,
            tokenizer=True,
            return_tensors="pt",
            padding=True,
            max_length=self.config.max_length
        )
        input_ids = seq_inputs["input_ids"]
        attention_mask = seq_inputs["attention_mask"]

        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # prompt_mask 用来遮住 sequence 里面 prompt 的部分
        seq_range = torch.arange(self.config.max_length).unsqueeze(0)
        prompt_mask = seq_range < prompt_len.unsqueeze(1)
        # distill_mask 用来遮住 sequence 里面非 response 的部分，包括 prompt 和 padding
        distill_mask = (~prompt_mask) & attention_mask.bool()
        labels.masked_fill_(prompt_mask, -100)

        with torch.no_grad():
            teacher_logits = self.teacher_model(input_ids, attention_mask=attention_mask).logits
        student_logits = self.student_model(input_ids, attention_mask=attention_mask).logits

        return {
            "teacher_logits": teacher_logits[:, :-1],
            "student_logits": student_logits[:, :-1],
            "labels": labels[:, 1:],
            "mask": distill_mask[:, 1:]
        }
```

为了提升泛化性和鲁棒性，对于负类不能像 one-hot 编码那样赶尽杀绝，需要适当给予一些概率，所以我们用 tempareture 调节概率的平滑性：

$$
p_i = \frac{\exp(\frac{z_i}{T})}{\sum_j \exp(\frac{z_j}{T})}
$$

## 黑盒蒸馏

黑盒蒸馏与白盒蒸馏不同，它不用访问模型内部的信息例如 logits等，它的整体流程大致为：
1. 准备多样化的 prompts
2. 把 prompts 送入教师 API，获得数据集
3. 数据清理：去重和去除低质量数据
4. 用 SFT 或者 RLHF 进行微调

```python
import openai
import json

client = openai.OpenAI(api_key="your_key")

def generate_data(prompts):
    dataset = []
    for p in prompts:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"请详细回答并展示思考过程：{p}"}]
        )
        dataset.append({"instruction": p, "output": response.choices[0].message.content})
    with open("synthetic_data.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    return dataset

# 后续：用HF TRL库或axolotl对学生SFT
# trainer = SFTTrainer(model=student, train_dataset=dataset, ...)
```