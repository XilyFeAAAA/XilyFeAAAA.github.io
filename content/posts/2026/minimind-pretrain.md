---
title: MiniMind 学习指北(三)：预训练
date: 2026-01-22T13:47:19+08:00
featuredImage: http://img.xilyfe.top/img/20260122134824760.png
authors:
  - Xilyfe
series:
  - minimind
tags:
  - 大模型
  - 深度学习
lastmod: 2026-02-10T02:28:23+08:00
---
## 数据集

>预训练我们采用的是 Teacher-Forcing，所以需要的数据格式应该是偏移的 `input_ids` 和 `labels`。

`__getitem__` 方法有几个需要注意的地方：
1. 在 input_ids 前后加上 bos 和 eos 两个 special token 可以帮助模型理解，句子该从哪里开始，在什么时候结束，不会在长文本胡言乱语。
2. 由于我们在模型的 `forward` 里面规定了：训练模式下将 logits 和 labels 进行偏移，所以在 Dataset 里面返回的 x 和 y 就不用额外的进行 shift 了。
3. padding 补充的 token 不参与 loss 计算，所以在 labels 里面把这部分 token 的 id 设为 -100，和 `F.cross_entropy(...,ignore_index=-100)` 一致。

```python
class PreTrainDataset(Dataset):
    def __init__(
        self, tokenizer, datapath: Union[str, PathLike[str]], max_length: int = 512
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(datapath)

    def load_data(self, datapath: Union[str, PathLike[str]]) -> list[str]:
        samples = []
        with open(datapath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(data)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> list[int]:
        sample = self.samples[index]
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length - 2,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = torch.tensor(
            [self.tokenizer.bos_token_id]
            + encoding["input_ids"]
            + [self.tokenizer.eos_token.id],
            dtype=torch.long,
        )
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels

```


预训练的代码中包含了很多 tricks，这里先放上源码：

```python
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=340)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=16)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_weight", type=str, default="pretrain")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--from_resume", type=int, default=0)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_test.jsonl")
    parser.add_argument("--from_weight", type=str, default="none")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser.parse_args()


def train_epoch(
    model: MiniMindForCausalLM,
    tokenizer: Tokenizer,
    epoch: int,
    loader: DataLoader,
    iters: int,
    sta_step: int = 0,
    wandb=None,
) -> None:
    start_time = time.time()
    for step, (input_ids, labels) in tqdm(enumerate(loader, start=sta_step), total=iters, desc=f"Epoch {epoch+1}"):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = input_ids != tokenizer.pad_token_id
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with context:
            res = model(input_ids, attention_mask, labels)
            loss = res.loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            tqdm.write(
                f"Epoch:[{epoch+1}/{args.epochs}]({step+1}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.3f}min"
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        if step % args.save_interval == 0 or step == iters - 1:
            model.eval()
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
            )
            model.train()

        del input_ids, labels, res, loss


if __name__ == "__main__":
    args = get_args()

    # 随机种子
    seed_everything(args.seed)

    # 配置目录、模型参数
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume
        else None
    )

    # 混合精度
    device = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    context = (
        nullcontext() if device == "cpu" else torch.amp.autocast(device, dtype=dtype)
    )

    # wandb
    wandb = None
    if args.use_wandb:
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.lr}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # model & tokenzier
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        print("torch.compile enabled")
    train_ds = PreTrainDataset(tokenizer, args.data_path, args.max_length)
    scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    assert tokenizer.pad_token_id is not None

    # recovery
    sta_epoch, sta_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        sta_epoch = ckp_data["epoch"]
        sta_step = ckp_data["step"]

    # 开始训练
    for epoch in range(sta_epoch, args.epochs):
        seed_everything(args.seed + epoch)
        loader = DataLoader(
            train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        
        it = iter(loader)
        if epoch == sta_epoch and sta_step > 0:
            for _ in range(sta_step):
                next(it)

            print(f"跳过前 {sta_step} 个 step。")
            train_epoch(
                model,
                tokenizer,
                epoch,
                it,
                iters=len(loader),
                sta_step=sta_step,
                wandb=wandb,
            )
        else:
            train_epoch(
                model, tokenizer, epoch, it, iters=len(loader), sta_step=0, wandb=wandb
            )
```

## ckp & resume

大型模型的训练耗时非常久，难免会出现间断的情况，所以我们需要定时将模型当前的状态进行保存，并且可以恢复继续训练，类似“断点续传”。

- checkpoint：通常只包含模型的权重 `model.state_dict()`，用于推理或者作为预训练权重被别人加载
- resume：通常包含训练的全部信息，包含模型权重、optimizer 权重、epoch、step 等等，用来恢复间断的训练

```python
state_dict = model.state_dict()
state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
ckp_tmp = ckp_path + ".tmp"
torch.save(state_dict, ckp_tmp)
os.replace(ckp_tmp, ckp_path)
```

这里用到工程化的设计，先把 ckp 的数据写到临时文件里，然后再把临时文件转存到目标路径，可以避免意外情况下留下半个有问题的 checkpoint。

```python
resume_data = {
    "model": state_dict,
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "step": step,
    "wandb_id": wandb_id,
}
for key, value in kwargs.items():
    if value is not None:
        if hasattr(value, "state_dict"):
	        resume_data[key] = value.state_dict()
        else:
            resume_data[key] = value
```

前面说到 resume 里面需要保存完整的训练环境，除了前面 checkpoint 里面保存的模型权重，还需要优化器权重等等。

---

等到训练的时候就需要根据 args 判断是不是需要读取 resume 继续训练：

```python
ckp_data = (
    lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
    if args.from_resume
    else None
)
sta_epoch, sta_step = 0, 0
if ckp_data:
    model.load_state_dict(ckp_data["model"])
    optimizer.load_state_dict(ckp_data["optimizer"])
    scaler.load_state_dict(ckp_data["scaler"])
    sta_epoch = ckp_data["epoch"]
    sta_step = ckp_data["step"]
```

每隔 `args.save_interval` 或者每个 batch 结束，就需要保存当前环境：

```python
if step % args.save_interval == 0 or step == iters - 1:
    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        step=step,
        wandb=wandb,
        save_dir="../checkpoints",
	)
```

## 混合精度训练

>在 cs224n 的帖子中，我已经详细记录了混合精度训练的知识点，具体可见 [[cs224n-lecture12]]。

简单来说，大模型的参数如果都用 FP32 来提高计算精度，可能导致显存不够出现内存溢出。如果浮点数精度用 FP16，性能不会大幅度下降，但是存在参数超出 FP16 表示范围的问题。解决方式是：一方面分情况使用 FP32 或者 FP16（在容易导致溢出的操作使用 FP16）；一方面在反向传播时候把 loss 扩大，避免参数小于 FP16 的表示范围，然后在更新参数之前再把梯度缩小。

具体训练中我们一般遵循下面框架：
```python
scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16"))

with torch.amp.autocast(device, dtype=dtype):
	loss = model(input_ids, attention_mask, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

首先说说 `torch.amp.autocast` 这个上下文管理器，它在前向传播也就是计算 loss 的过程中，判断哪些地方需要 FP16 哪些地方需要 FP32。例如 matmul / conv / attention 这些不容易导致参数计算溢出的地方，就使用 FP16，在 softmax / norm / reduction 这些地方就采用 FP32。

并且我们需要注意，`torch.amp.autocast` 控制的是“算的时候用什么”，不是“存的时候用什么”。也就是说模型存储的参数，其类型就是我们代码里面预先规定的。而前向计算时候碰到 softmax 这样的操作，它们会被临时 cast 成 FP16，然后用 FP16 的 kernel 跑；如果碰到矩阵乘法，参与的 tensor 就会被 cast 为 FP32，然后用 FP32 的 matmul kernel 处理。

>既然没有改变全部参数存储的类型，比如规定 FP32 存储还是 FP32，那如何解决占用显存大的问题？在绝大多数现代模型里：参数 + optimizer state ≈ 30–40%，而激活值（activations）+ 中间结果 ≈ 50–70%，所以 AMP 能大幅度减小峰值显存占用。

在反向传播的时候 backward kernel 被调用，它只能用 forward 保存下来的 dtype，对于 softmax 这些操作 backward 自然也是 FP16。

`torch.amp.GradScaler` 是为了解决反向传播更容易梯度消失的问题 - 求偏导更容易导致参数或者梯度超过 FP16 的精度。`scaler.scale(loss).backward()` 可以在反向传播之前把 loss 乘上一定倍数，这样放大的 loss 求偏导就不容易梯度消失，最后 `scaler.step(optimizer)` 会先 unscale 梯度，然后检查 inf / nan，如果安全则 `optimizer.step()` 更新参数。

## 余弦衰退学习率

$$
\frac{\text{lr}}{10} + 0.5 \times \text{lr} \times (1+\cos(\pi \times \frac{\text{cnt\_step}}{\text{total\_step}}))
$$
根据余弦函数的性质，我们可以看出 lr 呈逐渐减小的趋势。前期提高模型学习率，后期避免在最优点附近震荡所有减小学习率。

## 梯度累积

我们都知道，深度学习里面相对大的 batch_size 比小 batch_size 训练，loss 曲线更平滑噪声更小。但是对于参数量如此庞大的大模型，采用大 batch_size 可能会爆显存。而梯度累积则是一种 quick fix，它在时间维度上模拟更大的 batch，多次前向 + 反向传播只更新一次参数。

具体的做法如下：

```python
for step, (x, y) in enumerate(dataloader):
	logits = model(x)
	loss = loss_fn(y, logits) / step_accumulations
	loss.backward()
	
	if (step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

>注意：当我们执行 backward()，会将小 batch 的平均梯度添加到参数的 .grad 上。假如我们没有让 `loss /= step_accumulations`，那么经过 k 轮我们会得到小 batch 平均梯度的 k倍，而不是大 batch 的平均梯度。

## 梯度裁剪

梯度裁剪是为了解决：在反向传播时梯度是链式相乘的，深层模型里可能出现 loss 或者梯度达到天文数字超过范围的情况。它可以在 `optimizer.step()` 之前，将梯度缩小。

在混合精度中，需要用 scaler 将梯度先缩放回去在计算梯度的范数：
```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```


## 训练

继续我们模型、数据和预训练的脚步都准备好了，那肯定得跑一轮看看我们的模型是啥样了，于是我在 AutoDL 租了一张 RTX 5090 跑了两个小时。

![image.png](http://img.xilyfe.top/img/20260210143025198.png)

这里我按照 MiniMind 推荐的参数 `hidden_size=768, num_hidden_layers=16`：104M 的模型，跑了两个 epoch 最终 loss 稳定在 1.6 上下我就不浪费马内了。

{{< admonition type=question title="为什么 LLM 预训练通常是一到两个 epoch？">}} 
1. 现代大模型不同于以前的深度学习，训练使用万亿级 token 的大规模语料，所以如果和以前 DL 一样用几十上百个 epoch 进行训练一定会过拟合。
2. 模型参数巨大，训练的成本很高
{{< /admonition >}}