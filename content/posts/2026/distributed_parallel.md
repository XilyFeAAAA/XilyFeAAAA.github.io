---
title: 分布式训练技术
date: 2026-02-02T19:50:43+08:00
featuredImage: http://img.xilyfe.top/img/20260202195145685.png
authors:
  - Xilyfe
series:
  - LLM
tags: []
lastmod: 2026-02-06T12:12:22+08:00
---
我们先回忆一下传统的单机单卡训练模式：


![image.png](http://img.xilyfe.top/img/20260202195324965.png)

首先硬盘读取数据，CPU 处理数据，将数据组成一个 batch，再传入 GPU，网络前向传播算出 loss，再反向传播计算梯度，用梯度更新参数完成一次训练。这种传统模式在大参数量或者大数据量的情况下就容易陷入显存的瓶颈，于是就引出了多卡并行训练。

## DP


![image.png](http://img.xilyfe.top/img/20260202195640665.png)

DP(Data Parallel)，也就是数据并行。它的运行模式是：
1. 硬盘读取数据，由 CPU 处理之后，给每个 GPU 传不同的一部分 mini batch
2. 每个 GPU 各自计算进行前向传播，计算损失函数，然后反向传播计算梯度
3. 其它每个 GPU 都把梯度传给 GPU-0，计算全局平均梯度，然后更新自己的参数
4. 最后 GPU-0 把最新的参数传给其它 GPU，保证所有 GPU 上模型的一致。

Data Parallel 的问题在于数据的传输量太大了，并且都集中在 GPU-0 上压力太大了。假设参数量为 $\Psi$ 节点数量为 $N$，那么 GPU-0 需要传入梯度 $(N-1)\Psi$，传出参数量为 $(N-1)\Psi$；其他 GPU 传出梯度量为 $\Psi$ 传入参数为 $\Psi$。其次 DP 模型中给每个 GPU 分配一个线程，这就会出现 GIL 锁的问题。每个 GPU 所在线程进行前向计算或者反向传播时候，由于执行的是 CUDA 内核所以会解开 GIL 锁。但是 DP 里面还有很多纯 Python 层面的代码，例如模型复制 、输入切分、输出收集等等，这些操作由于 GIL 锁的存在就不能并行执行，效率很低。

## DDP

### Ring-AllReduce

![image.png](http://img.xilyfe.top/img/20260202231931700.png)




在介绍 DPP 之前，要先介绍一种集群通信方式 Ring-AllReduce：
1. 首先在第一个阶段 Scatter-Reduce，各个节点之间会相互传送部分信息，最终达到各个节点同步：
	1. 如图，第一步 GPU-0 将 $a_0$ 发给 GPU-1，GPU-1 将 $b_1$ 发给 GPU-2，GPU-2 把 $c_2$ 发给 GPU-0
	2. 第二步，GPU-0 把 $c_0+c_2$ 发给 GPU-1，GPU-1 把 $a_0+a_1$ 发给 GPU-2，GPU-2 把 $b_1+b2$ 发给 GPU-0
2. 第二个阶段是 AllGather：
	1. 此时 GPU-0/1/2 分别有了完整的 c/a/b 的信息，再进行两轮传播，三个 GPU 就掌握了全部信息。

>这样每个 GPU 都在同时发送和接受，最大限度的利用了每个显卡的上下行带宽。

### 训练过程

![image.png](http://img.xilyfe.top/img/20260202233932894.png)

首先 PyTorch 会把模型内的参数按照倒序排列（因为是反向传播求梯度，顺序和代码是相反的），然后将参数依次放在桶里。每个参数都会挂一个监听器，当参数求得梯度之后监听器被触发，此时检查桶内参数是不是全都计算好梯度了。如果梯度全部计算完成收集满一个桶，那么就用 Ring-AllReduce 对这个桶内参数的梯度进行同步。当全部桶都同步完整，各个 GPU 的模型就应该同步了，此时就可以调用优化器对参数进行更新。

>这里引入桶是因为，如果一个参数计算好梯度就同步，开销太大了。

假设参数量为 $\Psi$ 节点数量为 $N$，那么对于每个 GPU 有：
- Scatter-Reduce 阶段传入/传出：$(N-1)\frac{\Psi}{N}\approx\Psi$ 
- AllGather 阶段传入/传出：$(N-1)\frac{\Psi}{N}\approx\Psi$ 

回想之前三个 GPU，每个 GPU 有 a/b/c 三个参数。每个 GPU 都有 $\frac{\Psi}{N}$ 个块，每个块都会进行 $N-1$ 次传递，所以每个 GPU 总传入/传出约 $\Psi$ 次，AllGather 阶段同理，总大小约 $2\Psi$ 与集群大小无关。


### 代码实现

coding 之前我们必须先明确一个概念，不管我们是通过 `torch.multiprocessing.spawn` 还是 `torchrun` 来启动 DDP 分布式训练，脚本启动之后会 **复制出多个进程，每个进程都执行相同的代码**。

1. 在单卡训练的基础上我们需要额外导入三个模块：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
```

- `DistributedDataParallel`： PyTorch 中实现 DDP 机制的模块
- `DistributedSampler`： 用于把数据采样到不同的 GPU(进程) 上
- `torch.distributed`：提供分布式通信的基础模块（进程组初始化、rank 获取等）。

2. 进程组初始化

```python
def init_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size
```

首先我们配置每个进程都采用 NVIDIA 的 NCCL 通信库进行分布式通信（nccl 用 GPU 来通信，gloo 用 CPU 通信，相较而言 nccl 效率更高，但是 nccl 只适用于 Linux 环境），同时它会把当前的进程拉进同一个分布式通信组里。之后我们用 `torch.cuda.set_device` 帮我们把当前进程绑定到 GPU-rank 上，就不会出现多个进程共有 GPU 的情况。  

但是在多机多卡的情况下上述代码就会报错了，原因主要是我们搞混了 **rank 和 local_rank 的区别**。
rank 代表了自己是第几个进程，local_rank 代表了该占用哪张 GPU。在单机多卡的情况下，假如我们有四张卡，那么四个进程分别用 GPU-rank 是没问题的，例如 0 号进程就用 GPU-0。但是在多机多卡的情况下，假设你有两台机器，每台 4 张卡：
- 机器A：rank 0 1 2 3  
- 机器B：rank 4 5 6 7

这时机器 B 上的 rank-5 进程就不能绑定 GPU-5了，它绑定的应该是机器 B 上的 GPU-1，所以正确的通用代码为：

```python
def init_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(locak_rank)
    return rank, local_rank, world_size
```

3. 数据分布式采样

```python
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
```

这里有两个注意事项：
-  DataLoader 里面不需要再设置 `shuffle=True` 了，应该在每个 epoch 开始 `sampler.set_epoch(epoch)`。
-  DistributedSampler 不需要再手动填写 world_size 和 rank，我们在 `dist.init_process_group` 就已经注册了。

4. 用 DDP 包装模型

```python
model = model.cuda(local_rank)
model = DPP(model, device_ids=[local_rank])
```

这里用 DDP 把原始模型包装成分布式模型，这样每个进程都有完整模型副本，会自动在 backward() 时注册 hook，执行 AllReduce 操作，参数更新后，所有进程的模型保持一致。

5. 启动脚本

 单机多卡：
 
```lua
torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 train.py
```

多机多卡：

```lua
-- 机器 A

torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr=192.168.1.10 \
  --master_port=29500 \
  main.py

-- 机器 B
  
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=1 \
  --master_addr=192.168.1.10 \
  --master_port=29500 \
  main.py
```

- nnodes：机器数
- nproc_per_node：GPU 数量
- node_rank：当前机器是第几个节点
- master_addr：主节点IP  
- master_port： 通信端口

>torchrun 会自动计算 rank、local_rank 和 world_size 注入到环境变量里面，用 `os.environ` 就可以获取了。


### Accelerate

上面代码也可以看见 PyTorch 内置的 DDP 有点太麻烦了，我们可以用 Huggingface 封装的 accelerate 库替代，范例如下：

```python
from accelerate import Accelerator

def train():
	accelerator = Accelerator()
	model = ConvNet().cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
	train_dataset = Dataset(...)
	train_loader = Dataloader(train_dataset, batch_size, shuffle=False)
	train_loader, model, optimizer = accelector.prepare(train_loader, model, optimizer)
	
	for epoch in range(args.epoch):
		for x, y in train_loader:
			logits = model(x)
			loss = criterion(logits, y)
			accelector.backward(loss)
			optimizer.step()
			optimizer.zero_grad()
		
		if accelerator.is_main_process:
			print()

```

## DeepSpeed ZeRO

开始 DeepSpeed ZeRO 之前我们回顾一下 DP 和 DDP 优化了哪些地方。

- 首先 DP 采用单进程多线程运行，每个 GPU 独立计算梯度，汇总到 GPU-0 上平均梯度再更新参数。然后 GPU-0 再把更新后的参数分发出去。缺点由于 GIL 锁的问题，多个 GPU 通信是单进程无法利用 CPU 多核，其次通信量全部集中在 GPU-0 上压力大，最后通信量随着 GPU 数增加线性增长。
- DDP 采用 Ring-AllReduce 实现梯度同步。计算和同步并行，从输出层反向开始同步（离输出越近的参数梯度越先计算出来），为了降低频繁通信的开销，使用桶收集一定量梯度后同步一次，每个 GPU 的通行量约为 $2*\Psi$。

但是不管是 DP 还是 DDP，每个 GPU 都保存了完整的模型参数，中间激活值以及优化器状态，这里面优化器状态占用的显存最大，我们拿 AdamW 举例一共需要：
1. FP16 的参数、梯度（模型参数）
2. FP32 的梯度、一阶动量、二阶动量、Master Weight（优化器状态）

而 DeepSpeed ZeRO 的 ZeRO 含义是 Zero Redundancy Optimizer，其核心思想是 **消除冗余存储的优化器状态**，每个 GPU 中优化器状态是相同的，因此可以通过将优化器状态按离输出的位置关系进行分块，拆分到不同 GPU 上，实现零冗余。DeepSpeed 分为三个阶段，ZeRO-1 仅分区优化器状态，ZeRO-2 加入了梯度，ZeRO-3 加入了模型的参数。

### ZeRO-1


![image.png](http://img.xilyfe.top/img/20260204160715456.png)

我们从上图开始了解 DeepSpeed ZeRO-1。首先采样了不同的数据分配给每个 GPU，每个 GPU 保存了相同的 FP16 的参数和梯度（浅蓝色的两层），但是保存了不同分区的优化器状态（深蓝色区域）。假如我们有三个 GPU，模型一共有 9 层，那么 GPU-0 存前三层的优化器状态，GPU-1 存中间三层的优化器状态，GPU-2 存最后三层的优化器状态。训练开始：

1. 由于每个 GPU 都存储了完整的模型参数，所以可以分别独立的进行前向传播，计算得到 loss
2. 反向传播时，每个 GPU 都从后向前计算出每一层参数的梯度
3. 然后 GPU-0 和 GPU-1 分别将后三层计算出的梯度传给 GPU-2，GPU-2 计算得到后三层参数的平均梯度。同理，GPU-0 和 GPU-2 把中间三层计算的梯度传给 GPU-1，GPU-0 也是这样。最终三个 GPU 分别能够计算出自己对应层次的平均梯度。
4. 对 GPU-0 来说，它已经计算得到了前三层参数的梯度平均值，它又保存了前三层的优化器状态，所以它就可以将梯度转到 FP32 然后更新前三层优化器状态，再更新前三层的参数值，最后把更新后的前三层参数广播给其他两个 GPU。其他 GPU 也同样做上述操作将自己对应的参数更新了再广播给其他 GPU，使得不同 GPU 的模型最终保持一致。

假设参数量为 $\Psi$ 节点数量为 $N$，那么对于每个 GPU 有：
- 梯度收集阶段传入/传出：$(N-1)\frac{\Psi}{N}\approx\Psi$ 
- 参数广播阶段传入/传出：$(N-1)\frac{\Psi}{N}\approx\Psi$ 

最终总传入/传出参数量为 $2\Psi$ 和 DDP 通讯量相同。


![image.png](http://img.xilyfe.top/img/20260204164540785.png)

>上图可以看到 DeepSpeed ZeRO-1 通过将优化器状态分布在不同 GPU，大幅度降低了显存占用。


### ZeRO-2

![image.png](http://img.xilyfe.top/img/20260204170258315.png)


DeepSpeed ZeRO-2 相对于 ZeRO- 1 的核心优化在于进一步分区了梯度从而显著降低显存占用，想法很简单：每个 GPU 只负责更新对应的参数，那么只需要保存这部分参数的梯度就好了。训练过程如下：

1. 由于每个 GPU 都存储了完整的模型参数，所以可以分别独立的进行前向传播，计算得到 loss
2. 反向传播时，每个 GPU 都从后向前计算出每一层参数的梯度
3. 然后 GPU-0 和 GPU-1 计算出最后一层参数的梯度，它们会把这些梯度放到一个 bucket 里面，再传给 GPU-2。当 GPU-2 计算得到最后一层的平均梯度，GPU-0 和 GPU-1 就把这些梯度删除，因为不是自己需要的，以此减少了显存占用。倒数第二、三层也是如何，计算得到梯度再传给 GPU-2 计算平均梯度，然后自己再把不需要的这部分梯度删除。而 GPU-2 得到了后三层平均梯度，就可以更新自己对应的优化器状态，再更新参数。
4. 最后三个 GPU 再分别传递自己更新好的参数，使得每个 GPU 上的模型保存一致。

### ZeRO-3

![image.png](http://img.xilyfe.top/img/20260204173843936.png)

DeepSpeed ZeRO-3 又进一步分区了模型的参数，在前向传播时候通过其他 GPU 来广播自己所缺的那一部分参数。

假设参数量为 $\Psi$ 节点数量为 $N$，那么对于每个 GPU 有：
- 梯度收集阶段传入/传出：$(N-1)\frac{\Psi}{N}\approx\Psi$ 
- 参数广播阶段传入/传出：$2*(N-1)\frac{\Psi}{N}\approx2\Psi$ 

### 代码实现

第一个方法是手动创建 DataLoader + DistributedSampler，这样可以自己管理数据分片。

```python
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl")

def train_custom(epochs: int):
    model = Net().cuda()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model_engine, optimizer, *_ = deepspeed.initialize(
        model=model, optimizer=optimizer, config_json="deepspeed_config.json"
    )

    dataset = NetDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    criterion = MSELoss()
    model_engine.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)

        for i, (x, y) in enumerate(dataloader):
            inputs = x.to(model_engine.local_rank)
            labels = y.to(model_engine.local_rank)
            output = model_engine(inputs)
            loss = criterion(output, labels)
            model_engine.backward(loss)
            model_engine.step()
```

这种方法自己通过 `DistributedSampler` 对数据进行分布，根据 distributed 通讯组的信息(local_rank) 就能进行采样了，这种方法好处在于可以完全控制 dataloader，可加 collate_fn、自定义 sampler 等。而 `deepspeed.initialize` 会创建 DeepSpeedEngine 实例代替原来的 Module，负责优化器状态、梯度等信息的传递。


另一种方法是完全由 DeepSpeed 模块来自动处理数据分片和 shuffle，这里我们没有手动定义 optimizer，而是在 DeepSpeed 的配置文件里面填写优化器信息，然后由 `deepspeed.initialize` 方法直接生成：

```python
def train_default(epochs: int):
    model = Net().cuda()
    dataset = NetDataset()
    criterion = MSELoss()
    model_engine, optimizer, training_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=dataset,
        config="ds_config.json",
    )

    model_engine.train()
    for epoch in range(epochs):
        for x, y in training_dataloader:
            inputs = x.to(model_engine.local_rank)
            labels = y.to(model_engine.local_rank)
            output = model_engine(inputs)
            loss = criterion(output, labels)
            model_engine.backward(loss)
            model_engine.step()
```

>注意不管是哪种写法都不需要手动添加 `model.zero_grad()` ，因为 DeepSpeedEngine 已经接管了整个流程。它可以结合梯度累计的参数，自动进行 `engine.step()`。手动 zero_grad 会破坏 ZeRO 的分区状态，因为有可能梯度还没传播，其他 GPU 的分区还没有同步。


### 配置文件

#### ZeRO-2

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

大部分参数看名字就能意会了，这里介绍几个待情况而定的参数，具体可以参考官网：
1. `contiguous_gradients`：是否在内存中存储连续梯度，开启后可减少内存碎片化提高通信效率，但是增加显存开销。
2. `reduce_bucket_size`：图例中 bucket 的大小，若显存不足，减小值至 1e5 或 5e5，如果通信瓶颈明显，可适当增大值。
3. `offload_optimizer`：如果显存不足可以把优化器状态存在 CPU 上，只有 ZeRO-3 可以用。
