---
title: 分布式训练技术
date: 2026-02-02T19:50:43+08:00
featuredImage: http://img.xilyfe.top/img/20260202195145685.png
authors:
  - Xilyfe
series:
  - LLM
tags: []
lastmod: 2026-02-03T02:11:59+08:00
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

## Deepspeed ZeRO
