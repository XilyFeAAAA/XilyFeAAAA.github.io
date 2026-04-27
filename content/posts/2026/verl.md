---
title: 从零开始学 verl 框架
date: 2026-04-16T11:22:47+08:00
featuredImage: http://img.xilyfe.top/img/20260416112324552.png
authors:
  - Xilyfe
series:
  - RLHF
tags: []
lastmod: 2026-04-21T04:42:38+08:00
---
{{< admonition type=info title="Summary">}} 
这篇文章首先按照以下顺序展开：
1. Background 讲解，formulate 一下 verl 解决什么问题。
2. WalkThrough 部分，以 debugger 的视角从 entrypoint 开始看看程序在干什么，理解 verl 一次运行的行为。
3. 最后是讲解 verl 中 SPMD 这个并行计算模式。
{{< /admonition >}}

## 1. Prerequisite Knowledge

### 1.1 Hydra

Hydra 是一个**配置管理 + 实验管理**的 Python 框架，可以理解成加强版的 argparse。核心在于通过 YAML 文件进行参数配置，配置会被解析为 `DictConfig` 对象注入到函数中。

```python
"""
- trainer
  - config
    - ppo_trainer.yaml
  - main_ppo.py
"""
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
	# ...
    run_ppo(config)
```

### 1.2 Ray

Ray 是一个分布式框架。先看最基础的用法：

```python
@ray.remote
class Accumulator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x

    def get_value(self):
        return self.value
```

用 `@ray.remote` 装饰后，通过 `Accumulator.remote()` 获取的实例是一个 `ray.actor.ActorHandle`。调用方法时使用 `.remote()` 异步执行，返回的是 `ObjectRef`（类似 asyncio 的 Future），通过 `ray.get()` 阻塞取值：

```python
accumulator = Accumulator.remote()
value_ref = accumulator.get_value.remote()
# ObjectRef(16310a0f0a45af5c9e4edcae28cff4ef250feac40100000001000000)
value = ray.get(value_ref)  
# 0
```

verl 在 Ray 之上做了进一步封装，引入了以下几个概念：
- **Worker**：继承 `Worker` 基类的类，实例化后自动持有 `rank`、`world_size` 等环境变量，类似 `torchrun` 启动的进程。
- **ResourcePool**：管理可用的 GPU 资源。
- **WorkerGroup**：管理一组 Worker，是 SPMD 的执行单元。主进程通过 WorkerGroup 统一调度。

```python
@ray.remote
class GPUAccumulator(Worker):
    def __init__(self):
        super().__init__()
        self.value = torch.zeros(1, device="cuda") + self.rank

    @register(Dispatch.ONE_TO_ALL)
    def add(self, x):
        self.value += x
        return self.value.cpu()

resource_pool = RayResourcePool([1], use_gpu=True)
class_with_args = RayClassWithInitArgs(cls=GPUAccumulator)
worker_group = RayWorkerGroup(resource_pool, class_with_args)
print(worker_group.add(x=10))
```

`@register` 装饰器将 Worker 的方法注册到 WorkerGroup 上，之后直接通过 `worker_group.add(x=10)` 调用即可，无需手动管理 `.remote()`，至于 `Dispatch.ONE_TO_ALL` 这个参数在后面会具体说明。

### 1.3 DataProto

verl 把训练过程中需要的全部数据都存在 DataProto 这个数据结构里面，它会在不同的 worker 之间流转。

```python
class DataProto:
    batch: TensorDict          # tensor 类型的字段（input_ids, attention_mask 等）
    non_tensor_batch: dict     # 非 tensor 字段（data_source, ground_truth 等）
    meta_info: dict            # 元信息（seqlen 等）
```

## 2. Background

### 2.1 RLHF 的数据流复杂性

![](http://img.xilyfe.top/img/20260416135835748.png)

LLM 后训练的强化学习流程可以被定义为一个 DataFlowGraph，涉及：

- **多个模型角色**：

| 模型                     | 作用                    | 计算模式     |
| ---------------------- | --------------------- | -------- |
| Actor（策略模型）            | 生成回答（rollout），然后被训练更新 | 既要推理又要训练 |
| Critic（价值模型）           | 评估状态价值，辅助计算优势函数       | 既要推理又要训练 |
| Reference Policy（参考模型） | 计算 KL 散度，防止 Actor 偏离  | 只推理，冻结参数 |
| Reward Model（奖励模型）     | 给 Actor 的回答打分         | 只推理，冻结参数 |

- **多个阶段**：rollout（生成）、preparing experience（准备经验）、training（训练）
- **多种 Workload**：自回归生成、前向推理、梯度更新

### 2.2 推理与训练的并行策略冲突

我们回忆一下之前写过的 PPO Trainer：

```python
def train(self) -> None:
    """
    标准 RLHF PPO pipeline
    1. 用 actor 根据 prompt 生成回答
    2. 用 actor/ref 计算每个 token 的 logprobs
    3. 用 reward model 给整条回复打分
    4. 用 critic 估计 value 并计算优势（GAE）
    5. 用 PPO 更新 actor 和 critic
    """
    for step, batch in enumerate(self.dataloader):
        # 生成样本
        samples = self.get_sample(batch)
        # 生成经验，包括 rollout 和 evaluation 阶段的 adv rewards 等等
        experience = self.get_experience(samples)
        torch.cuda.empty_cache()
        # 每个 prompt 都要进行多轮训练
        for epoch in range(self.config.ppo_epochs):
            self.train_step(experience)
```

可以看到 `get_sample` 阶段就处于推理模式，得到需要的 reward、advantage 等信息后就会通过 `train_step` 进行训练，也就是对应推理模式。所以 Actor 和 Critic 需要在两种模式间频繁切换，而这两种模式对并行策略的需求截然相反：
- **训练模式**（FSDP/Megatron）：计算密集，需要较大的 TP/PP 来加速单步计算。
- **推理模式**（vLLM）：内存受限，需要较大的数据并行（DP）和较小的 TP/PP，以支持大 batch 和 KV Cache。

传统解决方案存在明显缺陷：
1. **拆分资源**：将 GPU 分为推理组和训练组，导致任一阶段都有大量 GPU 空闲，利用率极低。
2. **复制权重**：内存中同时保存两份模型分片，内存冗余严重；每次切换需要全集群 all-gather 进行参数重分片，通信开销巨大。

> **补充：为什么 vLLM rollout 后不能直接复用它的 logprobs？**
> 
> vLLM 在 logit 计算、采样、数值精度上与 FSDP 训练引擎存在细微差异。即使模型权重完全相同，vLLM 算出的 logprobs 与 FSDP 算出的也可能有明显偏差，这会隐式引入 off-policy 问题，导致梯度估计不准，训练不稳定甚至崩溃。因此 verl 的做法是：通过 vLLM 进行 rollout **只取生成的 tokens**，然后切换到 FSDP 训练引擎重新前向计算一次 logprobs。

### 2.3 HybridFlow

verl 采用的 HybridFlow 核心思想是**同一组 GPU 分时复用**，通过零冗余参数重分片技术让训练和推理阶段共享同一份参数。当 Actor 从 Generation 切换到 Training 时，并行策略可以无缝切换，例如从 `DP=8, TP=1` 变为 `DP=2, TP=4`，无须复制权重、无须拆分资源，做到资源利用率最大化。顶层采用 **single-controller** 设计，用户只需声明数据流图，底层的并行策略和分布式通信全部由 verl 处理：

```
定义: Actor rollout → ref/reward inference → critic inference → critic training → actor training
执行: verl 自动调度，分配 GPU，切换并行模式
```

## 3. WalkThrough

![image.png](http://img.xilyfe.top/img/20260421103249755.png)



我们通过 debugger 视角一步步看看 verl 的数据流是怎么走的。

### 3.1 Entrypoint

我们通过 `verl.main_ppo.py` 启动：

```python
# stage1 
if __name__ == '__main__':
    main()

# stage2
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)

# stage3
def run_ppo(config, compute_score=None):
	# ...
    ray.get(main_task.remote(config, compute_score))

# stage4
@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
	global_pool_id = "global_pool"
	resource_pool_spec = {
	  global_pool_id: ([config.trainer.n_gpus_per_node] * config.trainer.nnodes),
	}
	mapping = {
	  Role.ActorRollout: global_pool_id, Role.Critic: global_pool_id,
	  Role.RefPolicy: global_pool_id, Role.RewardModel: global_pool_id,
	}
	resource_pool_manager = ResourcePoolManager(
	  resource_pool_spec=resource_pool_spec, mapping=mapping)
	# ...
	trainer = RayPPOTrainer(
		config=config, 
	    resource_pool_manager=resource_pool_manager, # ...
	)
	trainer.init_workers()
	trainer.fit()
```

默认实现将所有 GPU 放入同一个资源池，各 workload 共享全部资源，因此各阶段**串行执行**。在大多数情况下这是效率最高的模式，因为同一时刻总有一个 workload 在满负荷使用所有 GPU。

### 3.2 Trainer init

`main_ppo.py` 里面主要干了 `trainer.init_workers` 和 `trainer.fit` 两件事。前者主要负责 worker group 的资源分配：

```python
def init_workers(self):
	all_wg = {}
	self.wg_dicts = []
	for resource_pool, class_dict in self.resource_pool_to_cls.items():
	    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
	    wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
	    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
	    all_wg.update(spawn_wg)
	    # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
	    self.wg_dicts.append(wg_dict)
```

因为 PyTorch 的显存管理机制不同进程无法共享已 reserve 的显存池，多进程场景下显存碎片化严重。verl 的解决办法是让每个 GPU 只运行一个进程（Worker），所有 workload 的方法都在这个进程内切换执行，从根本上消除了跨进程的显存浪费。

| 概念           | 对应实体                                   |
| ------------ | -------------------------------------- |
| Workload     | 一类具体任务，如 actor rollout、critic training |
| Worker       | 一个进程 = 一块 GPU                          |
| WorkerGroup  | 管理一组 Worker，负责 SPMD 调度                 |
| ResourcePool | 一组 GPU 资源的抽象                           |

例如 actor rollout 分配了 3 块 GPU，则对应的 WorkerGroup 会通过 Ray 启动 3 个独立进程（Worker），每个 Worker 独占一块 GPU，3 张卡的协作方式取决于配置（DP / TP 等）。

### 3.3 Training Loop

#### 3.3.1 分层概览

```
ray_trainer.py / fit()          ← 第4层：single-controller，只管调度逻辑
    ↓ 调用
RayWorkerGroup                  ← 第3层：multi-controller，类似 torchrun，把方法分布式广播给组内所有 Worker
    ↓ 调用
fsdp_workers.py                 ← 第2层：ActorRolloutRefWorker，调度层
  内部持有 ↓                          把 rollout、actor training、ref 三个角色合并管理
dp_actor.py                     ← 第1层：DataParallelPPOActor，执行层
                                      真正跑前向/反向传播的地方
```

- `dp_actor.py` 里的 `update_policy`、`compute_log_prob` 就是以往我们自己写的 RL 训练代码
- `fsdp_workers.py` 把上面的代码套进分布式框架，并额外处理 rollout 推理引擎的衔接
- `fit()` 就是训练主循环，我们不用了解分布式细节

#### 3.3.2 顶层数据流

verl 顶层采用 single-controller 的设计，`fit()` 里只写串行逻辑，不用考虑分布式。整体流程：

```
fit()
├── 初始化（加载checkpoint、验证）
└── 训练主循环（epoch → batch）
    ├── 生成阶段（Rollout）
    ├── 奖励计算阶段
    ├── 对数概率计算阶段
    ├── 优势估计阶段（Advantage）
    ├── Critic/Actor更新阶段
    └── 日志/验证/保存
```

数据用 DataProto 包裹，在各个 worker 之间传递。关键步骤代码如下：

```python
# 1. 准备数据，只取 prompt 部分去掉 response
batch = DataProto.from_single_dict(batch_dict)
gen_batch = self._get_gen_batch(batch)

# 2. rollout 生成回答
#    底层通过 Ray 进程间通信把 DataProto 传给 rollout worker，计算完再传回
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

# 3. 把生成结果合并回 batch
batch = batch.union(gen_batch_output)

# 4. 奖励 → KL 惩罚 → 优势估计
reward_tensor = extract_reward(batch)
batch = apply_kl_penalty(batch)
batch = compute_advantage(batch)    # GAE / GRPO 等

# 5. 更新网络
critic_output = self._update_critic(batch)
actor_output  = self._update_actor(batch)
self.checkpoint_manager.update_weights(...)  # 同步最新权重到 rollout worker
```

#### 3.3.3 FSDP Worker

接着来看看 `generate_sequences`、`update_actor`、`update_critic` 这些方法内部在做什么：

假如我们后端训练框架用的是 FSDP，那么 worker 用的就是 `fsdp_workers.py` 里面的 ActorRolloutRefWorker 和 CriticWorker 这两个 worker 类。ActorRolloutRefWorker 同时管理三个角色：

```python
ActorRolloutRefWorker
├── Actor    → 用 PPO 更新模型参数（训练）
├── Rollout  → 用当前模型生成回答（采样）
└── Ref      → 用原始模型算 KL 散度用的参考概率（不训练）
```

这三个角色可以合并在同一个 Worker 里，也可以拆开，由启动时的 `role` 参数控制。CriticWorker 则单独管理 Critic 模型的前向计算和参数更新。当 `fit()` 调用各方法时，**Worker Group 会把调用广播给组内所有 Worker 并行执行**，结果汇总后返回主进程。

verl 最核心的设计就是 **训练/推理模式切换**。训练用 FSDP 把参数切片分散到多卡，推理引擎（vLLM/SGLang）需要完整参数，所以每次 rollout 前后必须切换：

```python
async def rollout_mode(self):
    # 把 FSDP 分片的参数重新聚合成完整参数，推送给推理引擎
    params = self.actor_module_fsdp.state_dict()
    await self.rollout.update_weights(params)
    await self.rollout.resume(tags=["kv_cache"])   # 恢复 KV cache，准备生成

def generate_sequences(self, prompts):
    loop.run_until_complete(self.rollout_mode())           # 切换到推理模式
    output = self.rollout.generate_sequences(prompts)      # vLLM/SGLang 生成
    loop.run_until_complete(self.trainer_mode())           # 切回训练模式
    return output
```

这样同一批 GPU 在不同时间段分别承担训练和推理，实现了 GPU 分时复用，这也是 verl HybridFlow 名字的由来。

#### 3.3.4 DP Worker

`dp_actor.py` 里的 DataParallelPPOActor 是真正执行前向/反向传播的地方，被 ActorRolloutRefWorker 内部持有和调用。整体数据流：

```
DataProto (全量 batch)
    ↓  split
mini_batch × ppo_epochs        ← PPO 多轮更新
    ↓  split  
micro_batch × gradient_accum   ← 梯度累积（省显存）
    ↓
_forward_micro_batch()          ← 实际 forward
    ↓
loss.backward()
    ↓
_optimizer_step()               ← clip grad + step
```

`update_policy` 方法包含了训练的主训练：

```python
for epoch in ppo_epochs:
    for mini_batch in mini_batches:
        optimizer.zero_grad()
        for micro_batch in micro_batches:           # 梯度累积
            log_prob = forward(micro_batch)
            pg_loss = policy_loss_fn(old_log_prob, log_prob, advantages)
            loss = pg_loss - entropy * coeff + kl_loss * coeff
            (loss * scale).backward()
        clip_grad_norm + optimizer.step()
```

## 4. SPMD

### 4.1 环境变量管理

verl 不像 `torchrun` 那样自动注入 `RANK`、`WORLD_SIZE` 等环境变量，需要在创建 Worker 时手动配置：

```python
def _init_with_resource_pool(self, resource_pool, ray_cls_with_init):
  # ...
  rank = -1
  for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)): # Node
    for local_rank in range(local_world_size): # GPU
      rank += 1
      env_vars = {
        'WORLD_SIZE': str(world_size), 'RANK': str(rank), # More env vars ...
      }
      ray_cls_with_init.update_options(
        {'runtime_env': {'env_vars': env_vars}})
      # ...
      worker = ray_cls_with_init(placement_group=pg,
                                 placement_group_bundle_idx=local_rank)
      self._workers.append(worker)
  # ...
```

遍历所有节点和 GPU，为每个 Worker 注入对应的 `rank`/`world_size`，然后通过 Ray 的 placement group 将 Worker 绑定到指定 GPU。此后每个 Worker 进程内都能通过环境变量感知自己在集群中的位置。

### 4.2 register 装饰器

`register` 是 verl 中 WorkerGroup 方法的核心装饰器，用于声明方法被主进程调用时**数据如何分发、任务如何执行**。它将分布式调度的细节从业务逻辑中解耦——开发者只写单卡逻辑，`register` 负责切数据、发任务、收结果。

```python
def register(
	dispatch_mode=Dispatch.ALL_TO_ALL,
    execute_mode=Execute.ALL,
    blocking=True,
    materialize_futures=True
):

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        attrs = {
            'dispatch_mode':  dispatch_mode,
            'execute_mode':   execute_mode,
            'blocking':       blocking,
        }
        setattr(inner, MAGIC_ATTR, attrs)
        return inner

    return decorator
```

| 参数                    | 默认值          | 作用                                              |
| --------------------- | ------------ | ----------------------------------------------- |
| `dispatch_mode`       | `ALL_TO_ALL` | 数据分发策略（DP、TP 等），决定 `dispatch_fn` 和 `collect_fn` |
| `execute_mode`        | `ALL`        | 哪些 Worker 执行该方法                                 |
| `blocking`            | `True`       | 主进程是否阻塞等待结果                                     |
| `materialize_futures` | `True`       | 是否在函数入口立即解析 Ray `ObjectRef`                     |


{{< admonition type=info title="materialize_futures 对流水线效率的影响">}} 

Ray 的 `.remote()` 调用立即返回 `ObjectRef`，真正的数据在后台异步计算。`materialize_futures` 控制何时调用 `ray.get()` 阻塞取值：

```
materialize_futures=True（串行，默认）:
  Worker A: [Stage 1 计算][网络传输→]
  Worker B:                           [ray.get 阻塞][Stage 2 计算]

materialize_futures=False（通信与计算重叠）:
  Worker A: [Stage 1 计算][网络传输→]
  Worker B:               [Stage 2 启动][初始化/zero_grad][ray.get 等待][Stage 2 计算]
```

设为 `False` 时，Stage 2 可以提前启动，在传输数据的同时并行执行不依赖数据的初始化操作（如 `optimizer.zero_grad()`），从而掩盖通信延迟。

{{< /admonition >}}

### 4.3 dispatch_fn 与 collect_fn

`dispatch_mode` 决定了使用哪对 `dispatch_fn`/`collect_fn`：

```python
predefined_dispatch_mode_fn = {
    Dispatch.ONE_TO_ALL: {
        'dispatch_fn': dispatch_one_to_all,
        'collect_fn': collect_all_to_all,
    },
    Dispatch.ALL_TO_ALL: {
        'dispatch_fn': dispatch_all_to_all,
        'collect_fn': collect_all_to_all,
    },	
    Dispatch.DP_COMPUTE_PROTO: {
        'dispatch_fn': dispatch_dp_compute_data_proto,
        'collect_fn': collect_dp_compute_data_proto
    },
    # ...
}
```

以 `DP_COMPUTE_PROTO`（DataProto 数据并行）为例：

```python
# dispatch：把 DataProto 按 world_size 均匀切分，每个 Worker 拿到一片
def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(
        worker_group.world_size, *args, **kwargs
    )
    return splitted_args, splitted_kwargs

# collect：把所有 Worker 返回的 DataProto 片段拼回完整数据
def collect_dp_compute_data_proto(worker_group, output):
    return _concat_data_proto_or_future(output)

def _concat_data_proto_or_future(output):
    o = output[0]
    if isinstance(o, DataProto):
        return DataProto.concat(output)
    elif isinstance(o, ray.ObjectRef):
        return DataProtoFuture.concat(output)
    else:
        raise NotImplementedError
```

逻辑非常直观：`dispatch_fn` 是 chunk 切分，`collect_fn` 是 concat 合并，对称操作。

---


手动实现一个分布式的 inference：

```python
import ray
import math
from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.base.decorator import register, Dispatch, MAGIC_ATTR

# ─── Step 1: 定义自定义 dispatch/collect 函数 ────────────────────────────────

def dispatch_list_split(worker_group, *args, **kwargs):
    """把 prompts list 均匀切分成 world_size 份"""
    world_size = worker_group.world_size
    prompts = kwargs['prompts']
    
    chunk_size = math.ceil(len(prompts) / world_size)
    # 切成 world_size 个子列表，不足的 worker 分到空列表
    chunks = [
        prompts[i * chunk_size : (i + 1) * chunk_size]
        for i in range(world_size)
    ]
    # 返回的 kwargs 里 prompts 是个长度==world_size 的 list
    # execute_all_async 会把 chunks[i] 发给 worker[i]
    return args, {**kwargs, 'prompts': chunks}


def collect_list_concat(worker_group, output):
    """把各 Worker 返回的子列表拼回完整列表"""
    # output 是 [result_from_worker0, result_from_worker1, ...]
    result = []
    for sublist in output:
        result.extend(sublist)
    return result


# ─── Step 2: 把自定义逻辑注册到 Dispatch 映射表 ────────────────────────────────

# 扩展 predefined_dispatch_mode_fn（verl 内部的 dict）
from verl.single_controller.base.decorator import predefined_dispatch_mode_fn

# 用一个字符串 key 或自定义 Enum 值都可以，这里用字符串简单演示
DISPATCH_LIST_SPLIT = "LIST_SPLIT"

predefined_dispatch_mode_fn[DISPATCH_LIST_SPLIT] = {
    'dispatch_fn': dispatch_list_split,
    'collect_fn':  collect_list_concat,
}


# ─── Step 3: 在 Worker 里使用 ────────────────────────────────────────────────

@ray.remote
class InferWorker(Worker):
    def __init__(self):
        super().__init__()

	@register(dispatch_mode=Dispatch.ONE_TO_ALL):
	def load_model(self, model_name: str):
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name,
			dtype="auto",
			device=f"cuda-{self.rank}"
		)
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		
		return self.model.device, id(self.model)

    @register(dispatch_mode=DISPATCH_LIST_SPLIT)
    def infer(self, prompts: list[str]) -> list[str]:
        # 这里 prompts 已经是切分后的子列表，单卡逻辑
        print(f"[rank {self.rank}] received {len(prompts)} prompts: {prompts}")
        results = self.model.generate(prompts)
        return results


# ─── Step 4: 启动并调用 ─────────────────────────────────────────────────────

ray.init()

resource_pool   = RayResourcePool([2], use_gpu=True)  # 2 个 Worker
class_with_args = RayClassWithInitArgs(cls=InferWorker)
worker_group    = RayWorkerGroup(resource_pool, class_with_args)

model_info = worker_group.load_model("qwen3.5-2b")
print(model_info)

prompts = ["问题A", "问题B", "问题C", "问题D", "问题E"]
results = worker_group.infer(prompts=prompts)

print(results)
# ['[rank0] response to: 问题A', '[rank0] response to: 问题B', '[rank0] response to: 问题C',
#  '[rank1] response to: 问题D', '[rank1] response to: 问题E']
```

这个例子是想说明两个问题：
1. 当 worker group 初始化之后，组内的所有 worker 就会被实例化到对应的 GPU 上。并且这个 worker 在运行期间一直是 active 的，所以我们能看到 `worker_group.infer` 里面可以调用之前 `worker_group.load_model` 创建的 model 和 tokenizer。
2. `load_model` 方法不一定绑定 `@register` 装饰器，`@register` 的作用本质是把这个方法暴露成 可以被 `worker_group.xxx()` 远程调用的接口。


### 4.4 execute_fn

我们先理一下前面在干什么。假如现在要进行 actor rollout，这部分内容会交给 ActorRolloutWorkerGroup 处理，它内部会实例化多个 ActorRolloutWorker 进程，放在不同的 GPU 上运行，每个 worker 执行的都是 rollout workload。前面 `dispatch_fn` 解决的是如何把传入 worker group 的 DataProto 切分给每个 worker，`collect_fn` 解决的是如何把每个 worker 返回的数据合并起来向后传递，也就是实现的 SPMD，而 `execute_fn` 定义了每个 worker 具体执行的 workload 是什么。

```python
def execute_all_async(self, method_name: str, *args, **kwargs):
    # 这里我们假设，如果 args 和 kwargs 里面所有的参数都是 list，且所有的 list 长度都与 len(self._workers) 一致的话，我们会把
    # list 中的每一个分别发到对应的 worker 上去
    # print(f"execute_all_async: method {method_name}({args}, {kwargs})")
    length = len(self._workers)
    if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
        if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
            # print(f"splitting args and kwargs into {length} shards")
            result = []
            for i in range(length):
                sliced_args = tuple(arg[i] for arg in args)
                sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                remote_call = getattr(self._workers[i], method_name)
                result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
            return result
    
    return [getattr(worker, method_name).remote(*args, **kwargs) for worker in self._workers]
```

`method_name` 使得同一个 Worker 类可以动态切换行为，例如：

```python
execute_all_async("generate_sequences")  # rollout 阶段
execute_all_async("compute_log_prob")    # 计算 logprob 阶段
execute_all_async("update_actor")        # 训练阶段
```

这正是 ActorRolloutRefWorker 这类复合 Worker 的设计初衷——它在一个进程内同时承载推理和训练能力，通过 method_name 来切换。

### 4.5 func_generator

`func_generator` 是 multi-controller 最核心的部分，将 dispatch → execute → collect 三步封装成一个通用方法：

```python
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):

    def func(*args, **kwargs):
        # Step 1: 切分数据
        args, kwargs = dispatch_fn(self, *args, **kwargs)
        # Step 2: 分发给所有 Worker 异步执行
        output = execute_fn(method_name, *args, **kwargs)
        # Step 3: 阻塞等待（可选）
        if blocking:
            output = ray.get(output)
        # Step 4: 聚合结果
        output = collect_fn(self, output)
        return output

    return func
```

WorkerGroup 在初始化时扫描所有 Worker 上标记了 `@register` 的方法，为每个方法调用 `func_generator` 生成对应的代理方法，再通过 `setattr(self, method_name, func)` 挂载到自身。这样，整个 verl 形成了两层调度结构：

```
Driver（单进程）
  └── single-controller：顺序调用各 WorkerGroup 的方法
        └── WorkerGroup（multi-controller）：SPMD 调度
              └── dispatch → execute（多 Worker 并行）→ collect
```

调用方只需：

```python
actor_wg.update_actor(data)
```

看起来像本地函数调用，实际上它进行了数据切分 → 多 GPU 并行训练 → 数据聚合等一系列操作。

## 5. Programming Guide

### 5.1 数据集处理

verl 中标准的 RL 数据集有以下字段：

```json
{
	"data_source": used to chose reward function,
    "prompt": [{"role": ..., "content": ...}],
    "reward_model": {
        "style": "rule" or "reward",
        "ground_truth": ...
    },
    "extra_info": a dict containing extra information
}
```

其中 `prompt` 和 `reward_model` 字段是必须的，`data_source` 字段是用于标识数据集来源的字符串，比如 "gsm8k"、"math"、"code" 等等，reward function 内部可以根据不同的 source 给予不同的评分逻辑。具体可以看 `examples/data_preprocess` 部分的示例：

```python
def make_map_fn(split):

    def process_fn(example, idx):
        question_raw = example.pop('question')
        question = question_raw + ' ' + instruction_following
        answer_raw = example.pop('answer')
        solution = extract_solution(answer_raw)
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'answer': answer_raw,
                "question": question_raw,
            }
        }
        return data

    return process_fn

train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
```

我们对数据集进行预处理之后，需要把它转为 verl  支持的 `.parquet` 格式：

```python
train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
```

然后我们就可以在配置文件，或者通过参数把 `.parquet` 数据集的路径传给 verl：

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/geo3k/train.parquet \
    data.val_files=$HOME/data/geo3k/test.parquet \
    //...
```

### 5.2 奖励函数

#### 5.2.1 reward function

通过配置文件指定自定义奖励函数：

```python
custom_reward_function:
  path: /path/to/my_reward.py   # 你的文件路径
  name: my_reward_fn            # 函数名，如果叫 compute_score 可以不填
reward_model:
  reward_manager: naive
```

函数签名默认为**单条进、单个 float 出**：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "openai/gsm8k":
        return gsm8k_score(solution_str, ground_truth)
    elif data_source == "lighteval/MATH":
        return math_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Unknown data_source: {data_source}")
```

#### 5.2.2 reward manager

RewardManager 是包裹 reward function 的执行框架，负责：
- 从 `DataProto` 里拿到 token ids，decode 成文本
- 调用你的 reward function 计算 float score
- 把 score 转回 token-level reward tensor 返回给 trainer

verl 内置多种 RewardManager，这里介绍几个常用的：

- **NaiveRewardManager**：串行逐条评分，适用于 reward 是纯规则（EM/F1）且计算很快

```python
class NaiveRewardManager:
    def __call__(self, data: DataProto):
        for i in range(len(data)):
            data_item = data[i]
            sequences_str = self.tokenizer.decode(
                torch.cat((valid_prompt_ids, valid_response_ids))
            )
            score = self.compute_score(
                data_source=data_item.non_tensor_batch['data_source'],
                solution_str=sequences_str,
                ground_truth=data_item.non_tensor_batch['reward_model']['ground_truth'],
            )
            reward_tensor[i, valid_response_length - 1] = score
        return reward_tensor
```

- **PrimeRewardManager**：并发异步评分，适用于 reward 需要调外部 API / 代码执行

```python
class PrimeRewardManager:
    def __call__(self, data: DataProto):
        try:
            scores = asyncio.run(
                parallel_compute_score_async(
                    self.compute_score,
                    sequences_str, ground_truth, data_sources,
                    num_processes=64,
                )
            )
        except asyncio.TimeoutError:
            scores = [0.0] * len(sequences_str)
        # 写回 reward tensor ...
        return reward_tensor
```

- BatchRewardManager：可以接受一个 batch 的数据计算 reward，比如 GRPO 需要一个 group 来计算相对优势

```python
def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None):
	return [0.0] * len(solution_strs)
```

>这里注意把参数里面每个字段都加上 s。


#### 5.2.3 多奖励函数混合

verl 不像 trl 那样支持直接传入奖励函数列表，混合多个奖励信号有两种方式：

- **方式一**：在单个 reward function 内部合并

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    format_score      = check_format(solution_str)
    correctness_score = check_answer(solution_str, ground_truth)
    length_penalty    = -0.01 * max(0, len(solution_str) - 1000)
    return 0.2 * format_score + 0.7 * correctness_score + 0.1 * length_penalty
```

- **方式二**：自定义 RewardManager 编排多个函数

```python
class MultiRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None):
        self.tokenizer = tokenizer
        self.reward_fns = [
            (compute_format_score,      0.2),
            (compute_correctness_score, 0.7),
            (compute_length_score,      0.1),
        ]

    def __call__(self, data: DataProto):
        # decode、提取 ground_truth ...
        total = sum(
            w * fn(data_source, solution_str, ground_truth, extra_info)
            for fn, w in self.reward_fns
        )
        # 写回 reward tensor ...
        return reward_tensor
```

### 5.3 损失函数

#### 5.3.1 定义 loss_fn

verl 中所有损失函数都定义在 `verl.trainer.ppo.core_algos.py` 里面。`core_algos.py` 里的 loss 函数签名都很统一：

```python
def compute_policy_loss_xxx(
    old_log_prob,      # (bsz, response_len) rollout 时采样的 log prob
    log_prob,          # (bsz, response_len) 当前 policy 的 log prob  
    advantages,        # (bsz, response_len) GAE 优势估计
    response_mask,     # (bsz, response_len) 哪些位置是有效 response
    loss_agg_mode,     # "token_mean" / "seq_mean" / ...
    config,
    **kwargs
) -> (loss_scalar, metrics_dict)
```

然后我们需要用 `@register_policy_loss()` 装饰器把自定义的损失函数注册到路由表里。然后配置文件里启用：

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: compute_policy_loss_xxx
    entropy_coeff: 0.001
    use_kl_loss: false
```

#### 5.3.2 定义 advantages

不同算法的 advantages 各不相同，例如 PPO 用的是 GAE 计算优势，GRPO 通过组内归一化计算优势，所以如果需要自定义一个 `compute_advantages` 方法和自定义损失函数一样。优势估计函数需要符合以下定义：

```python
def compute_custom_advantage(  
    token_level_rewards: torch.Tensor,  
    response_mask: torch.Tensor,  
    index: np.ndarray = None,  
    config: Optional[AlgoConfig] = None,  
    **kwargs  
) -> tuple[torch.Tensor, torch.Tensor]:  
    """  
    计算自定义优势  
      
    Returns:  
        advantages: 优势值 (bs, response_length)  
        returns: 回报值 (bs, response_length)  
    """  
    # 实现你的优势计算逻辑  
    return advantages, returns
```

然后用装饰器注册函数：

```python
from verl.trainer.ppo.core_algos import register_adv_est, AdvantageEstimator  
  
@register_adv_est("custom_advantage")
def compute_custom_advantage(...):
    pass
```

并且在配置文件启用：

```yaml
algorithm:  
  adv_estimator: custom_advantage
```

#### 5.3.3 额外的字段

假设你的 loss 需要额外的信息，比如 token 级别的 reward，在 `dp_actor.py` 的 `update_policy` 里加上：

```python
select_keys = [
    "responses", "response_mask", "input_ids",
    "attention_mask", "position_ids",
    "old_log_probs", "advantages",
    "token_level_rewards",   # ← 新增字段
]

# 然后在 forward 循环里取出来传给 loss fn
token_rewards = model_inputs.get("token_level_rewards", None)

pg_loss, pg_metrics = policy_loss_fn(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,
    response_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
    config=self.config,
    token_level_rewards=token_rewards,  # ← 透传
)
```

### 5.4 配置参数

- Batch Size
	- `train_batch_size`：一次取多少条数据来生成经验
	- `ppo_mini_batch_size`：每次从 `train_batch_size` 中取多少数据来更新参数，模型每处理完一个 mini-batch，才会进行一次参数更新。
	- `ppo_micro_batch_size_per_gpu`：每张 GPU 每次实际 forward/backward 的样本数，实现的是梯度累积。
	- `ppo_max_token_len_per_gpu`：这是 `ppo_micro_batch_size_per_gpu` 的替代方案，与 `use_dynamic_bsz` 配合使用。系统会自动打包样本，直到总 Token 量接近这个阈值，形成一个动态的 micro batch size，从而稳定计算效率；无论长短样本，每个微批次的计算量都相对恒定。

```python
def train(self):
    for epochs in range(self.config.epochs):
        for batch in self.dataloader: 
            exp = self.get_experience(batch["prompts"])
            
            for _ in range(self.config.ppo_epochs):
                for mini_batch in split(exp, self.config.ppo_mini_batch_size):
	                for micro_batch in split(mini_batch, micro_batch_size_per_gpu * dp_size):
	                    self.optimizer.zero_grad()
	                    loss = self.compute_loss(micro_batch)
	                    loss.backward()
	                    self.optimizer.step()
```

我们从 dataloader 中取 `train_batch_size` 条数据用于 rollout 生成经验，然后每个 `ppo_epoch` 用 `ppo_mini_batch_size` 条数据来更新，每次更新 `ppo_micro_batch_size_per_gpu`，类似梯度累积。

- Rollout
	- `temperature`：越大采样随机性越高
	- `top_k`：在概率最高的k的token中进行采样
	- `topp`：从概率最高的token进行累加，直到累加概率和达到p,从这些token.里面进行采样
	- `n`：GRPO的组大小（非GRPO类算法为1）】
	- `ignore_eos`：为True时，在生成eos标记时不会停止，会继续生成直到最大长度
	- `gpu_memory_utilization`：rolloutt模型采样时的GPU使用显存占比，在I旧版本的vlm中是按照总显存进行计算
	- (一般设置在0.5左右)，新版本的vm中是按照剩余显存进行计算（可设置到0.85左右）
	- `layered_summon`：为True时节省显存，但是会更慢（时间换空间）
	- `tensor_model_parallel_size`：张量并行大小，一般是一个节点使用的GPU数量
	- `multi_turn.enable`：否使用agent loop,搭配rollout..mode=async
	- `mult_turn.max_assistant_turns`：assistant最大交互轮数
	- `mult_turn.tool_config_path`：工具配置路径
	- `multi_turm.max_user_turns`：user最大交互轮数
	- `multi_turn.max_tool_.response_length`：工具输出结果的最大长度
	- `multi_turn.tool_response_truncate_side`：如果工具输出结果过长，按照什么方式截断：left,middle,righ
	- `multi_turn.format`：工具调用的格式，一般为hermes
	- `enable_chunked_prefill`：分块处理非常长的Prompt,减少显存蜂值，但是降低吞吐量
- Algorithm
	- `clip_ratio`：新旧 `log_probs` 的裁剪比例
	- `clip_ratio_high`/`clip_ratio_low`：DAPO 里面提到的为了防止熵坍塌，对比例进行上下限不同的裁剪
	- `loss_agg_mode`：token-level-mean、sequence_level-mean 等等
	- `use_kl_loss`：是否在损失项里面加入 kl loss
	- `kl_loss_coef`：kl loss 的权重
	- `kl_loss_type`：用的是哪一种 kl 散度，k1、k2 还是 k3 估计
	- `gamma`：奖励折扣因子
	- `lam`：平衡 gae 和 td error
	- `adv_estimator`：优势估计方法，比如 PPO 对应 gae
	- `norm_adv_by_std_in_grpo`：要不要像 GRPO 一样对组内进行标准差归一化
- Trainer
	- `total_epochs`：总训练轮次
	- `total_training_steps`：如果没指定就是 `train_batch_size/ppo_mini_batch_size` 
	- `save_freqs`：多久保存一次
	- `n_gpus_per_node`：每个节点 gpu 数量
	- `nnodes`：共有多少个节点（机器）

### 5.5 Agent Loop

#### 5.5.1 整体架构概览
 
`ToolAgentLoop` 是 verl 框架中用于多轮工具调用强化学习训练的核心组件。它将一次完整的 agent 推理过程抽象为一个**有限状态机**，驱动模型与工具之间的多轮交互，最终输出带有 response mask 的 token 序列用于 RL 训练。
 
```
                    ┌─────────────────────────────────────────────┐
                    │              ToolAgentLoop.run()            │
                    └─────────────────────────────────────────────┘
                                          │
                              ┌───────────▼───────────┐
                              │    AgentState.PENDING │  ← 准备 prompt
                              └───────────┬───────────┘
                                          │
                           ┌──────────────▼──────────────┐
                           │   AgentState.GENERATING     │  ← LLM 推理
                           └──────────────┬──────────────┘
                                          │
                        ┌─────────────────┼───────────────────┐
                        │                 │                   │
               有 tool_calls        有 interaction        无任何后续
                        │                 │                   │
         ┌──────────────▼──────┐  ┌───────▼────────┐  ┌──────▼──────────┐
         │ PROCESSING_TOOLS    │  │  INTERACTING   │  │   TERMINATED    │
         │ (执行工具, 拼 token) │  │ (获取用户输入)  │  │   (输出结果)     │
         └──────────────┬──────┘  └───────┬────────┘  └─────────────────┘
                        │                 │
                        └────────┬────────┘
                                 │ 回到 GENERATING
                                 ▼
                      (直到 TERMINATED 退出循环)
```
 
---
 
#### 5.5.2 AgentData
 
`AgentData` 是贯穿整个 agent 生命周期的**数据容器**，所有状态都存储在这里。
 
```python
class AgentData:
    # ── 输入数据 ──────────────────────────────
    messages: list[dict]        # 对话历史（role/content 格式）
    image_data: list[Image]     # 多模态图像
    video_data: list[tuple]     # 多模态视频
    tools_kwargs: dict          # 工具初始化参数
 
    # ── 训练关键字段 ───────────────────────────
    prompt_ids: list[int]       # 完整的 token id 序列（prompt + 所有 response）
    response_ids: list[int]     # 当前轮次的 response token ids
    response_mask: list[int]    # 1=模型生成的 token, 0=工具/用户输入的 token
    response_logprobs: list[float]  # 对应 logprob，用于 PPO 等算法
 
    # ── 统计与奖励 ─────────────────────────────
    turn_scores: list[float]    # 每轮 interaction 产生的奖励
    tool_rewards: list[float]   # 每次工具调用产生的奖励
    user_turns: int
    assistant_turns: int
 
    # ── 临时状态 ───────────────────────────────
    tool_calls: list[FunctionCall]  # 当前轮次解析出的工具调用
    extra_fields: dict              # 自定义扩展字段
```
 
{{< admonition type=info title="response_mask 的含义">}} 
 
```
prompt:   [sys][user_msg]          ← 不在 response_mask 里
          ──────────────────────────────────────────────
          [asst_turn1][tool_resp1][asst_turn2][tool_resp2]
mask:          1 1 1 1    0 0 0 0    1 1 1 1    0 0 0 0
```
 
- mask=1 的 token 是模型**自主生成**的，参与梯度计算
- mask=0 的 token 是工具/环境返回的，**不参与梯度计算**

{{< /admonition >}}

#### 5.5.3 状态机各阶段详解

AgentLoop 可以在 verl 中被抽象为一个有限状态自动机，分为 PENDING、GENERATING、PROCESSING_TOOLS、INTERACTING、TERMINATED 五个状态：

```python
while state != AgentState.TERMINATED:
    if state == AgentState.PENDING:
        state = await self._handle_pending_state(agent_data, sampling_params)
    elif state == AgentState.GENERATING:
        state = await self._handle_generating_state(agent_data, sampling_params)
    elif state == AgentState.PROCESSING_TOOLS:
        state = await self._handle_processing_tools_state(agent_data)
    elif state == AgentState.INTERACTING:
        state = await self._handle_interacting_state(agent_data)
    else:
        logger.error(f"Invalid state: {state}")
        state = AgentState.TERMINATED
```

1. PENDING → GENERATING：将 messages + tool schemas 通过 chat template 转为 token ids，这是推理的起点。
 
```python
async def _handle_pending_state(self, agent_data, sampling_params):
    prompt_ids = await self.apply_chat_template(
        agent_data.messages,
        tools=schemas,      # 把工具 schema 嵌入 prompt
        images=...,
        videos=...,
    )
    agent_data.prompt_ids = prompt_ids
    return AgentState.GENERATING
```
 
2. GENERATING
 
```python
async def _handle_generating_state(self, agent_data, sampling_params):
    # 1. 调用 LLM 生成
    output = await self.server_manager.generate(
        prompt_ids=agent_data.prompt_ids, ...
    )
 
    # 2. 累积 token 序列
    agent_data.response_ids = output.token_ids
    agent_data.prompt_ids += agent_data.response_ids    # prompt 不断增长
    agent_data.response_mask += [1] * len(agent_data.response_ids)  # 模型输出 mask=1
 
    # 3. 检查终止条件
    if len(agent_data.response_mask) >= self.response_length: return TERMINATED
    if agent_data.assistant_turns >= self.max_assistant_turns: return TERMINATED
 
    # 4. 解析 tool_calls
    _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(...)
 
    # 5. 决定下一状态
    if agent_data.tool_calls:      return PROCESSING_TOOLS
    elif self.interaction_config:  return INTERACTING
    else:                          return TERMINATED
```

**终止条件**：
- response 总长度超出 response_length
- assistant_turns >= max_assistant_turns
- user_turns >= max_user_turns
- 没有工具调用且没有 interaction
- 模型生成了 EOS token（且 ignore_eos=False）

{{< admonition type=question title="为什么不直接用 OpenAI chat API 格式做多轮？">}} 

几乎所有 agent 框架（LangGraph、CrewAI 等）都用 OpenAI chat completion API 并把历史保存为 messages。但 veRL 团队在 DAPO 和 ReTool 训练中发现，把最终 messages 应用 `apply_chat_template` 得到的 token_ids，和每轮拼接 prompt_ids + response_ids 的结果并不相等——工具解析器会修改 content，decode-encode 过程也会引入不一致。这种不一致对 RL 训练至关重要，会导致轨迹偏离策略模型的分布。 所以 veRL 选择全程用 token_ids 操作，而不是 text message。
{{< /admonition >}}

3. PROCESSING_TOOLS：多轮 agentic 行为的核心，流程如下：
 
```
tool_calls (并行执行，上限 max_parallel_calls)
    │
    ▼
asyncio.gather(*tasks)    ← 并行调用所有工具
    │
    ▼
构造 tool 消息（text / 多模态）
    │
    ▼
apply_chat_template(tool_messages)  ← 转为 token ids
    │
    ▼
prompt_ids += tool_response_ids
response_mask += [0] * len(tool_response_ids)   ← 工具输出 mask=0
    │
    ▼
return GENERATING   ← 继续让模型生成
```
 
**工具调用的异常处理**：
 
```python
async def _call_tool(self, tool_call, tools_kwargs, agent_data):
    try:
        instance_id, _ = await tool.create(...)
        response, reward, res = await tool.execute(instance_id, tool_args, agent_data=agent_data)
    except Exception as e:
        return ToolResponse(text=f"Error when executing tool: {e}"), 0.0, {}
    finally:
        await tool.release(instance_id)   # 确保资源释放
```
 
**工具响应截断**：
 
```python
if len(tool_response_text) > self.max_tool_response_length:
    # 三种截断模式：left / right / middle
```

4. INTERACTING：用于模拟用户参与的对话场景（如 chatbot 训练）：
 
```python
async def _handle_interacting_state(self, agent_data):
    should_terminate, response, reward, metrics = \
        await agent_data.interaction.generate_response(...)
 
    # 将用户回复追加进去，mask=0（非模型生成）
    response_ids = await self.apply_chat_template([{"role": "user", "content": response}])
    agent_data.prompt_ids += response_ids
    agent_data.response_mask += [0] * len(response_ids)
 
    if should_terminate: return TERMINATED
    else: return GENERATING
```

#### 5.5.4 工具注册与选择
 
1. 工具注册：工具通过配置文件初始化，全局共享：
 
```python
tool_list = initialize_tools_from_config(tool_config_path)
self.tools = {tool.name: tool for tool in tool_list}
self.tool_schemas = [tool.tool_schema.model_dump(...) for tool in tool_list]
```
 
2. per-sample 工具选择：每个样本可以选择不同的工具子集，非常灵活：
 
```python
tool_selection = extra_info.get("tool_selection")
if tool_selection:
    agent_data._active_tools = {name: self.tools[name] for name in tool_selection}
    agent_data._active_tool_schemas = [...]
else:
    agent_data._active_tools = self.tools   # 使用全部工具
```

#### 5.5.5 Function Call 解析

当我们使用 ToolAgentLoop 时候，ToolParser 负责从模型输出的 token 序列里解析出 Function Call 的 `name` 和 `arguments`。不同模型训练格式不同，直接影响能否正确触发工具调用。如果我们训练的是 Qwen 系模型，直接用内置的 HermesToolParser：

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      format: hermes
```

如果用 DeepSeek 系模型，也可以用内置的 `format: deepseek`。但如果碰到自己训练的模型，或者自定义格式的 function call，我们需要实现自己的 ToolParser：

```python
# verl/experimental/agent_loop/tool_parser.py
import regex, json
from verl.experimental.agent_loop.tool_parser import ToolParser, FunctionCall

@ToolParser.register("my_format")
class MyModelToolParser(ToolParser):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 根据你模型实际输出的格式定义正则
        self.pattern = regex.compile(
            r"<tool_call>(.*?)</tool_call>", regex.DOTALL
        )

    async def extract_tool_calls(
        self, response_ids: list[int]
    ) -> list[FunctionCall]:
        text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        matches = self.pattern.findall(text)
        calls = []
        for m in matches:
            try:
                obj = json.loads(m)
                calls.append(FunctionCall(
                    name=obj["name"],
                    arguments=json.dumps(obj["arguments"], ensure_ascii=False)
                ))
            except Exception:
                pass   # 格式解析失败，忽略该次调用
        return calls
```

然后在 YAML 里配上：

```yaml
format: my_format   # 对应 @ToolParser.register 的 key
```

#### 5.5.6 最终输出结构
 
```python
AgentLoopOutput(
    prompt_ids   = prompt_ids,                         # 纯 prompt 部分
    response_ids = response_ids[:self.response_length], # 截断到最大长度
    response_mask= response_mask[:self.response_length],
    response_logprobs = ...,
    num_turns    = user_turns + assistant_turns + 1,
    metrics      = {...},
    extra_fields = {
        "turn_scores": [...],   # 每轮 interaction 奖励
        "tool_rewards": [...]   # 每次工具调用奖励
    }
)
```

#### 5.5.7 使用指南

讲了这么多，那怎么在 agentic rl 训练中用上 agent loop 呢？我们以 Search-R1 为例。

目前复现 Search-R1 有两种方法：

1. 使用 verl 自带的 ToolAgent

**Step1：自定义 Tool 类**

实现自己的 tool，继承 `verl.tools.base_tool.BaseTool`，`BaseTool` 需要实现以下三个方法：

```python
from verl.tools.base_tool import BaseTool, ToolResponse
from typing import Tuple, Dict, Any

class MySearchTool(BaseTool):

    def get_openai_tool_schema(self) -> dict:
        """
        返回 OpenAI Function Call 格式的工具描述，
        VERL 用它拼接到推理请求的 tools 字段里，告诉模型有哪些工具可用。
        """
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "搜索互联网，返回最多10条相关文档，每条包含 title 和 content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """
        工具的实际执行逻辑。
        AgentLoop 解析出 Function Call 后，调用此方法拿到工具返回值，
        再把结果拼回对话的 tool_response 里继续推理。
        
        name:      模型调用的函数名，比如 "web_search"
        arguments: 模型传入的参数，比如 {"query": "北京天气"}
        返回:      ToolResult，包含返回给模型的文本内容
        """
        query = arguments.get("query", "")
        # 替换成你真实的检索服务
        raw = await self._call_retrieval_service(query)
        return ToolResult(content=raw, is_error=False)

    def calc_reward(self) -> float:
        """
        （可选）基于工具调用状态计算过程奖励，例如：
        - 是否成功调用了工具（格式是否正确）
        - 工具调用次数是否合理（防止滥用）
        - 工具返回结果是否有效
        若不需要过程奖励，直接 return 0.0 即可。
        """
        return 0.0
```

**Step2：配置 tool config YAML**

```yaml
# search_tool_config.yaml
tools:
  - class_name: verl.tools.search_tool.SearchTool  # 或你的自定义类
    config:
      retrieval_service_url: http://127.0.0.1:8000/retrieve
      num_workers: 120
      rate_limit: 120
      timeout: 30
```

**Step3：Rollout YAML 配置**

```yaml
actor_rollout_ref:  
  rollout:
	name: vllm
    mode: async  # 异步模式，避免GPU空闲  
    multi_turn:  
      enable: True  
      max_user_turns: 3  # 最大用户轮次  
      max_assistant_turns: 3  # 最大助手轮次 
    tool_kwargs:
      tools_config_file: ./config/tool_config/search_tool_config.yaml
```

**Step4：在 dataset 里加 `agent_name` 字段**

```json
{
	"data_source": used to chose reward function,
    "prompt": [{"role": ..., "content": ...}],
    "reward_model": {
        "style": "rule" or "reward",
        "ground_truth": ...
    },
    "extra_info": a dict containing extra information,
    "agent_name": "tool_agent"
}
```

2. 另一种方法是继承 `AgentLoopBase` 自定义一个 AgentLoop

**Step1：实现 AgentLoopBase**

`AgentLoopBase` 只有一个必须实现的接口 `run()`，返回 `AgentLoopOutput`，其中 `response_mask` 是关键字段。

```python
from verl.experimental.agent_loop.base import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import ToolParser
import aiohttp, json

@register("my_search_agent")
class MySearchAgentLoop(AgentLoopBase):

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        """
        必须实现的唯一接口。
        负责完整执行一条样本的 Multi-Turn 推理+工具调用，
        返回 AgentLoopOutput，其中 response_mask 标记哪些 token 参与 loss 计算。
        
        VERL 框架只关心你最终返回什么 token，
        中间怎么调工具、循环几轮，完全由你控制。
        """
        messages = list(kwargs["raw_prompt"])   # 原始对话历史
        max_turns = self.config.get("max_turns", 5)

        all_response_ids = []
        all_response_mask = []   # 1=参与loss，0=不参与（工具返回内容不算模型生成）

        for turn in range(max_turns):
            # 1. 调推理引擎，生成一段回复
            output = await self.llm_engine.chat(
                messages=messages,
                sampling_params=sampling_params,
                tools=[self.tool_schema]
            )
            response_text = output.text
            response_ids  = output.token_ids

            # 模型自己生成的 token，全部参与 loss
            all_response_ids  += response_ids
            all_response_mask += [1] * len(response_ids)

            # 2. 检测是否有 Function Call
            tool_calls = await self.tool_parser.extract_tool_calls(response_ids)

            if not tool_calls:
                # 没有工具调用，模型直接给出答案，结束循环
                messages.append({"role": "assistant", "content": response_text})
                break

            # 3. 执行工具调用
            messages.append({"role": "assistant", "content": response_text})
            for fc in tool_calls:
                tool_result = await self._call_search(fc.arguments["query"])
                tool_response_ids = self.tokenizer.encode(tool_result)

                # 工具返回内容不是模型生成的，mask=0，不参与 loss
                all_response_ids  += tool_response_ids
                all_response_mask += [0] * len(tool_response_ids)

                messages.append({
                    "role": "tool",
                    "name": fc.name,
                    "content": tool_result
                })

        return AgentLoopOutput(
            response_ids=all_response_ids[: self.response_length],
            response_mask=all_response_mask[: self.response_length],
        )

    async def _call_search(self, query: str) -> str:
        url = self.config["retrieval_url"]
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json={"query": query}) as r:
                data = await r.json()
        return json.dumps(data, ensure_ascii=False)
```

**Step2：Rollout YAML 配置**

```python
data:
  return_raw_chat: True          # 必须，把原始 chat messages 传给 agent loop

actor_rollout_ref:
  rollout:
    mode: async                  # 必须，启用 server-based 异步 rollout
    name: sglang                 # 或 vllm
    
    agent_loop:
      # 告诉 verl 用哪个类处理 agent_name="my_search_agent" 的样本
      my_search_agent:
        class_name: my_search_agent_loop.MySearchAgentLoop
        config:
          retrieval_url: http://127.0.0.1:8000/retrieve
          max_turns: 5
      
      default:
        class_name: verl.agent_loop.SingleTurnAgentLoop
    
    # 并发控制
    agent_loop_kwargs:
      max_concurrent: 128
```

**Step3：在 dataset 里加 `agent_name` 字段**

```python
dataset = dataset.map(lambda x: {
    **x,
    "agent_name": "my_search_agent"  # 对应你注册的名字
})
```

## 6. References

- 视频
	- [verl 源码解读 & 客制化经验分享](https://www.bilibili.com/video/BV1CzbezREua/)
	- [VeRL强化学习实用教程：自定义奖励计算的若干方法，从简单到复杂，覆盖所有应用需求](https://www.bilibili.com/video/BV1seqHBFEGx)
	- [Agent is all you need，verl自定义Agent Loop](https://www.bilibili.com/video/BV18UBMBKEAV/)
	- [verl参数怎么看，今天就给你们来个保姆级的教程](https://www.bilibili.com/video/BV11WiFBYEV9/)
	- [强化学习（verl）训练日志怎么看，今天就给你们来个保姆级的教程](https://www.bilibili.com/video/BV1PwrrBtENk/)
	- [一个视频带你搞懂 verl里面的ray到底起什么作用？verl里面的worker怎么使用？](https://www.bilibili.com/video/BV116gozmEdn/)
	- [verl 源码解读 与 HybridFlow 编程范式讲解](https://www.bilibili.com/video/BV1Cs7WzNEDX)
- 博客
	- [全网第二细致的Verl GRPO实现拆解讲解](https://www.cnblogs.com/AikN/p/18893668)
	- [[AI Infra] VeRL 框架入门&代码带读](https://zhuanlan.zhihu.com/p/27676081245)
	- [VERL源码解读 &实操笔记](https://zhuanlan.zhihu.com/p/1931076626940139506)
	- [框架分享：Verl](https://www.xiaohongshu.com/explore/69ad8998000000002603cf96?source=webshare&xhsshare=pc_web&xsec_token=ABMeTFj3un3Nc4UH0C1sJhkTkw5WG5c7iz04k4IzR8FQk=&xsec_source=pc_share)