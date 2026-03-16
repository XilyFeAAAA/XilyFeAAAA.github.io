---
title: vLLM 原理
date: 2026-03-12T11:29:34+08:00
featuredImage: http://img.xilyfe.top/img/20260316120049695.png
authors:
  - Xilyfe
series:
  - 推理框架
tags:
  - Inference
  - 大模型
lastmod: 2026-03-16T10:41:50+08:00
---
>vLLM 是目前最受欢迎的开源 LLM 推理与服务引擎之一，它以 PagedAttention 为核心创新，彻底解决了传统 LLM Inference 中 KV Cache 内存碎片化的问题，让 throughput 提升 2~24×，同时内存利用率接近 100%。

## 前情提要

### KVCache

![](http://img.xilyfe.top/img/20260119121117689.png)

讲 vLLM 之前不得不先回忆一下 KVCache，我们发现 LLM Inference 的 Decode 阶段，我们每生成一个 token 它只依赖于 **$QK^T$ 矩阵的最后一行和 V 矩阵的乘积**，换句话说 **我们只需要计算增量部分，而不需要计算整个 $softmax(QK^T)V$**。通过额外记录 K 和 V 矩阵，我们就可以通过空间换时间，大幅减少计算开销。

但是这把双刃剑的副作用也是很明显的，我们通过简单推导可以得出 KVCache 所需的缓存为：$2*2*b*s*d*l$。假定参数用半精度存储，以 GPT-3-175B 为例，若 `batch_size=1`，序列长度为 100，则 KVCache 需要缓存 $2*2*100*12288*96=472\text{MB}$。

### 算力浪费

LLM Inference 分为两个阶段：Prefill 和 Decode。Prefill 阶段会计算所有 prompt 的 KVCache 然后存到显存里，Decode 阶段会逐 token 进行 generate，将新生成的 token 加到 KVCache 里，长度是动态增长的。但是这种方案存在着两个很严重的问题。

第一个问题是 **批处理的限制**：一个 batch 中不同推理请求完成的时间可能差异极大，正常情况下已经完成生成的请求需要等待所有请求完成才能释放显，导致 GPU 利用率低。

![image.png](http://img.xilyfe.top/img/20260312133254234.png)

传统框架采用固定 batch 的方式：
- 一次性把 N 个请求（prompt）打包成一个 batch。
- 送进 GPU 执行 **Prefill**（一次性算完所有 prompt 的 KV Cache）。
- 进入 **Decode** 阶段：**每一步（iteration）所有序列必须同步生成 1 个 token**。

由于整个 batch 的 KV Cache tensor、attention mask 还是一个大矩阵，短请求必须等最长请求也结束，整个 batch 才算完成。

```python
def max_length_processor(input_ids: torch.Tensor, logits: torch.Tensor, eos_token_id: int, initial_len: int, min_length: int) -> torch.Tensor:
    if input_ids.size(-1) - initial_len >= max_length:
        logits[:, eos_token_id] = float("inf")
    return logits
```

在 LLM Generate 那篇文章中我们就提过，假如短请求先完成（可能是输出 eos token 或者达到 max sequence length），它仍然需要占用 GPU 计算 next token，全量计算矩阵乘法和 softmax，只不过我们让新的 token 中 eos token 输出的概率调到最大，固定输出 eos token。

### 显存浪费

其次，当服务接收到一条请求时，它会为这条请求中的 prompts 分配 GPU 显存空间，其中就包括对 KVCache 的分配。由于**推理所生成的序列长度大小是无法事先预知的**，所以大部分框架会按照 $b*s$ 这样的固定尺寸，在 GPU 显存上预先为一条请求开辟一块连续的矩形存储空间。然而，这样的分配方法很容易引起 GPU 显存利用不足的问题.

![image.png](http://img.xilyfe.top/img/20260312134748094.png)

从上图的例子可以看到，我们为三个 prompt 组成的 batch 预分配了一大块内存，但实际生成中不一定能用满，这就导致出现了**内部碎片**。同时由于不同请求的 max sequence length 不同，就会导致 GPU 内存散列，空出来的小内存无法复用，出现 **外部碎片**。

## 优化方案

### PagedAttention

PagedAttention灵感来自于操作系统中虚拟内存和分页的经典思想，它可以允许在非连续空间立存储连续的 KVCache 张量。具体来说：
- PagedAttention 把每个序列的 KVCache 缓存进行了分块，每个块包含固定长度的 token，而在计算 attention 时可以高效地找到并获取那些块。
- 每个固定长度的块可以看成虚拟内存中的页，token 可以看成字节，序列可以看成进程。那么通过一个块表就可以将连续的逻辑块映射到非连续的物理块，而物理块可以根据新生成的 token 按需分配。

![image.png](http://img.xilyfe.top/img/20260312140255333.png)

- Prefill 阶段：
	- **划分逻辑块**：vLLM 拿到这条 prompt，先按照设定好的 block 大小，为 prompt 划分逻辑块。由于 prompt 中有 7 个 token，所以vLLM 用 2 个逻辑块（block 0， block 1）来装它们的 KVCache 值。其中在 block 1 中只装了"years", "ago", "hour"这 3 个 token，有 1 个位置是空余的，这个位置就被称为保留位。
	- **划分物理块**：划分好逻辑块后，我们就可以将其映射到物理块中去了。物理块是实际存放 KVCache 的地方。我们通过一张 block table 来记录逻辑块和物理块的映射关系，block table的主要内容包括：逻辑块和物理块的映射关系、物理块哪些部分被填充了。
	- 等我们计算出 KVCache 之后就将其填入物理块
- Decode 阶段：
	- 当我们生成第一个 token "fathers" 的时候，vLLM 会根据 block table 的映射关系读取 KVCache，在 sequence 角度看来它读取的 Cache 是连续存储的，但是底层是 PagedAttention 从离散的物理块上一个个获取的。
	- 然后再更新逻辑块、物理块和 block table。

---

除此之外 PagedAttention 还有一个优势，就是它可以让**内存共享更加高效**。

在传统框架里，Parallel Sampling 和 Beam Search 会让 KV Cache 内存爆炸，因为每个分支都要完整复制一份 prompt 的 KVCache。PagedAttention 通过 Ref Count 和 Copy-On-Write 机制解决了这个问题。

操作系统中的 Copy-On-Write（COW）机制是一种用于优化内存管理和进程创建的技术，它的核心思想是在某些情况下延迟数据复制，从而节省内存和提高效率。以进程创建为例：在创建新进程时，操作系统通常会复制父进程的地址空间到子进程。使用 Copy-On-Write，最初**并不实际复制物理内存页**，而是让父子进程**共享相同的物理内存页**。只有当新进程尝试修改这些共享页面时，才会复制这些页面，即“写时复制”。因为最初不复制内存，所以如果新进程没有修改任何内存页，那么系统就节省了这部分内存。即使新进程最终修改了内存，也只在实际需要时才复制，这仍然比一开始就复制整个内存空间要高效。

下面说明在 Parallel Sampling 的场景下，vLLM（PagedAttention）是怎么做到节省显存的。

![image.png](http://img.xilyfe.top/img/20260312155007222.png)

我们需要对 "Four score and seven years ago out" 这个 prompt 进行两段续写，A1 和 A2 都把这个 prompt 进行逻辑分块，分别得到了各自的 Block 0 和 Block 1，这两个块对于他们来说是不同的。然后存储到 GPU 上时，由于是同一个 prompt 内容相同，所以实际上只存储了一次，如图存储在物理块 Block 7 和 Block 1 上。A1 和 A2 分别有一张 block table，记录着它们的逻辑块 Block 0/1 映射到物理块 Block 7/1上，这时候物理块 Block 7/1 的 Ref Count 引用计数是 2。

当 A1 和 A2 分别生成了下一个不同的 token "fathers" 和 "mothers"，此时就发生变化了。我们假设 A1 先计算得到 "fathers" 对应的 KVCache，它会记录到逻辑块 Block 1 中，然后检查 block table 准备写入到对应的物理块 Block 1。此时 vLLM 检查物理块 Block 1，**发现它的 Ref Count > 1 触发了 COW**：
- 分配一个全新的空物理块 Block 3。
- 把**原物理 block1 当前的内容** 完整复制一份到新的 Block 3。
- 更新 A1 的逻辑块 Block 1 的指针，让它现在指向新的 Block 3。
- 原物理块 Block 1 的 ref count 从 2 → 1。
- 把 A1 刚刚算出来的 “fathers” KVCache 写进新的 Block 3。

此时由于物理块 Block 1 的 Ref Count = 1，当 A2 想把新 token "mothers" 写入到 Block 1 时就不会触发 COW 了。

>![image.png](http://img.xilyfe.top/img/20260312161058430.png)
>知乎上对这个过程写的很简略，最初我认为流程是谁先写谁就占用原块，然后另一个发现冲突再复制。


在 Beam Search 中 PagedAttention 同样能提高显存利用率。

![image.png](http://img.xilyfe.top/img/20260312162331893.png)

假设当前我们处于虚线右边，因为 `beam width = 4`，这意味着根据 Beam Search 算法，在当前阶段我们生成了 top 4 个概率最大的 toke：beam candidate 0/1/2/3，它们分别装在 Block 5，Block 6，Block 7 和 Block 8 中。接着下一轮 Beam Search，我们选择 top 4 的 next token，如图是 Block 9，Block 10，Block 11，Block 12。这时候 Block 8 被淘汰，导致之前的 Block 2/4/8 也相继被淘汰，它们的 Ref Count = 1，这些块对应的物理内存空间。这一路上，我们都根据最新时刻的 Beam Search结果，**释放掉不再被需要的逻辑块和对应的物理内存空间**，达到节省显存的目的。

### Continuous Batching

前面提到过，传统批处理一次把 N 个 request 凑成一个 batch，一起 prefill 和 decode。这就导致最长的那个 request 全部生成完，才能释放 GPU 资源给下一个 batch。有些短请求早早就 decode 结束，但 GPU 还得空等长请求。

vLLM 的解决方案是 Continuous Batching，它将 decode 从 sequence level 变成了 token level。什么意思呢？具体来说，每生成 1 个 token，就立刻检查当前 batch 里有没有 request 已经结束（EOS 或达到 max_tokens）。 一旦有空位，立即把新来的 request 插进来（可以是 prefill 阶段，也可以是正在 decode 的其他请求）。

![image.png](http://img.xilyfe.top/img/20260312200317659.png)

>如图，在 T5 时刻 S3 先完成了生成，这时候就会把 S5 插入来进行 Prefill。

---

但这里我就有新的问题了：Continuous Batching 会让一个 sequence 在 Decode 结束时候插入新的 sequence 进行 Prefill，但是 Prefill 和 Decode 完全是两个东西，Prefill 会把一整个 prompt 进行 forward，而 Decode 只会对 sequence 进行一次 generate，那么这两个怎么能并行计算呢？

经过 Grok 的提醒，我把 Continuous Batching 和 Chunked Prefill 混在一起了。Continuous Batching 指的是<mark>动态插入、token-level 更新、避免 sequence-level 等待</mark>。我们在 token level 进行 forward pass，每进行一个 step 就判断是否有新的 sequence 需要 Prefill，因为 Prefill 是计算密集型优先级高，而 Chunked Prefill 做的才是在同一个 forward pass 里同时跑 Prefill + Decode。

### Automatic Prefix Cache

对于同一个 System Prompt 的多个请求，正常情况下每个新 request 都要从头计算整个 prompt 的 KV Cache，这种前缀重复计算 KV Cache 的问题导致性能浪费严重。vLLM 的解决方案是：结合 PagedAttention 和 Hash-based，给每一个物理块计算一个 hash 值，当我们需要进行 Prefill 的时候，先判断这一段 token 的 hash 值是否已经存在了。如果 hash 存在则直接复用那个物理块，把它的 ref_count + 1，这样就不用重新计算并且分配显存。

APC 在下面两个场合能带来巨大的性能提升：
- 长文档查询：用户反复查询同一长文档（如软件手册或年报），但查询不同。在这种情况下，APC无需反复处理长文档，而是允许vLLM_只处理一次_长文档，之后所有请求都可以通过重复使用KV缓存来避免重新计算这份长文档。
- 多轮对话：用户可以在同一聊天会话中多次与应用聊天。在这种情况下，APC允许vLLM在所有未来轮次的对话中重复使用聊天历史的处理结果，

>需要注意：APC 仅减少 Prefill 阶段的处理时间，并不会缩短 Decode 阶段生成新 token 的时间。

![](http://img.xilyfe.top/img/20260313111142168.png)

如上图，每轮对话中都只有当前轮的 prompt 需要在prefill阶段进行计算。历史轮次中的Prefix + Generated KV Cache都会被缓存命中。

### Chunked Prefill

vLLM 里的 Chunked Prefill 是用来解决长 prompt 的 Prefill 阶段对整体推理性能影响的一项核心优化技术，尤其在高并发场景下特别有用。传统 Continuous Batching 里如果来一个超长 prompt 的 Prefill，它会独占一整个 batch，把正在 decode 的其他请求全部卡住，导致 Inter-Token Latency（ITL，每两个输出 token 之间的延迟） 剧烈波动，用户感觉生成突然卡顿。比如当前有 20 个 sequence 在进行 Decode，此时进来一个很长的 sequence 需要 Prefill，由于 Continuous Batching 里 Prefill 优先级高，下一个 step 就会先对这个 sequence 进行 Prefill，而需要 Decode 的那些 sequence 就需要进行等待。

Chunked Prefill 到底怎么干的？假设服务器上已经有 10 个正在 decode 的请求，每个 decode 每步生成 1 个 token。这时候又新来一个用户请求：prompt 长度 = 5000 tokens。
1. 首先会把这 10 个 token 组成一个 batch，再把剩下 `budget = 2048 - 10 = 2038 tokens` 的 token 留给 Prefill Chunk。
2. 下一个 step 10 个 token 也组成一个 batch，剩下位置可以 Prefill `prompt[2038:4076]`
3. 第三个 step Decode 10 个 token，并且 Prefill 结束，下一个回合就可以 Decode 11 个 token 了。




## Nano-vLLM

### 整体结构

![image.png](http://img.xilyfe.top/img/20260313233250755.png)

1. 主循环不断调用 `LLMEngine.step()` 进行一次 forward
2. `Scheduler.schedule()` 判断这一个 step 需要进行 Prefill or Decode（没有实现 Chunked Prefill）
3. `ModelRunner.run(seqs, is_prefill)`
	1. 准备好需要的上下文信息 `prepare_prefill` / `prepare_decode`
	2. 调用模型的 `forward()` 方法进行前向计算
	3. 对 logits 采样得到 next token
4. `Scheduler.postprocess()` 进行后处理
	1. 更新 sequence 状态
	2. 检查 eos token / max length
	3. 释放物理块

### Sequence

```python
class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

vLLM 中每一个 request 会被存储为一个 Sequence 对象，其中包括了 Sequence 的状态、长度、prompt、使用的物理块号等信息，用于管理单个推理请求的完整生命周期，从 token 序列的创建到完成生成。

### Block

```python
class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```

Block 就是我们之前提到的**物理块**，它存储了物理块号、引用计数、哈希值（用来快速判断是不是同一个 token）和存储的 token id。

### BlockManager

```python
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```

BlockManager 在初始化时，会预先创建 `num_blocks` 个 Block 对象，并维护：
- `hash_to_block_id`：哈希值 → 块 ID 的映射，用于快速查找已缓存的块
- `free_block_ids`：空闲块 ID 的队列，新块从队首取用
- `used_block_ids`：正在使用的块 ID 集合

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

vLLM 在做 prefix caching 的 hash 时采用链式哈希，原因在于：同一个 token 在不同上下文位置时，它对应的 KV cache 是不一样的。在 Transformer 里，每个 token 的 K/V 向量是通过 self-attention 计算得到的，而 self-attention 会依赖之前所有 token 的表示。例如，我们有两个 prompt："hello world" 和 "morning world"。在计算 token "world" 的时候，句子 1 看到的上下文是 \[hello]，句子 2 看到的上下文是 \[morning]。由于 `KV_cache("world"|"hello") ≠ KV_cache("world"|morning)`，所以我们不能直接复用句子 1 中 "world" 所在的物理块。

```python
def _allocate_block(self, block_id: int) -> Block:
    block = self.blocks[block_id]
    assert block.ref_count == 0
    block.reset()
    self.free_block_ids.remove(block_id)
    self.used_block_ids.add(block_id)
    return self.blocks[block_id]

def _deallocate_block(self, block_id: int) -> Block:
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)
```

底层分配和销毁物理块的方法很简单，就是 Block 在 `blocks`、`free_block_ids` 和 `used_block_ids` 里面来回移动，还有一些安全性检查。

```python    
def can_allocate(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= seq.num_blocks

def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

这两个方法都是容量检查：
- `can_allocate`：由调度器在 Prefill 调度前调用，检查空闲块数量是否 ≥ prompt 所需的总块数
- `can_append`：在 Decode 阶段调用，vLLM 生成了一个新 token 然后检查是否有足够空间追加这个新 token。如果当前 token 恰好是新块的第一个 token（`len(seq) % block_size == 1`），才需要额外申请一个新块

```python
def allocate(self, seq: Sequence):
    assert not seq.block_table
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```

当调度器将一个新序列从 `waiting` 移入 `running` 时，会调用 `allocate` 方法为该 sequence 分配所有需要的内存块。当处理 prompt 中第 i 个 Block 的时候，Block Manager 会取出这个 Block 对应的 token 加上 prefix 信息，计算当前 Block 的哈希值。如果 cache miss 那么就分配一个新的物理块，cache hit 则共享物理块，并且 ref count + 1。

>这里有一个细节，由于 Block Manager 采用的是懒删除，当物理块被移出时候不会清除内部的信息，也就是说我们可能用同一个 hash 值再次索引到它，此时如果它不在 `used_block_ids` 我们就需要通过 `self._allocate_block` 再次分配物理块。

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

`deallocate` 就是一个反向的过程，sequence 释放每一个 Block 后就把他的引用计数 - 1，如果这个物理块没被引用了就调用底层的释放物理块的方法。

### Scheduler

Scheduler 负责管理序列生命周期、协调资源分配，实现 Continuous Batching 机制。它维护两个队列和三种序列状态：

```python
class Scheduler:  
    def __init__(self, config: Config):  
        self.waiting: deque[Sequence] = deque()    # 等待队列  
        self.running: deque[Sequence] = deque()    # 运行队列
```

序列状态转换：`WAITING` → `RUNNING` → `FINISHED`，也支持 `RUNNING` → `WAITING` 的抢占回退。

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # prefill
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
            break
        num_seqs += 1
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True
    # decode
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False
```

每个 step vLLM 会先调用 `scheduler.schedule()` 判断这个 step 应该进行 Prefill or Decode。Nano vLLM 是 vLLM 的简化版本没有实现 Chunked Prefill，由于 Prefill 是计算密集型优先度更高，所以 `schedule()` 方法会先判断是否存在新的 sequence 需要 Prefill 然后再判断是否 Decode。

Prefill 阶段会取出处于 `WAITING` 状态并且数量少于 `max_num_seqs` 的 sequence，经过安全性判断后为其分配物理块，并且更新其状态。仅当 Prefill 队列为空时进入 Decode，遍历 `RUNNING` 队列，为每个序列尝试追加新的 KV Cache slot（`may_append`），并受 `max_num_seqs` 约束。结束后将本轮选出的序列重新写回 `RUNNING` 队列头部，这样下一轮 Decode 就会优先考虑这些 sequence。

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

前面我们说过 Block Manager 的 `can_append` 方法会判断最后一个 Block 能不能容纳 sequence 生成的 next token，换句话说就是我们需不需要为 next token 分配新的 Block。假如空间不足，就会不断通过抢占机制，抢占队尾（最低优先级）序列，释放 `RUNNING` 队列队尾 sequence 的 Blocks。

### KVCache

上面写了这么多你应该会有点疑惑：PagedAttention 里面 Block 是用来存放 KVCache 的，为什么在 Nano vLLM 里面物理块只存了 `token_ids`？因为它采用了一种<mark>逻辑管理与物理存储分离</mark>的设计：

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```


Block 中的 `token_ids` 是**元数据**，用于 Prefix Caching 的哈希计算和缓存验证；真正的 KV Cache 数据存储在 GPU 上的一个巨大张量中，也就是 `self.kv_cache`。这个张量的形状是 `[2, layers, num_blocks, block_size, heads, dim]`，也就代表了模型的每一层都有 `num_kvcache_blocks` 个物理块，每个物理块内存了 `block_size` 个 token 对应的 K 和 V Cache。

### ModelRunner

![image.png](http://img.xilyfe.top/img/20260313224125577.png)


```python
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()
```

ModelRunner 是模型的 wrapper，它附带了 DDP 和 KV Cache 管理等功能。类初始化时候会先设置分布式环境，然后初始化模型并且加载权重。

```python
def load_model(model: nn.Module, path: str):  
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})  
    for file in glob(os.path.join(path, "*.safetensors")):  
        with safe_open(file, "pt", "cpu") as f:  
            for weight_name in f.keys():  
                for k in packed_modules_mapping:  
                    if k in weight_name:  
                        v, shard_id = packed_modules_mapping[k]  
                        param_name = weight_name.replace(k, v)  
                        param = model.get_parameter(param_name)  
                        weight_loader = getattr(param, "weight_loader")  
                        weight_loader(param, f.get_tensor(weight_name), shard_id)  
                        break  
                else:  
                    param = model.get_parameter(weight_name)  
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)  
                    weight_loader(param, f.get_tensor(weight_name))
```

加载好权重之后，ModelRunner 会先预热模型，避免首次推理的性能波动。

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
    num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
    self.run(seqs, True)
    torch.cuda.empty_cache()
```

`warmup_model` 会满载的跑一轮 Prefill，它的好处在于：
1. vLLM 用 PagedAttention 把 KV Cache 分成一个个 block，满载跑一次才能把所有 block 提前分配好。
2. JIT 编译所有 Triton kernels。
3. 模拟最坏场景，判断是否会出现 OOM 的情况。

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

当 LLMEngine 调用 `generate()` 方法每个 step 进行 forward 的时候，ModelRunner 就会调用 `run()` 方法。`run()` 方法首先会根据 Prefill or Decode 预处理 sequence，这里我们插入一个知识点：FlashAttention 中要求传入的 Q/K/V 格式为 \[tokens, head, dim]，也就是说我们<mark>把所有 batch 的 sequence 都压成了一个长序列</mark>，通过一个张量记录每个 sequence 的结束位置，这样就不用对变长序列插入 PAD 浪费算力。

>把 batch 压成一个长序列会不会浪费 GPU 的并行能力？
>
>其实并不会，我们用 Prefill 阶段举例：假设有的请求 prefill 2000 token，有的 50 token，我们把他压成一条总长 5000 的序列 + cu_seqlens。这时候 GPU kernel 会自动把长序列切成多个 tile，短序列切成少量 tile。所有 tile 同时并行，不会因为某条序列短就让 GPU 空闲。而传统的 Pad 方法必须把每一序列 pad 到最长序列，大量 pad token 占用线程，并行度被 pad 拖累。

![image.png](http://img.xilyfe.top/img/20260315110203038.png)
		  
`input_ids` 和 `positions` 就摊平后的 token 和位置信息，需要注意由于 sequence 可能存在缓存过的公共前缀（KV Cache 缓存过了），所以需要记录的是这个 sequence 未缓存的 token（`seq[seq.num_cached_tokens:]`），由于 KVCache 增量计算是增加的 Q 乘以全部的 KV，所以 `seqlen_k` 是整个 sequence 长度。`cu_seqlens_q` 和 `cu_seqlens_k` 记录的是 Q 和 KV 在压平序列里的累计长度，比如 \[apple, banana] 记录的就是 \[0, 5, 11]。如果发现这个这个 sequence 存在缓存，那么就需要记录 slot_mapping。

![image.png](http://img.xilyfe.top/img/20260315120506639.png)

调试一下就能大概看出每一个变量的作用，这里着重说一下 `slot_mapping`。

`slot_mapping` 是<mark>序列中的 token 位置到 KV cache 的物理位置的映射</mark>，我们从头再梳理一遍：
1. 首先我们再 Block Manager 中初始化了 `num_blocks` 个空 Block，每个 Block 大小是 `block_size`。也就是说我们总共有 `num_blocks*block_size` 个槽位，每个槽位可以存储一个 token 的 KVCache。

```python
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

```

2. 同时我们在 ModelRunner 里面初始化了 `self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)`，也就是 `num_blocks*block_size` 个 token 的 KVCache。然后我们通过 `view` 让每个 Attention 层只拿到**自己那一层**的 slice，所有层共享同一个大缓冲区，但访问的是自己的 layer slice。

```python
self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

3. 在每个 sequence 的 Prefill 阶段，我们会通过 `allocate()` 方法为他分配物理块，不过这时候只是在 sequence 的类属性里面记录下，比如 `seq.block_table = [0, 1, 2]`

```python
for i in range(seq.num_blocks):
    token_ids = seq.block(i)
    h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
    block_id = self.hash_to_block_id.get(h, -1)
    if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
        cache_miss = True
    if cache_miss:
        block_id = self.free_block_ids[0]
        block = self._allocate_block(block_id)
    else:
        seq.num_cached_tokens += self.block_size
        if block_id in self.used_block_ids:
            block = self.blocks[block_id]
            block.ref_count += 1
        else:
            block = self._allocate_block(block_id)
    if h != -1:
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block_id
    seq.block_table.append(block_id)
```

4. 由于进行 Flash Attention 的时候需要传入 token 和其对应 KVCache 位置的映射。一个就是我们之前的 `slot_mapping`：标志每个 token 映射到哪个槽位；第二个是 `block_tables`，它形如 \[0, 1, 2, -1, 3, 4, 5, 6, 7, -1, -1, -1]，表示每个 sequence 对应哪些 Block，比如这个例子里 sequence 1 对应 \[0, 1, 2]，sequence 2 对应 \[3, 4, 5, 6]，sequence 3 对应 \[7]，需要用 -1 进行 PAD。
5. 然后就是 Attention Forward 的计算了。首先把传入的新的 KV 保存到 KVCache 中，然后把Q、**这一层的 KVCache**、摊平后的位置信息，以及 `block_tables` 都传入 Flash Attention，它内部就会根据位置偏移去这一层的 KVCache 这个大张量中索引每一个 token 对应槽位的 KVCache。

```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                      max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                      max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                      softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                            softmax_scale=self.scale, causal=True)
        return o

```

![image.png](http://img.xilyfe.top/img/20260315162328558.png)

`prepare_decode` 和 `prepare_prefill` 也有所不同，因为 **Decode 阶段每次 Q 长度都为 1**，并且**肯定会用到之前的 KVCache**，所以我们不需要记录摊平之后的位置，`slop_mapping` 也简单了很多。

### 生成流程

假设我们有两个 prompt 要进行推理，完整走一遍生成过程：
1. `outputs = llm.generate(prompts, sampling_params)`，无需多言
2. `generate()` 内部会通过 while 不断执行一个个 step 直到全部生成结束
3. 第一个 step：
	1. 通过 `scheduler.schedule()` 判断目前应该 Prefill，并且通过 `block_manager.allocate()` 为两个 seq 分配物理块（实际上内存在一开始都申请好了，现在不过是把 block id 记录到 seq 里面）
	2. 调用 `modelrunner.prepare_prefill()` 为两个 seq 记录 Flash Attention 需要的信息，并且把 batch 摊平为一个长序列
	3. 将两个 seq 送入模型进行前向传播
	4. 对输出的 logits 进行采样得到 next token 加到 seq 尾部
4. 开始 step 2：
	1. 此时判断处于 Decode 阶段，同时还会判断生成新 token 需不需要分配新的物理块
	2. 调用 `modelrunner.prepare_decode()` 为两个 seq 记录 Flash Attention 需要的信息，并且把 batch 摊平为一个长序列。注意两个方法的区别，在前面也有提到。
	3. 将两个 seq 送入模型进行前向传播
	4. 对输出的 logits 进行采样得到 next token 加到 seq 尾部
5. 经过多个 step 之后生成结束。
