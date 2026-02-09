---
title: MoE 混合专家模型
date: 2026-02-06T11:53:20+08:00
featuredImage: http://img.xilyfe.top/img/20260206115509890.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 大模型
lastmod: 2026-02-09T04:01:48+08:00
---

## MoE 是什么

MoE means Mixture of Experts，它是一种神经网络架构，可以把一个大模型拆分成多个小型的 expert，再用一个门控网络来决定每个输入该路由到哪些专家处理。
- 传统密集模型：所有参数在每次前向传播中都被激活，计算成本随参数量线性增长。
- MoE 模型：只有部分专家被激活（稀疏激活），总参数量可以很大，但每次推理的计算量相对小得多。

![](http://img.xilyfe.top/img/20260206120606934.png)

在我看来这个 MoE 机制有点像 MHA，MHA 用不同的注意力头负责学习文本中不同维度的信息，而 MoE 用不同的 expert 或者说不同区域的参数来负责处理不同类型的问题，例如某个专家就负责 coding 相关知识，某个增加就负责因果逻辑相关知识。而我们将 prompt 输入后，门控网络就会为我们选择合适的专家。

>在一个 16 层，`n_routed_experts=32` 的模型中一共会有 $16\times32=512$ 个专家，所以一个知识领域的表示并不是某个特定专家独立完成的，而是跨越不同网络深度的多个专家组合路径来完成的。
>![image.png](http://img.xilyfe.top/img/20260206123647136.png)
>
>从上图我们可以直观的看到，当涉及 GitHub 也就是 coding 方面的问题时，在 Layer-0 会调用 expert-18，它可能负责的是语义理解的部分；在中层 Layer-7 会调用 expert-0,3,4 等专家，可能负责思考解决方案等等。



## 工作原理

![image.png](http://img.xilyfe.top/img/20260206121522583.png)


在 Transformer 中，一个典型的 MoE 层通常替换 Feed-Forward Network 层。一般来说我们把注意力层得到的 hidden_states 传入一个 FFN 层，然后经过升维、降维、激活、残差连接这些操作处理向量，最后再输出。

而 Moe 会通过一个小型神经网络（门控网络），通过输入 x 得到一个概率分布表示每个 token 对应的每个专家的权重。然后用 Top-K 路由，选择分数最高的 K 个专家（K 通常是 1 或 2），被选中的专家对输入进行处理（每个专家就是一个小型 FFN），最后被选中专家的输出按门控权重加权求和，可以把上述过程简化表达为：

$$
y=\sum_{i=1}^{N}{G(x)_i \cdot E_i(x)}
$$

> 注意上文提到的 **概率分布表示每个 token 对应的每个专家的权重**，因为 MoE 是 token-level routing，每个 token 对应不同的专家，而不是对一个 sequence 的语义选择一个专家。

## 存在问题

1. 负载均衡
	- 在训练初期，某个专家可能运气好，处理了一些数据，使得权重更新得稍微好一点。于是模型把所有 Token 都发给它，最终导致这个专家承担 100% 计算，MoE 就退化成了一个小的稠密模型。
	- 解决方案方案就是引入辅助损失函数，加入 Auxiliary Loss，在训练过程中发现分配不均就惩罚模型，强制要求 Router 给每个专家分配大致相同的任务
2. 显存墙
	- 虽然推理时只计算 13B 参数，但总共 47B 的参数必须全部加载到显存中。这使得 MoE 模型对显存容量要求极高，而不是对显存速度或计算核心要求高。普通用户很难在消费级显卡上运行高性能MoE。
3. 通信开销
	- 多 GPU 训练或推理时，专家通常分布在不同的显卡上。问题是如果 token 在 GPU-1 上，但路由器把它分给了 GPU-2 上的专家，就需要跨卡传输数据，这会带来额外的延迟。
4. 共享专家
	- 有些知识是通用的比如基本的语法、逻辑，如果每个专家都学一遍，太浪费参数了。
	- 解决方案是设置一个共享专家 Shared Expert，它总是被激活处理通用知识。其他的专家只负责处理特定领域的知识。

## 具体实现

这里我们实现的是 SharedMoE，大致上分为 `MoERouter` 和 `SparseMoE` 两个模块，router 负责为每个 token 选择它对应的专家，`SparseMoE` 管理了整个过程，我们先看 router 部分：

```python
class MoERouter(nn.Module):
    def __init__(self, hidden_size: int, top_k: int, num_experts: int) -> None:
        super(MoERouter, self).__init__()

        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """
        hidden_states: [bs*len, dim]
        router_logits: [bs*len, num_experts]
        router_weights/selected_indices:  [bs*len, top_k]
        expert_masks: [bs*len, top_k, num_experts]
        """
        assert hidden_states.dim() == 2

        router_logits = self.gate(hidden_states)
        router_probs = nn.functional.softmax(router_logits, dim=-1)

        router_weights, selected_indices = torch.topk(input=router_probs, k=self.top_k, dim=-1)
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True).to(hidden_states.dtype)

        expert_masks = nn.functional.one_hot(selected_indices, num_classes=self.num_experts)

        return router_probs, router_weights, selected_indices, expert_masks
```

这里的 `hidden_states` 已经是 token-level 而不是 sample-level，然后通过线性变化加上 softmax 得到路由的概率分布，利用 `torch.topk` 就可以根据概率分布得到每个 token 的前 k 个专家。

```python
class SparseMoE(nn.Module):
    def __init__(self, config: Namespace):
        super(SparseMoE, self).__init__()

        self.config = config
        self.router = MoERouter(config.hidden_size, config.top_k, config.num_experts)
        self.experts = nn.ModuleList([FeedForwardNet(config) for _ in range(config.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() == 3
        bs, seq_len, hidden_size = hidden_states.size()

        tokens = hidden_states.reshape(-1, hidden_size)
        _, router_weights, _, expert_masks = self.router(tokens)

        final_output = torch.zeros(
            (bs * seq_len, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for expert_idx in range(self.config.num_experts):
            expert = self.experts[expert_idx]
            # 选择有专家 expert_idx 的 token
            token_indice, topk_indice = torch.where(expert_masks[..., expert_idx])
            if token_indice.numel() == 0:
                continue
            
            expert_output = expert(tokens[token_indice])

            # 乘上 expert 的权重
            expert_output = expert_output * router_weights[
                token_indice, topk_indice
            ].unsqueeze(-1)
            final_output.index_add_(0, token_indice, expert_output)
        return final_output
```

稀疏混合专家的过程其实也蛮简单：
1. 首先我们根据 router 得到每个 token 的 topk 专家
2. 然后我们遍历每一个专家，取出需要这个专家的全部 token
>这里不是遍历每个 token 而是遍历每个专家，效率更高，因为可以利用 PyTorch 并行计算张量的能力。
3. 我们根据 `expert_mask` 就能知道哪些 token 需要当前专家，并且获取这些 token 的索引。我们把每层 expert 的输出乘上权重累加起来就能得到结果。`tensor.index_add_()` 这个内置方法比 `tensor[indices] += ` 性能更好。

最后组合成 SharedExpert：

```python
class SharedMoE(nn.Module):
    def __init__(self, config: Namespace):
        super(SharedMoE, self).__init__()

        self.shared_experts = nn.ModuleList(
            [FeedForwardNet(config) for _ in range(config.num_shared_experts)]
        )
        self.sparse_moe = SparseMoE(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sparse_output = self.sparse_moe(hidden_states)
        shared_output = torch.stack(
            [shared_expert(hidden_states) for shared_expert in self.shared_experts],
            dim=0,
        ).sum(dim=0)

        return sparse_output + shared_output
```


## 优化

### 辅助负载均衡损失

在 Mixture of Experts 模型中，路由坍塌是一个核心问题：训练过程中，路由网络可能倾向于将几乎所有 token 都分配给少数几个
热门专家，而其他专家几乎不被使用。解决思路如同正则化防止过拟合一样，我们加入路由相关的损失项。辅助损失函数分为两种计算方式：序列级辅助损失和批级辅助损失。

**序列级辅助损失**

该方法以每个序列为单位计算负载，适用于 **长序列或者需要细粒度均衡的场景**

1. 计算每个序列中专家被选择的次数

$$
ce_{b,e} = \sum_{i=1}^{L}\sum_{j=1}^{k}{(\text{topk\_idx}_{b,i,j}=e)}
$$
2. 归一化为相对负载率

$$
\hat{ce}_{b,i}=\frac{ce_{b,i}}{L \cdot k / E}
$$

3. 计算每个序列中专家的平均分数

$$
\hat{s}_{b,e}=\frac{1}{L}\sum_{i=1}^{L}{s_{b,i,e}}
$$

4. 辅助损失

$$
L_{aux} = \alpha \cdot \frac{1}{B}\sum_{b=1}^B\sum_{e=1}^E \hat{c}_{b,e} \cdot \hat{s}_{b,e}
$$

```python
def compute_seq_aux_loss(
    router_probs: torch.Tensor,
    topk_idx: torch.Tensor,
    bs: int,
    seq_len: int,
    top_k: int,
    num_experts: int,
    alpha: float = 0.0
) -> torch.Tensor:  
    reshape_topk_idx = topk_idx.reshape(bs, -1)
    reshape_probs = router_probs.reshape(bs, seq_len, -1)
    
    expert_count = torch.zeros(bs, num_experts)
    expert_count.scatter_add_(1, reshape_topk_idx, torch.ones(bs, seq_len*top_k))
    expert_fraction = expert_count.div(seq_len * top_k / num_experts)
    expert_importance = reshape_probs.mean(dim=1)
    
    return alpha * (expert_fraction * expert_importance).sum(dim=1).mean()
```

{{< admonition type=question title="为什么需要把专家选择次数归一化呢？">}} 
根本目的是让专家被均匀选择时 ce\[i] 固定变成 1。
举个例子，比如一个 sequence 有：
- seq_len = 100
- top_k = 2
- n_experts = 8
如果均匀则每个专家被选中 $200 / 8 = 25$ 次 → ce\[i] = 25。进行归一化，即除以 25 那么 ce\[i] = 25 / 25 = 1。那么计算损失 `(ce * P_i).sum(dim=1)` 时 `sum=1`，aux_loss 就是 α。
{{< /admonition >}}

---

**批级辅助损失**

1. 计算每个专家的全局平均选择率

$$
f_e = \frac{1}{N\cdot k}\sum_{i=1}^{N \cdot k}{m_{i, e}}
$$

- N：总 token 数量
- k：top_k
- m：展平后 top_k 索引的 one-hot 编码

2. 归一化

$$
\hat{f_e} = f_e \cdot E
$$

- E：专家数量

3. 计算每个专家的全局平均分数

$$
p_e = \frac{1}{N}\sum_{i=1}^N{s_{i,e}}
$$

4. 批级辅助损失

$$
L_{aux}=\alpha \cdot \sum_{e=1}^E{\hat{f_e}\cdot p_e}
$$


```python
def compute_batch_aux_loss(
    router_probs: torch.Tensor,
    topk_idx: torch.Tensor,
    num_experts: int,
    alpha: float = 0.0
) -> torch.Tensor:    
    flat_topk_idx = topk_idx.reshape(-1)
    mask_ce = nn.functional.one_hot(flat_topk_idx, num_classes=num_experts)
    ce = mask_ce.float().mean(0)  # ce[i] = 专家i被选中的次数 / 总槽位数
    fi = ce * num_experts
    pi = router_probs.mean(0)
    return alpha * (fi * pi).sum()
```


>但是新的研究表明，过度强调均衡反而损害模型性能，因此 DeepSeek 提出了 **亲和度机制**，或是通过动态调整路由分数的偏置，在不引入额外损失项的情况下实现负载均衡。

### all-to-all
