---
title: "KVCache"
date: '2025-12-03T16:25:11+08:00'
authors: [Xilyfe]
series: ["LLM"]
tags: ["大模型", "Transformer"]
--- 
 

## 前情提要

**Teacher-Forcing vs 自回归生成**

Teacher-Forcing 做的时候和自回归生成非常相似，都是一个接一个 token 进行生成。但是我们需要注意，Teacher-Forcing 的计算是并行的，但是自回归生成的串行的，这是为什么？

- 主要是因为 Teacher-Forcing 位于训练阶段，在 Decoder 中我们已经知道了需要待生成的答案，为了实现**并行计算并且一个接一个 token 生成(预测)**，我们通过上三角矩阵进行掩码操作。
- 自回归生成位于推理阶段，这时候要生成的答案我们是不知道的，所以只能一次次预测下一个 token。具体操作是把 prompt 和已经预测的 token 进行拼接放进 Decoder 中，然后去最后一个隐藏状态，对应的就是预测的下一个 token，然后把这个 token 加入句子继续下一轮预测。

```python
def generate(model, input_ids, max_new_tokens, temperature=1.0):
    model.eval()
    input_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature # 只取最后一个token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

因此自回归生成中，这种串行计算每次计算 Q/K/V 是非常耗时的。

## KVCache 的作用

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/nokvcache.png" width="70%" />
</div>

上图为不加 KVCache 时候推理的过程：

1. 第一次循环计算 $QK^T$ 得到了 "中-中" 的注意力权重
2. 第二次循环计算 $QK^T$ 又重复计算了一次

所以 $QK^T$ 中存在重复计算，其次我们第二次循环实际上只想要得到 "人" 对应的 token，但是把 "华" 对应的 token 也重复计算了一遍。

<div style="text-align: center">
    <img src="../../../../resource/ai/llm/kvcache.png" width="70%" />
</div>

仔细分析循环的过程我们可以发现，对于第 i 词循环我们要生成 $token_i$，它只需要 $QK^T$ 这个下三角矩阵的最后一行和 $V$ 矩阵。再拆细一点，我们只需要 $Q_i$ 和 $K$ 矩阵相乘得到下三角矩阵最后一行还有 $V$，所以我们只需要缓存 $K$ 和 $V$ 矩阵。

> 把这个图例扩展到三维，也就是 \[batch_size, seq_len, d_model]的话，就是在 seq_len 维度上进行缓存。

## 代码

```python
class Attention(nn.Module):

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads

        self.W_pack = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:      
        batch_size, seq_len, _ = hidden_states.shape
        
        proj = self.W_pack(hidden_states)
        proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
        query_states = proj[0].view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = proj[1].view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = proj[2].view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
      
        kv_seq_len = key_states.shape[-2]
        
        if past_key_value is not None:
            # 更新 kv_seq_len
            kv_seq_len += past_key_value[0].shape[-2]
            # 合并 kv
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            

        past_key_value = (key_states, value_states) if use_cache else None
       
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)


        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
   

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value
```

第一次看这个代码时候我很奇怪，每次都计算 Q/K/V 三个矩阵那 Cache 在哪？其实我还是把 Teacher-Forcing 和 自回归生成搞混了。

假设有输入 `inp` 和输出 `tgt`：

- **Teacher-Forcing**：
	- seq2seq(Encoder-Decoder)：将 `inp` 输入到 Encoder，在 Decoder 部分用 `tgt[, 1:]` 当输入，`tgt[, :-1]` 当输出
	- GPT(Decoder Only)：将 `inp[, :-1]` 和 `tgt[, 1:]` 一起输入到模型
- **自回归生成**：将 prompt 输入到模型得到 prompt 的 hidden_state，然后传入模型得到下一个预测 token 的概率分布

所以带入这个代码，第一次把整个 prompt 传进去，生成 len(prompt) 的 KVCache，因为 `past_key_value=None` 所以更新 kv 长度和初始化缓存矩阵。后面开始，每一次循环传入上一次的 hidden_state 来预测下一个 token，也就是说每次传入的矩阵形状都是 \[batch_size, 1, d_model]，并没有重复计算。