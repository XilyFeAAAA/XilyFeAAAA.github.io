---
title: MiniMind 学习指北(四)：评估
date: 2026-01-25T13:47:19+08:00
featuredImage: http://img.xilyfe.top/img/20260122134824760.png
authors:
  - Xilyfe
series:
  - minimind
tags:
  - 大模型
  - 深度学习
lastmod: 2026-02-10T03:45:51+08:00
---
>这一章我们需要设计一个脚本来验证大模型的对话能力

## 评估脚本

我们预训练是让模型学会说话的能力，或者说词语接龙的能力，给他一个 prompt 它可以接着说下去。因此我们在处理 prompt 时候需要稍加处理：

```python
inputs = tokenizer.bos_tok_id + prompt
```

这样我们就能得到一个类似 "<im_start> 今天天气" 这样的输入文本了。 下一步我们用 tokenizer 把输入文本变长 token 序列：

```python
input_ids = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
```

在 [MiniMind 学习指北(一)：模型](../minimind-model) 这一节中，我们的模型继承了 Transformers 库的 `GenerationMixin` 这个基类，所以我们可以很方便的调用它的 `generate` 这个方法来生成文本。与此同时为了让模型输出更流程，我们可以调用它的 TextStream 这个类来实现流式输出：

```python
model, tokenizer = init_model()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generated_ids = model.generate(
	inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"].bool(),
    max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
)
```

okey，现在我们就可以使用之前预训练得到的模型权重进行评估，不出所料又爆 BUG 了。。。

## 修改模型

![image.png](http://img.xilyfe.top/img/20260210151653349.png)

我们先来看看这个 BUG 出在哪，Traceback 里面提示我们的保存 KVcache 的数组 `past_key_values` 是一个 **DynamicCache** 对象，不能用索引来取第一个元素。经过研究问题原来是 `GenerationMixin` 基类的 `generate` 方法会调用我们写好的 `forward` 方法，根据 `use_cache` 这个参数帮我们生成 `past_key_values`。然而新版本的 Transformers 库引入了 `DynamicCache` 作为 KV Cache 的标准实现，用于在自回归生成中高效管理注意力键值缓存，彻底替代旧版 `list[tuple]` 格式。

先了解一下 `DynamicCache` 里需要用到的方法：
- `get_seq_length(layer_idx: int) -> int`：获取第 i 层的 KV 长度
- `update(key: Tensor, value: Tensor, layer_idx: int) -> tuple`：类似之前的 `torch.concat` 帮我们更新 key 和 value，并且返回完整 key、value。

```python
class MiniMindModel(nn.Module):
	def forward(self, ...):
		# if past_key_values is None:
        #     past_key_values = [None] * len(self.layers)

        # sta_pos = 0 if past_key_values[0] is None else past_key_values[0][0].shape[2]
        if past_key_values is not None and use_cache:
            sta_pos = past_key_values.get_seq_length()
        else:
            sta_pos = 0
            if use_cache:
                past_key_values = DynamicCache()
	    #...
	    for idx, layer in enumerate(self.layers):
            hidden_states, past_key_values = layer(
                hidden_states,
                idx,
                position_embeddings,
                past_key_values,
                use_cache,
                attention_mask,
            )

class MiniMindBlock(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, past_key_values = self.attention(
            self.attn_norm(hidden_states),
            layer_idx,
            position_embeddings,
            past_key_values,
            use_cache,
            attention_mask,
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.ffn_norm(hidden_states))

        return hidden_states, past_key_values

class GQA(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        # if past_key_value is not None:
        #     key = torch.cat([past_key_value[0], key], dim=2)
        #     value = torch.cat([past_key_value[1], value], dim=2)
        # past_key_value = (key, value) if use_cache else None
        
        if use_cache:
            key, value = past_key_values.update(key, value, layer_idx)
```

## 评估结果

```
💬: 为什么天空是蓝色的
🤖: ？天空之所以呈现蓝色，是因为大气层中的氮气、氧气、水分子和其他气体分子会散射光线，使得天空看起来是蓝色的。

[Speed]: 38.49 tokens/s


💬: 请用Python写一个计算斐波那契数列的函数
🤖: 。函数可以使用任何一个函数来写，但是它需要一个大型的函数来存储，例如while()函数。因此，要编写程序时，要考虑到函数的复杂度，它需要大规模的函数来存储和处理数据。

[Speed]: 38.62 tokens/s


💬: 解释一下"光合作用"的基本过程
🤖: 。光合作用是指光能转化为化学能的过程。在光合作用中，植物、动物和微生物将太阳能转化为有机物，并释放出氧气。这个过程发生在光合作用的过程中，将二氧化碳和水转化为葡萄糖和氧气，同时释放出氧气。
光合作用的另一个基本过程是光合色素的合成。当光合色素通过光能被转化为化学能时，光合色素吸收光能并将其转化为化学能，从而生成葡萄糖和氧气。这个过程在光合作用中起到非常重要的作用，因为光合作用可以产生出植物和动物等光合生物所需的能量。

[Speed]: 36.80 tokens/s



💬: 解释什么是机器学习
🤖: ，并提供一个实际的应用。
机器学习是一种人工智能领域的算法和模型，通过训练计算机程序，使其能够从数据中学习，自动改进和适应，从而实现自动化决策和预测。      汽车，因为它们能够使汽车在道路上行驶，并提高行驶安全性。
机器学习是一种人工智能领域的算法和模型，通过训练计算机程序，使其能够从数据中学习，自动改进和适应，从而实现自动化决策和预测。      
一个实际应用的例子是图像识别。例如，利用深度学习技术，例如卷积神经网络（CNN）可以识别图像中的物体，并将其分类为正面或负面。       
一个实际的应用是自动驾驶汽车。自动驾驶汽车通过使用传感器和算法来实现自主驾驶，如激光雷达、激光雷达和摄像头。这些技术可用于自动驾驶
一个实际的应用是自动驾驶汽车。自动驾驶汽车通过使用传感器和算法来实现自主驾驶，如激光雷达、激光雷达和摄像头。这些技术可用于自动驾驶
汽车，因为它们能够使汽车在道路上行驶，并提高行驶安全性。

[Speed]: 36.99 tokens/s
```

可以看到模型在接龙方面还过得去，下一步我们需要进行监督微调 SFT 进一步提高模型对话能力。

