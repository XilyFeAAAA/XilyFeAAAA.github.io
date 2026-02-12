---
title: 模型的 generate 方法
date: 2026-02-11T11:19:33+08:00
featuredImage: http://img.xilyfe.top/img/20260211112140352.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 大模型
  - 深度学习
lastmod: 2026-02-13T12:01:27+08:00
---
在 MiniMind 系列的 eval 部分我们已经学习了如何通过 transformers 库里 GenerateMixin 基类来生成文本，这一章学习一下 `model.generate()` 方法到底是怎么实现的。

{{< link_ref "minimind-eval" >}}

## 为什么需要 Generate？

Transformer 模型在训练时有一个 `forward` 方法，是用于针对模型的输入来产生输出，从而计算损失 loss，更新模型的参数。既然有这么一个生成的函数了，为什么 Transformer 中还有专门设计 `generate` 方法来负责在推理时生成文本呢？

这里面主要有两个原因：

1. **模型训练与生成的差异：** 一般情况下（比如在分类任务上），`forward` 方法与 decoding 或者 prediction 时的处理是一致的。但是在 LM 的训练过程中，训练阶段的 `forward` 方法中通常并不采用真正的递归的方式进行逐步处理。在理论上，LM模型通过输入的一组 token，来预测下一个 token。 然后再将新的预测出的 token 与前面的 token 序列进行链接，用来继续预测下一个 token。但是在实际训练中，上面的自回归的训练方式训练的效率非常的低，无法更好的利用GPU的并行处理能力，因此实际训练中一般采用 Teacher-Forcing 这种方式并行的训练。
2. **LM Decoding方法的复杂性：** 相比分类任务，LM 这种自递归的生成任务的解码通常都比较复杂。比如在 LM 或者 ASR 任务中解码方法有很多，比如 `greedy_search(),` `contrastive_search()`，`sample()`，`beam_search()`，`beam_sample()` 等等。通过使用更加复杂的解码的方法，通常可以得到更好的效果。比如 beam_search 的结果比 greedy_search 的结果通常会更好。但是在训练过程中，采用的方法更加类似于 greedy search。因此，设计更好的解码算法也是一个很重要的研究方向。

## 常见的解码方式

### Greedy Search

![image.png](http://img.xilyfe.top/img/20260212215049963.png)

Greedy Search 顾名思义是贪心搜索，即每步选择概率最大的 token。如上图所示，从单词 The 开始，该策略每步都会选择下一步概率最大的词，最后会得到输出序列 The nice woman，总概率是 0.5 * 0.4 = 0.2。Greedy Search 速度最快但也有如下几个缺点：

1. 它可能会**错过全局概率最大的序列**。比如上图中，The dog has 的总概率更大，是0.4 * 0.9 = 0.36。
2. 由于缺少随机性，模型在输出一个重复的 token 之后，有较大可能**陷入重复输出序列**的循环。
3.  Greedy Search 解码方式非常接近模型训练时候的 objective，因此容易**复述训练数据**，缺少了创造性。

### Beam Search

![image.png](http://img.xilyfe.top/img/20260212215305949.png)

为了解决 Greedy Search 可能错过全局最大概率序列的问题，Beam Search 策略经常会被采用，即维护 `beam=n`，保留当前最佳的n 个序列，并且对于每个序列，都在计算最好的 n 个 next token，然后再从 $n*n$ 个结果中，保留 n 个概率乘积最大的序列。比如上图中，假设 beam=2，从 The 开始，会保留 \[The dog, The nice] 两个序列，接着每个序列选取 2 个最佳的 next token，得到4个序列，再从中选择2个最佳序列 \[The dog has, The nice woman]。然而，beam Search 有以下缺点：

1. 在 NLP 任务中一般将 eos_token 视为文本的结尾，也就是 absorbing state。如果某个候选序列达到这个 absorbing state，就不再扩展它。这就会造成 Beam Search 通常会倾向于更短的序列，因为长序列算概率乘积后，数值会相对短序列更小。因此，一般会在得分函数中引入 length normalization 对长度进行归一化。
2. 由于缺少随机性，Beam Search 仍然很可能掉入重复序列的循环。因而一些工作引入了 n-grams penalty 来缓解。最常见的方法是通过将已经看到的 n-gram 的下一个单词的概率设置为 0，来确保没有 n-gram 出现两次。n 是一个超参数，如果 n 设为2，则 2-gram 序列，比如 New York 不会在解码中出现两次。
3. 最后，相比于人类语句一般不太可预测，Beam Search 生成的序列缺少惊喜，因此在需要创造性的生成场景中不是非常合适。

### Random Sampling

![image.png](http://img.xilyfe.top/img/20260212215839262.png)

随机采样策略根据当前的概率来抽签选择 next token。如上图所示，任何词都有一定概率被选择。该方案生成的序列充满了创造性，也相对较少出现重复序列循环问题。但是它生成的语句却很可能不通顺。

这里一般会引入 temperature 来改变生成 next token 的概率分布，使其更偏向于 high probability token。具体来说是通过 $prob=\text{softmax}(\frac{logits}{T})$ 控制文本生成的多样性
- T = 1：原始分布。
- T < 1：分布变尖，大概率的 token 更占优势。
- T > 1：分布变平，小概率的 token 被抬高。
- T → 0：接近 one-hot ，只剩 logit 最大的 token，接近 Greedy Search。

{{< admonition type=question title="temperature=0是不是等同于 Greedy Search？">}} 
错误的，在实践中我们会发现把 temperature 设为 0，每次输出的文本还是有所不同，主要是有三个原因。

首先是因为 **浮点数加法不满足结合律**。我们都知道计算机内部浮点数 float 是用尾数+精度的方法存储的，所以浮点数进行加法可能导致溢出。例如在 python 中 `(1 + 1e-16) - 1e-16=0` 和 `1 + (1e-16 - 1e-16) = 1` 两者答案就不同。在推理过程中，多卡多机并发的通讯延迟，不同线程之间计算速度导致的先后差异，都可能导致计算的先后顺序不同，最终导致结果不同。

另一方面，只从 DeepSeek 问世之后 MoE 架构被广泛采用。如果是我们自己部署推理没啥问题，但是如果用的是大型厂商的服务，他们往往为了提高效率把多个请求合并成一个大 batch 进行推理。在 MoE 中为了效率考虑，每个专家通常都用容量限制，一个专家同时只能处理有限数量的 token。当我们的 prompt 里某个 token 和别人 prompt 里面某个 token 同时竞争一个专家时，谁被分配到备用专家就不一定了，所以输出也不一定。
{{< /admonition >}}

除此之外，在 Random Sampling 基础上还提出了多种采用策略：
1. Top-p Sampling：按照概率从高到低选择 n 个词，使它们的累计概率为 p，在这 n 个词里面随机采样。
2. Top-K Sampling：限制了模型只能从概率最高的 k 个词中按照 Basic Sampling 采样，优点是避免了低概率词汇的影响，提高了文本质量，但 k 可能会导致词汇过于局限，不如 Top-p Sampling 灵活。


### Rank Beam Sampling

![image.png](http://img.xilyfe.top/img/20260212220450547.png)

Adiwardana et al., 2020 提出了 sample-and-rank 解码策略，该方法在对话领域效果很好。其思想是先通过 random sampling（结合temperature调整概率分布）生成出 N 个 sentence，然后再从 n 个 sentence 中选择概率乘积最大的。这种方式通过 random sampling 保留了生成结果的多样性和创造性，后又通过 rank 过滤掉了不通顺的序列。下面两个表格对比了 sample 的结果和 beam search 的结果。明显地，sample 结果多样性会更好。

beam sample 方法是 sample and rank 的改进，原理上类似，相比 sample and rank 在最后才对结果排序去获得最佳的 n 个序列，beam sample **在每一步保留当前最佳的 n 个序列**，既保证了多样性和创造性，又可以**减少在 rank 阶段需要过滤掉的句子**。


### Group Beam Search

![image.png](http://img.xilyfe.top/img/20260212220543236.png)

group beam search 同样是为了解决 beam search 多样性不足的问题，如上图所示，可以发现 beam search 生成的图像描述几乎是重复的，这是由于在搜索树中具有相似的共享路径，导致最终的变化很小。相比之下，group(diverse) beam search 生成的结果则更多样化，也更加类似描述图像的人际差异。

![image.png](http://img.xilyfe.top/img/20260212220555429.png)

group beam search 主要思路是通过将 beam search 中的候选路径进行分组，**在各组内去寻找最优解**。如上图所示，beam search 的候选路径有6条，group beam search 将这6条候选路径两两作为一组，分为三组。每一步都在各组内的词表空间下去取 top-2 的结果作为当前预测的 token，对于当前组来说，通过**对之前组已生成的 token 进行惩罚**，来保证当前组生成的 token 与之前组不同的概率更大，从而更具**多样性**。


## 代码实现

这次我们参考 transformers 库实现一个 GenerateMixin，核心就是 `model.generate()` 方法，我们就实现 Greedy Search 和 TopP、TopK Sampling 这几种比较简单的解码策略。

```python
@torch.inference_mode()
def generate(
    self,
    input_ids: Union[list[int], torch.Tensor],
    attention_mask: Optional[torch.tensor] = None,
    max_new_tokens: int = 8192,
    min_length: int = -1,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: float = DEFAULT_TOP_K,
    eos_token_id: Optional[int] = None,
    use_cache: bool = True,
    num_return_sequences: int = 1,
    do_sample: bool = True,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    past_key_values: Optional[DynamicCache] = None,
) -> torch.Tensor:
```

这里有几个参数需要介绍一下：
1. `max_new_tokens`：控制生成的 token 数量
2. `min_length`：控制文本的最短长度
3. `use_cache`：是不是采用 KVCache
4. `num_return_sequences`：一次生成多条 response
5. `repetition_penalty`：惩罚重复生成的 token

**1. 参数预处理**

```python
if isinstance(input_ids, list):
    input_ids = torch.tensor(input_ids)

initial_len = input_ids.size(-1)
input_ids = input_ids.repeat(num_return_sequences, 1)
if attention_mask is not None:
    attention_mask = attention_mask.repeat(num_return_sequences, 1)
```

这里比较简单，就是把序列复制 `num_return_sequences` 个，这样就可以利用 GPU 并行推理生成多个序列。

**2. 推理生成**

```python
for _ in range(max_new_tokens - initial_len):
	past_len = past_key_values.get_seq_length() if past_key_values else 0
	output = self.forward(
	    input_ids=input_ids[:, past_len:],
	    attention_mask=attention_mask,
	    labels=None,
	    past_key_values=past_key_values,
	    use_cache=use_cache,
	    logits_to_keep=1,
	)
	logits = output.logits[:, -1, :]
	if min_length != -1:
	    logits = min_length_processor(input_ids, logits, eos_token_id, initial_len)
	if repetition_penalty != DEFAULT_REPETITION_PENALTY:
	    logits = repetition_penalty_processor(input_ids, logits, repetition_penalty)
	if do_sample:
	    if temperature != DEFAULT_TEMPERATURE:
	        logits = temperature_warper(logits, temperature)
	    if top_k != DEFAULT_TOP_K:
	        logits = top_k_warper(input_ids, logits, top_k)
	    if top_p != DEFAULT_TOP_P:
	        logits = top_p_warper()
	if do_sample:
	    probs = torch.softmax(logits, dim=-1)
	    next_token = torch.multinomial(probs, num_samples=1)
	else:
	    # greedy_search
	    next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [bs, 1]
```

因为我们采用了 KVCache 所以要对 `input_ids` 进行切片：第一次推理时候就是对整个 prompt 进行推理，后面每次就只需要传入 1 个 token 就好了。这里 KVCache 采用了 transformers 库的 DynamicCache 而不是老版的 `list[tuple]`，所以读文本长度的方式有所不同。然后调用 `forward` 方法生成注意力分数 logits 之后，我们就应用不同的策略包括温度调节、重复惩罚等等，这后面会具体说。 最后根据 `do_sample` 判断是采样 or Greedy Search 来生成 next token。

{{< admonition type=question title="我们用了 KVCache 那么 `logits_to_keep=1` 是不是没啥用了？">}} 
是这样的，我们回忆一下 `logits_to_keep` 这个参数的作用。

```python
 slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
 logits = self.lm_head(hidden_states[:, slice_indices, :])
```

在模型的推理阶段，假如不开启 KVCache，那么每次 `hidden_states` 的形状为 \[batch_size, seq_len, hidden_size]，`lm_head` 的计算开销就是 `(batch_size × seq_len × hidden_dim) @ (hidden_dim × vocab_size)`。但实际上在推理阶段我们只需要最后一个 token 的信息，也就是说 `hidden_states[:, -1, :` 才是我们需要的，这就是 `logits_to_keep` 这个参数的意义。但是有了 KVCache 之后，第二次推理开始我们只传入一个 token，也就是说 `seq_len` 固定为 1 了，那么 `logits_to_keep` 就没有意义了。
{{< /admonition >}}

讨论两个写代码时候有点纠结的问题：attention mask 和 eos_token。

attention mask 别名 padding mask，用来标记 `input_ids` 里面哪些 token 是 pad 上去没有语义的，让每个 token 的注意力不浪费在那些无意义的填充字符上。我们在 softmax 之前对注意力分数进行处理，把 pad 的部分，置为一个非常大的负数。但是因为我们采用了 KVCache，scores 的形状从 \[batch_size, seq_len, seq_len] 变成了 \[batch_size, q_len, v_len]，也就是 \[batch_size, 1, v_len]。我最直接的想法就是，那我们是不是也需要对 attention mask 进行切片，让它长度为 1？

```python
# Padding Mask
if attention_mask is not None:
    assert attention_mask.dim() == 2
scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

```

深入代码之后发现其实是不用的。使用 KVCache 之后，我们生成的 attention mask 的形状其实是 \[batch_size, v_len]。然后我们在第一维 unsqueeze 形状就是 \[batch_size, 1, v_len] 了（这里是多头注意力所以还有一个 head 维度，多了一个 unsqueeze）。

第二个问题是 eos_token。`num_return_sequences` 参数规定了 batch 的数量，我们同时推理生成多个 response。但是由于每个 sequence 都是并行同时推理的，假如一个 sequence 生成了 eos_token 我们怎么控制这个 sequence 后续不生成了呢？transformers 库的解决办法是记录下来哪些 sequence 已经结束了，然后把这些 sequence 的 next token 都设为 eos_token。

```python
unfinished_sequence = torch.ones(num_return_sequences)

for _ in range():
	output = self.forward()
	# ...
	
if eos_token_id is not None:
	unfinished_sequence = unfinished_sequence.mul((next_token != eos_token_id).long())
	next_token = next_token * unfinished_sequence + eos_token_id * (1 - unfinished_sequence)
```

---

```python
def temperature_warper(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / temperature

def top_k_warper(input_ids: torch.Tensor, logits: torch.Tensor, top_k: int) -> torch.Tensor:
    top_k = min(top_k, input_ids.size(-1))

    mask = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(mask, float("-inf"))

def top_p_warper():
    pass

def repetition_penalty_processor(input_ids: torch.Tensor, logits: torch.Tensor, penalty: float) -> torch.Tensor:
    score = torch.gather(logits, 1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    scores_processed = logits.scatter(1, input_ids, score)
    return scores_processed

def min_length_processor(input_ids: torch.Tensor, logits: torch.Tensor, eos_token_id: int, initial_len: int, min_length: int) -> torch.Tensor:
    if input_ids.size(-1) - initial_len < min_length:
        logits[:, eos_token_id] = float("-inf")
    return logits
```

temperature、top_k、top_p 还有 repetion penalty 都是改变了 logits 的分布，min_length 是通过将 eos_token 的概率设为 0 来实现的。


**3. 更新参数**

```python
input_ids = torch.cat([input_ids, next_token], dim=-1)
past_key_values = output.past_key_values if past_key_values else None
if attention_mask is not None:
    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], -1)
```

最后更新一下 KVCache，然后把 `input_ids` 和 `attention_mask` 合并上去。