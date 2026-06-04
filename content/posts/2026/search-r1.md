---
title: Search-R1 学习指北
date: 2026-05-21T20:26:00+08:00
featuredImage: http://img.xilyfe.top/img/20260522112900397.png
authors:
  - Xilyfe
series:
  - 项目笔记
tags: []
lastmod: 2026-06-01T04:24:28+08:00
---
## 1. 背景

### 1.1 和 RAG 区别

传统 RAG 的流程是：
1. 把提示词拿去算向量相似度
2. 把检索出来的内容拼接到 Prompt 模板里
3. 发送给大模型

```text
【系统提示词】
你是一个严谨的助手。请根据以下给出的【参考资料】来回答用户的【问题】。
如果参考资料中没有相关信息，请直接回答“不知道”，不要胡思乱想。

【参考资料开始】
资料 [1]: 蒂姆·库克（Tim Cook），1960年出生，现任苹果公司CEO...
资料 [2]: 奥本大学（Auburn University）成立于1856年，位于美国阿拉巴马州...
资料 [3]: 苹果公司由史蒂夫·乔布斯等人于1976年创立...
【参考资料结束】

【用户问题】
苹果现任CEO的母校是哪年建校的？
```

但是 RAG 的缺陷也很明显，如果检索召回阶段失败了，后面能力再强的 LLM 也是巧妇难为无米之炊。如果我们提问的是苹果现任 CEO 母校的建校时间？那么 RAG 最初只检索到的 **蒂姆·库克** 就不能为后续提供帮助了。而 Search-R1 的思考过程是这样的：

```text
<think>
用户想知道苹果现任 CEO 母校的建校时间。
第一步：我需要先确定苹果现任 CEO 是谁。
-> 发起搜索：【苹果现任 CEO】
-> 收到结果：蒂姆·库克（Tim Cook）。

第二步：我知道名字了，接下来要查他的母校。
-> 发起搜索：【蒂姆·库克 母校 大学】
-> 收到结果：他毕业于奥本大学（Auburn University）。

第三步：最后查这所大学的建校时间。
-> 发起搜索：【奥本大学 建校时间】
-> 收到结果：1856年。

第四步：信息完整，可以回答。
</think>
苹果现任 CEO 蒂姆·库克的母校是奥本大学，该校建于 1856 年。
```

可以看到 Search-R1 解决的是大模型在面对未知或动态信息时，缺乏自主规划和深度推理的问题。而传统 RAG 解决的是“大模型没有企业私域数据/没有实时数据”的问题，它是一个知识搬运工。

### 1.2 为什么用 RL 而不是 SFT

1. SFT 的本质是**行为克隆**，它需要人类或更强的模型（如 GPT-4）提供近乎完美的标准轨迹数据，但面对复杂的、未知的研究型问题，**怎么拆解、换什么关键词、搜几轮能拼出真相，根本没有标准路径**。
2. 高质量的多轮搜索+长思考的 SFT 数据极难标注，成本非常高。而 RL 只需要给模型一个干净的 QA 数据集，模型自己去 rollout 试错，在成千上万种搜索关键词组合中，它自己把最能提炼出正确答案的那条路径给找出来
3. 在传统的 SFT 中，模型是一步一步进行 next token prediction。如果一个复杂问题需要连搜 3 次，模型在第 1 次搜索时如果出现了一点点偏差，这个偏差会在第 2 次、第 3 次搜索时被无限放大，最终彻底脱轨，而 RL 关注的是**长程回报**。

## 2. 数据构造

### 2.1 数据来源

Search-R1 用的数据集是 Huggingface 上的 FlashRAG，这个数据集提供了 question 和 golden_answer，正好适用于 Search-R1 这种 ORM 的 RL 训练。FlashRAG 包含多种类型的 QA 数据：

| 数据集             | 说明                       | 训练/测试用途         |
| --------------- | ------------------------ | --------------- |
| nq              | Natural Questions，单跳事实问答 | 训练 + 测试         |
| hotpotqa        | 多跳问答                     | 训练 + 测试         |
| triviaqa        | 开放域事实问答                  | 测试              |
| popqa           | 长尾实体问答                   | 测试              |
| 2wikimultihopqa | 多跳问答                     | 测试              |
| musique         | 组合式多跳问答                  | 测试              |
| bamboogle       | 多跳推理问答                   | 测试              |
| strategyqa      | 是/否推理问答                  | v0.3 格式奖励版本支持测试 |

>这里多条问答的意思就是 **问题需要经过多步思考**，例如前面提到的"苹果现任 CEO 母校的建校时间"，需要先思考 CEO 是谁，然后它的母校是哪个，然后再查找学校的建校时间。而单跳 QA 就是例如 "现在的美国总统是谁" 这种单次思考就能解决的问题。

### 2.2 数据格式设计

verl 里的数据处理在之前的文章里面提到过，每条样本会被整理成如下结构：

```python
data = {
    "data_source": data_source,
    "prompt": [{
        "role": "user",
        "content": question,
    }],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "target": example["golden_answers"]
        }
    },
    "extra_info": {
        "split": split,
        "index": idx,
    }
}
```

| 字段                               | 含义                              |
| -------------------------------- | ------------------------------- |
| data_source                      | 当前样本来自哪个数据集，例如 `nq`、`hotpotqa`  |
| prompt                           | 对话格式 prompt，通常只有一个 user message |
| ability                          | 任务类型，这里统一为 `fact-reasoning`     |
| reward_model.style               | 奖励类型，当前为 `rule`，表示使用规则奖励函数      |
| reward_model.ground_truth.target | 标准答案列表，用于 EM 奖励计算               |
| extra_info.split                 | 当前样本属于 train/test               |
| extra_info.index                 | 样本在原数据集中的索引                     |

然后 Search-R1 通过 `Search-R1\scripts\data_process\qa_search_train_merge.py` 和 `Search-R1\scripts\data_process\qa_search_test_merge.py` 两个脚本将 FlashRag 数据集中各种 source 的 QA 数据转换为 verl 格式的 parquet 类型数据。用 `Search-R1\scripts\data_process\nq_search.py` 构造符合 verl 格式 parquet 类型的单跳数据集。

>`nq_search` 是早期 NQ-only Search-R1 实验，用来只在 NQ 上训练动态搜索能力，后续 Search-R1 都是用多跳 QA 数据集来训练的。

此外 Search-R1 对提示词也有加工：

```python
def make_prefix(dp, template_type):
    question = dp['question']

    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix

def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            # ...
```

在 prompt 中 Search-R1 就引导模型要求模型先思考，然后需要外部知识时主动发起搜索，最后通过 answer 标签抽取最终答案。


{{< admonition type=info title="数据清洗的 trick">}} 
每条样本在构造前会对 question 做一个简单标准化处理：
1. 去掉问题前后的空白字符；
2. 如果问题没有以问号结尾，则补上 `?`；
3. 保证 prompt 中的问题格式相对统一。

这一步很简单，但对于模型学习稳定输出格式是有帮助的，因为 prompt 末尾始终是规范的问题形式。
{{< /admonition >}}

### 2.3 Parquet 文件如何进入训练

1. 训练脚本中通过如下参数指定数据路径

```bash
data.train_files=$DATA_DIR/train.parquet
data.val_files=$DATA_DIR/test.parquet
```

2. RLHFDataset 加载

verl 用 RLHFDataset 这个类管理数据，它本质就是 `torch.utils.data.Dataset`，内部会用 pandas 库读取 parquet 类型数据集，然后在 `__getitem__` 里面会自动应用 chat_template，并且通过 tokenizer 输出包含 `input_ids`、`attention_mask`、`position_ids` 等信息的字典。

3. rollout

Search-R1 的数据构造阶段并不会真的执行搜索，也不会提前生成 `<information>`。在训练时 verl 会把 batch_size 大小的数据组成一个 DataProto 对象，然后先进行 rollout，这时候才会发生搜索。

```python
final_gen_batch_output = generation_manager.run_llm_loop(
    gen_batch=gen_batch,
    initial_input_ids=first_input_ids,
)
```

4. 奖励

在计算 reward 时候，verl 会根据数据项的 `data_source` 属性选择不同的奖励函数：

```python
if data_source in [
    'nq', 'triviaqa', 'popqa', 'web_questions',
    'hotpotqa', '2wikimultihopqa', 'musique',
    'bamboogle', 'strategyqa'
]:
    return qa_em_format.compute_score_em
```

## 3. 搜索交互机制

这部分主要说明 Search-R1 如何实现多轮 LLM-环境交互循环的。前面提到过 Search-R1 的 rollout 流程是让大模型在需要了解外部知识时候，通过 `<search></search>` 标签进行检索，然后外部环境将检索到的知识用 `<information></information>` 加入 prompt 让大模型继续进行 next token prediction，直到模型获取所有需要的知识，将答案用 `<answer></answer>` 输出。所以第三部分会分 **检索服务如何实现** 以及 **verl 内部的多轮 rollout 流程** 两方面展开。

### 3.1 检索服务

#### 3.1.1 整体设计

Search-R1 把检索器设计为一个独立的 HTTP 服务，训练代码通过 HTTP 调用。这样做的好处是：

1. 检索器和训练进程解耦，可以独立部署和扩展
2. 统一 API 接口，底层可以切换 BM25 / Dense / Google / SerpAPI

```python
@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    # 格式化返回
    ...
```

verl 在 rollout 需要检索时，向 HTTP 服务发送类似 POST 请求，类似：

```json
{
  "queries": ["What is the capital of France?", "Who wrote Dune?"],
  "topk": 3,
  "return_scores": true
}
```

然后得到返回：

```json
{
  "result": [
    [
      {"document": {"title": "France", "text": "...", "contents": "\"France\"\n..."}, "score": 0.95},
      {"document": {"title": "Paris", "text": "...", "contents": "\"Paris\"\n..."}, "score": 0.88},
      ...
    ],
    [...]
  ]
}
```

#### 3.1.2 Retriever 设计

>Retriver 服务支持多种检索后端，包括 Dense Retriver、BM25 Retriver 以及网络检索的 Retriever，这些和大模型关系不大就简单了解一下。

Dense Retriever 的核心是**用神经网络把 query 和 document 都编码成低维稠密向量，通过向量相似度检索**：`score(q, d) = sim(Encoder(q), Encoder(d)) = cosine / dot-product`。

1. 加载数据集的 FAISS 索引和 Encoder 模型
2. 收到请求时：
	1. 用 Encoder 将 query 编码为向量
	2. 用 FAISS 找到 top-k 相似的文档
	3. 返回文档内容以及相似度分数

---

BM25 核心是基于**词频统计**的经典算法，本质是改进版 TF-IDF。

```python
class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)

    def _search(self, query, num, return_score):
        hits = self.searcher.search(query, num)
        # 从 hits 中提取文档内容
        all_contents = [
            json.loads(self.searcher.doc(hit.docid).raw())['contents']
            for hit in hits
        ]
        results = [
            {'title': content.split("\n")[0], 'text': "\n".join(content.split("\n")[1:]), 'contents': content}
            for content in all_contents
        ]
        return results, scores
```

### 3.2 多轮生成流程

Search-R1 用的是老版本的 verl，没有 agent_loop 来生成工具调用的 trajectory，所以他自己手写了一个 rollout generator，这个章节就看看 Search-R1 是怎么进行 rollout 的。

#### 3.2.1 整体设计

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           Multi-Turn Rollout Loop                                  │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                        │
│  │    Prompt    │────▶│   LLM 生成    │────▶│  后处理响应   │                        │
│  │  (rolling)   │     │ generate_seq │     │ postprocess  │                        │
│  └──────────────┘     └──────────────┘     └──────┬───────┘                        │
│                                                    │                               │
│                                                    ▼                               │
│                                         ┌────────────────────┐                     │
│                                         │     解析 Action     │                     │
│                                         │                    │                     │
│                                         │  <search> query    │                     │
│                                         │  <answer> answer   │                     │
│                                         └─────────┬──────────┘                     │
│                                                   │                                │
│                                ┌──────────────────┴──────────────────┐             │
│                                │                                     │             │
│                                ▼                                     ▼             │
│                      ┌────────────────┐                  ┌────────────────┐        │
│                      │ action=search  │                  │ action=answer  │        │
│                      │                │                  │                │        │
│                      │  调用检索服务   │                  │   标记完成      │         │
│                      └───────┬────────┘                  └────────────────┘        │
│                              │                                                     │
│                              ▼                                                     │
│                    ┌──────────────────────┐                                        │
│                    │ 拼接检索结果到       │                                          │
│                    │ rolling state        │                                        │
│                    └──────────┬───────────┘                                        │
│                               │                                                    │
│                               ▼                                                    │
│                           下一轮循环                                                │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

多轮生成流程本质上是一个 **LLM + 环境交互的循环**，LLM 在每一轮中决定是继续搜索还是给出最终答案。假如模型输出 `<search></search>` 标签，那么就会向 Retriever 进行检索，然后把检索到的信息加入 prompt 里面重新进行 rollout，直接模型输出最终结果或者循环次数达到上限。

#### 3.2.2 主循环

- 保存原始 prompt：`{'input_ids': initial_input_ids[:, -max_start_length:]}`
- 右侧所有生成内容 + 检索结果的累积：`{'responses': empty, 'responses_with_info_mask': empty}`
- 活跃样本掩码：`torch.ones(batch_size, dtype=torch.bool)`
- 每个样本的轮次数：`turns_stats = torch.ones(batch_size)`
- 有效动作计数：`valid_action_stats = torch.zeros(batch_size)`
- 搜索动作计数：`valid_search_stats = torch.zeros(batch_size)`

```python
for step in range(max_turns):
    if not active_mask.sum():
        break  # 所有样本都已完成

    # Step 1: 裁剪 rolling state 到有效长度
    rollings.batch = cut_to_effective_len(rollings.batch)

    # Step 2: 仅对活跃样本进行 LLM 生成
    rollings_active = DataProto({k: v[active_mask] for k, v in rollings.batch})
    gen_output = generate_with_gpu_padding(rollings_active)

    # Step 3: 后处理响应（截断到 </search> 或 </answer>）
    responses_ids, responses_str = postprocess_responses(gen_output.batch['responses'])

    # Step 4: 执行环境交互（解析动作 + 调用搜索）
    next_obs, dones, valid_action, is_search = execute_predictions(responses_str)

    # Step 5: 更新活跃掩码
    active_mask = active_mask * ~dones

    # Step 6: 处理检索结果为 token ids
    next_obs_ids = process_next_obs(next_obs)

    # Step 7: 更新 rolling state（拼接响应 + 检索结果）
    rollings = update_rolling_state(rollings, responses_ids, next_obs_ids)

    # Step 8: 更新右侧累积输出
    original_right_side = update_right_side(original_right_side, responses_ids, next_obs_ids)
```


>可以注意到新版本 verl 的 AgentData 里面只保存了每个 trajectory 的 `input_ids` 没有保存 `attention_mask`，因为每条 trajectory 是独立的进行推理的，而不是像 Search-R1 AgentLoop 一样拼成一个大 batch 送入 vLLM。


#### 3.2.3 数据预处理

大模型中不同的场景用不同的 padding 方式，inference 时用的是 left padding，而 training 过程中用的是 right padding。在推理阶段假如我们用 right padding，那么长度较短的句子右侧会充满 `<PAD>`，此时 GPU 并行计算时模型会去预测 `<PAD>` 后面的 Token，或者注意力机制被右侧一堆无意义的 `<PAD>` 干扰，直接导致生成逻辑崩盘。在训练阶段一般都是 Teacher Forcing，我们的数据是 prompt + response，`<PAD>` 填充的部分一般是这个句子已经结束了，所以没有影响。

推理阶段的 left padding 就会存在一个问题，假如某个 batch 的 prompt 长度都很短，而我们设置的 `max_prompt_len` 很大，就会导致 batch 左边填充了很多无意义的 `<PAD>`：

```
sample_0: [0, 0, 0, 0, 5, 8, 3, 7, 2, 6]   有效长度 6
sample_1: [0, 0, 0, 0, 0, 0, 4, 9, 1, 3]   有效长度 4
sample_2: [0, 0, 0, 7, 2, 5, 8, 3, 6, 1]   有效长度 7
```

`cut_to_effective_len` 方法就是找到这个 batch 中最长的句子长度，然后把 batch 张量裁剪到这个长度。这样在送入 vLLM 生成前先裁掉这些无效 token，就可以减少计算量：

```
sample_0: [0, 5, 8, 3, 7, 2, 6]
sample_1: [0, 0, 4, 9, 1, 3, 3]
sample_2: [7, 2, 5, 8, 3, 6, 1]
```

#### 3.2.4 调用 verl 推理引擎

veRL 的多 GPU inference 通常要求 `batch_size` 能被 GPU 数整除，方便按 data parallel rank 均匀切分。如果直接丢给 veRL，某些 rank 分到 2 条，某些 rank 分到 1 条，甚至有些逻辑会因为 shape 不一致出问题。而在多轮 rollout 过程中，如果某些 trajectory 很早就推理完成了，可能会出现最初 `batch_size` 可以被整除，但后续 `active_batch` 变少导致无法整除的情况，所以 Search-R1 做了一层 padding：

```python
if num_gpus <= 1:
    return self.actor_rollout_wg.generate_sequences(active_batch)

batch_size = active_batch.batch['input_ids'].shape[0]
remainder = batch_size % num_gpus
if remainder == 0:
    return self.actor_rollout_wg.generate_sequences(active_batch)

# padding 补充 batch_size
padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
# 清除多余的 padding 部分
return padded_output
```

---

然后先看看 Search-R1 是怎么对 batch 进行处理的：

```python
padding_size = num_gpus - remainder
padded_batch = {}

for k, v in active_batch.batch.items():
    # Use first sequence as padding template
    pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
    padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
padded_active_batch = DataProto.from_dict(padded_batch)
for key in padded_active_batch.batch.keys():
    padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
# Generate with padded batch
padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
```

假设：

```python
input_ids.shape = [13, 4096]
attention_mask.shape = [13, 4096]
position_ids.shape = [13, 4096]
```

那么 Search-R1 会把 batch 0 复制 `num_gpus - remainder` 分加在 DataProto 的后面，再交给 vLLM 进行推理，最后生成结束再把这部分删掉：

```python
trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
padded_output.batch = trimmed_batch
```

---

在 verl 那篇文章里面我已经梳理过整个 inference 的流程了，这里简单再过一遍。

```text
Search-R1
  -> actor_rollout_wg.generate_sequences()
    -> FSDP ActorRolloutRefWorker.generate_sequences()
      -> self.rollout.generate_sequences()
        -> vLLMRollout.generate_sequences()
          -> self.inference_engine.generate()
```

当我们调用 `actor_rollout_wg` 也就是 actor worker group 的 `generate_sequences` 方法时，verl 会将数据进行拆分，然后通过 ray 分发给不同 GPU 上的 worker。ActorRolloutRefWorker 是 veRL 的一个 Ray worker 类，它有三个不同的角色 actor、rollout 和 ref，会根据不同的配置承担不同职责。当它负责 rollout 时候，它就依赖于内部的成员变量 `self.rollout`，也就是调用链中的 vLLMRollout。vLLMRollout 内部会初始化 vLLM inference engine，它就是 verl 对 vLLM 的封装。

```python
self.inference_engine = LLM(
    actor_module,
    tokenizer=tokenizer,
    model_hf_config=model_hf_config,
    tensor_parallel_size=tensor_parallel_size,
    dtype=config.dtype,
    enforce_eager=config.enforce_eager,
    gpu_memory_utilization=config.gpu_memory_utilization,
    skip_tokenizer_init=False,
    max_model_len=config.prompt_length + config.response_length,
    load_format=config.load_format
)
```

简单来说，当 Search-R1 的 LLMGenerationManager 调用 `actor_rollout_wg` 的推理方法时候，verl 会通过 ray 在各个 GPU 上同时启动 vLLM 进行推理。

#### 3.2.5 响应处理

`_postprocess_responses` 负责在生成的文本第一个 `</search>` 或 `</answer>` 处截断：

```python
if '</search>' in resp:
    resp = resp.split('</search>')[0] + '</search>'
elif '</answer>' in resp:
    resp = resp.split('</answer>')[0] + '</answer>'
```

这确保每轮生成只包含一个完整的动作。

#### 3.2.6 环境交互

`execute_predictions` 将所有动作为 search 的 trajectory 收集起来，然后发送 HTTP 请求获取检索结果，检索返回的文档被格式化为结构化文本：

```python
def _passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference
```

最终包裹在 `<information>` 标签中作为 observation 返回。如果发现 LLM 输出格式不正确（既没有 `<search>` 也没有 `<answer>`），Search-R1 会把错误提示加进 prompt 里面：

```text
My previous action is invalid. 
If I want to search, I should put the query between <search> and </search>. 
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.
```

#### 3.2.7 更新状态

每轮生成后，`_update_rolling_state` 将**当前响应**和**检索结果**拼接到 rolling state 中，作为下一轮的输入：

```python
new_input_ids = concatenate_with_padding([
    rollings.batch['input_ids'],  # 之前的 rolling state
    cur_responses,                 # 本轮 LLM 响应
    next_obs_ids                   # 本轮检索结果
])

# 裁剪到 max_prompt_length
new_input_ids = new_input_ids[:, -max_prompt_length:]
new_rollings = DataProto.from_dict({
    'input_ids': new_input_ids[:, -max_len:],
    'position_ids': new_position_ids[:, -max_len:],
    'attention_mask': new_attention_mask[:, -max_len:]
})
```

这里 Search-R1 还会维护每个 trajectory 的 `info_mask`，它的作用是在计算 loss 时让检索结果不参与梯度计算，类似 `attention_mask` 不让 `<PAD>` 参与梯度计算。

#### 3.2.8 存在的问题

Search-R1 的 multi-turn rollout 本质上是把整个对话历史（system prompt + 多轮 `<think>`/`<search>`/`<result>` + 当前生成）拼成一个长序列，再送到 vLLM 继续生成。当这个序列超过 `max_prompt_length` 时，vLLM 默认**从左侧截断** 保留最近的 token：

```python
max_len = min(self.config.max_prompt_length, effective_len)  
new_rollings = DataProto.from_dict({  
    'input_ids': new_input_ids[:, -max_len:],  # 取最右侧，即丢弃最左侧（system prompt + 原始问题）  
    ...  
})
```

这有可能导致 rollout 出现错误：
1. 生成格式崩坏：system prompt 通常定义了输出格式（比如要用 `<think>`, `<search>`, `<answer>` 等 XML tag）。截掉之后模型就失去了这部分指令，很容易退化回普通对话格式，导致后续的 tool parser 解析失败，或者直接输出 raw text。
2. 搜索历史丢失：模型看不到前几轮已经搜了什么，会重复发出相同的 `<search>` query，浪费 turns。

所以需要在 swanlab 上监控每一轮 rollout 的长度，如果被截断了很可能训练出问题，需要调整 `max_prompt_length`。但这个问题在训练时不会有影响，因为 Search-R1 rollout 结束后返回的是完整未截断的整个 trajectory：

```python
def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
	# original_left_side  → 保存初始 prompt 的固定副本（不变）  
	# original_right_side → 累积所有 responses（从右侧增长）
	return self._compose_final_output(original_left_side, original_right_side, meta_info)
```


## 4. 奖励设计

>Search-R1 的灵感来源于 Deepseek-R1，就是单单通过一个 outcome-based reward（最终答案是否正确）让模型学会什么时候搜索、搜什么、如何利用检索结果，而不需要细粒度的 prm。

之前 verl 的笔记里面提到过，在 verl 里面自定义奖励函数的方法很多，而 Search-R1 是自定义了奖励函数并且重写了 Reward Manager。

### 4.1 方案一

Search-R1 默认训练脚本使用的是 `qa_em.compute_score_em` 这个 reward function：

```python
def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError
```

对于上述 data_source 的训练数据集他都会用 `qa_em.compute_score_em` 来打分，假如抽取出来的答案为正确答案就给 1 分，打错 0 分，没有像 r1 一样给格式奖励：

```python
def compute_score_em(solution_str, ground_truth, format_score=0., score=1.):
    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
```

然后 `extract_solution` 这个抽取答案的函数实际上就是用 re 从文本中提取被 `<answer></answer>` 包裹的文本。但是要注意，由于 Search-R1 的 prompt 是一个 one-shot prompt，所以正常来说会抽取出 2 个被包裹的文本，第二个才是模型生成的：

```text
[prompt 中的示例 <answer> Beijing </answer>]
[模型生成的 <answer> final answer </answer>]
```

然后 `em_check` 也不是单纯的判断两个字符串是不是相等，而是会进行一些 normalization 的操作：
1. 转小写；
2. 移除标点；
3. 移除英文冠词 `a / an / the`；
4. 合并多余空格。

### 4.2 方案二

方案一的缺点在于奖励设计的太过严苛了，一是对格式依赖强，其次对 EM 较严格，所以 Search-R1 还有一个备用方案位于 `verl/trainer/main_ppo_format.py`：

|条件|返回值|直觉|
|---|---|---|
|无法抽取答案，格式合法，检索正确|`structure_format_score + retrieval_score`|虽未回答，但流程正确且搜到答案|
|无法抽取答案，格式合法，检索不正确|`structure_format_score`|流程正确，但没有有效答案|
|无法抽取答案，格式非法|`0`|完全失败|
|答案正确，格式合法|`score`|最优，默认 1|
|答案正确，格式非法|`score - structure_format_score`|答案对，但流程格式不好|
|答案错误，格式合法，检索正确|`structure_format_score + retrieval_score`|搜到了，但没答对|
|答案错误，格式合法，检索不正确|`structure_format_score`|流程对，但搜索/回答无效|
|答案错误，格式非法|`final_format_score`|至少输出了答案标签，但整体结构差|

### 4.3 reward flow

Search-R1 的 Reward Manager 对每条样本做以下事情：

```python
prompt_ids = data_item.batch['prompts']
response_ids = data_item.batch['responses']

valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

valid_prompt_ids = prompt_ids[-valid_prompt_length:]
valid_response_ids = response_ids[:valid_response_length]

sequences = torch.cat((valid_prompt_ids, valid_response_ids))
sequences_str = tokenizer.decode(sequences)
```

首先它去除了 prompt 和 response 两侧的 padding token，然后组合到一起进行 decode，之后送入 reward function 打分。veRL 的 PPO/GRPO 训练期望 reward 是一个和 `responses` 同 shape 的 tensor：

```python
reward_tensor.shape == responses.shape
```

但 Search-R1 的规则奖励是 outcome reward，只有一个标量分数。因此 `RewardManager` 会创建一个全 0 的 tensor，然后把分数写到**最后一个有效 response token** 上，最后把 reward tensor 放进 DataProto 向后流动：

```python
reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
reward_tensor[i, valid_response_length - 1] = score
batch.batch['token_level_scores'] = reward_tensor
# ...
batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
```

之后就进入计算 advantage 的部分了，`compute_advantage` 会把计算得到的优势和回报一起汇入 DataProto 里面：
- 假如用的是 PPO 那么就会用 GAE 反向计算 advantage。
- 假如用的是 GRPO 那么就会计算组内归一化的均值，然后平摊给每一个 token 位置。

```python
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data
```


## 5. RL 算法选择

### 5.1 PPO &  GRPO

$$
\begin{align}
L^{PPO}(\theta) &= \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \ \text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right] \\
L^{\text{GRPO}}(\theta) &= \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right) - \beta D_{\text{KL}} \right]
\end{align}
$$


观察两个公式我们可以发现，GRPO 和 PPO 的 actor loss 实际上是一个形式，只不过他们两个的优势 advantage 计算方式不同，而且 GRPO 加了一个 KL 惩罚。所以 verl 中二者共用同一个 PPO clipped policy loss：

```python
def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl
```

>这里可能你会发现一个很奇怪的问题，GRPO 的 policy loss 不是先对每条序列做 token 平均，再对 G 条序列做平均吗？为什么他可以和 token-level 的 PPO 复用一个 policy_loss 计算函数呢？
>原因我们在 DAPO 的文章里面提到过，GRPO 原始公式的 seq-mean-token-mean 会导致在长 CoT 序列中梯度贡献被稀释，无法学习关键推理步骤。所以 verl 在 GRPO 中就参考 DAPO 把 loss 改成 token-level 了，和 PPO 都用 `compute_policy_loss` 计算 policy loss。

### 5.2 core_algos

我们回忆一下训练的调用链：

```text
train_ppo.sh / train_grpo.sh
  ↓
verl.trainer.main_ppo
  ↓
RayPPOTrainer.fit()
  ↓
LLMGenerationManager.run_llm_loop()
  ↓
actor_rollout_wg.generate_sequences()
  ↓
多轮 search rollout 得到完整 responses
  ↓
actor_rollout_wg.compute_log_prob()
  ↓
ref_policy_wg.compute_ref_log_prob()    可选
  ↓
critic_wg.compute_values()              PPO 需要
  ↓
reward_fn(batch)
  ↓
compute_advantage(...)
  ├── GAE  → PPO
  └── GRPO → group outcome advantage
  ↓
_create_loss_mask()
  ↓
actor_rollout_wg.update_actor()
  ↓
DataParallelPPOActor.update_policy()
  ↓
core_algos.compute_policy_loss()
```

由于 verl 中采用的训推分离，推理时需要重新计算 response 的 `log_probs`。PPO 用 critic 计算完 value 和 reward 之后就可以计算 advantage 了。PPO 用的是 GAE 计算优势：

```python
def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor, gamma: torch.Tensor, lam: torch.Tensor):
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns
```

GAE 的计算就是一个 反向遍历，在手写 PPO Trainer 那个文章讲过。对于 GRPO 来说它不需要 critic，只要有了 outcome-reward 它就可以计算组内相对优势：

```python
def compute_grpo_outcome_advantage(
	token_level_rewards: torch.Tensor,
	eos_mask: torch.Tensor,
	index: torch.Tensor,
	epsilon: float = 1e-6
):
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores
```

### 5.3 kl penalty

verl 中 kl penalty 有两种处理方式。第一种方法就是把 kl penalty 直接加在 reward 里面：

$$
r'_t = r_t - \beta \text{KL}_t
$$

例如用 PPO 训练就会设 `actor_rollout_ref.actor.use_kl_loss = False`，这样就会调用 `apply_kl_penalty` 把 kl penalty 加入 loss：

```python
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = (
        data.batch["info_mask"]
        if "info_mask" in data.batch
        else data.batch["attention_mask"]
    )
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"],
            data.batch["ref_log_prob"],
            kl_penalty=kl_penalty,
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}

    return data, metrics
```


第二种方法就是像 GRPO 一样，把 kl penalty 加到 actor loss 里面。当我们在脚本中设置：

```bash
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

训练器不在 reward 中扣 KL：

```python
batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
```

而是在 actor loss 中加 kl penalty：

```python
kld = core_algos.kl_penalty(
    logprob=log_prob,
    ref_logprob=ref_log_prob,
    kl_penalty=self.config.kl_loss_type,
)
kl_loss = masked_mean(kld, response_mask)
policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
```

## 6. 训练流程与架构

>verl 的具体细节在 [从零开始学 verl 框架](https://xilyfeaaaa.github.io/posts/verl/) 里面已经介绍过了，这边就整体串联一遍。

首先，veRL 顶层是一个 **single-controller** 的结构。也就是说，整个 PPO/GRPO 训练 step 的逻辑顺序由一个 driver 进程控制。在 Search-R1 里这个 controller 主要就是 `RayPPOTrainer.fit()`。它自己不直接在本进程里跑所有 GPU 计算，而是按流程调用不同的 Ray WorkerGroup：

- `actor_rollout_wg.generate_sequences()`：rollout 生成 trajectory。
- `ref_policy_wg.compute_ref_log_prob()`：计算 reference policy logprob。
- `critic_wg.compute_values()`：PPO/GAE 下计算 values。
- `critic_wg.update_critic()`：更新 critic。
- `actor_rollout_wg.update_actor()`：更新 actor。

训练循环一开始就是从 dataloader 取一个 batch：

```python
for epoch in range(self.config.trainer.total_epochs):
    for batch_dict in self.train_dataloader:
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        batch = batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.n_agent,
            interleave=True,
        )

        gen_batch = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"]
        )
```

取出一个 batch 之后，driver 会调用 `actor_rollout_wg.generate_sequences()` 进行 rollout 生成 trajectory：

```python
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
batch.non_tensor_batch["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
    dtype=object,
)
batch = batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n,
    interleave=True,
)
batch = batch.union(gen_batch_output)
```

这里的 `actor_rollout_wg` 是一个 Ray WorkerGroup。调用它的方法时，driver 并不是自己执行生成，而是通过 Ray 把任务分发到多个 GPU worker 上。

{{< admonition type=info title="相关概念：Ray、Worker、WorkerGroup、Ray Actor" >}}
Ray 是一个分布式执行框架。veRL 用 Ray 在多个 GPU 上启动多个远程进程，每个远程进程通常绑定一张 GPU。

在这套代码里可以这么理解：

```text
Driver / Single Controller
  └── actor_rollout_wg  （driver 本地的 WorkerGroup 代理）
        ├── Ray actor rank 0 / GPU 0 / ActorRolloutRefWorker
        ├── Ray actor rank 1 / GPU 1 / ActorRolloutRefWorker
        ├── Ray actor rank 2 / GPU 2 / ActorRolloutRefWorker
        └── Ray actor rank 3 / GPU 3 / ActorRolloutRefWorker
```

几个概念的关系是：

| 概念 | 在代码里的形态 | 作用 |
|---|---|---|
| Driver | `RayPPOTrainer.fit()` 所在进程 | 控制整个 PPO/GRPO step 的逻辑顺序 |
| WorkerGroup | `RayWorkerGroup` | driver 侧的代理，负责把一次调用分发到多个 worker |
| Ray actor | Ray 远程 Python 进程 | 真正运行在 GPU 上的进程 |
| Worker | `ActorRolloutRefWorker` / `CriticWorker` 等 | Ray actor 里面执行模型计算的对象 |

当我们调用 WorkerGroup 的 `generate_sequences()` 方法时，WorkerGroup 会：

1. 根据 worker 数量把 `DataProto` 切成多份。
2. 把每份数据通过 Ray RPC 发给对应 GPU 上的 worker。
3. 每个 worker 都执行自己的 `generate_sequences()`。
4. 最后把每个 worker 的生成结果 concat 起来，返回给 driver。
{{< /admonition >}}

前面说到，driver 调用 WorkerGroup 以后，每个 GPU 上的 `ActorRolloutRefWorker` 都会执行自己的 `generate_sequences()`：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def generate_sequences(self, prompts: DataProto):
    prompts = prompts.to('cuda')
    recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

    assert self._is_rollout

    if self._is_offload_param:
        load_fsdp_param_and_grad(
            module=self.actor_module_fsdp,
            device_id=torch.cuda.current_device(),
            load_grad=self._is_offload_grad,
        )

    prompts.batch = prompts.batch.cuda()
    prompts.meta_info.update({
        'eos_token_id': self.tokenizer.eos_token_id,
        'pad_token_id': self.tokenizer.pad_token_id,
    })

    with self.rollout_sharding_manager:
        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)

    if self._is_actor and recompute_log_prob:
        old_log_probs = self.actor.compute_log_prob(data=output)
        output.batch['old_log_probs'] = old_log_probs

    output = output.to('cpu')

    if self._is_offload_param:
        offload_fsdp_param_and_grad(...)

    torch.cuda.empty_cache()
    return output
```

verl 是一个训推分离的框架，训练用的是 FSDP 推理用的是 vLLM，而参数更新发生在 FSDP 的模型上。所以当一个 step 训练结束下个 step 进行 rollout 推理时，verl 需要把 FSDP 模型的权重拷贝到 vLLM 上，保证一致性。由于显存有限，所以 verl 中默认开启 vLLM 和 FSDP 权重的 offload，也就是使用结束自动把模型权重从 GPU 上 offload 到 CPU。所以如果 FSDP 参数之前 offload 到 CPU，就先 load 回 GPU，再进入 sharding manager 里从 FSDP 导出权重并同步给 vLLM，用 vLLM 做 rollout。

{{< admonition type=info title="Hybrid Engine">}} 
这里详细介绍一下 verl 里面的训推分离机制，前面我们说到 `ActorRolloutRefWorker` 这个 worker 同时负责 actor 训练和推理，它是怎么做到的呢？ `ActorRolloutRefWorker` 有两个很重要的成员变量：

1. `self.actor_module_fsdp`： HuggingFace causal LM 包一层 FSDP。
2. `self.rollout`：如果用的是 vLLM 进行推理就是一个 vLLMRollout 类，负责推理。

当我们调用 `ActorRolloutRefWorker` 的 `update_actor` 方法时，它就会对 `self.actor_module_fsdp` 进行训练，包括计算 logprobs，计算 policy loss，forward 和 loss backward 等等。当我们调用 `generate_sequence` 方法时就需要 vLLM 了，此时 verl 就会通过 rollout_sharding_manager 从 FSDP `state_dict()` 导出参数给 vLLM，我们用 `self.rollout` 这个 vLLM 对象就可以进行推理了。

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def generate_sequences(self, prompts: DataProto):
    prompts = prompts.to('cuda')
    recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

    if self._is_offload_param:
        load_fsdp_param_and_grad(...)

    prompts.batch = prompts.batch.cuda()
    prompts.meta_info.update({
        'eos_token_id': self.tokenizer.eos_token_id,
        'pad_token_id': self.tokenizer.pad_token_id,
    })

    with self.rollout_sharding_manager:
        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)

    if self._is_actor and recompute_log_prob:
        old_log_probs = self.actor.compute_log_prob(data=output)
        output.batch['old_log_probs'] = old_log_probs

    output = output.to('cpu')
    if self._is_offload_param:
        offload_fsdp_param_and_grad(...)
    torch.cuda.empty_cache()
    return output
    
class FSDPVLLMShardingManager(BaseShardingManager):
	def __enter__(self):
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        params = self.module.state_dict()
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        # Copy, not share memory
        load_format = 'hf' if self.full_params else 'dtensor'
        self.inference_engine.sync_model_weights(params, load_format=load_format)
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del params
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        self.inference_engine.offload_model_weights()
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)
        
        self.module.train()
        torch.cuda.empty_cache()
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
```

从 `fsdp_workers.py` 的代码中可以看到，进入 FSDPVLLMShardingManager 这个上下文管理器之后，会先把 FSDP 的模型权重 copy 到 vLLM 里面。然后才调用 `self.rollout.generate_sequences(prompts=prompts)` 用 vLLM 进行推理。等退出上下文管理器后再把 vLLM 的模型权重 offload 掉，减少显存占用。
{{< /admonition >}}

如果你深入代码可能会发现一个问题，`actor_rollout_wg.generate_sequences()` 会按 DP 切分 DataProto，但是 vLLM 推理是 TP 切分的啊，所以不应该用 TP 切分吗？假如把 batch 按照 DP 切分为 n 份，每个 worker 持有 $\frac{1}{n}$ 的数据，那他也只拥有一个 GPU 怎么能 TP 并行推理呢？实际上 vLLM 的 TP 不是发生在“一个 worker 内部”，而是发生在“多个 worker 组成的 TP group 之间”。

这里举个例子，假如有 4 个 GPU 并且 TP=2。一开始整个大 batch 被切分为 $\frac{1}{4}$ 给每个 worker，由于 TP=2 所以 4 个 GPU 也会被分为 2 个 TP Group。等进入推理时候，相同 TP Group 的 GPU 会进行 allreduce，这样 GPU0 和 GPU 1 都会得到 $\frac{1}{2}$ batch 的数据，然后他们就可以在组内进行 TP 并行了。

{{< admonition type=question title="TP 和 DP 是什么">}} 
假如你不了解 TP 和 DP，这里简要补充前置知识。
1. DP 也就是 Data Parallel 数据并行，把一个大 batch 拆为多个 micro batch 在多个 GPU 上 forward，这是用通信量换时间。
2. TP 是 Tensor Parallel 张量并行，把一个模型的完整权重拆为多个部分放在多个 GPU 上，这是用通信量换显存。

训练的瓶颈是**梯度同步和参数更新**，batch size 越大越好（减少梯度噪声），DP 天然 scale batch。而推理的瓶颈是 **KV Cache 显存** 和 **单序列的自回归延迟**，长序列的 KV Cache 可能撑爆单卡显存，所以适合用 TP。但是张量并行的通信很重，所以尽可能的减少 TP。
{{< /admonition >}}

这样我们就成功 rollout 完成了，接下来计算 actor 的 logprobs、update actor 等也是由 ray 统一方法任务，在多个 GPU 上并发进行。

## 7. 工程细节

### 7.1 冷启动

对于参数量小的模型，例如 Qwen2.5-3B，它的 instruction follow 能力是比较差的，即使我们在 system prompt 里面要求按照 `<think>`、`<search>` 等标签输出它可能也会出现问题，导致强化学习根本没办法收敛，组内 reward 都是零。所以 SFT 冷启动就是先用一些标注好的轨迹数据做一轮监督微调，让模型学会基本的标签格式和搜索行为模式。

我用的是 Llama-Factory 做的 sft，具体可以看 LlamaFactory 的 README：

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

```yaml
### model
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: identity,alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048
max_samples: 1000
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen3-4b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# eval_dataset: alpaca_en_demo
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500

```

这些参数还是比较基础的，需要注意一下 llamafactory 支持的数据集格式为 sharegpt 或者 alpaca：

```python
# alpaca
[
    {
        "instruction": "任务指令",
        "input": "可选的输入上下文",
        "output": "期望的输出响应"
    }
]
# sharegpt
[
    {
        "conversations": [
            {
                "from": "human",
                "value": "用户说的话"
            },
            {
                "from": "gpt",
                "value": "助手的回复"
            },
            {
                "from": "human",
                "value": "用户下一句话"
            }
        ],
        "system": "可选的系统提示词"
    }
]
```

然后使用这些自定义数据集需要我们重写 `dataset_info.json` 文件：
- 如果数据集是标准的 alpaca 格式，那么只需要定义文件名即可

```json
{
  "my_dataset": {
    "file_name": "my_data.json"
  }
}
```

- 如果是标准的 sharegpt 格式，那么需要指定列名

```json
{
  "chat_dataset": {
    "file_name": "chat.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

- 如果是 GPT 的 OpenAI messages 格式，我们需要指明各个 tag 的名字

```json
{
  "openai_dataset": {
    "file_name": "openai.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

### 7.2 state masking

Search-R1 用 state masking 来保证只有模型生成的 token 会参与训练，也就是说我们检索得到的 `<information></information>` 不会计算 kl penalty 或者 loss 进行反向传播。state masking 类似 attention mask，attention mask 是记录哪些 token 是 pad token，而 state masking 就是用 0/1 掩码标记哪些 token 是由模型生成的，接下来看看它的实现逻辑。

首先在 rollout 过程中，Search-R1 会一直维护两个变量：

```python
original_left_side = {
    'input_ids': initial_input_ids[:, -self.config.max_start_length:]
}

original_right_side = {
    'responses': initial_input_ids[:, []],
    'responses_with_info_mask': initial_input_ids[:, []]
}
```

- `original_left_side` 记录了初始 prompt，在整个 rollout 过程中不变
- `original_right_side` 中 `responses` 保存完整右侧序列，包括模型输出和搜索返回的信息，`responses_with_info_mask` 把非模型生成的 token 也就是 information 部分填充为 PAD token。

在每一轮 rollout 中都会得到 `cur_responses` 和 `next_obs_ids` 就是这一轮生成的 token 和 search 结果，然后 Search-R1 会用 `_update_right_side` 方法更新 `original_right_side`。等到 rollout 结束，Search-R1 就会用 `_compose_final_output` 方法把这些信息整合为 DataProto，里面包含了 `input_ids` 等信息：

```python
    def _compose_final_output(self, left_side: Dict, right_side: Dict, meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output
```

代码里面的 `info_mask` 就是前面提到的 state masking，非模型生成的 token 用 PAD token 填充了，`create_attention_mask` 这个方法会生成对应的 0/1 mask 张量。

>为什么不能把 `attention_mask` 和 `info_mask` 合在一起呢？
>因为 `attention_mask` 是在推理中使用的，目的是 transformer 结构中让每个 token 的注意力不浪费在那些无意义的填充字符上，在 softmax 之前对注意力分数进行处理。把注意力分数里那些不希望关注的部分置为一个非常大的负数，这样 softmax 之后它们的注意力权重就会接近于 0。而 `info_mask` 是在计算 loss 的时候用，所以两个不能混在一起。

最后在 update actor 计算 policy loss 时候，Search-R1 会先用 `_create_loss_mask` 把 `info_mask` 截长之后保存到 `loss_mask`：

```python
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch["responses"].shape[-1]
        response_mask = batch.batch["attention_mask"][:, -response_length:]

        loss_mask = batch.batch["info_mask"][:, -response_length:]
        batch.batch["loss_mask"] = loss_mask

        metrics.update(
            {
                "state_tokens/total": loss_mask.sum().item(),
                "state_tokens/coverage": (loss_mask.sum() / response_mask.sum()).item(),
            }
        )

        return batch, metrics

```

然后计算 loss 时候直接乘上 `loss_mask` 就好了：

```python
def _compute_loss(self, batch):
    loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()
    labels = batch['input_ids'][:, 1:].cuda()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = self.fsdp_model(input_ids=batch['input_ids'],
                                 attention_mask=batch['attention_mask'],
                                 position_ids=batch['position_ids'],
                                 use_cache=False)  # prevent model thinks it it generating
    logits = output.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels.contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    loss = loss * loss_mask
```

### 7.3 超参

| 参数                       | 说明                                                                                                                                     |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `n_agent`                | 每个 prompt 采样几个 trajectory，如果设置太小可能导致组内 reward 相同方差为 0 训练失败，如果太大会导致 OOM。                                                                |
| `temperature`            | GRPO 算法需要保证探索性，这样组内不同 trajectory 不同才会产生方差，促使 RL。                                                                                       |
| `lr`                     | 学习率一般设置在 1e-6，小于 PT 和 SFT，如果学习率过大可能导致灾难性遗忘或者 reward hack。                                                                              |
| `critic.lr`              | critic model 的学习率默认是 1e-5 比较小，因为 critic model 是从头开始训练预测 return 的能力。                                                                    |
| `train_batch_size`       | 每个 step 取出的样本数量，实际上每个 step 训练的样本数还需要乘上 `n_agent`。                                                                                      |
| `ppo_mini_batch_size`    | 每个 step 训练的样本数量实际为 `train_batch_size * n_agent`，GRPO 会切成 mini-batch，所以有点 off-policy。                                                   |
| `ppo_micro_batch_size`   | 这个的作用是梯度累计，比如 n 条训练样本但是显存只能放下 m 条，那么就可以通过 $\frac{n}{m}$次梯度累计达到相同效果。                                                                    |
| `max_turns`              | rollout 的最多轮次，需要在 wandb 里面注意训练中每轮 rollout 还剩下多少 trajectory。如果大量样本 rollout turn 很短，那么可能他们根本没有搜索，如果大量样本达到 max_turn 还没有结束，说明可能陷入了循环哪里出错了。 |
| `max_prompt_length`      | prompt 的最大长度。                                                                                                                          |
| `max_response_length`    | 单次生成的最长 token 数量。                                                                                                                      |
| `topk`<br>               | 检索返回的 document 数量。                                                                                                                     |
| `gpu_memory_utilization` | vLLM 的 GPU 利用率，由于除了推理框架还有别的部分占用 GPU，所以 vLLM 的 GPU 利用率不好设置，太高容易 OOM 太低效率低。需要根据 batch_size 和 模型大小等多次修改。                                  |

超参设定的几个 tips：
1. `warmup_ratio` 的默认值为 0.285 会导致大部分时间都在预热（学习率从 0 逐渐提高到 1e-6），实际上 RL 不需要这么长的预热，降低 `warmup_ratio` 到 0.015 提高效率
2. `max_turens` 默认值为 2 轮，但复杂问题需要多接几轮。加到 4 轮之后显存压力明显增加一因为上下文变长了。所以需要在`max_response_length`和`max_obs_length`上做取舍。根据 Search-R1 提供的公式，在默认配置的情况下增加 2 轮多需要 2000 token。

```python
max_prompt_length = max_start_length + max_response_length * (max_turns - 1) + max_obs_length * max_turns
```

{{< admonition type=question title="多个 batch_size 的关系">}} 
每个 step 都会取 `train_batch_size` 个样本，如果采用 GRPO 那么会对每个 prompt 进行 repeat `n_agent` 次，所以每个 step 实际训练用到的样本总数是 `train_batch_size * n_agent`。由于显存有限没办法一次性训练，所以 verl 会把这些样本拆成大小为 `ppo_mini_batch_size` 的小块。我们都知道让 `batch_size` 适度增大训练效果更好，梯度估计的近似越准、噪声越低，但是它也受到 GPU 显存的限制。所以 verl 把 `ppo_mini_batch_size` 的小块再切成 `ppo_micro_batch_size` 更小块进行梯度累计，在数学层面没有任何影响。
{{< /admonition >}}

## 8. 评估

假如在配置文件中指定了 `val_only` 那么 Search-R1 会直接复用 `_validate()` 方法进行评估：

```python
if self.val_reward_fn is not None and self.config.trainer.get(
    "val_before_train", True
):
    val_metrics = self._validate()
    pprint(f"Initial validation metrics: {val_metrics}")
    logger.log(data=val_metrics, step=self.global_steps)
    if self.config.trainer.get("val_only", False):
        return
```

Search-R1 原论文里面是用 7 个 QA 数据集对直接回答、CoT、RAG、SFT 等多个方法进行对比进行 exact match 评分，最后结论是：
1. Search-R1 相较于 RAG 有 24% 的提升。
2. 模型参数量越大 Search-R1 的提升越明显
3. GRPO 和 PPO 相比，PPO 效果更好一些后期更稳定。GRPO 收敛更快，但是由于后期组内 reward 都比较高了，方差低梯度小，训练没那么稳定。不过 GRPO 少了 critic model 显存占用少了很多。

## 9. Notes

### 9.1 训练监控

RL 的 loss 曲线没办法反应训练的效果，一方面要看 reward 等参数变化，另一方面需要进行抽样输出，每次打印几条完整轨迹，看看模型的行为模式有没有在变好。

### 9.2 Ray 残留

Ray有一个反复出现的问题。训练跑完或者中途断了之后，再启动就卡住不动，也不报错。后来发现是上一次的Ray进程没清干净，`ray stop--force` 一下就好了。


## 10. 优化 Search-R1

官方仓库实现的 Search-R1 是基于老版本的 verl，所以考虑在新版 verl 上复现 Search-R1，改进方向有以下几点：

1. 新版本 verl 的 AsyncAgentLoop 可以进行 async rollout，减少了 tokenizer decode 操作或者环境交互带来的 GPU 闲置，rollout 效率大幅度提高。
2. 新版本 verl 的 ToolAgentLoop 实现了 tool agent rollout 的完整流程，从自定义文本协议变成标准的 tool calling，把检索操作变成了一个 tool。
3. Search-R1 里有 `info_mask` / `state_masking` 这种项目内自定义逻辑。SearchAgent-Zero 直接用新版 verl AgentLoop 的 `response_mask`，非模型生成的 token，padding 和异常处理的 token 都赋值为 0。
4. 检测异常轨迹（效果待定，感觉有问题）：
	1. 如果句子长度超过上限，不是进行左截断而是直接把这条 trajectory mask 掉不参与训练
	2. 如果 tool call turn 超过上限，把这条 trajectory mask 掉
	3. 如果发现模型生成的 tool call 有问题，例如 json 无法解析/tool call 格式不对/query 过多对象。那么就把这轮 turn 之前的全部 token mask 掉，也就是只保留存在问题的 token。一般 rollout 有问题的句子adv都是负的，那么就是降低这些有问题的 token 的概率。
