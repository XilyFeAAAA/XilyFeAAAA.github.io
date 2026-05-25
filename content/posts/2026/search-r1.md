---
title: Search-R1 学习指北
date: 2026-05-21T20:26:00+08:00
featuredImage: http://img.xilyfe.top/img/20260522112900397.png
authors:
  - Xilyfe
series:
  - 项目笔记
tags: []
lastmod: 2026-05-25T01:57:43+08:00
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


## 6. RL 算法选择

### 6.1 PPO vs GRPO 的区别

### 6.2 GRPO 的 outcome-level advantage 归一化

### 6.3 KL penalty 的作用和配置

### 6.4 adv_estimator 选择对训练稳定性的影响

## 7. 训练流程与架构

### 7.1 verl 框架的 actor-critic-ref-rm 四角色设计

### 7.2 Ray 分布式调度

### 7.3 rollout → reward → update 的完整 loop

### 7.4 FSDP 并行策略

## 8. 工程细节

### 8.1 token_level_scores 的写入位置

### 8.2 多轮 rollout 中 attention_mask 的处理

### 8.3 prompt 模板对模型行为的影响

### 8.4 训练超参

## 8. 评估方法

### 8.1 在线验证

### 8.2 离线评估

### 8.3 指标

## 9. 消融实验与关键结论

### 9.1 有无 search 的对比

### 9.2 格式奖励的必要性

### 9.3多跳 vs 单跳数据混合训练的效果

### 9.4 不同基座模型的表现差异
