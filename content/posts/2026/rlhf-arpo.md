---
title: LLM 中的强化学习：ARPO
date: 2026-03-18T11:26:32+08:00
featuredImage: http://img.xilyfe.top/img/20260311113437709.png
authors:
  - Xilyfe
series:
  - RLHF
tags:
  - 大模型
  - 强化学习
lastmod: 2026-04-30T08:28:30+08:00
---
> **摘要**：**Agentic Reinforced Policy Optimization (ARPO)** 是一种专为训练多轮工具调用 LLM 智能体而设计的强化学习算法。现有 RL 算法（如 GRPO、DAPO）在模型的长程推理能力与多轮工具交互能力之间难以取得良好平衡，通常只在整条轨迹末尾提供奖励信号，忽略了步骤级别的探索。验观察到，LLM 在与外部工具交互之后，生成 token 的熵分布会显著升高，表现出高度不确定的行为。ARPO 的核心思路是：在模型"不确定该怎么用工具"的关键时刻，加大探索力度，从而更高效地学会正确的工具使用方式。


## 1. 研究背景与动机

### 1.1 现有方法的局限

近年来，以 DeepSeek-R1、QwQ 为代表的大模型在**单轮推理任务**上通过 RLVR（可验证奖励强化学习）取得了巨大突破。但在现实的开放场景中，LLM 往往需要调用外部工具（搜索引擎、代码解释器、浏览器等）来辅助解题，也就是所谓的 **Agentic RL** 场景。

目前主流的 Agentic RL 方法如 GRPO、DAPO、REINFORCE++，都属于 trajectory-level 的 RL 算法：

- 先让模型生成完整的工具调用轨迹；
- 只根据最终输出给予奖励信号；
- 完全忽略每一步工具调用之间的细粒度行为。

这种做法有两个核心问题：

1. **稀疏奖励**：只有轨迹结束才有反馈，无法对中间每一次工具交互进行精细引导；
2. **忽略步骤级探索**：多轮工具调用中，每次工具返回结果后，模型的行为空间会发生显著变化，但现有方法完全没有针对这一特点设计探索机制。

### 1.2 关键实验发现


推动 ARPO 提出最关键的是一个实验发现：作者在实验中发现 agentic rl 在 rollout 阶段，每次把 tool call result 加到历史记录后，模型生成的前 10~50 个 token 熵值**急剧升高**。

![image.png](http://img.xilyfe.top/img/20260428145152231.png)

作者对这个现象的思考是：
1. 外部工具返回 `<obs></obs>` 不是模型自己生成的内容，造成了分布偏移，引入了大量不确定性。
2. 工具反馈带来的不确定性超过了原始输入本身

## 2. 解决方案

ARPO 的核心思路是：**在模型熵值飙升的工具调用步骤处，自适应地进行分支采样，从而在高不确定性时刻扩大探索空间。**

### 2.1 基于熵的自适应 Rollout

1. 展开初始化：给定总采样预算 $M$，首先生成 $N$ 条完整的全局轨迹，剩余 $M-N$ 的预算留给后续的分支采样。同时记录每条轨迹开头 $k$ 个 token 的熵值，形成初始熵矩阵 $H_{\text{initial}} \in \mathbb{R}^{1 \times k}$。
2. 熵变监测： 在模型执行工具调用后，让其继续生成 $k$ 个 token，计算此时的熵矩阵 $H_t$​，并计算归一化的熵变化：$\Delta H_t = \text{Normalize}(H_t - H_{\text{initial}})$。其中，归一化是指将 Δ​H 的所有值求和并除以词表大小 V。正值的 Δ​H 表示工具调用步骤 k 之后不确定性增加，而负值则反映了不确定性的降低。
3. 基于熵的自适应束搜索：根据熵变化，计算当前步骤的分支采样概率：

$$P_t = \alpha + \beta \cdot \Delta H_t，\begin{array}{c} \text{Action}(P_t) = \begin{cases} \text{Branch}(Z), & \text{if } P_t > \tau \\ \text{Continue}, & \text{otherwise} \end{cases} \end{array}$$

- $\alpha$：基础采样概率（保证基线探索）；
- $\beta$：熵的权重系数；
- $\tau$：触发分支的阈值；
- $Z$：分支出来的额外路径数量。

当 $P_t$ 超过阈值 $\tau$，模型就从当前节点"分叉"出 $Z$ 条新的推理路径，覆盖更多样化的工具使用行为。

4. 终止：该过程不断迭代，直到满足以下条件之一：（1）如果分叉路径的总数 $\hat{Z}$ 达到部分采样预算 $M−N$，则停止分支并继续采样，直到生成最终答案；（2）如果所有路径在达到 $M−N$ 之前终止，我们将补充 $M−N−Z$ 个额外的轨迹级样本。

通过利用这种高效的展开机制，ARPO 促进了具有不确定性意识的探索，使 LLM 能够更有效地识别步骤级工具调用行为。 同时，假设全局扩展规模和每条轨迹的 Token 数为 n，ARPO 将每次展开的计算复杂度从轨迹级强化学习的 $O​(n^2)$ 降低到 $O​(n​log⁡n)$ 和 $O​(n^2)$ 之间


### 2.2 优势归因估计

自适应 Rollout 会产生一种特殊的轨迹结构：**部分 token 是多条路径共享的，部分 token 是各路径独有的**。ARPO 提出了两种方案来处理不同类型 token 的优势计算

#### 2.2.1 硬优势估计

- 独有 token：使用各自轨迹的归一化奖励计算优势 $\hat{A}_{i,t}$；
- 共享 token：使用所有包含该共享段的轨迹的平均优势 $\hat{A}^{\text{shared}}_{i,t} = \frac{1}{d}\sum_{i=1}^{d} \hat{A}_{i,t}$。

举个例子，假如模型第一个 tool call 生成的 token 为 $[a,b,c,d]$，此时熵变超过阈值需要进入分支，后续生成了两条 trajectory $[e,f,g]$ 和 $[h,i,j]$。ARPO 也是 outcome-based reward，根据 result 两个 trajectory 都计算出他们各自的优势 $A_1$ 和 $A_2$。如果是 GRPO，$A_1$ 和 $A_2$ 会被均摊到各自 trajectory 的每一个 token 上。而在 ARPO 的硬优势估计中，两个 trajectory 的独有 token 会被分配到 $A_1$ 和 $A_2$，它们共享的 token 分配到 $\frac{A_1+A_2}{2}$。

#### 2.2.2 软优势估计

我们回顾一下 GRPO 的目标函数：

$$J_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G}  \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right)  - \beta D_{\text{KL}} \right]$$

然后沿用上面的例子，目标函数即为：

$$J(\theta) = \frac{1}{2}\left[ \frac{1}{|y_1|} \sum_{t} r_{1,t}(\theta)\hat{A}_1 + \frac{1}{|y_2|} \sum_{t} r_{2,t}(\theta)\hat{A}_2 \right]$$

这时候我们可以发现一个有意思的问题，对于共享的 token，它们的重要性采样系数是相同的：

$$r_{1,s}(\theta) = \frac{\pi_\theta(c \mid x,\ \text{ab})}{\pi_{\text{ref}}(c \mid x,\ \text{ab})} = r_{2,s}(\theta)$$

所以我们对共享 token 求梯度可以得到：

$$
\begin{align}
\nabla_\theta J \Big|_s &= \frac{1}{2}\left[ \frac{1}{|y_1|} r_s(\theta)\hat{A}_1 + \frac{1}{|y_2|} r_s(\theta)\hat{A}_2 \right] \nabla_\theta \log\pi_\theta(c \mid x, \text{ab}) \\
&= \frac{r_s(\theta)}{2} \left[ \frac{\hat{A}_1}{|y_1|} + \frac{\hat{A}_2}{|y_2|} \right] \nabla_\theta \log\pi_\theta(c \mid x, \text{ab}) \\
& \approx r_s(\theta) \cdot \underbrace{\frac{\hat{A}_1 + \hat{A}_2}{2}}_{\text{平均 advantage}} \cdot \frac{1}{L}\nabla_\theta \log\pi_\theta(c \mid x, \text{ab})
\end{align}
$$

当两个 trajectory 的长度接近时候，这正好等价于给共享 token 赋予平均 advantage $\frac{\hat{A}_1 + \hat{A}_2}{2}$，即硬优势估计的做法。

对独有 token 计算梯度时候，我们以 token "e" 为例：

$$\nabla_\theta J \Big|_t = \frac{1}{2} \cdot \frac{1}{|y_1|} \cdot r_{1,t}(\theta) \cdot \hat{A}_1 \cdot \nabla_\theta \log\pi_\theta(e \mid x, \text{abcd})$$

由于后半部分不含这个 token 所以梯度不会从 $\hat{A_2}$ 经过，只用了 $\hat{A_1}$​，与硬估计中独有 token 用自己的 advantage 完全一致。

![image.png](http://img.xilyfe.top/img/20260428160335696.png)

ARPO 的论文中也通过实验比较了软硬优势估计的区别，可以看到软优势估计得到的奖励曲线更高更加平稳。它们认为是硬优势估计需要在代码里显式地找到分叉点，判断哪些 token 是共享的、哪些是独有的，容易引入噪声。而软优势估计把这个逻辑藏进了 importance sampling ratio 的结构里——共享 token 因为 prefix 相同导致 ratio 相等，梯度自然叠加平均；独有 token 因为 prefix 不同导致 ratio 不同，梯度天然隔离。

### 2.3 奖励函数

ARPO 采用层次化奖励，考虑了正确性和格式，还加入了**多工具协作奖励**：

$$
\begin{align}
\begin{array}{c} R &= \begin{cases} \max(\text{Acc} + r_M,\ \text{Acc}) & \text{格式正确 \& Acc} > 0 \\ 0 & \text{格式正确 \& Acc} = 0 \\ -1 & \text{otherwise} \end{cases} \end{array} \\
\begin{array}{c} r_M &= \begin{cases} 0.1 & \text{如果同时使用了} \langle\text{search}\rangle \text{ 和 } \langle\text{python}\rangle \\ 0 & \text{否则} \end{cases} \end{array}
\end{align}
$$

## 3. verl 复现

论文在 verl 框架上复现了 ARPO 算法，正好有助我们学习如何对 verl 进行二次开发。

### 3.1 整体改动

我们从 ARPO 的训练脚本看看它对参数做了哪些修改：

- `algorithm.adv_estimator=grpo`：ARPO 的采样软优势估计，所以和 GRPO 完全相同：计算 outcome-based reward 之后组内归一化得到 advantage，最后平摊到之前的每一个 token 上。
- `actor_rollout_ref.rollout.mode=sync_with_tool`：ARPO 最大的改动就是基于熵的自适应 rollout，ARPO 重写了 verl 的 rollout 过程，这一部分需要重点关注。
- `custom_reward_function.path=.../deep_research.py`：ARPO 也重写了奖励函数，这部分比较简单。


### 3.2 奖励函数

先从最简单的奖励函数开始看看 `verl/utils/reward_score/deep_research.py`：

```python
def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None):
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0
    }
    
    response = solution_str
    valid_template, reason = validate_format(response)
    
    if not valid_template:
        result["score"] = -1
        result["reason"] = f"bad format: {reason}"
        return result
    
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]
    
    answer_part = extract_answer(response)
    if answer_part is None:
        result["score"] = -1
        result["reason"] = "cannot extract answer"
        return result
    
    try:
        answer = remove_boxed(last_boxed_only_string(answer_part))
        result["answer"] = answer
    except Exception as e:
        result["score"] = -1
        result["reason"] = f"find box error: {e}"
        return result
    
    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score
    print(f"f1_score: {f1_score}, answer: {answer}, ground_truth: {ground_truth}")
    
    if f1_score > 0 and "</search>" in response and "</python>" in response:
        result["score"] = f1_score + 0.1
        result["reason"] = f"correct answer and calling search and python at the same time, get score: {f1_score + 0.1}"
    elif f1_score > 0:
        result["score"] = f1_score
        result["reason"] = f"correct answer, get f1 score: {f1_score}"
    else:
        result["score"] = 0
        result["reason"] = f"wrong answer but good format: {answer}"
    
    return result
```

这部分就是 verl 里面很标准的一个奖励函数设计，我们实现一个 `compute_score` 方法然后返回一个 float 类型的分数或者返回一个字典用于记录数据。然后可以看看 ARPO 如何计算正确性奖励的：

```python
def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])

    return final_metric['f1']

```

它用 token-level 的 F1 来衡量生成答案和标准答案的重合程度。计算 f1 分数时候它用 `normalize_answer` 进行了文本归一化，做了小写、去标点、去冠词、去多余空格等操作，还对特殊答案（yes/no/noanswer）进行了处理，如果不相同则完全不匹配。


### 3.3 自定义 rollout

我们先回顾一下 verl 里面 rollout 过程的整个链路，然后看看 ARPO 是如何自定义 rollout 流程的：

1. 主程序位于 `trainer/main_ppo.py`，里面会先进行一系列初始化：
	1. `RayPPOTrainer.init_workers()` 会进行资源分配并且调用内部 woerker 的 `init_model` 方法
	2. `ActorRolloutRefWorker.init_model()` 会根据 config 选择 rollout 类
2. `RayPPOTrainer.fit()` 方法是训练的主循环
	1. trainer 会调用 worker group 进行 rollout
	2. worker group 内部会通过 rollout 类的 `generate_sequence` 方法进行 rollout 生成 response
	3. 更加底层的来说，rollout 类会通过 inference engine 进行 token 的生成，verl 会管理 inference 和 training 模型参数的转换


所以我们需要自定义 rollout 流程需要实现一个自定义的 rollout 类，它需要继承 vLLMRollout 基类，然后实现 `generate_sequence` 方法：

```python
class vLLMRolloutWithTools(vLLMRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):

    def __del__(self):

    def _extract_content(self, text: str, tag: str) -> str:

    def _execute_tool_with_retry(self, tool, content):
    
    def _calc_entropy(self, logprobs):

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
```

1. 进入主函数 & 状态初始化：generate_sequences 是整个引擎的核心。函数一开始为每条输入样本分配 initial_rollouts 条并行轨迹，构造五个平行列表：
- curr_inputs：当前 token 序列（随着生成不断增长）
- init_inputs：原始 prompt（保持不变，用来算 response 长度）
- result_masks：哪些 token 是「模型生成的」（工具结果 = 0，模型输出 = 1）
- call_counters：每条轨迹已调用工具几次
- active_indices：还没结束的轨迹索引

```python
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    input_ids   = prompts.batch['input_ids']   # shape: (batch, seq_len)
    batch_size  = input_ids.size(0)
    
    # ── 为每条输入样本展开 initial_rollouts 条轨迹 ────────
    curr_inputs    = []   # 实时增长的 token 列表
    init_inputs    = []   # 固定的 prompt token
    result_masks   = []   # 1=模型生成, 0=工具结果
    call_counters  = []   # 工具调用计数
    active_indices = []   # 尚未结束的轨迹
    
    initial_rollouts = min(self.initial_rollouts, num_samples)
    
    for i, ids in enumerate(prompt_token_ids_list):
        for _ in range(initial_rollouts):          # ← 每个 prompt 复制 N 份
            curr_inputs.append(ids.copy())
            init_inputs.append(ids.copy())
            result_masks.append([])
            call_counters.append(0)
            active_indices.append(len(curr_inputs) - 1)
    
    # 跟踪每个原始样本已有多少 rollout
    rollouts_per_sample = [initial_rollouts] * batch_size
    sample_to_indices   = {
        i: [i * initial_rollouts + j for j in range(initial_rollouts)]
        for i in range(batch_size)
    }
```

2. 主生成循环：主循环每轮只对 active_indices（未完成轨迹）调用一次 vLLM generate：

```python
max_len = self.config.response_length   # e.g. 4096

while active_indices:
    active_prompts = [curr_inputs[i] for i in active_indices]
    
    with self.update_sampling_params(
        n          = 1,
        stop       = self.stop_sequences,       # ← 遇到 </search> 等停下
        max_tokens = max(1, max(
            max_len - (len(curr_inputs[i]) - len(init_inputs[i]))
            for i in active_indices              # ← 每条轨迹剩余 budget
        )),
        detokenize = True,
        logprobs   = self.logprobs               # ← top-10 logprob 用于熵
    ):
        outputs = self.inference_engine.generate(
            prompt_token_ids = active_prompts,
            sampling_params  = self.sampling_params,
            use_tqdm         = False
        )
    
    # 每个 output 对应 active_indices[i]
    for i, out_idx in enumerate(active_indices):
        output           = outputs[i]
        generated_tokens = output.outputs[0].token_ids
        
        curr_inputs[out_idx].extend(generated_tokens)
        result_masks[out_idx].extend([1] * len(generated_tokens)
```

3. 熵监控：生成后立刻从 logprobs 中计算当前步的信息熵，并与这条轨迹的初始熵对比。

```python
# ── 熵计算辅助函数 ─────────────────────────────────────
def _calc_entropy(self, logprobs):
    p_list  = [math.exp(l) for l in logprobs]        # logprob → prob
    entropy = -sum(p * l for p, l in zip(p_list, logprobs))
    return entropy

# ── 循环内：对每条 active 轨迹算熵 ─────────────────────
vocab_size          = len(self.tokenizer.get_vocab())
entropy_norm_factor = math.log(vocab_size)   # 归一化到 [0,1]

current_entropy_dict = {}
for i, out_idx in enumerate(active_indices):
    output = outputs[i]
    logprobs = []
    tokens = output.outputs[0].token_ids
    for j in range(min(20, len(tokens))):         # ← 取前 20 个 token
        logprob_info   = output.outputs[0].logprobs[j]
        token_logprobs = [t.logprob for t in logprob_info.values()]
        logprobs.extend(token_logprobs)
    
    entropy = self._calc_entropy(logprobs) / entropy_norm_factor
    current_entropy_dict[out_idx] = entropy
    
    if out_idx not in self.initial_entropy_dict:   # ← 首次记录初始熵
        self.initial_entropy_dict[out_idx] = entropy
```

4. 工具调用检测：每条轨迹生成结束后检查 finish_reason。

```python
tool_requests: Dict[str, List[Dict]] = {tag: [] for tag in self.tools}
next_active_indices = []

for i, out_idx in enumerate(active_indices):
    finish_reason = output.outputs[0].finish_reason   # 'stop' | 'length'
    stop_reason   = output.outputs[0].stop_reason     # 触发的 stop 字符串
    
    is_tool_call = (finish_reason == 'stop' 
                    and stop_reason in self.stop_sequences)
    
    if is_tool_call:
        tag = stop_reason.strip("</>")   # e.g. "</search>" → "search"
        
        if call_counters[out_idx] < self.tool_call_limit:
            call_counters[out_idx] += 1
            full_text = self.tokenizer.decode(curr_inputs[out_idx])
            content   = self._extract_content(full_text, tag)
            
            tool_requests[tag].append({"index": out_idx, "content": content})
            next_active_indices.append(out_idx)
            tool_metrics["tools/total_calls"] += 1
        else:
            # 超出调用上限 → 强制 EOS
            curr_inputs[out_idx].append(eos_token_id)
            result_masks[out_idx].append(1)
            tool_metrics["tools/call_limit_reached_count"] += 1
    
    elif finish_reason == 'length':
        next_active_indices.append(out_idx)  # 继续下一轮生成
    
    elif finish_reason == 'stop':             # EOS，正常结束
        pass
```

5. 工具并行执行：所有工具请求被一次性提交到线程池，并行执行（IO 密集型，适合多线程）。

```python
for future in concurrent.futures.as_completed(futures):
    idx, tag = futures[future]["index"], futures[future]["tag"]
    result   = future.result(timeout=self.tool_timeout)
    
    result_text = result["result"] or f"Tool({tag}) returned empty output."
    
    # 工具结果包裹进 <result> 标签
    formatted_result = f" <result>\n{result_text}\n</result>"
    result_tokens    = self.tokenizer.encode(formatted_result)
    
    curr_inputs[idx].extend(result_tokens)
    result_masks[idx].extend([0] * len(result_tokens))  # ← 0 = 不计 loss
```

6. 自适应 Beam 分支：工具结果追加完毕后，判断是否需要从当前轨迹「分裂」出新分支。

```python
for orig_sample, active_idxs in active_by_sample.items():
    remaining_slots = num_samples - rollouts_per_sample[orig_sample]
    if remaining_slots <= 0:
        continue
    
    branches_created = 0
    for source_idx in active_idxs:
        branches_per_idx = min(beam_size - 1, remaining_slots - branches_created)
        if branches_per_idx <= 0:
            break
        
        for _ in range(branches_per_idx):
            # ── 熵自适应分支概率 ──────────────────────────
            entropy_now   = current_entropy_dict.get(source_idx, 0.0)
            entropy_init  = self.initial_entropy_dict.get(source_idx, 0.0)
            entropy_delta = entropy_now - entropy_init
            
            prob = random.random() - self.entropy_weight * entropy_delta
            prob = max(0.0, min(1.0, prob))
            
            if prob > self.branch_probability:   # ← 不满足则跳过此分支
                continue
            # ─────────────────────────────────────────────
            new_inputs.append(curr_inputs[source_idx].copy())
            new_result_masks.append(result_masks[source_idx].copy())
            new_call_counters.append(call_counters[source_idx])
            rollouts_per_sample[orig_sample] += 1
            branches_created += 1
```

{{< admonition type=warning title="论文和代码的差异">}} 

**论文描述的逻辑：**

```
工具返回结果 → 模型看到结果后生成 k 个 token → 
计算这 k 个 token 的 H_t → 
ΔH_t = H_t - H_initial → 决定是否分支
```

即论文里的 $H_t$​ 是模型**读完工具结果之后**立刻产生的熵，反映的是工具结果带来的即时不确定性。

**代码实现的逻辑：**

```
第 N 轮 generate（此时 context 已含上轮工具结果）→ 
计算本轮输出前 20 个 token 的 entropy →
执行工具调用 → 插入 result →
根据刚才算的 entropy_delta 决定 branch →
第 N+1 轮 generate（context 含本轮工具结果）→ 
计算新 entropy...
```

关键区别：**用于 branch 决策的 entropy，是当前轮 generate 的 entropy，不是插入工具结果之后的 entropy**。工具结果对 entropy 的影响要到下一轮才体现。

{{< /admonition >}}

7. 最后是一些后处理，计算一下 mask，padding 等等，然后返回 DataProto。

```
┌─────────────────────────────────────────────────────────────┐
│                    generate_sequences                       │
│                                                             │
│ ① 初始化阶段                                                  │
│   - 每个 prompt 生成 initial_rollouts 条轨迹                  │
│                                                             │
│ ② 主循环：while active_indices 非空                           │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ (a) 调用 vLLM.generate(stop=tags)                    │   │
│   │                                                     │   │
│   │ (b) 计算当前序列熵 current_entropy                    │   │
│   │                                                     │   │
│   │ (c) 检查 finish_reason                               │   │
│   │     ├─ tool call → 收集 tool 请求                    │   │
│   │     ├─ length    → 继续下一轮生成                      │   │
│   │     └─ EOS       → 从 active_indices 移除            │   │
│   │                                                     │   │
│   │ (d) 并行执行工具调用（线程池）                           │   │
│   │     - result_mask[tool] = 0                         │   │
│   │                                                     │   │
│   │ (e) 基于熵的自适应 Beam 分裂                           │   │
│   │     - 新分支加入 active_indices                       │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│ ③ 后处理阶段                                                  │
│   - 对齐输出（padding）                                       │
│   - stack 成 batch                                          │
│   - 构造 loss_mask                                           │
│                                                             │
│ ④ 输出                                                       │
│   - DataProto(batch, meta_info["metrics"])                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```