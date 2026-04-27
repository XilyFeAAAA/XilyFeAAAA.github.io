---
title: ToolBrain 学习指北
date: 2026-04-13T11:09:35+08:00
featuredImage: http://img.xilyfe.top/img/20260414110340462.png
authors:
  - Xilyfe
series:
  - 项目笔记
tags: []
lastmod: 2026-04-16T11:05:23+08:00
---
{{< admonition type=info title="前言">}} 
思考了很长时间还是准备走 LLM 算法，如果我去做 agent 开发做 RAG，那和直接学后端学 Java 有什么区别。LLM 算法这块 agenticRL 的 bar 比较低也不需要什么 paper，所以准备继续研究这个方向。本来打算搞完 MedicalGPT 之后复现一个 Search-R1，做 Agentic Search 方向。但是仔细看了下发现，现在大部分的项目都是基于 verl、slime 这些 RL 框架开发的，对于我这种不了解 Agentic RL 的看起来太复杂了。所以我准备按照下面几步入门：
1. 通过 ToolBrain 项目学会 Agentic RL 的基本流程
2. 学会 verl 框架，以及二次开发，参考[文章](https://www.cnblogs.com/AikN/p/18893668 )
3. 复现一个 agentic rl 项目：Search-R1、Deepresearch、ReTool 等等。
{{< /admonition >}}

## 1. 概览

先通过 ToolBrain 的 example 看看这个项目在做什么：

```python
# 1. Imports and Component Definition
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match
from smolagents import CodeAgent, TransformersModel, tool

@tool
def multiply(a: int, b: int) -> int:
    return a * b

# Define a standard agent (CPU-compatible)
model = TransformersModel("Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=128)
agent = CodeAgent(model=model, tools=[multiply], max_steps=1)

# 2. Initialize the Brain with your agent and a built-in reward
brain = Brain(
    agent,
    reward_func=reward_exact_match,
    algorithm="GRPO"
)

# 3. Define a task and start training!
dataset = [{"query": "What is 8 multiplied by 7?", "gold_answer": "56"}]
brain.train(dataset, num_iterations=10)

# 4. Save your trained agent
brain.save("./my_first_trained_agent")
```

可以看到 ToolBrain 是一个用强化学习来训练 Agent 更好使用工具的框架：小模型在用调用工具时，常常连简单任务都会失败。光靠 prompt engineering 并不能从根本上解决问题，因为 Agentic 系统工作流一旦变化，工具调用准确率就会崩。 ToolBrain 的目标就是用 RL 从根本上解决这个问题。

其次我们需要明确一个 Agentic RL 和 LLM RL 的区别：
- LLM Reinforce Learning 是一个退化的单步马尔科夫决策过程，我们把 prompt 输入给 LLM 它直接返回一段文本回答
- Agentic Reinforce Learning 是一个多步决策过程，传入 prompt 以后 Agent 会识别调用工具的意图，然后把工具调用的结果加入 messages 里面继续 generate，这个 loop 持续到生成结束。

ToolBrain 里面分为 Agent 和 Algorithm。Agent 包含三个部分，首先它包含模型的本体，比如 TransformerModel，Agent 用它来生成 completions；其次是工具 tool，Agent 识别模型的意图调用对应的工具，将工具的结果传给模型进行后续生成；最后就是 AgentEngine，它是一个实现 "思考 → 调工具 → 观察结果 → 再思考" 的 ReAct 循环，或者说它负责生成整个 trajectory。然后 Algorithm 就是 DPO、GRPO 等强化学习算法，它负责具体的算法逻辑，用来给模型参数优化。

## 2. 主流程

我们以 GRPO 为例，从 `brain.train()` 梳理一遍训练的过程：

1. 训练 `num_iterations` 轮，对每个 example 进行训练。

```python
def train(self, dataset: List[Dict[str, Any]], num_iterations: int = 1):
    for i in range(num_iterations):
        for example in dataset:
            if self.algorithm in GRPOALiasNames or self.algorithm in DPOALiasNames:
                query = example.get("query")
            elif self.algorithm in SupervisedALiasNames:
                query = example
            self.train_step(query=query, reward_kwargs=example)
```

2. `train_step()` 内部会先进行采样，用 Agent 收集完整 trace，然后利用 trace 进行 GRPO 训练。

```python
def train_step(self, query: Any, reward_kwargs: Dict[str, Any]):
    num_group_members = self.config.get("num_group_members", 2)
    
    if self.algorithm in GRPOALiasNames or self.algorithm in DPOALiasNames:
        traces, rewards, rl_inputs = self.get_trace(query, reward_kwargs)

    if self.algorithm in GRPOALiasNames:
        self.learning_module.train_step(rl_inputs, rewards)
```

3. `get_trace()` 方法为一个 query 生成多条 Agent 执行轨迹 Trace，同时计算每条轨迹的奖励，供后续 RL 训练使用，它类似 PPO / GRPO / RLOO 等 RLHF 方法里的 rollout 阶段。

```python
def get_trace(self, query: str, reward_kwargs: Dict[str, Any]):
    traces: List[Trace] = []
    rl_inputs: List[Any] = []
    raw_memory_collection: List[List[Any]] = []  # Collection of raw memory steps
    num_group_members = self.config.get("num_group_members", 2)
    
    for i in range(num_group_members):
        trace, rl_input, raw_memory_steps = self.agent_adapter.run(query)
        traces.append(trace)
        rl_inputs.append(rl_input)
        raw_memory_collection.append(raw_memory_steps)
        torch.cuda.empty_cache()
        gc.collect()
    
    if self.judge_model_id is not None:
        enhanced_reward_kwargs = {
            **reward_kwargs,
            "raw_memory_collection": raw_memory_collection,
            "judge_model": self.judge_model_id,
        }
    else:
        enhanced_reward_kwargs = {
            **reward_kwargs,
            "raw_memory_collection": raw_memory_collection,  # List of raw memory steps for each trace
        }
    rewards = self.reward_func.get_batch_scores(
        traces, **enhanced_reward_kwargs
    )
    
	    return traces, rewards, rl_inputs
```


4. GRPO 的 `train_step()` 就是参数更新的具体流程了。

```python
def train_step(
    self,
    segments: List[List[ChatSegment]],
    rewards: List[float],
) -> None:
    device = self.device
    pi_theta = self.policy
    
    batch = build_inputs(
        segments=segments, rewards=rewards, tokenizer=pi_theta.tokenizer
    )
    input_ids = batch.input_ids.to(device)  # shape: (B, L)
    attention_mask = batch.attention_mask.to(device)  # shape: (B, L)
    completion_mask = batch.completion_mask.to(device)  # shape: (B, L)
    advantages = batch.advantages.to(device)  # shape: (B, L)
    
    chunk_len = self.config.get("chunk_len", None)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            pi_theta_old_logps = pi_theta.get_per_token_logps(
                input_ids=input_ids,  # shape: (B, L)
                attention_mask=attention_mask,  # shape: (B, L)
                chunk_len=chunk_len,
            )  # shape: (B, L-1)
            pi_ref_logps = self.pi_ref.get_per_token_logps(
                input_ids=input_ids,  # shape: (B, L)
                attention_mask=attention_mask,  # shape: (B, L)
                chunk_len=chunk_len,
            )  # shape: (B, L-1)
    
    for _ in range(self.config["opt_steps"]):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            pi_theta_logps = pi_theta.get_per_token_logps(
                input_ids, attention_mask, chunk_len=chunk_len
            )  # shape: (B, L-1)
            loss = grpo_loss(
                pi_theta_logps=pi_theta_logps,  # shape: (B, L-1)
                pi_theta_old_logps=pi_theta_old_logps,  # shape: (B, L-1)
                pi_ref_logps=pi_ref_logps,  # shape: (B, L-1)
                advantages=advantages[:, 1:],  # shape: (B, L-1)
                completion_mask=completion_mask[:, 1:],  # shape: (B, L-1)
                epsilon=self.config["epsilon"],
                beta=self.config["beta"],
            )
        pi_theta = self._update_policy(pi_theta, loss)
        pi_theta_old_logps = pi_theta_logps.detach()
        del pi_theta_logps, loss
        torch.cuda.empty_cache()
    
    self.policy = pi_theta
    self.training_steps += 1
```

4. `_update_policy()` 就是一个反向传播，优化器更新的过程，没什么好说。

```python
def _update_policy(self, pi_theta: Policy, loss: torch.Tensor) -> Policy:
    model = getattr(pi_theta, "llm", None) or getattr(pi_theta, "model", None)

    model.train()
    self.optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), self.config["max_grad_norm"])
    self.optimizer.step()

    torch.cuda.empty_cache()
    return pi_theta
```

---

现在我就大致了解 AgenticRL 的训练流程了：

1. 生成 trace，类似 rollout：对于每一条训练 prompt，模型会并行生成 `num_group` 条 trace。每条 trace 的生成过程是一个标准的 ReAct 循环：
	- 模型接收 prompt，生成初步输出（推理 + 意图识别）
	- 识别到需要调用工具，输出工具调用指令
	- 外部环境执行工具并返回结果，结果被填入上下文
	- 模型继续生成，直到输出 final answer
2. 计算奖励：目前大部分的 Agentic RL 采用的还是 trajectory-level 的 reward，这就会遇到 credit assignment 问题，后面会在提到。
3. 强化学习优化：这一步和 LLM 的强化学习基本一致，通过 DPO/PPO/GRPO 等强化学习算法做 policy gradient 更新。

{{< admonition type=question title="为什么 DPO 也可以需要生成 Trace？">}} 
研究源码时候我有一个疑惑，DPO 不是用用现成的正反例数据进行 offline 训练吗，为什么代码里面还需要 `get_trace()` 呢？

ToolBrain 实现的是 **online DPO**（也叫 self-play DPO），而不是传统的 offline DPO。它的逻辑是把 N 条 traces 按 reward 降序排列，前一半作为 chosen，后一半作为 rejected。所以正反例完全来自当前 policy 的 rollout，不需要任何预先标注的数据集。

```
同一个 query → agent 跑 N 次 → N 条 traces + N 个 rewards  
                                        ↓  
                              make_dpo_pairs(rl_inputs, rewards)  
                                        ↓  
                    按 reward 排序，上半部分 = chosen，下半部分 = rejected  
                                        ↓  
                              DPOAlgorithm.train_step(chosen, rejected)  
```

{{< /admonition >}}

## 3. Agent&Trace

梳理完了训练的整体流程，我发现 Agent 如何生成一条 Trace 是我们最需要了解的，其他部分的代码就是之前 LLM RL 的内容。

```python
import json
import re
from typing import Any, Optional

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that can use tools to answer questions.

Available tools:
{tools_description}

STRICT RULES - you MUST follow these:
1. You cannot fabricate answers for unknown knowledge points;
2. Instead, use tools to retrieve relevant information.

To use a tool, respond in this EXACT format:
<think>[your reasoning, explain WHY you need to call the tool]</think>
<tool><tool_name>[tool_name]</tool_name><arguments>{{"param1": "value1"}}</arguments></tool>

After receiving the tool result, THEN respond:
<think>[your reasoning based on the tool result]</think>
<answer>[your answer]</answer>

Only call one tool at a time."""


class Tool:
    def __init__(self, name: str, func: callable, desc: Optional[str] = None) -> None:
        self.name = name
        self.desc = desc
        self.func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def to_schema(self) -> dict:
        import inspect

        sig = inspect.signature(self.func)
        params = {}
        for param_name, param in sig.parameters.items():
            annotation = param.annotation
            if annotation is int:
                type_str = "integer"
            elif annotation is float:
                type_str = "number"
            elif annotation is bool:
                type_str = "boolean"
            else:
                type_str = "string"
            params[param_name] = {"type": type_str}

        return {
            "name": self.name,
            "description": self.desc,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": list(params.keys()),
            },
        }


def to_tool(name: Optional[str] = None, desc: Optional[str] = None):
    def decorator(func):
        tool_name = name or func.__name__
        return Tool(tool_name, func, desc)

    return decorator


class Agent:
    def __init__(
        self,
        max_steps: int,
        tools: list[Tool] = [],
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        test: bool = False,
    ) -> None:
        self.max_steps = max_steps
        self.model_id = model_id
        self.test = test

        self.temperature = 0.9
        self.max_new_tokens = 1024
        self.tools = {tool.name: tool for tool in tools}

        tools_desc = "\n\n".join(
            f"Tool: {t.name}\nDescription: {t.desc}\nParameters: {json.dumps(t.to_schema()['parameters']['properties'])}"
            for t in tools
        )
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tools_description=tools_desc)

        if test:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.generate = self.generate_openai
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.generate = self.generate_transformers

    def generate_transformers(self, messages: list[dict]) -> str:
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def generate_openai(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def parse_output(self, text: str) -> dict:
        result = {
            "thought": None,
            "tool_name": None,
            "arguments": None,
            "final_answer": None,
            "raw": text,
        }

        # 解析 <think>
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            result["thought"] = think_match.group(1).strip()

        # 解析 <answer>，优先返回
        answer_match = re.search(
            r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE
        )
        if answer_match:
            result["final_answer"] = answer_match.group(1).strip()
            return result

        # 解析 <tool><tool_name>...</tool_name><arguments>...</arguments></tool>
        tool_match = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
        if tool_match:
            tool_body = tool_match.group(1)

            tool_name_match = re.search(
                r"<tool_name>(.*?)</tool_name>", tool_body, re.DOTALL
            )
            if tool_name_match:
                result["tool_name"] = tool_name_match.group(1).strip()

            args_match = re.search(
                r"<arguments>(.*?)</arguments>", tool_body, re.DOTALL
            )
            if args_match:
                try:
                    result["arguments"] = json.loads(args_match.group(1).strip())
                except json.JSONDecodeError:
                    result["arguments"] = {}

        return result

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"

        tool = self.tools[tool_name]
        try:
            return tool(**arguments)
        except Exception as e:
            print("工具调用失败，报错：", e)
            return f"Error: Tool '{tool_name}' execute failed"

    def run(self, query: str) -> tuple[str, list]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        steps = []

        for _ in range(self.max_steps):
            messages_snapshot = [m.copy() for m in messages]

            model_output = self.generate(messages)
            parsed = self.parse_output(model_output)

            step_info = {
                "messages_before": messages_snapshot,
                "model_output": model_output,
                "parsed": parsed,
                "tool_output": None,
            }

            if parsed["final_answer"] is not None:
                steps.append(step_info)
                return parsed["final_answer"], steps

            if parsed["tool_name"] is not None:
                tool_result = self.execute_tool(
                    parsed["tool_name"], parsed["arguments"] or {}
                )
                step_info["tool_output"] = tool_result

                messages.append({"role": "assistant", "content": model_output})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool '{parsed['tool_name']}' returned: {tool_result}\nContinue solving the problem.",
                    }
                )
            else:
                step_info["parsed"]["final_answer"] = model_output
                steps.append(step_info)
                return model_output, steps

            steps.append(step_info)

        last_output = steps[-1]["model_output"] if steps else "Max steps reached"
        return last_output, steps


if __name__ == "__main__":

    @to_tool("retriver", "返回仓库中指定商品的库存")
    def retriver(item_id: str):
        return 100

    agent = Agent(
        max_steps=5,
        model_id="gemini-3.1-flash-lite",
        api_key="your-api-key-1",
        base_url="http://localhost:8317/v1",
        tools=[retriver],
        test=True,
    )
    last_output, steps = agent.run("商品 t99 的库存是多少")
```

ReAct Agent 的主要流程为：
1. 用 API 或者本地模型进行 generate
2. 从 completions 里解析出思考过程、工具调用、最终答案
3. 记录这一 step 的操作，假如是工具调用则把调用结果加入 messages，假如是最终答案则输出
4. 重复这个 loop，当返回最终答案或达到 step 上限时退出

这里面如何让模型按照要求输出比较关键，我用 GPT 的 API 测试了两次，它不会按照我的要求调用 `retrieve` 工具，而是询问商店的 ID，后面我改成 Gemini 就正常了，这可能和 SYSTEM PROMPT 也有关系。


## 4. 复现 ToolBrain

```python
class Brain:
    def __init__(
        self,
        agent: Agent,
        dataset: Union[list, Dataset],
        rewards: list[Reward] = [],
        algorithm: Literal["grpo", "ppo", "dpo"] = "grpo",
        config: dict = dict(
            learning_rate=3e-5,
            batch_size=1,
            eps=0.2,
            num_group_members=2,
            beta=0.1,
            max_grad_norm=1.0,
            use_bitsandbytes=False,
            max_seq_len=4096,
            lora_config=None,
            device="cuda",
        ),
    ) -> None:
        self.agent = agent
        self.rewards = rewards
        self.algorithm = algorithm
        self.config = config

        if isinstance(dataset, list):
            self.dataset = Dataset.from_list(dataset)
        else:
            self.dataset = dataset

        self._setup_training()

    def _setup_training(self):
        self.actor = self.agent.get_model()
        self.ref = copy.deepcopy(self.actor)
        self.tokenizer = self.agent.get_tokenizer()

        assert self.actor is not None
        assert self.tokenizer is not None

        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.config["learning_rate"],
        )

    def train(self, iters: int = 1):
        print("\n🚀 Starting training...")
        for it in range(iters):
            print(f"\n--- Iteration {it + 1}/{iters} ---")
            for example in self.dataset:
                self.train_step(example)

        print("\n🎉 Training finished!")

    def train_step(self, example: dict):
        traces, rewards = self.get_trace(example)
        batch = self.build_input(traces, rewards)
        device = self.config["device"]

        input_ids = batch["input_ids"].to(device)  # shape: (B, L)
        attention_mask = batch["attention_mask"].to(device)  # shape: (B, L)
        completion_mask = batch["completion_mask"].to(device)  # shape: (B, L)
        advantages = batch["advantages"].to(device)  # shape: (B, L)

        with torch.no_grad():
            ref_logprobs = self.get_logprobs(
                self.ref,
                input_ids=input_ids,  # shape: (B, L)
                attention_mask=attention_mask,  # shape: (B, L)
            )

        logprobs = self.get_logprobs(
            self.actor,
            input_ids=input_ids,  # shape: (B, L)
            attention_mask=attention_mask,  # shape: (B, L)
        )

        self.actor.train()
        self.optimizer.zero_grad()
        loss = self.grpo_loss(
            logprobs=logprobs,
            old_logprobs=logprobs.detach(),
            ref_logprobs=ref_logprobs,
            advantages=advantages[:, 1:],
            completion_mask=completion_mask[:, 1:],
        )
        loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.config["max_grad_norm"])
        self.optimizer.step()

    def build_input(self, traces: list[dict], rewards: list[float]) -> dict:
        all_input_ids: list[list[int]] = []
        all_attention_masks: list[list[int]] = []
        all_completion_mask: list[list[int]] = []
        all_advantages: list[list[float]] = []

        if isinstance(rewards, list):
            rewards = torch.tensor(rewards).to(self.config["device"])
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        normalized_rewards = (rewards - mean) / (std + 1e-8)

        for idx, trace in enumerate(traces):
            seq_ids: list[int] = []
            seq_attn: list[int] = []
            seq_comp_mask: list[int] = []
            seq_advs: list[float] = []

            for _, segment in enumerate(trace):
                # if segment["role"] != "assistant" and i != 0:
                #     continue  # Skip non-assistant segments except the first one
                segment_ids = self.tokenizer.encode(
                    segment["content"], add_special_tokens=False
                )
                if len(seq_ids) + len(segment_ids) > 4096:  # for 14B
                    continue
                seq_ids.extend(segment_ids)
                # Attention mask: 1 for every real token
                seq_attn.extend([1] * len(segment_ids))
                # Completion mask: 1 only for model_completion tokens
                if segment["role"] != "assistant":
                    seq_comp_mask.extend([0] * len(segment_ids))
                else:
                    seq_comp_mask.extend([1] * len(segment_ids))

                # Expand the per-trace normalized reward along this turn's tokens
                seq_advs.extend(
                    [float(normalized_rewards[idx].item())] * len(segment_ids)
                )

            # Accumulate per-trace sequences
            all_input_ids.append(seq_ids)
            all_attention_masks.append(seq_attn)
            all_completion_mask.append(seq_comp_mask)
            all_advantages.append(seq_advs)

        # Pad to batch-first tensors
        input_ids = pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in all_input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(seq, dtype=torch.int) for seq in all_attention_masks],
            batch_first=True,
            padding_value=0,
        )
        completion_mask = pad_sequence(
            [torch.tensor(seq, dtype=torch.int) for seq in all_completion_mask],
            batch_first=True,
            padding_value=0,
        )
        advantages = pad_sequence(
            [torch.tensor(seq, dtype=torch.float) for seq in all_advantages],
            batch_first=True,
            padding_value=0.0,
        )

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            advantages=advantages,
        )

    def get_trace(self, example: dict):
        traces = []
        outputs = []
        rewards = []
        query = example["query"]
        num_group_members = self.config.get("num_group_members", 2)

        for i in range(num_group_members):
            try:
                print(f"    📝 Trace {i + 1}/{num_group_members}")
                output, trace = self.agent.run(query)
                traces.append(trace[-1]["messages"])
                outputs.append(output)
                torch.cuda.empty_cache()
                gc.collect()

                rw = 0
                for reward in self.rewards:
                    rw += reward(trace=trace, output=output, **example)
                rewards.append(rw)

            except Exception as e:
                print(f"    ❌ Error during agent iteration: {e}")
                continue

        return traces, rewards

    @staticmethod
    def get_logprobs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids: [B, L]
        """
        B, L = input_ids.shape

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()

        logprobs = F.log_softmax(logits, dim=-1)
        logprobs = logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        return logprobs.view(B, L - 1)

    def grpo_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
    ):
        ratio = torch.exp(logprobs - old_logprobs)
        k3 = -torch.exp(logprobs - ref_logprobs) + ratio - 1
        surr_1 = ratio * advantages
        surr_2 = (
            torch.clamp(ratio, 1 - self.config["eps"], 1 + self.config["eps"])
            * advantages
        )
        per_token_loss = (
            -torch.min(surr_1, surr_2) + self.config["beta"] * k3
        ) * completion_mask
        loss = per_token_loss.sum(dim=-1) / (completion_mask.sum(dim=-1) + 1e-8)
        return loss.mean()
```

需要注意我们只对 assistant 部分计算 loss，所以需要比较复杂的遍历 `messages` 数组，构造 `completion_mask`。

## 5. 问题&收获

- 问题
	1. ToolBrain 的 `get_trace` 是串行的效率非常低，通用它只能支持同一 group 并行训练，不支持多个 prompt 一起。
	2. ToolBrain 的奖励是 trajectory-level 的粒度，它只能对整个句子进行奖励，假如工具调用正确但是答案错了还是给零分，这就会出现credit assignment 问题：模型无法区分哪一步的 tool call 是关键的，好的 reasoning step 和坏的 reasoning step 得到相同的梯度信号。
- 收获：
	1. agentic rl 训练包含多轮对话，并且对话中含有 CoT 思维链，导致 seq_len 非常大，所以我们采样后计算 logprobs 的显存开销是非常大的。ToolBrain 的解决办法是进行分 chunk 计算 logprobs。

## 6. 补充

GitHub 项目 [GRPO](https://github.com/junfanz1/GRPO/blob/main/grpo.py) 手写复现了 Search-R1 的全链路。

```python
"""
Search-R1 (GRPO Edition)
=========================
Reinforcement learning training loop based on GRPO (Group Relative Policy Optimization)
for training an Agentic RAG / Search-Enhanced Language Model.

Key Features:
- Combines search actions with answer generation actions
- Uses GRPO (group-relative PPO) instead of standard policy gradients
- No critic network required
- Compares multiple candidate trajectories
- KL regularization to stabilize training
- Detailed annotations aligned with Search-R1 / DeepSeek-R1 experimental design
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List


# ============================================================
# 🧱 Data Structures
# ============================================================

@dataclass
class TokenStep:
    """Record of a single token generation"""
    token_id: int
    token_text: str
    log_prob: float
    position: int  # Position in the full sequence (used for recomputing log probs)


@dataclass
class Trajectory:
    """Complete trajectory (including question, search, and answer generation)"""
    question: str
    answer: str
    token_steps: List[TokenStep]
    generated_text: str  # Model-generated text including intermediate search commands
    full_input_ids: List[int]
    generated_positions: List[int]
    reward: float = 0.0  # Reward value (computed later)


# ============================================================
# 🔍 Search Engine (replaceable with real RAG)
# ============================================================

class SearchEngine:
    """Simple local knowledge base search engine"""

    def __init__(self):
        self.knowledge_base = {
            "machine learning": "Machine learning is a subset of AI that enables computers to learn from experience.",
            "neural networks": "Neural networks are computing systems inspired by biological neural networks.",
            "deep learning": "Deep learning is a subset of machine learning using artificial neural networks.",
            "transformer": "Transformers are neural network architectures using self-attention mechanisms.",
            "reinforcement learning": "Reinforcement learning involves agents learning through environment interaction.",
        }

    def search(self, query: str) -> str:
        """Return a simple search result"""
        query_lower = (query or "").lower().strip()
        for key, value in self.knowledge_base.items():
            if key in query_lower:
                return value
        return f"No information found for: {query}"


# ============================================================
# 🧩 Search-R1 Trainer (GRPO-based)
# ============================================================

class SearchR1Trainer:
    def __init__(self, model_name="Qwen/Qwen2.5-7B", device=None):
        # Load language model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # --------------------------------------------------------
    # 🧠 Trajectory Generation (agent interaction)
    # --------------------------------------------------------
    def generate_trajectory(self, question: str, search_engine: SearchEngine) -> Trajectory:
        """
        Simulate agent answering process: may perform searches (<search> ... </search>) before generating answers.
        """
        self.model.eval()
        current_text = question
        input_ids = self.tokenizer.encode(question, return_tensors="pt").to(self.device)

        token_steps = []
        full_input_ids = input_ids[0].tolist()
        generated_positions = []

        max_steps = 100
        done = False

        for _ in range(max_steps):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)

            # Sample next token from probability distribution
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze()
            log_prob = torch.log(probs[0, next_token_id])
            token_text = self.tokenizer.decode([next_token_id.item()], skip_special_tokens=True)

            # === Fix position indexing ===
            future_pos = len(full_input_ids)
            generated_positions.append(future_pos)

            # Record token generation
            token_steps.append(TokenStep(
                token_id=next_token_id.item(),
                token_text=token_text,
                log_prob=log_prob.item(),
                position=future_pos,
            ))

            # Update input sequence
            full_input_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)

            # Append to current text
            current_text += token_text

            # Check if entering search mode
            if "<search>" in current_text and "</search>" in current_text:
                query = self.extract_search_query(current_text)
                search_result = search_engine.search(query)
                # Inject search result into context
                context_addition = f"\n[Search result]: {search_result}\n"
                current_text += context_addition
                search_tokens = self.tokenizer.encode(context_addition, add_special_tokens=False)
                full_input_ids.extend(search_tokens)
                input_ids = torch.tensor([full_input_ids], device=self.device)

            # Termination condition
            if any(end in current_text.lower() for end in ["</answer>", "<eos>", "end of answer"]):
                done = True
                break

        return Trajectory(
            question=question,
            answer=current_text,
            token_steps=token_steps,
            generated_text=current_text,
            full_input_ids=full_input_ids,
            generated_positions=generated_positions,
        )

    # --------------------------------------------------------
    def extract_search_query(self, text: str) -> str:
        """Extract query between <search> ... </search>"""
        start_tag, end_tag = "<search>", "</search>"
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag)
        if start >= len(start_tag) and end > start:
            return text[start:end].strip()
        return ""

    # --------------------------------------------------------
    # 🎯 Reward Function (replaceable with semantic similarity)
    # --------------------------------------------------------
    def compute_reward(self, prediction: str, ground_truth: str) -> float:
        """
        Simple reward: 1 if prediction contains ground_truth, else 0.
        Can replace with embedding similarity or LLM evaluation.
        """
        return 1.0 if ground_truth.strip().lower() in prediction.strip().lower() else 0.0

    # --------------------------------------------------------
    # 🔁 Recompute trajectory log_probs (for KL and new policy)
    # --------------------------------------------------------
    def recompute_log_probs(self, trajectories: List[Trajectory]) -> List[torch.Tensor]:
        self.model.eval()

        input_ids_list, attention_masks, adjusted_positions = [], [], []
        for traj in trajectories:
            input_ids_tensor = torch.tensor(traj.full_input_ids, dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids_tensor, device=self.device)
            input_ids_list.append(input_ids_tensor)
            attention_masks.append(attention_mask)
            adjusted_positions.append(traj.generated_positions)

        max_len = max(len(ids) for ids in input_ids_list)
        input_ids_padded = torch.stack([
            F.pad(ids, (0, max_len - len(ids)), value=self.tokenizer.pad_token_id)
            for ids in input_ids_list
        ])
        attention_mask_padded = torch.stack([
            F.pad(mask, (0, max_len - len(mask)), value=0)
            for mask in attention_masks
        ])

        with torch.no_grad():
            outputs = self.model(input_ids_padded, attention_mask=attention_mask_padded)
            logits = outputs.logits  # [batch, seq_len, vocab]

        all_log_probs = []
        for i, (traj, positions) in enumerate(zip(trajectories, adjusted_positions)):
            log_probs = []
            for pos, token_step in zip(positions, traj.token_steps):
                pred_index = max(pos - 1, 0)  # causal LM correction
                log_prob_tensor = F.log_softmax(logits[i, pred_index], dim=-1)[token_step.token_id]
                log_probs.append(log_prob_tensor)
            all_log_probs.append(torch.stack(log_probs))
        return all_log_probs

    # --------------------------------------------------------
    # 🧮 KL Divergence (stable implementation)
    # --------------------------------------------------------
    def compute_kl_divergence(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
        """KL(old || new) = sum(p_old * (log_old - log_new))"""
        p_old = torch.exp(old_log_probs)
        kl = torch.sum(p_old * (old_log_probs - new_log_probs))
        return kl

    # --------------------------------------------------------
    # 🧩 GRPO Update Step (core)
    # --------------------------------------------------------
    def update_policy(self, trajectories: List[Trajectory], beta: float = 0.01) -> torch.Tensor:
        """
        Compute GRPO loss:
            - Compute group mean reward
            - Compute relative advantage
            - Add KL regularization
        """
        rewards = torch.tensor([t.reward for t in trajectories], dtype=torch.float32, device=self.device)
        mean_reward = rewards.mean()
        advantages = rewards - mean_reward  # Group-relative advantage

        # Recompute new policy log_probs
        new_log_probs_list = self.recompute_log_probs(trajectories)
        old_log_probs_list = [
            torch.tensor([step.log_prob for step in traj.token_steps], dtype=torch.float32, device=self.device).detach()
            for traj in trajectories
        ]

        policy_losses, kl_losses = [], []
        for adv, old_lp, new_lp in zip(advantages, old_log_probs_list, new_log_probs_list):
            seq_len = min(len(old_lp), len(new_lp))
            old_lp, new_lp = old_lp[:seq_len], new_lp[:seq_len]
            kl = self.compute_kl_divergence(old_lp, new_lp)
            kl_losses.append(kl)
            policy_losses.append(-adv * torch.sum(new_lp))

        policy_loss = torch.stack(policy_losses).mean()
        kl_loss = torch.stack(kl_losses).mean()
        total_loss = policy_loss + beta * kl_loss

        return total_loss

    # --------------------------------------------------------
    # 🚀 Single Training Step (GRPO)
    # --------------------------------------------------------
    def train_step(self, queries, answers, search_engine, optimizer, num_candidates=4, beta=0.01):
        """
        Perform GRPO update for a batch:
        - Generate num_candidates trajectories per query
        - Compute group-relative reward
        - Compute loss and backpropagate
        """
        self.model.train()
        batch_trajectories = []

        for question, ground_truth in zip(queries, answers):
            group_trajs = []
            for _ in range(num_candidates):
                traj = self.generate_trajectory(question, search_engine)
                traj.reward = self.compute_reward(traj.generated_text, ground_truth)
                group_trajs.append(traj)

            batch_trajectories.extend(group_trajs)

        loss = self.update_policy(batch_trajectories, beta=beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # --------------------------------------------------------
    # 🔁 Full Training Loop
    # --------------------------------------------------------
    def train(self, dataset, search_engine, epochs=3, lr=1e-5, num_candidates=4, beta=0.01):
        optimizer = Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0.0
            for question, answer in dataset:
                loss = self.train_step([question], [answer], search_engine, optimizer, num_candidates, beta)
                total_loss += loss
            print(f"[Epoch {epoch+1}] Avg Loss = {total_loss / len(dataset):.4f}")

if __name__ == "__main__":
    # Simple dataset
    dataset = [
        ("What is deep learning?", "Deep learning is a subset of machine learning."),
    ]

    search_engine = SearchEngine()
    trainer = SearchR1Trainer(model_name="Qwen/Qwen2.5-7B")  # can swap for smaller model for testing
    trainer.train(dataset, search_engine, epochs=2, lr=1e-5, num_candidates=3, beta=0.02)
```

可以看到 Search-R1 的训练流程和 ToolBrain 基本相似，它把采样 trajectory 的过程从 agent 集成到了一起：

```
开始
  ↓
输入数据 (question, ground_truth)
  ↓
for 每个 question:
  ↓
  采样 N 条 trajectory（group）
  ↓
  for 每条 trajectory:
      ↓
      初始化 context = question
      ↓
      循环生成 token:
          ↓
          模型 forward → logits → 采样 token
          ↓
          记录 log_prob + position
          ↓
          拼接到 context

          ↓
          判断是否出现 <search>...</search>
              ↓
              是:
                  提取 query
                  ↓
                  调用 SearchEngine
                  ↓
                  将搜索结果注入 context
                  ↓
                  继续生成
              
              否:
                  继续生成

          ↓
          判断是否结束（</answer> / eos）
              ↓
              是 → 结束该 trajectory
              否 → 继续循环

      ↓
      得到完整 trajectory
      ↓
      计算 reward（基于 ground_truth）
  
  ↓
  得到一个 group（N条 trajectories）
  ↓
  计算 group mean reward
  ↓
  计算 advantage:
      advantage_i = reward_i - mean_reward

  ↓
  重新计算 new_log_probs（当前模型）
  ↓
  获取 old_log_probs（采样时记录）

  ↓
  for 每条 trajectory:
      ↓
      计算 KL(old || new)
      ↓
      计算 policy loss:
          -adv * sum(new_log_probs)

  ↓
  聚合:
      policy_loss = mean(...)
      kl_loss = mean(...)
      total_loss = policy_loss + β * kl_loss

  ↓
  反向传播:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  ↓
进入下一个 question

↓
结束 epoch
↓
输出训练好的模型
```

Search-R1 就是把 ToolBrain 的工具调用改成了搜索，其它大同小异。