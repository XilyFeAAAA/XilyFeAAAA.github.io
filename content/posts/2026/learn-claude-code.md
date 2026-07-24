---
title: ClaudeCode 学习指北
date: 2026-07-04T10:02:23+08:00
featuredImage: http://img.xilyfe.top/img/20260704115226287.png
authors:
  - Xilyfe
series:
  - 项目笔记
tags:
  - claudecode
  - agent
  - harness
lastmod: 2026-07-04T10:02:23+08:00
---
## 1. AgentLoop


![image.png](http://img.xilyfe.top/img/20260704102608179.png)

```python
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ── Tool definition: just bash ────────────────────────────
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def agent_loop(messages):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_bash(block.input["command"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```


## 2. ToolUse

对于更多的工具，我们只需要在 `TOOLS` 里面添加字段，并且提供调用的函数就好了。

```python
TOOLS = [
    {"name": "bash",       "description": "Run a shell command.", ...},
    {"name": "read_file",  "description": "Read file contents.",  ...},
    {"name": "write_file", "description": "Write content to file.", ...},
    {"name": "edit_file",  "description": "Replace text in file once.", ...},
    {"name": "glob",       "description": "Find files by pattern.", ...},
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write, "edit_file": run_edit, "glob": run_glob,
}

def run_bash():	pass
def read_file(): pass
def run_write(): pass
def run_edit(): pass
def run_glob(): pass

def agent_loop(messages):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
	            handler = TOOL_HANDLERS[block.name]
	            output = handler(**block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})

```

{{< admonition type=warning title="">}} 
由于大语言模型幻觉的问题，我们需要进行安全性的保证，防止模型输出一些 `rm -rf` 之类的命令。可以通过 **目标路径是否在当前目录下来进行判断**。

```python
WORKDIR = Path.cwd()

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"
```

{{< /admonition >}}

ClaudeCode 里面对 ToolUse 的实现实际上比刚刚说的复杂的多：
1. 我们刚刚用的是 `Tools` 数组和 `TOOL_HANDLERS` 字典还有单独函数的方式定义的工具，但是在 Claudecode 里面每个工具都是一个单独的对象，包含了 schema、验证、权限、执行方法等信息。
2. 上面的 agent 在执行工具调用时候是串行的效率很低，Claudecode 的方法是通过一个 `isConcurrencySafe(input)` 方法判断各个 ToolUse 之间能否并发，然后把并发安全的 ToolUse 编入同一个 batch，batch 内真正并发执行。遇到非并发安全的就开新 batch 串行执行，各个 batch 之间严格顺序。
3. 演示代码在执行 ToolUse 时候直接用 llm 请求的参数 execute 了，而 Claudecode 的每个工具调用经过 5 步验证：
	1. Zod schema 验证：参数类型/结构检查
	2. 工具级验证：参数值验证（如路径是否在工作区内）
	3. PreToolUse hooks：钩子可以返回消息、修改输入、阻止执行
	4. 权限检查：canUseTool + checkPermissions → allow/deny/ask
	5. 执行 `tool.call()`
4. Claudecode 采用流式 ToolUse，当检测到 response 里包含 ToolUse 就会执行工具调用，大幅度提高性能。


## 3. Permission

>前面提到安全不能靠信任模型，需要要靠代码在工具执行之前做判断。一种简化版的 permission check 就是通过 blacklist+rule match+user ask 的方式实现权限审核。

![image.png](http://img.xilyfe.top/img/20260704111202446.png)

```python
DENY_LIST = [
    "rm -rf /", "sudo", "shutdown", "reboot",
    "mkfs", "dd if=", "> /dev/sda",
]

def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None
```

```python
PERMISSION_RULES = [
    {
        "tools": ["write_file", "edit_file"],
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "Writing outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
        "message": "Potentially destructive command",
    },
]

def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None
```

```python
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n⚠  {reason}")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"
```

然后我们在 agent loop 里需要进行工具调用前进行一个 check_permission 就行了：

```python
def check_permission(block) -> bool:
    # 闸门 1: 硬拒绝
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n⛔ {reason}")
            return False

    reason = check_rules(block.name, block.input)
    if reason:
        decision = ask_user(block.name, block.input, reason)
        if decision == "deny":
            return False

    return True

for block in response.content:
    if block.type == "tool_use":
        if not check_permission(block):
            results.append({... "content": "Permission denied."})
            continue
        output = TOOL_HANDLERS[block.name](**block.input)
        results.append(...)
```

## 4. Hook

我们设想一个场景，假如我们想要记录每次 bash 调用、操作后自动 git add，那么我们可能需要把 agent loop 修改成这样：

```python
def agent_loop(messages):
    while True:
        # ... LLM call ...
        for block in response.content:
            if block.type != "tool_use":
                continue
            log_to_file(block)
            check_permission(block)
            notify_slack(block)
            output = execute(block)
            auto_git_add(block)
```

Hook 的思路是定义多个 **事件**，然后在事件发生的位置调用注册的函数。

```python
HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:   # 返回值 ≠ None → hook 说"停"
            return result
    return None
```

例如上面代码定义的 `UserPromptSubmit` 事件就是用户输入提交后、进入 LLM 前触发，`PreToolUse` 事件在 ToolUse 前触发，我们就可以把之前的 Permission Check 注册到 `PreToolUse` 事件里：

```python
def permission_hook(block):
    if block.name == "bash":
        for pattern in DENY_LIST:
            if pattern in block.input.get("command", ""):
                return "Permission denied by deny list"
    if block.name in ("write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None

def log_hook(block):
    print(f"[HOOK] {block.name}(...)")

def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(f"[HOOK] ⚠ Large output from {block.name}")

register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
```

然后修改 agent loop 代码，在事件发生的 trigger 就好了：

```python
for block in response.content:
    if block.type != "tool_use":
        continue

    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(blocked)})
        continue

    handler = TOOL_HANDLERS.get(block.name)
    output = handler(**block.input) if handler else f"Unknown: {block.name}"
    trigger_hooks("PostToolUse", block, output)
    results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
```

## 5. ToDo

>在 agentic 环境下大模型的上下文长度更长，工具结果不断填满上下文，可能导致系统提示的影响力被稀释。`todo_write` 这个工具和 reminder 机制的作用就是，让 agent 在思考之前先理清思路该干什么。Claudecode 里面有两个任务系统并存，先来将简易版的 `todo_write`。

`todo_write` 的思路很简单，就是**让模型输出当前需要完成哪些任务**，哪些已经完成了，哪些还在进行中，这样就能在上下文中让模型不断回顾需要干什么。

```python
CURRENT_TODOS: list[dict] = []

def run_todo_write(todos: list) -> str:
    global CURRENT_TODOS
    CURRENT_TODOS = todos

    lines = ["\n## Current Tasks"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "▸", "completed": "✓"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"
    
TOOLS = [
    {"name": "bash",       ...},
    {"name": "read_file",  ...},
    {"name": "write_file", ...},
    {"name": "edit_file",  ...},
    {"name": "glob",       ...},
    {"name": "todo_write", "description": "Create and manage a task list ...",
     "input_schema": {
         "type": "object",
         "properties": {
             "todos": {
                 "type": "array",
                 "items": {
                     "type": "object",
                     "properties": {
                         "content": {"type": "string"},
                         "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                     },
                 },
             },
         },
     },
    },
]

TOOL_HANDLERS["todo_write"] = run_todo_write
```

为了防止提供了 `todo_write` 工具但是模型一直不调用，可以连续 $N$ 轮没调 `todo_write` 时，自动注入一条提醒：

```python
if rounds_since_todo >= 3 and messages:
    messages.append({
        "role": "user",
        "content": "<reminder>Update your todos.</reminder>",
    })
    rounds_since_todo = 0
```

Claudecode 还设计一个 Task System，里面包含文件持久化、依赖图、并发锁、ownership 等机制，会在后面说明。

## 6. SubAgent

>agent 在修一个 bug 时读了 30 个文件来追踪调用链，中间聊了 60 轮。messages 列表涨到 120 条，其中大部分是"追踪调用链"的中间过程，和"修 bug"这个最终目标无关。这些中间过程占着上下文位置，让 Agent 越来越"健忘"，它记不住最初的问题是什么了。subagent 的作用就是开一个新的 agent 负责一些子任务，然后把最终的结果返回给 parent agent，减少无关信息的污染


![image.png](http://img.xilyfe.top/img/20260704115529461.png)

subagent 功能通过新增的 `task` 工具来实现。调用 `task` 工具时，spawn 一个 sub agent。开启一个 subagent 听起来很麻烦，实际上只要 **给一个全新的 `messages[]`，跑自己的循环** 就是 subagent 了，结束后只把摘要文本回传给 parent agent。

>主要注意 sub agent 的工具受限，不能递归 spawn 新的 sub Agent

```python
def spawn_subagent(description: str) -> str:
    sub_tools = [
        {"name": "bash", ...}, {"name": "read_file", ...},
        {"name": "write_file", ...}, {"name": "edit_file", ...},
        {"name": "glob", ...},
    ]
    messages = [{"role": "user", "content": description}]

    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUB_SYSTEM,
            messages=messages, tools=sub_tools, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                blocked = trigger_hooks("PreToolUse", block)
                if blocked:
                    results.append({... "content": str(blocked)})
                    continue
                handler = SUB_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown"
                trigger_hooks("PostToolUse", block, output)
                results.append({... "content": output})
        messages.append({"role": "user", "content": results})

    return extract_text(messages[-1]["content"])
```

然后把他封装为一个 tool 供 parent agent 调用：

```python
TOOLS = [
    {"name": "bash", ...},
    {"name": "read_file", ...},
    {"name": "write_file", ...},
    {"name": "edit_file", ...},
    {"name": "glob", ...},
    {"name": "todo_write", ...},
    {
	    "name": "task",
	     "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
	     "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required":["description"]}
	},
]

TOOL_HANDLERS["task"] = spawn_subagent
```

## 7. Skill

>在一个全栈项目中，假如我们前端有一份 React 组件规范，后端有一份 Java 开发规范，数据库有一份 MySQL 开发规范，如果希望 Agent 自动遵守这些规范就需要把这些文档一股脑的喂进 SYSTEM PROMPT，这会浪费大量的上下文。skill 的思想就是把这些文档像 ToolUse 一样，将它们的 frontmatter 注入到 SYSTEM PROMPT 里面，大模型就可以在合适的时机选择读取这些 skill。


![image.png](http://img.xilyfe.top/img/20260704154911395.png)

这里先简要介绍一下 skill，它的结构一般为：

```python
skills/
  agent-builder/
	  SKILL.md
	  references/
	  scripts/
```

- SKILL.md 是一个文档的规范，他由 frontmatter 部分和 markdown 部分组成。frontmatter 部分包含 skill 的名字和 description，直接注入到 SYSTEM PROMPT 里面，让模型进行选择。然后 markdown 部分一般包含了这个 skill 如何使用，例如有 example、workflow 等等。
- references 是 skill 使用的一些参考，例如一个生成前端 UI 的 skill，它的 references 就是各种风格的页面 UI。
- scripts 是一些脚本，例如一个自动提交 git 的 skill，就可以把完整的 bash 命令放在 scripts 里面，模型直接取出来用就好了，可以避免幻觉或者生成错误。


```python
SKILL_REGISTRY: dict[str, dict] = {}

def _scan_skills():
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", d.name)
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}

_scan_skills()

def list_skills() -> str:
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())

def build_system() -> str:
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill tool to get full details when needed."
    )

SYSTEM = build_system()

def load_skill(name: str) -> str:
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]
```

## 8. Compact

learn-claude-code 项目里面提到了五层 compact 操作。

第一层 compact 是 `snip_compact`，当对话轮数超过限制时删除中间的对话，保留头部和尾部，并且不能把 `assistant(tool_use)` 和后面的 `user(tool_result)` 拆开：

```python
def snip_compact(messages, max_messages=50):
    if len(messages) <= max_messages:
        return messages
    head_end, tail_start = 3, len(messages) - (max_messages - 3)
    if head_end > 0 and _message_has_tool_use(messages[head_end - 1]):
        while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
            head_end += 1
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1
    snipped = tail_start - head_end
    placeholder = {"role": "user", "content": f"[snipped {snipped} messages from conversation middle]"}
    return messages[:head_end] + [placeholder] + messages[tail_start:]
```

第二层 compact 是 `micro_compact`，负责把早期的 tooluse 对话删除：

```python
KEEP_RECENT_TOOL_RESULTS = 3

def micro_compact(messages):
    tool_results = collect_tool_result_blocks(messages)
    if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS:
        return messages
    for _, _, block in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
        if len(block.get("content", "")) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages
```

>这里有很值得提到的一点，在 claudecode 中 Anthropic API 的 prompt cache 本质上是按前缀做增量缓存的——服务端会对 messages 数组算一个类似 hash chain 的东西，只要某一段前缀的内容跟上次请求完全一样，这段前缀就能命中缓存，不用重新计算 KV。如果按照刚刚说的方式直接把中间内容截断，会导致从这条消息开始，**后面所有消息的缓存哈希全部失效**，所以 claudecode 中采用的是调用 `cache_edit` API 来清除旧结果而不影响缓存，或者在 cache 的 TTL 到期时再直接删除早起的 tooluse。

第三层 compact 是 `tool_result_budget`，如果最后一次的 tool result 包含大量内容，那么会把超过大小阈值的内容落盘到文件里，在上下文中留下标记，让模型后面再按需读取：

```python
def tool_result_budget(messages, max_bytes=200_000):
    last = messages[-1]
    blocks = [(i, b) for i, b in enumerate(last["content"])
              if b.get("type") == "tool_result"]
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_bytes:
        return messages
    ranked = sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True)
    for idx, block in ranked:
        if total <= max_bytes:
            break
        block["content"] = persist_large_output(block["tool_use_id"], str(block["content"]))
        total = recalculate_total(blocks)
    return messages
```

第四层 compact 是 `compact_history`，当前三层 compact 都无法有效缩减上下文时候，就通过 LLM 对上下文进行总结。流程包含三步：
1. 把上下文的完整内容用 jsonl 落盘到文件里当备份
2. 调用 LLM 进行总结，要求保留当前目标、重要发现、已改文件、剩余工作、用户约束等关键信息。
3. 把旧的上下文替换为总结的内容。

>真实 Claude Code 会在 compact 后重新附加部分最近文件、计划、agent/skill/tool 等上下文。

```python
def compact_history(messages):
    transcript_path = write_transcript(messages)  # 先保存完整对话
    summary = summarize_history(messages)          # LLM 生成摘要
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]
```

刚刚讲的前四层 compact 操作都是在 **调用 API 请求之前主动发起的**，每轮自动调用 L1/L2/L3 compact，如果发现超过阈值再进行 L4 compact，然后再发起 API 请求。L5 reactive compact 解决的是 **已经把 messages 发给 API 然后返回报错 `prompt_too_long` 的情况**。

```python
def reactive_compact(messages):
    transcript = write_transcript(messages)
    tail_start = max(0, len(messages) - 5)
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1
    summary = summarize_history(messages[:tail_start])
    return [{"role": "user",
             "content": f"[Reactive compact]\n\n{summary}"}, *messages[tail_start:]]
```

把上面各种 compact 操作合起来就是：

```python
def agent_loop(messages):
    reactive_retries = 0
    while True:
        messages[:] = tool_result_budget(messages)    # L3: 大结果落盘
        messages[:] = snip_compact(messages)          # L1: 裁中间
        messages[:] = micro_compact(messages)         # L2: 旧结果占位

        # 还不够？LLM 摘要（1 API 调用）
        if estimate_token_count(messages) > THRESHOLD:
            messages[:] = compact_history(messages)

        try:
            response = client.messages.create(...)
        except PromptTooLongError:
            if reactive_retries < MAX_REACTIVE_RETRIES:
                messages[:] = reactive_compact(messages)  # 应急
                reactive_retries += 1
                continue
            raise  # 超过重试上限，抛出异常
        # ... 工具执行 ...

        # compact 工具：模型主动调用时触发 compact_history
        if block.name == "compact":
            messages[:] = compact_history(messages)
            results.append({..., "content": "[Compacted. History summarized.]"})
            messages.append({"role": "user", "content": results})
            break  # 结束当前 turn，用压缩后的上下文开始新一轮
```

## 9. Memory

>compact 操作解决的是上下文信息太多，超过模型限制的问题，但这种压缩操作是 **有损** 的，Memory 解决的是如何保存一些的重要信息。


Memory 系统会把上下文中的重要信息总结成四类记忆：

| 类型        | 回答什么   |
| --------- | ------ |
| user      | 你是谁    |
| feedback  | 怎么做事   |
| project   | 正在发生什么 |
| reference | 东西在哪找  |

然后各种的 memory 都会被保存在 `.memory/` 目录下，每条记忆是一个独立的 `.md` 文件，文件开头带 YAML frontmatter：

```markdown
---
name: user-preference-tabs
description: User prefers tabs for indentation
type: user
---

User prefers using tabs, not spaces, for indentation.
```

因为记忆文件会越来越多，不可能每次都把所有文件内容塞进 context，所以搞一个 `MEMORY.md` 作为**目录/索引**：

```markdown
- [user-preference-tabs](user-preference-tabs.md) — User prefers tabs for indentation
```

接下来就涉及到两个问题，memory 如何使用和如何提取。首先是如何使用的问题：
1. 一方面 claudecode 会把 `MEMORY.md` 里面的内容放进 SYSTEM PROMPT 当做索引，让模型知道有哪些记忆在
2. 另一方面，在每轮开始时，claudecode 会做一次**轻量的 LLM side-query**，把"最近对话"+"记忆目录"一起发给 LLM，让它选出"这轮跟哪几条记忆相关"(最多 5 条)，然后**只把选中的那几条完整内容**读出来，临时塞进当前 user turn。

记忆提取是在每一轮对话结束后，假如这一轮的 response 中没有工具调用，那么就可以调用一次 LLM，让它给最近几条对话总结 memory。记忆文件会越攒越多，可能出现重复、过时、互相矛盾的内容。所以设了一个阈值，一旦文件数超过这个阈值就触发一次"整理"：把所有记忆发给 LLM，让它去重、合并冲突、砍掉过时的，然后重写这些文件。

## 10. System Prompt

>按需组装 SYSTEM PROMPT

简单地说 SYSTEM PROMPT 可以分为 identity、tools、workspace 和 memory 这几个大类。把一大段字符串拆成字典，每个 key 是一个主题：

```python
PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}
```

当前没有记忆文件，加载 memory section 只是浪费 token。根据 context 的真实状态决定加载哪些：

```python
def get_system_prompt(context: dict) -> str:
    sections = []

    # 始终加载
    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(PROMPT_SECTIONS["tools"])
    sections.append(PROMPT_SECTIONS["workspace"])

    # 按需加载 — 基于真实状态，不是关键词
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")

    return "\n\n".join(sections)
```

context 上下文里面存的是当前的状态，例如是否启用工具，是否有记忆索引等等：

```python
def update_context(context: dict, messages: list) -> dict:
    memories = ""
    if MEMORY_INDEX.exists():
        content = MEMORY_INDEX.read_text().strip()
        if content:
            memories = content
    return {
        "enabled_tools": list(TOOL_HANDLERS.keys()),
        "workspace": str(WORKDIR),
        "memories": memories,
    }
```

最终的 agent loop 里面 SYSTEM PROMPT 就不是写死的一段字符串了，变成通过函数自动拼接：

```python
def agent_loop(messages: list, context: dict):
    system = get_system_prompt(context)
    while True:
        response = client.messages.create(
            model=MODEL, system=system, messages=messages,
            tools=TOOLS, max_tokens=8000)
        context = update_context(context, messages)
        system = get_system_prompt(context)
```

## 11. Recovery

>这里针对常见的几种 error 进行处理


### max_tokens error

1. 第一次把 `max_tokens` 从 8K 变成 64K，让模型输出完整内容。
2. 如果 64K 的 `max_response_token` 还不够的话，就让模型多次输出阶段内容，然后不断追加到 messages 

```python
if response.stop_reason == "max_tokens":
    if not state.has_escalated:
        max_tokens = ESCALATED_MAX_TOKENS
        state.has_escalated = True
        continue
    messages.append({"role": "assistant", "content": response.content})
    if state.recovery_count < MAX_RECOVERY_RETRIES:
        messages.append({"role": "user", "content":
            "Output token limit hit. Resume directly — "
            "no apology, no recap. Pick up mid-thought."})
        state.recovery_count += 1
        continue
    return
# Normal: append after max_tokens check
messages.append({"role": "assistant", "content": response.content})
```

### prompt_too_long error

这里就是前面 compact 章节提到的 reactive compact，如果经过 L1/L2/L3/L4 compact 之后 API 还报错上下文过长就会调用 reactive compact，对五段对话前的上下文进行压缩：

```python
except PromptTooLongError:
    if not state.has_attempted_reactive_compact:
        messages[:] = reactive_compact(messages)
        state.has_attempted_reactive_compact = True
        continue
    return
```

### 11. API error

对于模型自身的故障，例如 429 限流、529 过载等等，可以通过指数回避+重试来解决：

```python
def retry_delay(attempt, retry_after=None):
    if retry_after:
        return retry_after
    base = min(500 * (2 ** attempt), 32000) / 1000
    return base + random.uniform(0, base * 0.25)

def with_retry(fn, state, max_retries=10):
    for attempt in range(max_retries):
        try:
            return fn()
        except (RateLimitError, OverloadedError):
            delay = retry_delay(attempt)
            time.sleep(delay)
            if is_overloaded:
                state.consecutive_529 += 1
                if state.consecutive_529 >= 3 and FALLBACK_MODEL:
                    state.current_model = FALLBACK_MODEL
    raise MaxRetriesExceeded()
```

## 12. Task System

>前面提到的 Todo 本质上是**一次性的、当前会话的执行清单**——写在内存里，给这一轮任务打个勾。它没有"依赖"概念,你只能按写的顺序自己脑内排序,而且**关掉会话就没了**。真实项目往往是"先打地基,再盖楼"这种有先后顺序的多步骤工作：先建 schema，再写 API，再写测试。如果没有依赖关系的约束，Agent 很容易写到一半发现前置条件没做，然后回头补。

Task System 要解决的有两件事，一个是让 task 之间有 DAG 的先后依赖关系，第二个是让 task 可以持久化保存。对于持久化保存，可以让每个任务独立一个文件，存在 `.tasks/{id}.json` 里面：

```python
@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str          # pending | in_progress | completed
    owner: str | None    # 谁在做这个任务
    blockedBy: list[str] # 依赖哪些任务的 ID
```

然后 Task System 通过五个操作组成了一个状态机：
1. `create_task`：创建任务，并且声明它依赖于谁
2. `can_start`：检查这个任务的所有依赖是否都属于 completed 状态
3. `claim_task`：判断任务状态是不是 in_progress 并且调用 `can_start` 判断依赖是不是完成，如果没问题就认领任务
4. `complete_task`：标记任务完成，并且扫一遍所有任务，找出因为这个任务完成而"解锁"的下游任务，类似迪杰斯特拉算法。
5. `get_task`：获取具体任务信息

## 13. Background Task

对于 LLM 的每个工具调用操作，需要区分它是耗时的还是可以立刻获得结果的。有两个办法：
1. 首先 claudecode 在进行工具调用时候会返回一个 `run_in_background` 字段，标志它希不希望用后台操作
2. 其次可以进行启发式判断，也就是判断一些关键词例如 `install`、`build`、`compile` 等等

```python
def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """Fallback heuristic: commands likely to take > 30s."""
    if tool_name != "bash":
        return False
    cmd = tool_input.get("command", "").lower()
    slow_keywords = ["install", "build", "test", "deploy", "compile",
                     "docker build", "pip install", "npm install",
                     "cargo build", "pytest", "make"]
    return any(kw in cmd for kw in slow_keywords)

def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """Model explicit request takes priority; fallback to heuristic."""
    if tool_input.get("run_in_background"):
        return True
    return is_slow_operation(tool_name, tool_input)
```

然后在 agent loop 中的执行流程类似：

```python
results = []
for block in response.content:
    if block.type != "tool_use":
        continue
    if should_run_background(block.name, block.input):
        bg_id = start_background_task(block)
        results.append({"type": "tool_result",
            "tool_use_id": block.id,
            "content": f"[Background task {bg_id} started] "
                       f"Result will be available when complete."})
    else:
        output = execute_tool(block)
        results.append({"type": "tool_result",
            "tool_use_id": block.id, "content": output})

# 通知和工具结果合入同一条 user 消息
user_content = []
bg_notifications = collect_background_results()
if bg_notifications:
    for notif in bg_notifications:
        user_content.append({"type": "text", "text": notif})
user_content.extend(results)
messages.append({"role": "user", "content": user_content})
```

当 LLM 需要通过 background task 执行工具调用时候，就会通过 `start_background_task` 处理后台任务，然后返回一个任务 ID。同时在 tool result 里面说明任务 ID 在进行中。当后续任务完成，就会直接在 messages 上下文里面把任务的结果插入进去。

```python
_bg_counter = 0
background_tasks: dict[str, dict] = {}   # bg_id → {tool_use_id, command, status}
background_results: dict[str, str] = {}   # bg_id → output
background_lock = threading.Lock()

def start_background_task(block) -> str:
    """Run tool in a daemon thread. Returns background task ID."""
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"

    def worker():
        result = execute_tool(block)
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": block.input.get("command", ""),
            "status": "running",
        }
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return bg_id


def collect_background_results() -> list[str]:
    """Collect completed results as task_notification messages."""
    with background_lock:
        ready_ids = [bid for bid, task in background_tasks.items()
                     if task["status"] == "completed"]
    notifications = []
    for bg_id in ready_ids:
        with background_lock:
            task = background_tasks.pop(bg_id)
            output = background_results.pop(bg_id, "")
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{output[:200]}</summary>\n"
            f"</task_notification>")
    return notifications
```

## 14. Corn Schedule

claudecode 把调度和执行两个操作解耦，一个线程专门看当前时间有没有需要执行的 schedule，另一个线程看队列里面有没有 schedule 需要执行，也就是经典的生产者消费者模型。

scheduler 每秒都拿当前时间去跟所有注册的 cron 任务比对，如果时间对上了就把这个任务扔进队列：

```python
def cron_scheduler_loop():
    while True:
        time.sleep(1)
        now = datetime.now()
        for job in scheduled_jobs:
            if cron_matches(job.cron, now):
                cron_queue.append(job)   # 扔进队列,自己不执行
```

queue processor 不断检查队列里有没有新任务，并且当前 agent 是不是空闲，如果都满足条件那么就会执行这轮任务：

```python
def queue_processor_loop():
    while True:
        time.sleep(0.2)
        if not has_cron_queue():
            continue 
        if not agent_lock.acquire(blocking=False):
            continue
        run_agent_turn_locked()
```

然后 `run_agent_turn_locked` 这个函数具体干的事情，就是从队列里把已经触发的任务捞出来，包装成一条消息塞进上下文：

```python
fired = consume_cron_queue()
for job in fired:
    messages.append({"role": "user", "content": f"[Scheduled] {job.prompt}"})
```

## 15. Agent Team

之前的章节介绍过 sub agent 这个概念，简单来说就是一些子任务可能会占据长上下文，所以分发给 sub agent 完成，然后把结果再回传。但是 sub agent 也有局限性，比如：
	1. sub agent 是阻塞式的，parent agent 发出调用后 通常得等它返回结果才能继续往下走
	2. sub agent 是一次性的，得到结果后就销毁上下文了，如果 parent agent 看完结果后还想追问细节，只能重新 spawn 一个全新的 sub agent，而这个新的 sub agent 对之前那一轮的过程**完全没有记忆**，只能把结论重新讲一遍给它听.

所以 agent team 这个概念就是让 lead agent 可以同时 spawn 多个 team agent，每个 team agent 都是**持续存在的**，始终维持着自己的上下文 messages，并且每个 agent 之间都可以通过信箱互相沟通，实现协作、分发任务、汇报。

---

每个 agent 都通过 `.jsonl` 文件来充当信箱，发消息就是往对方的文件里添加一行 JSON，读消息就是读文件 + 删除操作：

```python
class MessageBus:
    def send(self, from_agent: str, to_agent: str,
             content: str, msg_type: str = "message"):
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time()}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a") as f:
            f.write(json.dumps(msg) + "\n")

    def read_inbox(self, agent: str) -> list[dict]:
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        msgs = [json.loads(line) for line in inbox.read_text().splitlines()]
        inbox.unlink()
        return msgs
```

spawn 一个 team agent 的操作可以设计成一个 tool，类似：

```python
def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    system = f"You are '{name}', a {role}. Use tools to complete tasks."

    def run():
        messages = [{"role": "user", "content": prompt}]
        sub_tools = [bash, read_file, write_file, send_message]
        for _ in range(10):           # 最多 10 轮
            inbox = BUS.read_inbox(name)
            if inbox:
                messages.append({"role": "user",
                    "content": f"<inbox>{json.dumps(inbox)}</inbox>"})
            response = client.messages.create(
                model=MODEL, system=system, messages=messages[-20:],
                tools=sub_tools, max_tokens=8000)
        BUS.send(name, "lead", summary, "result")

    threading.Thread(target=run, daemon=True).start()
```

1. 在 claudecode 里面 team agent 是无限循环的，每跑完一轮任务就会发生后发 `idle_notification` 信号，等 inbox 消息，收到后继续。这里就只训练 10 轮。
2. 这里的工具集简化了，claudecode 里面 team agent 也有 TaskCreate、TaskUpdate 等工具，任务系统是团队共享的

然后 lead agent 在每轮循环后检查信箱，如果 team agent 发来了信息就把他 append 到上下文后：

```python
# 主循环结束后
inbox = BUS.read_inbox("lead")
if inbox:
    inbox_text = "\n".join(
        f"From {m['from']}: {m['content'][:200]}" for m in inbox)
    history.append({"role": "user",
                    "content": f"[Inbox]\n{inbox_text}"})
```

>在 claudecode 里面是每秒都会检查 inbox，防止用户没有输入就不处理信箱。


## 16. Team Protocol

>前面提到了怎么实现 team agent 之间的分工合作，但是他们发送的信息没有结构化的协议。例如 lead agent 希望 team agent 关机，但是 team agent 任务还没完成，关机就会导致数据丢失，这就需要类似 tcp 握手的动作。

每个协议请求创建一条状态记录，记录谁发的、发给谁、当前状态、附带内容：

```python
@dataclass
class ProtocolState:
    request_id: str      # 唯一 ID，如 "req_004281"
    type: str            # "shutdown" | "plan_approval"
    sender: str          # 发起方
    target: str          # 接收方
    status: str          # pending | approved | rejected
    payload: str         # 计划文本或关机原因
    created_at: float    # 时间戳

pending_requests: dict[str, ProtocolState] = {}
```

现在我们需要考虑一个问题，如何设计 `spawn_teammate_agent` 这个工具

```python
def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    if name in active_teammates:
        return f"Teammate '{name}' already exists"

    system = (f"You are '{name}', a {role}. "
              f"Use tools to complete tasks. "
              f"Check inbox for protocol messages (shutdown_request, etc).")

    def handle_inbox_message(name: str, msg: dict, messages: list) -> bool:
        """Dispatch incoming protocol messages by type.
        Returns True if teammate should stop."""
        msg_type = msg.get("type", "message")
        meta = msg.get("metadata", {})
        req_id = meta.get("request_id", "")

        if msg_type == "shutdown_request":
            BUS.send(name, "lead", "Shutting down gracefully.",
                     "shutdown_response",
                     {"request_id": req_id, "approve": True})
            print(f"  \033[35m[protocol] {name} approved shutdown "
                  f"({req_id})\033[0m")
            return True  # stop the loop

        if msg_type == "plan_approval_response":
            approve = meta.get("approve", False)
            if approve:
                messages.append({"role": "user",
                    "content": f"[Plan approved] Proceed with the task."})
            else:
                messages.append({"role": "user",
                    "content": f"[Plan rejected] Feedback: {msg['content']}"})

        return False  # continue

    def run():
        messages = [{"role": "user", "content": prompt}]
        sub_tools = [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object",
                              "properties": {"command": {"type": "string"}},
                              "required": ["command"]}},
            {"name": "read_file", "description": "Read file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"}},
                              "required": ["path"]}},
            {"name": "write_file", "description": "Write file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["path", "content"]}},
            {"name": "send_message",
             "description": "Send message to another agent.",
             "input_schema": {"type": "object",
                              "properties": {"to": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["to", "content"]}},
            {"name": "submit_plan",
             "description": "Submit a plan for Lead approval.",
             "input_schema": {"type": "object",
                              "properties": {"plan": {"type": "string"}},
                              "required": ["plan"]}},
        ]
        sub_handlers = {
            "bash": run_bash, "read_file": run_read, "write_file": run_write,
            "send_message": lambda to, content: (BUS.send(name, to, content),
                                                  "Sent")[1],
            "submit_plan": lambda plan: _teammate_submit_plan(name, plan),
        }

        shutdown_requested = False
        while not shutdown_requested:
            # Check inbox for protocol messages
            inbox = BUS.read_inbox(name)
            should_stop = False
            non_protocol = []
            for msg in inbox:
                if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                    should_stop = handle_inbox_message(name, msg, messages)
                    if should_stop:
                        break
                else:
                    non_protocol.append(msg)
            if should_stop:
                shutdown_requested = True
                break
            if non_protocol:
                inbox_json = json.dumps(non_protocol)
                messages.append({"role": "user",
                    "content": "<inbox>" + inbox_json + "</inbox>"})

            # LLM turn
            try:
                response = client.messages.create(
                    model=MODEL, system=system, messages=messages[-20:],
                    tools=sub_tools, max_tokens=8000)
            except Exception:
                break

            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                # Idle: wait for inbox messages instead of exiting
                # Real CC sends idle_notification to Lead here
                while not shutdown_requested:
                    time.sleep(1)
                    inbox = BUS.read_inbox(name)
                    if not inbox:
                        continue
                    for msg in inbox:
                        if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                            should_stop = handle_inbox_message(name, msg, messages)
                            if should_stop:
                                shutdown_requested = True
                                break
                        else:
                            non_protocol.append(msg)
                    if shutdown_requested:
                        break
                    if non_protocol:
                        inbox_json = json.dumps(non_protocol)
                        messages.append({"role": "user",
                            "content": "<inbox>" + inbox_json + "</inbox>"})
                        break  # back to LLM turn with new messages

            # Execute tool calls
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    handler = sub_handlers.get(block.name)
                    output = handler(**block.input) if handler else "Unknown"
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": str(output)})
            messages.append({"role": "user", "content": results})

        # Send final summary to Lead
        summary = "Done."
        for msg in reversed(messages):
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for b in msg["content"]:
                    if getattr(b, "type", None) == "text":
                        summary = b.text
                        break
                else:
                    continue
                break
        BUS.send(name, "lead", summary, "result")
        active_teammates.pop(name, None)
        print(f"  \033[32m[teammate] {name} finished\033[0m")

    active_teammates[name] = True
    threading.Thread(target=run, daemon=True).start()
    print(f"  \033[36m[teammate] {name} spawned as {role}\033[0m")
    return f"Teammate '{name}' spawned as {role}"
```

当 lead agent 通过 tooluse 启动一个 team agent 时候，他干的实际上是这些事情：
1. 启动一个 agent loop
2. 在收到 team agent 的关闭命令前，不断循环检测 inbox 里面有没有新的信息
3. inbox 里面的信息有两种，一种就是普通的命令，比如 lead agent 希望 team agent 去 pip install；另一种就是 protocol，比如上面提到的 shutdown team agent。如果是普通的 inbox message，就会直接插入 team agent 的上下文中，如果是 protocol message，就会执行对应操作，然后返回 protocol response 给 lead agent。
4. 当 team agent 的任务完成就会进入 idle 状态，等待 inbox 里面传来新的信息

下面看一个例子：

```text
s16 >> Spawn alice as a backend dev. Ask her to create a file. Then request her shutdown.
> spawn_teammate
  [teammate] alice spawned as backend dev
Teammate 'alice' spawned as backend dev
  [alice] thinking (round 1)...
  [alice] > write_file({'path': 'D:\\dev\\llm\\alice.txt', 'content': 'hello from alice'})
  [alice] result: Wrote 16 bytes to D:\dev\llm\alice.txt
  [alice] thinking (round 2)...
> send_message
  [bus] lead → alice: (message) Please create the file named alice.txt in D:\dev\l
Sent to alice
  [alice] says: Done.
  [alice] idle, waiting for inbox...
  [alice] woke up: 1 new message(s)
  [alice] thinking (round 3)...
  [alice] says: 已完成。
  [alice] idle, waiting for inbox...
> request_shutdown
  [bus] lead → alice: (shutdown_request) Please shut down gracefully.
  [protocol] shutdown_request → alice (req_910701)
Shutdown request sent to alice (req: req_910701)
  [alice] inbox: (shutdown_request) Please shut down gracefully.
  [bus] alice → lead: (shutdown_response) Shutting down gracefully.
  [protocol] alice approved shutdown (req_910701)
  [alice] wrapping up: 已完成。
  [bus] alice → lead: (result) 已完成。
  [teammate] alice finished
已启动 alice，已让她创建文件，并已请求她关闭。
  [protocol] unknown request_id: req_267789
  [protocol] shutdown ✓ (req_910701: approved)

[Inbox: 4 messages injected]
```

可以看到，lead agent 在 spawn team agent Alice 时候，就把创建一个文本文件的 prompt 塞进它的上下文了。但是 lead agent 同时又思考需要让 Alice 创建文件，所以出现了两次创建文件的操作。Alice 在收到 lead agent 的 inbox 消息时，根据上下文发现文件已经创建好了，所以直接返回已完成。