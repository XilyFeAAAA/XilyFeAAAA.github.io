---
title: "Python 异步编程"
date: '2025-10-19T17:59:11+08:00'
authors: [Xilyfe]
series: ["Python 技巧"]
tags: ["Python", "异步", "asyncio"]
--- 


## 事件循环

### 获取当前事件循环

| 方法                         | 适用场景                            | 是否要求协程中运行     | Python 3.10+     | 行为                       |
| ---------------------------- | ----------------------------------- | ---------------------- | ---------------- | -------------------------- |
| `asyncio.get_running_loop()` | 在协程/任务中获取当前运行的事件循环 | ✅ 必须在事件循环中调用 | ✅ 推荐           | 抛错如果没有运行的 loop    |
| `asyncio.get_event_loop()`   | 早期通用方式                        | ❌ 可在同步代码中用     | ⚠️ 不推荐新项目用 | 自动创建事件循环（旧行为） |

---

**官方推荐（Python 3.10+）**

- **在异步代码中**，优先使用：`asyncio.get_running_loop()`
  
- **在同步代码中**，用 `asyncio.new_event_loop()` 显式创建，并用 `asyncio.run()` 管理事件循环

```python
import asyncio

async def my_async_task():
    await asyncio.sleep(1)
    print("任务完成")

def main():
    # 创建一个新的事件循环
    loop = asyncio.new_event_loop()

    # 可选：将其设置为当前线程默认事件循环
    asyncio.set_event_loop(loop)

    try:
        # 执行异步任务直到完成
        loop.run_until_complete(my_async_task())
    finally:
        # 最后关闭事件循环
        loop.close()

if __name__ == "__main__":
    main()

```