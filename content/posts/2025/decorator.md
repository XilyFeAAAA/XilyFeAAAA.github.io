---
title: "Python 装饰器"
date: '2025-10-19T17:59:11+08:00'
authors: [Xilyfe]
series: ["Python 技巧"]
tags: ["Python", "装饰器"]
--- 


## 最基本的装饰器

```python
def simple_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before the function runs.")
        func(*args, **kwargs)
        print("After the function runs.")
    return wrapper

def say_hello():
    print("Hello!")

decorated = simple_decorator(say_hello)
decorated()
```
输出：
```python
Before the function runs.
Hello!
After the function runs.
```
## 语法糖

上例可以简化为：
```python
@simple_decorator
def say_hello():
    print("Hello!")

say_hello()
```

使用 `@decorator_name` 等价于：`say_hello = simple_decorator(say_hello)`

## 带参数的装饰器

```python
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet():
    print("Hi!")

greet()
```

## 类装饰器

> 类装饰器可使用类的实例属性轻松保存状态，可以更好组织复杂逻辑。

```python
class Decorator:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("Before call")
        result = self.func(*args, **kwargs)
        print("After call")
        return result

@Decorator
def greet(name):
    print(f"Hello, {name}")

greet("Alice")
```

### 类成员函数装饰器

> 要让 Python 装饰器装饰 **类的成员函数** 并获取到它的 `self`，需要让装饰器的 wrapper 接受并传递 `self` 参数

```python
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):  # 注意第一个参数是 self
        print(f"[DEBUG] Before calling {func.__name__}")
        result = func(self, *args, **kwargs)  # 保留 self
        print(f"[DEBUG] After calling {func.__name__}")
        return result
    return wrapper

```

## 使用 `functools.wraps`

> 使用装饰器会丢失原函数的元信息，如 `__name__`、`__doc__`，可以通过 `functools.wraps` 修复。

```python
from functools import wraps

def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```
否则:
```python
@decorator
def foo():
    """This is foo."""
    pass

print(foo.__name__)  # 默认会输出 wrapper 而不是 foo
```