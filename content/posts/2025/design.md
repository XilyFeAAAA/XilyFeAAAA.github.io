---
title: "Python 设计模式"
date: '2025-10-19T17:59:11+08:00'
authors: [Xilyfe]
series: ["Python 技巧"]
tags: ["Python", "设计模式"]
--- 


## 单例模式

### 实现方式

1. **模块实现单例**

```python
# a.py
class Singleton:
	pass

singleton = Singleton()

# b.py
from a import singleton
```

1. **装饰器实现单例**

```python
def singleton(cls):
    _instance = {}
    def decorated(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return decorated
```

2. **使用类的方式创建装饰器**

```python
class Singleton(object):
    def __init__(self,cls):
        self._cls = cls
        self._instance = {}

    def __call__(self, *args):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls(*args)
        return self._instance[self._cls]
```

3. **使用类方法实现单例**

```python
class Person(object):
    @classmethod
    def get_instance(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = cls(*args, **kwargs)
        return cls._instance
```

4. **使用`__new__`方法实现单例**

```python
class Person(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance
```

5. **使用元类`metaclass`实现单例**

```python
class Singleton(type):
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class Person(metaclass=Singleton):
    def __init__(self,name, age):
        self.name = name
        self.age = age
```


> [!Warning]
> 在多线程和异步的环境下，有可能出现多个线程同时拿到`_instance=None`这个值，导致创建多个实例，解决办法就是通过锁来限制。

1. 多线程

```python
class Person(object):
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        with cls._lock:
            if not hasattr(cls, '_instance'):
                cls._instance = cls(*args, **kwargs)
        return cls._instance
```

2. 异步

```python
class Person(object):
    _lock = asyncio.Lock()
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        async with cls._lock:
            if not hasattr(cls, '_instance'):
                cls._instance = cls(*args, **kwargs)
        return cls._instance

```