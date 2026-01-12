---
title: "排序算法"
date: '2025-11-04T23:30:20+08:00'
authors: [Xilyfe]
series: ["算法分析与设计"]
tags: ["算法", "排序"]
---

## 快速排序

### 算法实现

```python
def quick_sort(arr, l, r):
    if l < r:
        idx = partition(arr, l , r)
        quick_sort(arr, l, idx - 1)
        quick_sort(arr, idx + 1, r)

def paritition(arr, left , right):
    pivot = arr[left]
    low, high = left, right
    while low < high:
        while arr[high] > pivot and low < high: high -= 1
        while arr[low] < pivot and low < high: low += 1
        if low < high:
            arr[low], arr[high] = arr[high], arr[low]
    
    arr[left], arr[low] = arr[low], arr[left]
    return low
```

### 优化方案

**1. 使用插入排序**

当输入数据已经“几乎有序”时，使用插入排序速度很快。我们可以利用这一特点来提高快速排序的速度。当对一个长度小于k的子数组调用快速排序时，让她不做任何排序就返回。上层的快速排序调用返回后，对整个数组运行插入排序来完成排序过程。

```python
def quick_sort(arr, l, r, k):
    if l >= r: return
    if r - l < k:
        insertion_sort(arr, l, r)
    else:
        idx = partition(arr, l , r)
        quick_sort(arr, l, idx - 1)
        quick_sort(arr, idx + 1, r)

def insertion_sort(arr, l, r):
    
```

