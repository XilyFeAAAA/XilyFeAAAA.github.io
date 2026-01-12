---
title: "计算几何学"
date: '2025-11-04T23:30:20+08:00'
authors: [Xilyfe]
series: ["算法分析与设计"]
tags: ["算法", "几何学"]
---

# 计算几何学

## 最近点对问题

### 算法设计思路


先按 x 坐标排序数组（一次O(nlogn)）。递归把数组对半分成左右两部分，分别求出最小距离 dl 和 dr，取 d=min(dl,dr)。考虑跨分割线 midx 的点对：只需考虑距离分割线小于 d 的点（形成一个带状区域），然后按 y 排序。对带内每个点，最多检查后常数个点即可。合并阶段总开销 O(n)，因此总体 O(nlogn)。

### 源码

```python
from dataclasses import dataclass
import math
import time

FILEPATH = r"D:\school\ustc\assignment\algorithm\实验二：求平面上n个顶点的最近点对问题\实验二：求平面上n个顶点的最近点对问题\data.txt"
MIN_LEN = 3

@dataclass
class Point:
    x: float
    y: float
    idx: int

def readfile(fp: str) -> list[Point]:
    res = []
    for line in open(fp, "r",  encoding='utf-8').readlines():
        idx, x, y = line.strip().split()
        res.append(Point(float(x), float(y), int(idx)))
    return res


def simple(arr: list[Point]) ->  tuple[float, Point, Point]:
    n = len(arr)
    res, p1, p2 = float('inf'), None, None
    if n < 2:
        return res, p1, p2
    for i in range(n):
        x1, y1 = arr[i].x, arr[i].y
        for j in range(i+1, n):
            x2, y2 = arr[j].x, arr[j].y
            dis = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if dis < res:
                res, p1, p2 = dis, arr[i], arr[j]
    return res, p1, p2

def partition(arr: list[Point]) -> tuple[float, Point, Point]:
    """分治法 时间复杂度O(nlogn)"""

    res, p1, p2 = float("inf"), None, None
    
    if (n := len(arr)) < 2:
        return res, p1, p2
    
    if n > MIN_LEN:
        mid = n // 2
        midx = arr[mid].x
        dl, lp1, lp2 = partition(arr[:mid])
        dr, rp1, rp2 = partition(arr[mid:])
        if dl <= dr:
            best_res, best_pair = dl, (lp1, lp2)
        else:
            best_res, best_pair = dr, (rp1, rp2)
        strip = [p for p in arr if (p.x - midx) ** 2 < best_res]
        strip.sort(key=lambda p: p.y)
        m = len(strip)
        for i in range(m):
            ix, iy = strip[i].x, strip[i].y
            for j in range(i+1, m):
                jx, jy = strip[j].x, strip[j].y
                dis = (ix - jx) ** 2 + (iy - jy) ** 2
                if dis < best_res:
                    best_res, best_pair = dis, (strip[i], strip[j])
                if (iy - jy) ** 2 > dis: break
        return best_res, *best_pair
    else:
        # 长度小于Min_Len暴力
        for i in range(n):
            x1, y1 = arr[i].x, arr[i].y
            for j in range(i+1, n):
                x2, y2 = arr[j].x, arr[j].y
                dis = (x1 - x2) ** 2 + (y1 - y2) ** 2
                if dis < res:
                    res, p1, p2 = dis, arr[i], arr[j]
        return res, p1, p2
            




if __name__ == "__main__":
    arr = readfile(fp=FILEPATH)
    arr.sort(key=lambda p: p.x)
    sta = time.time()
    dis2, p1, p2 = partition(arr[:])
    print(f"分治法: 最近点 p1-{p1.idx} p2-{p2.idx} 距离为{math.sqrt(dis2)} 耗时: {time.time() - sta}s")
    sta = time.time()
    dis2, p1, p2 = simple(arr[:])
    print(f"朴素算法: 最近点 p1-{p1.idx} p2-{p2.idx} 距离为{math.sqrt(dis2)} 耗时: {time.time() - sta}s")
    
```

### 结果分析

实验输出如下:

```
分治法: 最近点 p1-7119 p2-5826 距离为2.807526491415896 耗时: 0.15185284614562988s
朴素算法: 最近点 p1-7119 p2-5826 距离为2.807526491415896 耗时: 16.47048258781433s
```

可以看出在数据较大的情况下，分治法相对于朴素的二重遍历速度提高非常大，但是算法还可以优化：当前每次合并操作都会对条状带进行y的排序，这个操作可以在递归的过程中传递的。