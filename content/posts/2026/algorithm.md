---
title: "算法刷题记录"
date: 2026-01-21T17:27:00+08:00
featuredImage: ""
authors:
  - Xilyfe
series: []
tags: []
lastmod: 2026-01-21T05:59:59+08:00
---
## 数组

### 二分查找

二分查找的原理就是不断收缩 left 和 right 指针，使得最终希望找的的元素 target 在他们围成的区域里。所以这就引申出两种写法，如果区域是左闭右闭的 \[left, right]，那么最终 target 就是 `left==right` 的那个值；如果是左闭右开的 \[left, right)，那么当 `left+1==right` 时候 target 就在左端点的位置。

#### 左闭右开

- 如果 `left==right`，那么说明已经没有意义 target 不存在了，所以循环条件应该是 `while left < right`。
- 出现 `A[mid] > target` 时候，我们应该收缩右边界。由于规定了右边界是开的，所以可以让 `right=A[mid]`。

```python
def binaery_search(arr, tgt):
	l, r = 0, len(arr)
	while l < r:
		mid = (l + r) >> 1
		if arr[mid] == tgt:
			return mid
		elif arr[mid] > tgt:
			r = mid
		else:
			l = mid + 1
	return -1
```

#### 左闭右闭

- 由于右端点是闭的，所以 `left==right` 时 target 仍然可能存在，需要 `while left <= right` 判断
- 出现 `A[mid] > target` 时候，我们明确 `A[mid]` 不可能为 target 了，所以不能放在右端点

```python
def binaery_search(arr, tgt):
	l, r = 0, len(arr)-1
	while l <= r:
		mid = (l + r) >> 1
		if arr[mid] == tgt:
			return mid
		elif arr[mid] > tgt:
			r = mid + 1
		else:
			l = mid + 1
	return l
```

#### lower_bound

lower_bound 标准定义是：在一个有序数组 A\[0…n−1] 中，返回第一个**大等于** x 的位置下标。假设我们使用左闭右开区间 \[left, right)，那么我们就需要保持以下的循环不变量：
1. 区间左侧 $[0, l)$中的元素一定满足 A\[i] < x  
2. 区间右侧 $[r, n)$ 中的元素一定满足 A\[i] > x

```python
def lower_bound(nums, tgt):
	l, r = 0, len(nums)
	while 
```

#### upper_bound

upper_bound 标准定义是：在一个有序数组 A\[0…n−1] 中，返回第一个**严格大于** x 的位置下标。假设我们使用左闭右开区间 \[left, right)，那么我们就需要保持以下的循环不变量：
1. 区间左侧 $[0, l)$中的元素一定满足 A\[i] ≤ x  
2. 区间右侧 $[r, n)$ 中的元素一定满足 A\[i] > x

因此，真正的 upper_bound 位置一定在 $[l, r)$ 中，我们通过不断收缩 left 和 right 就能找到 upper_bound。

```python
def upper_bound(nums, tgt):
	l, r = 0, len(nums)
	while l < r:
		mid = (l + r) >> 1
		if nums[mid] > tgt:
			r = mid
		else:
			l = mid + 1
	return l
```


## 链表

## 哈希表

## 字符串

## 双指针

## 栈&队列

## 二叉树

## 回溯

## 贪心

## 动态规划

## 单调栈

## 图论