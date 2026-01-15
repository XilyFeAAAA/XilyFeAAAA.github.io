---
title: 动态规划
date: 2026-01-12T11:31:48+08:00
lastmod: 2026-01-13T12:39:32+08:00
featuredImage: http://img.xilyfe.top/img/202601122278.png
authors:
  - Xilyfe
series:
  - 算法分析与设计
tags:
  - 算法
  - dp
---

## 思想&步骤

- 解决的是寻找问题的**一个最优解**
- 具备的两个要素：**最优子结构和子问题重叠**
	- 问题的最优解由相关子问题的最优解组合而成
	- 子问题重叠：例如斐波那契数列，重复求同一个子问题
- 步骤：
	- 识别最优解的特征
	- 递归的定义最优解的值（就是状态转移方程）
	- 自底向上求解

## 和分治法区别

- 分治法是分解为互不相干的子问题求解之后合并
- 动态规划是重叠的子问题

## 算法设计

### 攒硬币

```text
例：给定不同面额的硬币coins和一个总金额amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回-1。你可以认为每种硬币的数量是无限的。
```

假设我们用 `Func(coins, amount)` 求解所需的最少的硬币个数，并且 硬币为 1 和 2，那么可以把他分解为，求解 amount-1 所需最少硬币和 amount -2 所需最少硬币的最小值 + 1

```python
def MinimumCoins(coins, amount):
	dp = [INF] * (amount+1)
	dp[0] = 0
	for i in range(1, len(dp)):
		for coin in coins:
			if i - coin < 0: continue
			dp[i] = min(dp[i], dp[i-coin]) + 1
	return dp[amount]
```

### 钢条切割


<div style="text-align: center">     <img src="http://img.xilyfe.top/img/20260110125444743.png" width="90%" /> </div>

这个题思路和凑硬币完全一样，把最小值变成最大值，就行了。

### 矩阵链乘法


<div style="text-align: center">     <img src="http://img.xilyfe.top/img/20260110135416958.png" width="80%" /> </div>

m\[i]\[j] 依赖于区间更短的子问题，所以必须按**区间长度递增**计算，常见顺序是：

```
for l = 2 to n // 链长度  
 for i = 1 to n−l+1  
  j = i + l − 1  
  计算 m[i][j]
```

这是一个典型的**区间 DP**。

```python
MatrixMultiply()
dp = [[INF for i in range(n+1)] * (n+1)]
for i in range(n+1):
	dp[i][i] = 0
for l in range(2, n+1):
	for i in range(1, n+1-l):
		j = i + l - 1
		for k in range(i, j):
			cost = dp[i][k]+dp[k+1][j]+m[i].p*m[k].q*m[j].q
			if cost < dp[i][j]:
				dp[i][j] = cost
				res[i][j] = k
```

### 最大子串和

```
例：给定一个长度为 n 的整数序列 A[1…n]（可正可负），要求找一个**连续子段** A[i…j]，使其元素和最大。
1 2 -5 6 2
```

关键想法： **以位置 i 结尾的最大子段和只和 i−1 有关**。

```python
ans = -INF
dp = [0]*(n+1)
dp[1] = A[1]
for i in range(2, n+1):
	dp[i] = A[i] if dp[i-1] < 0 else A[i] + dp[i-1]
	ans = max(ans, dp[i])
return ans

ABCD
AAAB
```
### 最长公共子序列

<div style="text-align: center">     <img src="http://img.xilyfe.top/img/20260110144927916.png" width="90%" /> </div>

 **空间复杂度O(mn)**

用 dp 数组，dp\[i]\[j] 代表 A\[:i] 和 B\[:j] 的 LCS，如果 A\[i] 和 B\[j] 相等，那么说明可以向后移动一位；如果不相等，那么要么是 dp\[i-1]\[j] 要么是 dp\[i]\[j-1]。

```python
def LCS(a: str, b: str) -> str:
    m, n = len(a), len(b)
    dp = [[0 for j in range(n+1)] for i in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i][j-1], dp[i-1][j])
    return dp[m][n]
```

> 注意子序列和子串的区别：==子序列不要求连续==，所以如果前面出现了 LCS， 后面有相同字符的话 LCS 只会更长，不会出现==被前面较短的匹配卡住的例子==

**空间复杂度O(2\*min(m,n))**

计算时第 i 行只依赖上一行和当前行左边的值，所以只需保存两行：`prev` (代表 dp\[i-1]\[_]) 和 `cur` (代表 dp\[i]\[_])。将列数设为较短的字符串长度能最小化空间，因此空间为 $2*(\min(m,n)+1)$，写成大 O 即 $O(2\cdot\min(m,n))$。

```python
def LCS_2min(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    prev = [0 for _ in range(m+1)]
    for i in range(1, n+1):
        cur = [0 for _ in range(n+1)]
        for j in range(1, m+1):
            if b[i-1] == a[j-1]:
                cur[j] = prev[j-1]
            else:
                cur[j] = max(cur[j-1], prev[j])
        prev = cur
    return cur[m]
```

**空间复杂度O(min(m,n))**

空间复杂度还可以进一步压缩，刚刚说过：计算时第 i 行只依赖上一行和当前行左边的值，实际上是==依赖上一行的 dp\[i-1]\[j-1]，dp\[i-1]\[j] 和当前行左边 dp\[i]\[j-1] 的值==。我们用 dp记录当前行的值，diag 记录左上方的值，那上方的值怎么办呢？实际上 dp 数组 j 位置还未更新时候，dp\[j]就是上一行的值，也就是 dp\[i-1]\[j]。

```python
def LCS_min(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    dp = [0 for _ in range(m+1)]
    for i in range(1, n+1):
        diag = 0
        for j in range(1, m+1):
            tmp = dp[j]
            if b[i-1] == a[j-1]:
                dp[j] = 1 + diag
            else:
                # max里面的dp[j] 代表上一行还未更新的dp[i-1][j]
                dp[j] = max(dp[j], dp[j-1])
            # 对dp[i][j+1] 来说 diag 是dp[i-1][j] 用tmp记录
            diag = tmp
    return dp[m]
```

### 凑数字


```
例：给你一个整数 `n` ，返回和为 `n` 的完全平方数的最少数量。**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。
```

```python
import math
class Solution:
    def numSquares(self, n: int) -> int:
        pow = [i**2 for i in range(1, math.floor(math.sqrt(n))+1)]
        dp = [float("inf") for _ in range(n+1)]
        dp[0] = 0
        for i in range(1, n+1):
            for k in pow:
                if i-k >= 0:
                    dp[i] = min(dp[i], dp[i-k] + 1)
        return dp[n]
```
