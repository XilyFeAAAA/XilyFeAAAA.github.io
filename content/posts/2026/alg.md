---
title: 面试算法速通
date: 2026-05-18T14:06:04+08:00
featuredImage: http://img.xilyfe.top/img/20260520102520276.png
authors:
  - Xilyfe
series: []
tags: []
lastmod: 2026-05-30T12:08:57+08:00
---
## 1. 动态规划

### 1.1 01背包

{{< admonition type=summary title="背景">}} 
有 $n$ 件物品和一个最多能背重量为 $w$ 的背包。第 $i$ 件物品的重量是 `weight[i]`，得到的价值是 `value[i]` 。每件物品只能用一次，求解将哪些物品装入背包里物品价值总和最大。
{{< /admonition >}}

- DP 数组： `dp[i][j]`，`[0,i]` 的物品放进重量为 $j$ 的包中的最大价值
- 递推式：`dp[i][j] = max(dp[i-1][j], dp[i-1][j-weight[i]] + value[i])`
- 可以看到 `dp[i][j]` 依赖于左上角，所以第一列和第一行需要初始化

```cpp
for(int i = 1; i < weight.size(); i++) { // 遍历物品
    for(int j = 0; j <= bagweight; j++) { // 遍历背包容量
        if (j < weight[i]) dp[i][j] = dp[i - 1][j];
        else dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);
    }
}
```

### 1.2 完全背包

{{< admonition type=summary title="背景">}} 
有 $n$ 件物品和一个最多能背重量为 $w$ 的背包。第 $i$ 件物品的重量是 `weight[i]`，得到的价值是 `value[i]` 。每件物品**能用无数次**，求解将哪些物品装入背包里物品价值总和最大。
{{< /admonition >}}

- DP 数组： `dp[i][j]`，`[0,i]` 的物品放进重量为 $j$ 的包中的最大价值
- 递推式：`dp[i][j] = max(dp[i-1][j], dp[i][j-weight[i]] + value[i])`
- 可以看到 `dp[i][j]` 依赖于左上角，所以第一列和第一行需要初始化

我们思考一下，对于 `dp[i][j]` 我们可以选择放入物品 $i$ 或者不放物品 $i$。假如不放物品 $i$ 那么就是 `dp[i-1][j]`，我们希望得到 `[0, i-1]` 的物品放进重量为 $j$ 的包中的最大价值。假如我们放物品 $i$，背包容量为 `j - weight[i]`，`dp[i][j - weight[i]]` 为背包容量为 `j - weight[i]` 且不放物品 $i$ 的最大价值

```cpp
for(int i = 1; i < weight.size(); i++) { // 遍历物品
    for(int j = 0; j <= bagweight; j++) { // 遍历背包容量
        if (j < weight[i]) dp[i][j] = dp[i - 1][j];
        else dp[i][j] = max(dp[i - 1][j], dp[i][j - weight[i]] + value[i]);
    }
}
```

### 1.3 零钱兑换

{{< admonition type=summary title="背景">}} 
给定不同面额的硬币 coins 和一个总金额 amount。计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。你可以认为每种硬币的数量是无限的。
{{< /admonition >}}

- DP 数组： `dp[i]`，凑成总金额 $i$ 所需要的最少硬币数量
- 递推式：`dp[i] = min(dp[i], dp[i - coin[j]]+1)`
- 可以看到 `dp[i]` 依赖于左侧并且有一个 min 操作，所以把所有赋值为 inf 并且令 `dp[0]=0`

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)  # 创建动态规划数组，初始值为正无穷大
        dp[0] = 0  # 初始化背包容量为0时的最小硬币数量为0

        for coin in coins:  # 遍历硬币列表，相当于遍历物品
            for i in range(coin, amount + 1):  # 遍历背包容量
                if dp[i - coin] != float('inf'):  # 如果dp[i - coin]不是初始值，则进行状态转移
                    dp[i] = min(dp[i - coin] + 1, dp[i])  # 更新最小硬币数量

        if dp[amount] == float('inf'):  # 如果最终背包容量的最小硬币数量仍为正无穷大，表示无解
            return -1
        return dp[amount]  # 返回背包容量为amount时的最小硬币数量
```

### 1.4 最长上升子序列

{{< admonition type=summary title="背景">}} 
给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。
{{< /admonition >}}

- DP 数组：`dp[i]`，$i$ 之前包括 $i$ 的以 `nums[i]` 结尾的最长递增子序列的长度
- 递推式：`dp[i] = max(dp[i], dp[j]+1)`
- 每一个 $i$，对应的 `dp[i]`（即最长递增子序列）起始大小至少都是1

```cpp
for (int i = 1; i < nums.size(); i++) {
    for (int j = 0; j < i; j++) {
        if (nums[i] > nums[j]) dp[i] = max(dp[i], dp[j] + 1);
    }
    if (dp[i] > result) result = dp[i]; // 取长的子序列
}
```

### 1.5 最长递增子串

{{< admonition type=summary title="背景">}} 
给定一个未经排序的整数数组，找到最长且连续递增的子序列，并返回该序列的长度。这里连续的子序列也就是子串。
{{< /admonition >}}

- DP 数组：`dp[i]`，$i$ 之前包括 $i$ 的以 `nums[i]` 结尾的最长递增子串的长度
- 递推式：`dp[i] = dp[i-1] + 1 if nums[i] > nums[i-1] else 1`
- 每一个 $i$，对应的 `dp[i]`（即最长递增子序列）起始大小至少都是 1

>这里面 dp 数组的定义最为关键，`dp[i]` 一定是以下标 $i$ 为结尾，并不是说一定以下标 0 为起始位置。

```cpp
for (int i = 1; i < nums.size(); i++) {
    if (nums[i] > nums[i - 1]) { // 连续记录
        dp[i] = dp[i - 1] + 1;
    }
}
```

### 1.6 最大子序和

{{< admonition type=summary title="背景">}} 
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
{{< /admonition >}}

- DP 数组：`dp[i]` 以 `nums[i]` 结尾的最大子序和

这个题目看似和 **最大子串和** 是一个思路

```python
for i in range(1, n):
    num = nums[i]
    dp_max.append(max(dp_max[i - 1] * num, dp_min[i - 1] * num, num))
    dp_min.append(min(dp_max[i - 1] * num, dp_min[i - 1] * num, num))

```



### 1.7 乘积最大子数组

{{< admonition type=summary title="背景">}} 
给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续 子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
{{< /admonition >}}

- DP 数组：`dp[i]` 以 `nums[i]` 结尾的最大乘积

这个题目看似和 **最大子串和** 是一个思路

```python
for i in range(1, n):
    num = nums[i]
    dp_max.append(max(dp_max[i - 1] * num, dp_min[i - 1] * num, num))
    dp_min.append(min(dp_max[i - 1] * num, dp_min[i - 1] * num, num))

```



### 1.8 最长重复子数组

{{< admonition type=summary title="背景">}} 
给两个整数数组 `nums1` 和 `nums2` ，返回 _两个数组中 **公共的** 、长度最长的子数组的长度_ 。
{{< /admonition >}}

这题说的是子数组，元素都是连续的，所以要想到 **最大子序和** 这一题，也就是 `dp[i]` 代表 `nums[i]` 元素结尾的最大子串和。所以这一题里面 `dp[i][j]` 代表 `nums1 [i]` 和 `nums2 [j]` 结尾的最大重复子数组长度。

```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        n, m = len(nums1), len(nums2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        ans = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if nums1[i-1] == nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    ans = max(dp[i][j], ans)
        return ans
```

### 1.9 最长公共子序列





### 1.10 分割等和子集

{{< admonition type=summary title="背景">}} 
给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
{{< /admonition >}}

这题要求把数组分成两个元素和相等的子集，换句话说就是 **找到一个子集，使得元素和是全部元素和的一半**，因此有两个 DP 思路：
1. 布尔 DP：`dp[i][j]` 表示从 `nums[0...i]` 中是否能选出若干元素，使和恰好为 `j`。如果 `dp[i][j]=True` 那么要不然不选物品 `i` 的时候元素和就是 `j` 也就是 `dp[i - 1][j]=True`，要不然就是加上物品 `i` 之后仍元素和等于 `j`，即 `dp[i - 1][j - nums[i - 1]]=True`。
2. 01背包：`dp[i][j]` 表示容量为 `j` 的背包，从 `nums[0...i]` 中能凑出的“最大和”，这里的最大和意思是**不超过 j 的前提下，能够凑出的最大总和**。


```python
class Solution:
    def canPartition(self, nums: list[int]) -> bool:
        total = sum(nums)
        if total % 2:
            return False

        n = len(nums)
        target = total // 2
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = True

        
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                if j < nums[i - 1]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
        return dp[n][target]
```

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2:
            return False

        n = len(nums)
        target = total // 2

        dp = [[0] * (target + 1) for _ in range(n)]
        for j in range(nums[0], target + 1):
            dp[0][j] = nums[0]

        for i in range(1, n):
            for j in range(1, target + 1):
                if j < nums[i]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - nums[i]] + nums[i])
            if dp[i][target] == target:
                return True
        return False
```

>[最后一块石头的重量 II](https://leetcode.cn/problems/last-stone-weight-ii/) 这个问题也是 **分割等和子集** 的变种。返回这一堆石头 **最小的可能重量**，也就是尽可能把石头分成两堆总重量相等的，所以也就是找到重量和接近总重量一半的子数组。

### 1.11 排列数

{{< admonition type=summary title="背景">}} 
给你一个由 **不同** 整数组成的数组 `nums` ，和一个目标整数 `target` 。请你从 `nums` 中找出并返回总和为 `target` 的元素组合的个数。
{{< /admonition >}}

- 关键点：排列问题不能单纯套用 01 背包的办法，假如我们先遍历物品再遍历背包容量，那么每次外层遍历都会确定一个物品的位置，这会导致你的结果里只能出现像 `[1, 1, 2]` 这样升序的组合，而永远无法产生像 `[1, 2, 1]` 这样数字交替出现的**排列**。所以对于排列问题我们只需要用一维 DP，然后先遍历容量再遍历物品。

```python
class Solution:
    def combinationSum4(self, nums: list[int], target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(1, target + 1):
            for num in nums:
                if i - num >= 0:
                    dp[i] += dp[i - num]
        return dp[target]

```

### 1.12 树形 DP

{{< admonition type=summary title="背景">}} 
小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 `root` 。除了 `root` 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果 **两个直接相连的房子在同一天晚上被打劫** ，房屋将自动报警。给定二叉树的 `root` 。返回 _**在不触动警报的情况下** ，小偷能够盗取的最高金额_ 。
{{< /admonition >}}

- 关键点：树形 DP 实际上就是一种特殊的递归，我们让节点返回两个值：选择节点和不选节点当前子树能提供的最大值。

```python
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:


        def dfs(node: Optional[TreeNode]) -> tuple[int, int]:
            if node is None:
                return 0, 0
            
            l, l_not = dfs(node.left)
            r, r_not = dfs(node.right)

            return node.val + l_not + r_not, max(l, l_not) + max(r, r_not)
            
        rt, rt_not = dfs(root)
        return max(rt, rt_not)
```


## 2. 单调栈

>适用范围：查找元素**下一个**比它大 or 小的元素。**更直白来说就是用一个栈来记录我们遍历过的元素**，因为我们遍历数组的时候，我们不知道之前都遍历了哪些元素，以至于遍历一个元素找不到是不是之前遍历过一个更小的，所以我们需要用一个容器来记录我们遍历过的元素。

假如我们需要找到下一个更大的元素，那么就需要一个单调递减栈。假如数组是 `[1,3,2,4]`，首先我们把 1 放进栈里，然后再放 3。由于要保持栈内单调递减，所以把栈顶的 1 弹出，这时候 3 就是 1 的下一个更大元素。


```python
nums = [1, 3, 2, 4]
st = []
for i, num in enumerate(nums):
	while st and nums[st[-1]] < num:
		prev = st.pop()
		print(f"{prev} -> {i}")
	st.append(i)
```


## 3. Boyer-Moore 投票

{{< admonition type=summary title="背景">}} 
Boyer-Moore 投票算法解决的问题是：假定有一个长度为 $n$ 的数组，其中存在一个众数，然后返回这个众数。众数是指在数组中出现次数**大于** `⌊ n/2 ⌋` 的元素。
{{< /admonition >}}

算法的思路可以用「擂台赛」打比方：

1. 擂主登场：`nums[0]` 成为初始擂主，生命值为 1。
2. 挑战者出现：遍历后续元素，作为挑战者。
3. 比武：如果挑战者与擂主属于同一门派（值相同），那么擂主生命值加 1，否则擂主生命值减 1。
4. 擂主更迭：如果比武后，擂主生命值降为 0（同归于尽），那么下一个挑战者成为新的擂主，生命值为 1。
5. 最后在擂台上的那人，便是武林盟主（绝对众数）。

我们思考一下，假如长度为 $n$ 的数组众数是 $a$，它出现了 $i$ 次，其余元素出现了 $n-i$ 次，根据众数的定义满足 $i > n - i$，所以所有挑战者都减一后擂主的血量仍然大于零。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        ans = hp = 0
        for x in nums:
            if hp == 0:  # x 是初始擂主，生命值为 1
                ans, hp = x, 1
            else:  # 比武，同门加血，否则扣血
                hp += 1 if x == ans else -1
        return ans
```

## 4. 中心扩散法

{{< admonition type=summery title="背景">}} 
给你一个字符串 `s`，找到 `s` 中最长的回文子串。
{{< /admonition >}}

中心扩散法的思想是：从每一个位置出发，向两边扩散即可，遇到不是回文的时候结束。

![image.png](http://img.xilyfe.top/img/20260518223329368.png)

```python
def longestPalindrome(self, s: str) -> str:
    n = len(s)
    ans = ""
    max_len = 0
    for i in range(n):
        left, right = i - 1, i + 1
        while left >= 0 and s[left] == s[i]:
            left -= 1
        while right < n and s[right] == s[i]:
            right += 1
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        l = right - left - 1
        if l > max_len:
            max_len = l
            ans = s[left + 1: right]
    return ans
```


## 记录

- [ ] 最长连续序列：注意题目要求+去重
- [ ] 接雨水
- [ ] 三数之和：双指针+去重
- [x] 无重复字符的最长子串：
- [ ] 找到字符串中所有字母异位词：滑动窗口又错了
- [ ] 和为 K 的子数组
- [ ] 滑动窗口最大值
- [ ] 最小覆盖子串
- [ ] 最大子数组和
- [ ] 缺失的第一个正数
- [ ] 矩阵置零
- [ ] 搜索二维矩阵 II
- [ ] 反转链表
- [ ] 环形链表 II
- [ ] 合并两个有序链表
- [ ] 排序链表
- [ ] 对称二叉树
- [ ] 将有序数组转换为二叉搜索树
- [ ] 验证BST
- [ ] 从前序与中序遍历序列构造二叉树
- [ ] 路径总和 III
- [ ] 二叉树的最近公共祖先
- [ ] 课程表
- [ ] 单词搜索
- [ ] 搜索插入位置
- [ ] 搜索旋转排序数组
- [ ] 最小栈
- [ ] 字符串解码
- [ ] 每日温度
- [ ] 柱状图中最大的矩形
- [ ] 数组中的第K个最大元素
- [ ] 前 K 个高频元素
- [ ] 数据流中位数
- [ ] 只出现一次的数字
- [ ] 多数元素
- [ ] 下一个排列