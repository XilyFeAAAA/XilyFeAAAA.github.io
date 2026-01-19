---
title: 算法分析与设计复习
date: 2026-01-08T11:27:53+08:00
featuredImage: http://img.xilyfe.top/img/20260113133012452.jpg
authors:
  - Xilyfe
series:
  - 期末复习
tags:
  - 算法
  - 复习
lastmod: 2026-01-17T01:26:28+08:00
---

## 第二章 算法初步

### 插入排序

```python
arr = []
for i in range(1, n):
	key = arr[i]
	j = i - 1
	while j and arr[j] > key:
		arr[j] = arr[j-1]
		j -= 1
	arr[j+1] = key
```

>[!Note]
>假设前面元素已经有序，不断将新元素插入到合适的位置，保证依然有序

{{< admonition type=question title="能不能用二分查找优化最坏情况的时间复杂度？">}} 
不行。因为查找的时间降下来了，但是还有移动元素的时间。
{{< /admonition >}}

### 循环不变式

比如插入排序，每次循环从数组 A 中取出第 j 个元素插入有序区 A\[1 .. j-1]，然后递增 j。这样A\[1 .. j-1] 的有序性始终得到保持，这就是所谓的“循环不变”了。

要用循环不变式证明一个循环的正确性，通常需要证明以下三点：
- 初始化：循环的第一次迭代之前，它为真。 
- 保持：如果循环的某次迭代之前它为真，那么下次迭代之前它仍为真。 
- 终止：在循环终止时仍为真，不变式为我们提供一个有用的性质，该性质有助千证明算法是正确的

**例：用循环不变式证明插入排序算法的正确性**

循环不变式：在每次循环 i 开始前，数组的 \[0, ..., i-1] 都是有序的
1. 初始化：在第一次循环之前，`i=1`，数组 \[0,...,i-1] 只有一个元素，所以是有序的
2. 保持：在每次迭代过程中，假设 \[1…i-1]是已经排好序的序列，待排序的元素 A\[i] 依次与A\[i-1]、A\[i-2] 进行比较，如果 A\[i-1] 等大于 A\[i]，则依次将其向右移动一位 A\[j+1]<---A\[j]，当遇到开始小于 A\[i] 的元素时，则 A\[i] 找到了合适的插入位置，插入之后，整个序列又是排好序的了。
3. 当 `i=n+1` 时，循环结束，此时A\[1…n]中已经有n个元素，且已经排好序。

### 复杂性分析

![](http://img.xilyfe.top/img/20260108142513977.png)

对于插入排序，我们可以得到每条代码执行次数如上图，所以总运行时间为：

$$
T(n)=c_1n+c_2(n-1)+c_4(n-1)+c_5\sum_{j=2}^nt_j+c_6\sum_{j=2}^n(t_j-1)+c_7\sum_{j=2}^n(t_j-1)+c_8(n-1)
$$
当数组基本有序时候，会出现最佳情况 $t_j=1$，总运行时间为 

$$
T(n)=c_1n+c_2(n-1)+c_4(n-1)+c_8(n-1)=(c_1+c_2+c_4+c_5+c_8)n-(c_2+c_4+c_5+c_8)
$$

可以表示为 $T(n)=an+b$，是关于 n 的线性函数。但是当数组反向排序时候出现最坏情况，每次都需要和 A\[0,...,j-1] 对比，所以这时候 $t_j=j$，变成二次函数。

>具体证明如书p15。

一般来说平均时间复杂度和最坏时间复杂度一样坏，最坏情况时间复杂度用 $\Theta$  来表示，插入排序就是 $\Theta(n^2)$ 。

### 归并排序及其时间复杂度

```python
def MergeSort(arr, p, r):
	if p < r:
		q = (q + r) / 2
		MergeSort(arr, p, q)
		MergeSort(arr, q+1, r)
		Merge(arr, p, q, r)
def Merge(arr, p, q, r):
	n1 = q - p + 1
	n2 = r - q
	l = arr[p:q+1]
	l.append(INF)
	r = arr[q+1:r+1]
	r.append(INF)
	i = j = 0
	for k in range(p, r+1):
		if l[i] > r[j]:
			arr[k] = r[j]
			j += 1
		else:
			arr[k] = l[i]
			i += 1
```

>原本的代码需要判断 i 和 j 是否超过了边界，现在用哨兵 INF 就能解决这个问题。


$$
T(n) =
\begin{cases}
\Theta(1), & n = 1 \\
2T\!\left(\frac{n}{2}\right) + \Theta(n), & n > 1
\end{cases}
$$

每个规模为 n 的问题可以被分解为两个规模为 $\frac{n}{2}$ 的子问题，加上合并的时间复杂度 $\Theta(n)$ ，就可以得到归并排序的时间递推式。具体 T(n) 怎么推出时间复杂度为 $O(nlogn)$ 在后面说。

### 课后题

![image.png](http://img.xilyfe.top/img/20260114125934526.png)


```text
31 41 59 26 41 58
31 41 59 26 41 58
31 41 58 26 41 58
26 31 41 58 41 58
26 31 41 41 58 58
26 31 41 41 58 58
```

---

![image.png](http://img.xilyfe.top/img/20260114125919472.png)

```text
def linear_find(A, v):
	for i, x in enumerate(A):
		if v == x: return i
	return -1
```

循环不变式：每次循环 i 前，数组 `A[0, .. i-1]` 不含元素 v
1. 初始：第一次循环，循环不变式为空，肯定不含 v
2. 维护：之后每一次循环，如果新元素等于 v 就返回退出了；如果不等于 v 才加入循环不变式，所以  `A[0, .. k-1]` 肯定不含元素 v
3. 结束：当 `i=n+1` 时候循环结束，此时循环内 n 个元素都不为 v，满足循环不变式。如果含元素 v 则在结束前退出。

---

![image.png](http://img.xilyfe.top/img/20260114125857624.png)

```python
def SelectSort(arr):
	n = len(arr)
	for i in range(0, n):
		min_val = INF
		min_idx = -1
		for i in range(i+1, n):
			if arr[i] < min_val:
				min_val = arr[i]
				min_idx = i
		swap(arr[i], arr[idx])
```

- 最坏情况 $O(n^2)$
- 最好情况 $O(n)$

循环不变式：每次循环 i 前，数组 `A[0, ..., i-1]` 递增
1. 初始：第一次循环前，循环不变式为空，所以满足条件
2. 维护：每一次循环 k 之后，会从数组 `A[k-1:n]` 中挑选最小的元素，也就是数组第 k 小的元素插入，保证了依旧是单调递增
3. 结束：当 `i=n+1` 时候循环结束，数组内元素依次为第一小，第二小，...，依次递增，满足循环不变式

---

![image.png](http://img.xilyfe.top/img/20260114130005755.png)

```python
def Merge(arr, p, q, r): 
	n1 = q - p + 1 
	n2 = r - q 
	l = arr[p:q+1]
	r = arr[q+1:r+1]
	i = j = 0
	k = p
	while i < n1 and j < n2:
		if l[i] > r[j]: 
			arr[k] = r[j] 
			j += 1 
		else: 
			arr[k] = l[i] 
			i += 1
		k += 1
	while i < n1:
		arr[k] = l[i]
		k += 1
		i += 1
	while j < n2:
		arr[k] = r[j]
		k += 1
		j += 1
```

---

![image.png](http://img.xilyfe.top/img/20260114130019488.png)


不行，因为查找的时间降下来了，但是还有移动元素的时间。

## 第三章 函数增长率

### 不同渐进符号

| 记号       | 含义   | 理解  |
| -------- | ---- | --- |
| $\theta$ | 紧确界  | =   |
| O        | 上界   | <=  |
| o        | 非紧上界 | <   |
| $\Omega$ | 下界   | >=  |
| $\omega$ | 非紧下界 | >   |
定义 $f(n) = \Theta(g(n))$，存在常数 c₁, c₂ > 0 和 n₀，使得对所有 n ≥ n₀：

$$
0 \le c_1 g(n) \le f(n) \le c_2 g(n)
$$


![](http://img.xilyfe.top/img/20260108151837332.png)

### 和式界的证明方法


#### 等差和
$$
\sum_{i=1}^n i = \frac{n(n+1)}{2} = \Theta(n^2)
$$
#### 等比和

$$
\sum_{i=0}^{n} 2^i = 2^{n+1} - 1 = \Theta(2^n)
$$
#### 调和级数

$$
\sum_{i=1}^n \frac{1}{i} = \Theta(\log n)
$$
#### 放缩法

$$
\sum_{i=1}^n \log{i} < \sum_{i=1}^n \log{n} = n\log{n}
$$

$$
\sum_{i=1}^n \log{i} > \sum_{i=\frac{n}{2}}^n \log{i} > \frac{n}{2} log{\frac{n}{2}}
$$

所以紧确界是 $\Theta(n\log n)$ 。

### 课后题

![image.png](http://img.xilyfe.top/img/20260114134530306.png)

根据多项式展开定理：
$$
(n + a)^b = C_0^b n^b a^0 + C_1^b n^{b - 1} a^1 + \cdots + C_b^b n^0 a^b
$$
经过放缩：
$$
a_0 x^0 + a_1 x^1 + \cdots + a_n x^n \le (a_0 + a_1 + \cdots + a_n) x^n
$$
所以：
$$
C_0^b n^b \le C_0^b n^b a^0 + C_1^b n^{b - 1} a^1 + \cdots + C_b^b n^0 a^b \le (C_0^b + C_1^b + \cdots + C_b^b) n^b = 2^b n^b
$$


---

![image.png](http://img.xilyfe.top/img/20260114134546363.png)

$$
2^{n+1}  =2*2^n \le c*2^n(c \ge 2)
$$

对于等式 2，找不到一个 c 可以使 $2^{2n} \le c*2^n$ 所以不成立。

---

>3.2-3 和 3.2-5 太偏数学了肯定不考。

## 第四章 递归关系式

### 代入法

对于之前归并排序的例子，我们假设它的解是 $T \left(n\right) = O \left( n lgn \right)$ ，即我们需要证明 $T\left( n \right) \le c_1nlgn$。

根据数学归纳法，首先我们要确定当 n 比较小时该猜测成立。当 $n=1$ 时，$T\left(1\right) = \Theta \left(1 \right) = d_1$  ，其中 $d_1$ 是某个大于 0 的常数。根据猜测，我们希望 $T\left(1\right) \le c_1 lg1 = 0$ ，可是，无论怎样取 c ，该式都不可能成立，因为 $T\left(1\right) = d_1$ 必然大于 0。数学归纳法还没开始就失败了。不过不必担心，这只是一个 $lg1 = 0$ 导致的特殊情况，我们完全可以把数学归纳法的初始状态放在 $n \gt 1$ 的位置，同时不影响数学归纳法的结果。因为我们只关心当 n 足够大时 $T\left(n\right)$  的渐进性质，而不关心初始阶段。

现在令 n=2 ，则 $T\left(2\right) = 2T\left(1\right) + d_2 = 2d_1 + d_2$ ，我们希望 $T\left( 2 \right) \le c_12lg2$ 成立，化简后得 $c_1 \ge d_1 + d_2/2$ ，由于 $d_1$ 和 $d_2$ 是常数，因此这样的 $c_1$ 是存在的，初始情况成立。

接下来，数学归纳法需要假设解在 $n/2$ 处成立，即 $T\left( \frac{n}{2} \right) \le c_1 \frac{n}{2} lg\frac{n}{2} = \frac{1}{2}c_1n\left(lgn - 1\right)$ 成立。然后我们来证明 $T\left( n \right) \le c_1nlgn$ 也成立。

$$
\begin{equation} 
\begin{aligned} 
T\left(n\right) & = 2 T\left(\frac{n}{2} \right) + \Theta\left(n\right) \\ 
& \le 2  \frac{1}{2}c_1n\left(lgn - 1\right) + d_2n \\ 
& = c_1nlgn + \left(d_2 - c_1\right) n \\ 
& \le c_1nlgn 
\end{aligned} 
\end{equation} 
$$


同理只需要证明 $T \left(n\right) = \Omega \left( n lgn \right)$，就可以得证 $T \left(n\right) = \Theta \left( n lgn \right)$

### 递归树

步骤：
1. 把递归式中分成递归部分 $T(\frac{n}{2})$ $2T(\frac{n}{4})$ 和每层的处理时间 $O(n)$
2. 递归树每一层都按照递归部分的规则，把上一层瓜分；第一层瓜分的是 $O(n)$
3. 统计每一层的总耗时，最后累加
4. 通过放缩得到最后时间复杂度

![1901d6398175b71ed698c8f0bed61cae.jpg](http://img.xilyfe.top/img/20260114141442209.jpg)


### 主定理

![](http://img.xilyfe.top/img/20260108160544005.png)

简单的理解： 
1. 若 f(n) 增长较慢（情况 1），则递归部分占主导。 
2. 若 f(n) 和递归增长相等（情况 2），需额外乘以 log n。 
3. 若 f(n) 增长较快且满足平衡条件，则非递归部分占主导。

>[!Warning]
>主定理在某些情况下不适用：
>1. 假如 $f(n) = n · (1 + sin(log n))$ 不是渐进平滑的，那么不能用情况 2。情况 2 隐含了：在递归树的每一层，总代价大致是“同一个量级”，然后一层一层累加出一个 log n 因子。而这个 $f(n)$ 根据不同的 n $sin(log n)$ 取值不同，所以不行。
>2. 对于 $Tn=2T(\frac{n}{2}))+nlgn$ ，由于 $n\lg n$ 不是多项式大于 n（主定理真正比较的是 $f(n)$ 和 $n^{1+ε}$ 的大小关系），所以不能套。

### 最大子数组

>最大子数组的三种情况：1 在左半边，2 在右半边，3 横跨（从 mid 向两边扩展即可，线性的）

```python
def FindMaxSubarray(arr, l, r):
	if l == r:
		return l, r, arr[l]
	else:
		mid = (l + r) / 2
		left_l, left_r, left_sum = FindMaxSubarray(arr, l, mid)
		right_l, right_r, right_sum = FindMaxSubarray(arr, mid+1, r)
		mid_l, mid_r, mid_sum = FindMaxCross(arr, l, mid, r)
		return ...

def FindMaxCross(arr, l, mid, r):
	left_max = -INF
	right_max = -INF
	sum = 0
	for i in range(mid, l, -1):
		sum += arr[i]
		if sum > left_max:
			left_max = sum
			max_left = i
	sum = 0
	for i in range(mid+1, r):
		sum += arr[i]
		if sum > right_sum:
			right_sum = sum
			max_right = i
	return max_left, max_right, left_max+right_max
```

`FindMaxCross` 时间复杂度是 $O(n)$，每次都分成两个子任务，所以递推式微 $T(n)=2T(n/2)+ \Theta(n)$，时间复杂度为 $O(n\lg n)$。

### Stranssen 矩阵乘法

给定两个 $n \times n$ 矩阵 A,B，计算：

$$
A \times B,\quad C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
$$
- 三重循环
- 每个 $C_{ij}$ 需要 n 次乘法
- 一共 $n^2$ 个元素

假设 n 是 2 的幂（不是也能 padding）。

把矩阵分成四个 $\frac{n}{2} \times \frac{n}{2}$ 子矩阵：

$$
\begin{array}{c} A = \begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix}, \quad
B = \begin{bmatrix}
B_{11} & B_{12} \\
B_{21} & B_{22}
\end{bmatrix} \end{array}
$$


普通分治乘法：

$$
\begin{aligned} C_{11} &= A_{11}B_{11} + A_{12}B_{21} \\\\ C_{12} &= A_{11}B_{12} + A_{12}B_{22} \\\\ C_{21} &= A_{21}B_{11} + A_{22}B_{21} \\\\ C_{22} &= A_{21}B_{12} + A_{22}B_{22} \end{aligned}
$$

递推式是 $T(n)=8T(n/2)+Θ(n2)$  还是没有改进，Stranssen 的思路是：**矩阵加减法是 $O(n^2)$，比乘法便宜，用加法代替乘法。**

$$
\begin{align*}
M_1 &= (A_{11} + A_{22})(B_{11} + B_{22}) \\
M_2 &= (A_{21} + A_{22}) B_{11} \\
M_3 &= A_{11} (B_{12} - B_{22}) \\
M_4 &= A_{22} (B_{21} - B_{11}) \\
M_5 &= (A_{11} + A_{12}) B_{22} \\
M_6 &= (A_{12} - A_{22}) (B_{21} + B_{22}) \\
M_7 &= (A_{11} - A_{21}) (B_{11} + B_{12})
\end{align*}
$$

就能得到：

$$
\begin{align*}
C_{11} &= M_1 + M_4 - M_5 + M_7 \\
C_{12} &= M_3 + M_5 \\
C_{21} &= M_2 + M_4 \\
C_{22} &= M_1 - M_2 + M_3 + M_6
\end{align*}
$$

所以只需要 7 次矩阵乘法加上若干次矩阵加减法，$T(n) = 7T(n/2) + \Theta(n^2)$，根据主定理递推部分时间复杂度比较大，所以为 $\Theta(n^{log_2^7})$。

### 课后题

![image.png](http://img.xilyfe.top/img/20260114144017454.png)

出现 n 不是 2 的幂的情况，只需要填充 0 即可。Strassen 算法会把规模为 n 的矩阵分裂为 4 个 规模为 n/2 的子矩阵，进行 7 次乘法和若干次加法，所以递推式为 $T(n)=7T(\frac{n}{2})+O(n)$，根据主定理，时间复杂度为 $O(n^{\lg7})$。

---

![image.png](http://img.xilyfe.top/img/20260114144034710.png)

既然 Strassen 算法把规模为 n 的矩阵分裂为 4 个 规模为 n/2 的子矩阵，进行 7 次乘法和若干次加法，递推式写为 $T(n)=7T(\frac{n}{2})+O(n)$，那可以把第一种方法写为 $T(n)=143640T(\frac{n}{70})$，根据主定理得到 $n^{log_{70}^{143640}} \approx n^{2.7951284873613815}$。其他方法同理，和 $\log_2^7$ 对比即可。

---

![image.png](http://img.xilyfe.top/img/20260114144145418.png)

猜测：T(n) >= c · n · lgn

证明：n = 1 为递归式基本情况，T(1) = 2T(⌊n/2⌋) + n = 1 。当 n >= 2 时，有 T(n) = 2T(⌊n/2⌋) + n >= 2 · c · ( ⌊n/2⌋ · lg⌊n/2⌋ ) + n >= ？，我的思路到这就卡着了，因为向下取整的符号会让左边 <= 右边，让接下去的推导无法成立，。如果这里是向上取整的符号的话，就可以推出 2 · c · ( ⌈n/2⌉ · lg⌈n/2⌉ ) + n >= c · n · (lgn - 1)+ n =  c · n · lgn 。

看来是要换猜想了。加上些常数如何？参考了别人的解法，新猜测：$T(n) \ge c(n+2)\lg(n+2)$

证明：还是一样，n = 1 为递归式的基本情况，T(1) = 2T(⌊n/2⌋) + n = 1 。当 n >= 2 时，T(n) = 2T(⌊n/2⌋) + n >= 2c · ((⌊n/2⌋ + 2) · (lg⌊n/2⌋ + 2) + n >= 2c · ((n/2 - 1 + 2) · (lg(n/2)  - 1 + 2) + n = 2c · ((n/2 + 1) · lgn) + n = c · (n + 2) · lgn + n >= c · n · lgn ，其中存在 c > 0，使得最后一步推导成立，所以 T(n) = Ω( n·lgn )。

---

![image.png](http://img.xilyfe.top/img/20260114144156639.png)

证法类似 4.3-3，如果发现上界/下界整不出来，可能是猜测太紧了，可以加减常数。 

---

![image.png](http://img.xilyfe.top/img/20260114144216608.png)

![image.png](http://img.xilyfe.top/img/20260114152905064.png)

---

![image.png](http://img.xilyfe.top/img/20260114144229855.png)

![28073924becb514232245d1975669aca.jpg](http://img.xilyfe.top/img/20260114153432116.jpg)


---

![image.png](http://img.xilyfe.top/img/20260114144300387.png)

不行，$n^{log_b^a}=n^2$ 但是 $n^2\lg n$ 并不是多项式大于 $n^2$ 。
## 第六章 堆排序

### Heapify

堆的维护从上到下，时间复杂度为 $O(\lg n)$ ，就是二叉堆的高。

```python
def MaxHeapify(arr, i):
	l = i.left
	r = i.right
	if l <= size and arr[l] > arr[i]: largest = l
	elif r <= size and arr[r] > arr[i]: largest = r
	else: largest = i
	
	if largest != i:
		swap(arr[i], arr[largest])
		MaxHeapify(arr, largest)
```

非递归的 MaxHeapify 用 **while True** 即可，每次 `i = largest`。

### BuildHeap

由于数组 \[n/2, ..., n] 都是叶节点，叶节点可以看作一个元素的堆，所以建堆时候，只需要维护 \[1, ..., n/2] 的非叶节点就好。

```python
def BuildHeap(arr):
	for i in range(n/2, 1, -1):
		MaxHeapify(arr, i)
```

每次调用 `MaxHeapify` 时间复杂度是 $O(\lg n)$，需要调用 $O(n)$ 次，所以时间复杂度为 $O(n\lg n)$。

### HeapSort

不断把堆顶（最大元素）放到队尾，并且堆大小减一。
```python
def HeapSort(arr):
	BuildHeap(arr)
	for i in range(arr.size):
		swap(arr[1], arr[-1])
		arr.size -= 1
		MaxHeapify(arr, 1)
```
### 优先队列

假设你有一组数据，每个元素都有一个**优先级**，你需要反复做两件事：

1. 插入新元素
2. 快速找到并删除当前优先级最高（或最低）的元素

|数据结构|插入|找最大/最小|删除最大/最小|
|---|---|---|---|
|普通数组|O(1)|O(n)|O(n)|
|排序数组|O(n)|O(1)|O(1)|
|链表|O(1)|O(n)|O(n)|
|**优先队列（堆）**|**O(log n)**|**O(1)**|**O(log n)**|

因为堆的插入只需要对插入位置向上进行维护，树高为 $O(\log n)$；取最大最小元素只需要取堆顶元素就好。删除的话需要堆顶和队尾交换，然后自上而下维护，也是 $O(\log n)$ 。

```python
def Insert(heap, x):
	heap.size += 1
	heap[heap.size] = 0
	Increase(heap, heap.size, x)

def Maximum(heap):
	return heap[1]

def ExtractMax(heap):
	swap(heap[1], heap[-1])
	heap.size -= 1
	Heapify(heap, 1)
	return heap[-1]

def Increase(heap, i, x):
	heap[i] += x
	while i > 1 and heap[i // 2] < heap[i]:
		swap(heap[i//2], heap[i])
		i = i // 2
```

### 课后题

![image.png](http://img.xilyfe.top/img/20260114153606641.png)

非递归的 MaxHeapify 用 **while True** 即可，每次 `i = largest`，如果发现没有交换，那么退出循环。

```python
def MaxHeapify(heap, i):
	while True:
		l = i.left
		r = i.right
		if l <= size and heap[l] > heap[i]: largest = l
		elif r <= size and heap[r] > heap[i]: largest = r
		else: largest = i
		
		if largest == i:
			return
		swap(arr[i], arr[largest])
		i = largest
```

---

![image.png](http://img.xilyfe.top/img/20260114153623254.png)

![image.png](http://img.xilyfe.top/img/20260114182817887.png)

---

> 6.3-3 偏数学不看了

---

![image.png](http://img.xilyfe.top/img/20260114153704774.png)

最坏情况下，从 $\lfloor \frac{n}{2} \rfloor$ 到 1 的每个非叶子结点进行 Heapify 都需要进行到根节点，也就是 $\sum_{k=1}^{\lfloor \frac{n}{2} \rfloor}{h(k)}=\Omega(n\lg n)$ ，这里 $h(k)$ 是节点 k 的高度。

---

![image.png](http://img.xilyfe.top/img/20260114153721993.png)

![image.png](http://img.xilyfe.top/img/20260114183643047.png)

---

![image.png](http://img.xilyfe.top/img/20260114153737799.png)

假设是最小堆
```python
def HeapDelete(heap, i):
	heap[i] = INF
	MinHeapify(heap, i)
	heap.size -= 1
```

## 第七章 快速排序

### 代码

```python
def QuickSort(arr, l, r):
	if l < r:
		mid = Partition(arr, l, r)
		QuickSort(arr, l, mid-1)
		QuickSort(arr, mid+1, r)
def Partition(arr, l, r):
	pivot = arr[r]
	left = l - 1
	for i in range(l, r+1):
		if arr[i] <= pivot:
			left += 1
			swap(arr[left], arr[i])
	swap(arr[left+1], arr[r])
	return left + 1
```

### 性能

快排的最坏情况发生在，pivot划分两个区域，一边元素为0，一边为 n-1。这时候快排的递推式为：$T(n)=T(n-1)+T(0)+O(n)$ 最终得到 $O(n^2)$。最好情况就是 pivot 均分，这时候 $T(n)=2T(n/2)+O(n)$ 得到 $O(n\log n)$。

快速排序的平均情况更接近 **最好情况**，因为任何一种常数比例的划分都会生成高度为 $\Theta(\lg n)$ 的递归树，每一层的时间代价都是 $\Theta(n)$，所以运行时间是 $\Theta(n\lg n)$。

{{< admonition tip "为什么快速排序比归并排序更好" true >}}
尽管归并排序在理论上具有稳定的 $O(n \log n)$ 时间复杂度，并且是稳定排序，但**快速排序在多数场景下仍然被更广泛使用**
1. 空间开销显著更小：归并排序在数组实现下需要额外的 O(n) 辅助空间，而快排是原地排序
2. 常数因子更小，实际运行更快：归并排序在“合并”阶段需要频繁地进行数组拷贝与写入辅助数组，常数开销明显更大
3. 缓存局部性更好：快速排序在划分过程中，主要在当前子数组上进行顺序访问和局部交换，具有良好的空间局部性，能充分利用 CPU cache。归并排序在合并阶段需要在多个数组之间来回读写，缓存命中率较低。
 {{< /admonition >}}

### 随机性

普通 quicksort 的性能**强烈依赖 pivot 的选择**：如果 pivot 每次都选到最小 / 最大 ，会导致划分极不平衡， 递归深度 n，最终时间复杂度为 $O(n^2)$。随机快排就是 pivot 随机选择一个元素，而不是固定头尾或者中间。

### 课后题

![image.png](http://img.xilyfe.top/img/20260114184433172.png)

所有元素都相同时，返回的 q 是选择的 pivot 的下标，所以只有选择中间的元素当 pivot 就行了。

---

![image.png](http://img.xilyfe.top/img/20260114184456428.png)

$$
\begin{aligned}
            & n\lg n \ge nk + n\lg{n / k} \\
\Rightarrow & \lg n \ge k + \lg n - \lg k \\
\Rightarrow & \lg k \ge k.
\end{aligned}
$$

由于无法实现，所以加了常数：
$$
\begin{aligned}
            & c_qn\lg n \ge c_ink + c_qn\lg(n / k) \\
\Rightarrow & c_q\lg n \ge c_ik + c_q\lg n - c_q\lg k \\
\Rightarrow & \lg k \ge \frac{c_i}{c_q}k.
\end{aligned}
$$

## 第八章 线性时间排序

### 基于比较的排序算法下界

最坏情况下，任何比较排序都需要进行 $\Omega(n\lg n)$ 次比较。

![](http://img.xilyfe.top/img/20260109144011628.png)


>没看懂，记住结论

### 计数排序

```python
def CountingSort(arr, n):
	b, c = [0] * (n+1), [0] * (n+1)
	for i in range(n):
		b[arr[i]] += 1
	for i in range(1, n+1): # 因为是 0-n 一共 n+1 个元素
		b[i] += b[i-1]
	for i in arr[::-1]:
		c[b[i]] = i
		b[i] -= 1
	return c
```

> 这个是简易版本，还需要用 max 和 min 计算偏移，这样可以减少空间浪费。

计数排序最简单的思路就是，假如有数组 0,1,2,4,2，那么记录元素 0-4 分别出现 1,1,2,0,1 次，只需要按 0-4 的顺序，输出对应次数的元素即可。但是存在一个问题，就是这个排序是不稳定的。现在需要记录累计次数，1,2,4,4,5 ，然后按照逆向遍历原数组即可。例如，首先遍历到元素 2，找到它的前缀数组值为 4，所以 4 的位置放 2，前缀值减一。那么下一次再遍历到 2 的时候，它就会根据前缀值 3 放在 3 的位置。

时间复杂度是 $O(n+k)$，n 是元素个数，k 是区间范围。

**缺点**：
- 只适用于整数（或可映射到整数）。
- 范围 k 太大时空间浪费严重。
- 不适合稀疏数据。
### 基数排序

假设有 n 个元素，每个元素为 d 位

```python
def RadixSort(arr, d):
	for i in range(d):
		SortOnDigit(arr, i)
```

对于每一次循环，假如对第 i 位采用 $O(n+k)$ 的稳定排序（每个元素有 k 种取值），例如计数排序，那么总耗时 $\Theta(d(n+k))$。

>记住从低位到高位排序

### 桶排序

![](http://img.xilyfe.top/img/20260109152902759.png)

每个桶内再用其他的排序算法进行排序（比如快排），这样子时间复杂度不还是 $O(n\log n)$ 吗？
如果要排序的数据有 n 个，我们把它们分在 m 个桶中，这样每个桶里的数据就是 $k=\frac{n}{m}$。每个桶内排序的时间复杂度就为 $O(k\log k)$。m 个桶就是 $m * O((n / m)*log(n / m))=O(nlog(n / m))$。当桶的个数 m 接近数据个数 n 时，$log(n/m)$就是一个较小的常数，所以时间复杂度接近O(n)。

### 课后题

![image.png](http://img.xilyfe.top/img/20260114191811512.png)

前缀和，得到累积数组之后，用  `c[b] - c[a]` 就可以了。

---

![image.png](http://img.xilyfe.top/img/20260114191829974.png)

当桶排序所有元素都被集中在一个桶的时候，桶排序就会退化为快速排序，如果出现快排的最坏情况：每次选择的 pivot 都是最大最小值，就会出现 $O(n^2)$ 的时间复杂度。可以通过归并排序或者桶排序，使得最坏情况下也是 $O(n\lg n)$。

## 第九章 中位数和顺序统计

### 最值

- 单独获得最大值或者最小值，最少需要 n-1 次比较
- 同时获得最大和最小值不需要 2(n-1) 次比较，只需 $3*\lfloor\frac{n}{2}\rfloor$  ，方法就是同时取两个输入，然后将他们先进行一次毕竟，然后拿较小的元素和最小值比较，较大的元素和最大值进行比较，只需要三次比较。

### 期望时间为线性的选择算法

选择算法：给定无序数组 A 和整数 k（1 ≤ k ≤ n），找出数组中第 k 小的元素（即排序后位于位置 k 的元素）。

思想类似快排，

- 随机选择一个主元 pivot，对数组进行分区（partition）：左边 < pivot，右边 > pivot，pivot 自己放中间。
- 分区后，pivot 落在最终位置 rank（假设 rank 是 pivot 的排名，从1开始）。
- 如果 rank == k，直接返回 pivot。
- 如果 k < rank，递归在左子数组找第 k 小。
- 如果 k > rank，递归在右子数组找第 (k - rank) 小。

```python
def SelectK(arr, l, r, k):
	if l == r: return arr[l]
	
	pivot_idx = random.randint(l, r)
	swap(arr[pivot_idx], arr[r])
	
	i = l - 1
	for j in range(l, r+1):
		if arr[j] <= arr[r]:
			i += 1
			swap(arr[i], arr[j])
		swap(arr[i], arr[r])
	i += 1
	rank = i - l + 1
	
	if rank == k:
		return arr[]
	elif rank > k:
		return SelectK(arr, l, rank-1, k)
	else:
		return SelectK(arr, rank+1, r, k - rank)
```

### 最坏时间为线性的选择算法及其时间分析

Bfprt 算法的思路是选择一个合适的 pivot 使得最坏情况下，时间复杂度还是线性的。选择合适 pivot 的方法是，将数据分为多个区间，递归调用 Bfprt 找到每个区间中位数的中位数。

```python
def bfprt(arr, low, high, k):
    n = high - low + 1
    if n <= 5:
        # 排序子数组并返回第 k 小
        sub = sorted(arr[low:high+1])
        return sub[k-1]
    
    # 步骤2-3: 分组找每组中位数
    medians = []
    for i in range(low, high+1, 5):
        group = arr[i:min(i+5, high+1)]
        group.sort()
        medians.append(group[len(group)//2])
    
    # 步骤4: 递归找中位数的中位数 mom
    mom = bfprt(medians, 0, len(medians)-1, len(medians)//2 + 1)
    
    # 步骤5: 用 mom 分区（类似快速排序 partition）
    # ...（交换 mom 到末尾，partition，返回 rank）
    
    # 步骤6: 递归判断
    if k == rank:
        return mom
    elif k < rank:
        return bfprt(arr, low, pivot_pos-1, k)
    else:
        return bfprt(arr, pivot_pos+1, high, k - rank)
```

### 课后题

![image.png](http://img.xilyfe.top/img/20260114193806889.png)

采用锦标赛制，对元素进行两两比较，最终经过 $\lceil \log_2 n \rceil$ 轮 n-1 次比较得到最小元素。这时候至少有 $\lceil \log_2 n \rceil$ 个元素直接输给过最小元素，第二小元素只有可能在这里面。这时候比较 $\lceil \log_2 n \rceil - 1$ 次就能得到里面最小元素，也就是第二小元素。

---

![image.png](http://img.xilyfe.top/img/20260114193936716.png)

>k分位数是大小为n的集合（比如数组）里面的k-1个数，它们把有序的集合分为k个分组，任何两个个分组之间的大小之差的绝对值不超过1（有点类似于平衡二叉树），比如集合{3， 5， 9， 4， 2， 1， 6， 8， 9， 10， 12， 7， 6}，排序后为{1， 2， 3， 4， 5， 6， 6， 7， 8， 9， 9， 10， 12}，它的4（k = 4）分位数为{4， 6， 9}， 分组后的子集合分别为{1， 2， 3， 4}， {5， 6， 6}， {7， 8， 9}， {9， 10， 12}。要求从集合中找出这k-1个数，并且时间复杂度为O(nlgk)。

思路：如果对这 k-1 个数分别使用 Order Statistics 算法，第一次找出第4小的数，第二次找出第7小的数，第三次找出第10小的数，虽然每次的时间复杂度为 O(n)，但 k-1 次则为$O(nk)$，不是 $O(nlgk)$。所以可以采用分治的思路。

1. 假如需要找到 k 分位数，k=4
2. 那么就先减半找到 k=2 分位数，这时候时间复杂度 $O(n)$
3. 然后我们递归的处理左边部分和右边的分位数

## 第十三章 红黑树

二叉搜索树没有控制树高，红黑树在二叉搜索树基础上，通过在节点数增加颜色，控制没有一条路径会比其他路径长出 2 倍，使得近乎平衡。

- 根叶黑
- 不红红
- 黑路同

- 有 n 个节点的红黑树，高度至多为 $2lg(n+1)$
- 一棵黑高为 K 的红黑树中，结点最多为 $2^{2k+1}-1$（红黑交替），最少 $2^{k+1}-1$ （全黑）

>不考画图，随便看看

## 第十四章 数据结构的扩张

### 顺序查找树


![](http://img.xilyfe.top/img/20260109181752650.png)

前面学的 Order-Statistics 算法可以在 $O(n)$ 的时间内找到第 i 个元素，但是如果需要进行多次查找操作，时间复杂度还是蛮高的。顺序查找树能做到一次预处理之后，每次查找的时间复杂度为 $O(\lg n)$。

它在红黑树基础上，每个节点维护了一个 size 属性，标志了以自己为根的子树个数，它的功能包括：
- 根据元素 x 获得它的 rank
- 根据 rank 获取 x

```python
def Rank2Ele(t, r):
	rank = t.left.size + 1
	if rank == r: return
	elif rank > r: return Rank2Ele(t.left, r)
	else: return Rank2Ele(t.right, r - rank)
	
def Ele2Rank(t, x):
	if t == x: return t.left.size + 1
	elif t > x: return Ele2Rank(t.left, x)
	else: return t.left.size + 1 + Ele2Rank(t.right, x)
```

![](http://img.xilyfe.top/img/20260109183127292.png)

简单来说：**如果扩张属性只影响到父节点或只影响到子节点，就可以在红黑树上扩张**。

> 能不能拓展深度属性呢？
> 不行，假如根节点删除了，那么下面所有子节点都需要修改深度--，影响的是 $O(n)$。
### 区间树

- 区间树以红黑树为基础
- 区间树的节点关键词存储的是一个区间 `i.left, i.right`
- 区间树节点附加信息是 `max`，代表节点所在的所有子树，最大的右端点

由于 `x.max = max(x.left.max, x.int.right, x.right.max)` ，根据定理可以在红黑树上扩展。

```python
def IntervalSearch(T, i):
	res = []
	x = T.root
	if x != T.nil and overlap(x.i, i):
		res.append(x)
	if x.left != T.nil and x.left.max > i.left:
		IntervalSearch(x.left, i)
	elif x.right != T.nil:
		IntervalSearch(x.right, i)
```

>[!Note]
>区间树的左节点区间的左端点一定比父节点的左端点小，但是右端点不一定，并不是说左端点整个区间都在父节点区间的左侧。

### 课后题

![image.png](http://img.xilyfe.top/img/20260115113957870.png)

```python
def i_after_x(T, x, i):
	idx = Ele2Rank(T, x)
	return Rank2Ele(T, i + idx)
```

根据元素查 rank 和你操作的时间复杂度都是 $O(\lg n)$，所以还是 $O(\lg n)$。

---

![image.png](http://img.xilyfe.top/img/20260115114022269.png)

可以，因为当一个节点的颜色反转时候，它只会影响父节点的黑高，例如红变黑，那么父节点黑高++，只会在这颗子树向上或者向下影响，最多 $O(\lg n)$。

---

![image.png](http://img.xilyfe.top/img/20260115114042145.png)

```python
def min_overlap(T, i):
	res = T.nil
	rec = INF
	x = T.root
	while x != T.nil:
		if overlaps(x.int, i) and x.int.left < rec:
			rec = x.int.left
			res = x
		if x.left != T.nil and x.left.max >= i.right:
			x = x.left
		else:
			x = x.right
	return res
```

## 第十五章 动态规划

### 思想&步骤

- 解决的是寻找问题的**一个最优解**
- 具备的两个要素：**最优子结构和子问题重叠**
	- 问题的最优解由相关子问题的最优解组合而成
	- 子问题重叠：例如斐波那契数列，重复求同一个子问题
- 步骤：
	- 识别最优解的特征
	- 递归的定义最优解的值（就是状态转移方程）
	- 自底向上求解

### 和分治法区别

- 分治法是分解为互不相干的子问题求解之后合并
- 动态规划是重叠的子问题

### 算法设计

**例：给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回-1。你可以认为每种硬币的数量是无限的。**

假设我们用 `Func(coins, amount)` 求解所需的最少的硬币个数，并且 硬币为 1 和 2，那么可以把他分解为，求解 amount-1 所需最少硬币和 amount -2 所需最少硬币的最小值 + 1。写成递推式即为：$T(n)=min(T(n-1)+T(n-2))+1$

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

---

![](http://img.xilyfe.top/img/20260110125444743.png)

这个题思路和凑硬币完全一样，把最小值变成最大值，就行了。

---

![](http://img.xilyfe.top/img/20260110135416958.png)

m\[i]\[j] 依赖于区间更短的子问题，所以必须按**区间长度递增**计算，常见顺序是：

```
for l = 2 to n // 链长度  
 for i = 1 to n−l+1  
  j = i + l − 1  
  计算 m[i][j]
```

这是一个典型的**区间 DP**。

```python
def MatrixMultiply(m):
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

---

**例：给定一个长度为 n 的整数序列 A\[1…n]（可正可负），要求找一个连续子段 A\[i…j]，使其元素和最大。**

```python
ans = -INF
dp = [0]*(n+1)
dp[1] = A[1]
for i in range(2, n+1):
	dp[i] = A[i] if dp[i-1] < 0 else A[i] + dp[i-1]
	ans = max(ans, dp[i])
return ans
```

>[!Note]
>这里我们不应该纠结于，假如后一个数字是负数，后面会不会有更大的正数来弥补。我们需要注意，子串和这个概念代表了“**以位置 i 结尾的最大子段和只和 i−1 有关**”。如 1 2 -5 6 这个例子，假如我们纠结怎么判断 -5 之后 6 加上去子串和更大就会乱，我们应该在 -5 的视角，对于 -5 来说如果前面的子串大于 0，那么对它就是有利的应该加上去。

---

![](http://img.xilyfe.top/img/20260110144927916.png)

>注意子序列和子串区别

```python
for i in range(1, m+1):
	for j in range(1, n+1):
		if x[i] == y[j]:
			dp[i][j] = dp[i-1][j-1] + 1
		else:
			dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

---

```text
找出两个长度分别为m和n字符串序列的最长公共字串（字串为下标连续的子 序列），试： 
（1） 先给出朴素算法的算法思想、伪代码及计算时间复杂度 
（2） 再给出算法改进思路或一个更有效的算法
```

这个问题可以结合 **最长公共子序列** 和 **最长连续子串** 两个问题。令 `dp[i][j]` 为 以 X\[i] 和 Y\[j] 结尾的最长公共子串长度，若字符相等：`dp[i][j] = dp[i−1][j−1] + 1` ，若字符不等：`dp[i][j] = 0`。

```
LongestCommonSubstring(X, Y):
    n = length(X)
    m = length(Y)
    create array dp[0..n][0..m]
    maxLen = 0
    endPos = 0

    for i = 0 to n:
        dp[i][0] = 0
    for j = 0 to m:
        dp[0][j] = 0

    for i = 1 to n:
        for j = 1 to m:
            if X[i] == Y[j]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > maxLen:
                    maxLen = dp[i][j]
                    endPos = i
            else:
                dp[i][j] = 0

    return maxLen   // 子串为 X[endPos-maxLen+1 .. endPos]
```

---

**例：给你一个整数 `n` ，返回和为 `n` 的完全平方数的最少数量。完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。**

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

### 课后题

![image.png](http://img.xilyfe.top/img/20260115125710532.png)

![image.png](http://img.xilyfe.top/img/20260115132239486.png)

---

![image.png](http://img.xilyfe.top/img/20260115125830364.png)

因为归并排序这种分治算法不会出现重叠子问题，所以不需要备忘技术记录重复结果。

---

![image.png](http://img.xilyfe.top/img/20260115131012283.png)

1 0 0 1 1 0

---

![image.png](http://img.xilyfe.top/img/20260115131511100.png)

- 对于 $2 \times \min(m,n)$ 空间复杂度的算法，由于每次遍历只需要用到 dp 数组的当前行和上一行，所以可以只需要保留这两行
- 对于 $\min(m,n)$ 空间复杂度的算法，我们可以仅仅保留一行 dp 数组，执行到  `dp[i][j] = max(dp[i-1][j], dp[i][j-1])` 时候，由于 `dp[i][j-1]` 还没被 `dp[i][j]` 覆盖可以直接读取；`dp[i][j] = dp[i-1][j-1] + 1` 里面的 `dp[i-1][j-1]` 已经被 `dp[i-1][j]` 覆盖，所以需要单独的变量记录就可以了。

## 第十六章 贪心算法

### 思想&对比

- 核心思想：在构造解的过程中，每一步都做出当前看来最优（局部最优）的选择，并期望通过一系列局部最优选择最终得到全局最优解。
- 和动态规划的区别：dp 只要求有最优子结构，贪心还要求满足**贪心选择性质**：**存在一个最优解，其第一步选择等于贪心算法所做的选择。**

### 哈夫曼树

贪心正确性证明：假设树叶节点频率最小的两个节点 a，b 的频率分别是 fa 和 fb，全部元素中，频率最小的结点为 x 和 y，频率为 fx 和 fy。那么用 xy 替换 ab，他们仍然在最深处，并且加权路径长度只可能小等于 原先的 ab，所以得到了另一课最优的编码树。

### 算法设计

**例：给定 n 个活动， 活动 i 的开始时间为 $s_i$，结束时间为 $f_i$，选择尽可能多的互不冲突的活动。**

1. 按活动结束时间递增排序；
2. 选择第一个结束最早的活动；
3. 在剩余活动中，选择开始时间 ≥ 当前活动结束时间的、结束最早的活动；
4. 重复直到无法选择。

贪心性质证明：

设 A* 是任意一个最优解（即包含最多活动的集合），其中第一个活动是 b，其结束时间为 f_b。由于 a 是**所有活动中结束时间最早的活动**，因此有：f_a ≤ f_b。

分两种情况讨论：

1. 如果 a = b：那么 A* 本身已经包含贪心选择 a，命题成立。
2. 如果 a ≠ b：那么可以用 a 替换 b，因为 a 比 b 早结束，不会影响后面活动。并且替换后活动数量一致，仍然是最优解。

---

**例：有 n 个物品，物品 i 的价值为 $v_i$，重量为 $w_i$，背包容量为 W，每个物品可以取任意比例（可分割）。**

1. 按照单位价值从高到低排序
2. 依次选择物品加入背包
3. 如果最后一件物品无法全部装入，则计算可以装入的比例，然后按比例装入。

贪心性质证明：设 a 为单位重量价值最大的物品。若某最优解不包含 a，则从中取出重量为 x 的低单位价值物品，用同重量的 a 替换，总价值不减。反复交换可得包含 a 的最优解，故贪心选择性质成立，小数背包贪心算法正确。

### 课后题

![image.png](http://img.xilyfe.top/img/20260115141631359.png)

- 算法思路：按照重量排序，每次选重量最小的
- 贪心性质证明：
	- 设 $S^*$ 是任意一个最优解，$S^*$ 选择的第一个元素下标是 i
	- 假设按照贪心算法选择的第一个元素下标为 0
	- 假如 `i==0`，那么最优解第一个元素就是贪心
	- 假如 `i != 0`，那么最优解的第一个元素肯定比 0 重价值低，那么替换为 0 就可以得到更优解。

---

![image.png](http://img.xilyfe.top/img/20260115141644607.png)

- 算法思路：对点集进行排序，每次都取剩余点集中最靠左的点当新区间的左端点
- 贪心性质证明：
	- 假设：$x_1$​ 是当前最左的点，贪心算法选择的第一个区间为 $I_a = [x_1,\; x_1+1]$
	- 设 $S^*$ 是任意一个最优解，覆盖 $x_1$ 的区间为 $I_b = [l_b,\; l_b+1]$，那么必有 $l_b \le x_1 \le l_b + 1$。
	- 按照思路，贪心区间的右端点为 $r_a = x_1 + 1$，有 $r_b \le r_a$。
	- 假设 $I_a = I_b$ 则最优解 $S^*$ 本身已经包含贪心选择的区间，命题成立。
	- 假设 $I_a \ne I_b$，则新区间 $S' = (S^* \setminus \{I_b\}) \cup \{I_a\}$，这是一个更优解。


---

![image.png](http://img.xilyfe.top/img/20260115141709861.png)


- 斐波那契数列构造的哈夫曼树，所有节点左子树都只有一个节点。
- 最优前缀码为：0,01,001,0001

## 实验二 回溯法

### 解空间树

根据问题的不同，解空间通常表现为以下三种形式之一：

1. 子集空间（子集树）
	- 每个元素只有“选”或“不选”两种状态。
	- 常见于 0-1 背包、子集和问题。    
	- 解空间树通常是一棵二叉树。
2. 排列空间（排列树）
	- 元素的顺序不同即构成不同解。
	- 常见于全排列、TSP 等问题。
	- 树的分支数随层数递减。
3. 组合空间
	- 介于子集和排列之间。
	- 常见于从 n 个元素中选 k 个的组合问题。
### 算法设计
### n 后问题

**问题描述： 在 n×n 的棋盘上放置 n 个皇后，使得任意两个皇后不在同一行、同一列或同一对角线上。**

```python
def n_queen(chess, line):
	if line == n+1 and check(chess):
		print(chess)
	for i in range(n):
		chess[line][i] = True
		# 可以在这里剪枝，用col记录那些列有皇后了
		# if i in col:
		#     continue
		# else:
		#     col[i] = True
		
		n_queen(chess, line+1)
		chess[line][i] = False
		# col[i] = False
```

### 0-1背包

**问题描述： 每个物品只能选或不选，在容量限制下最大化总价值。**

```cpp
void dfs(int i, int cw, int cv) {
    if (i == n) {
        ans = max(ans, cv);
        return;
    }

    // 不选第 i 个
    dfs(i + 1, cw, cv);

    // 选第 i 个
    if (cw + w[i] <= W) {
        dfs(i + 1, cw + w[i], cv + v[i]);
    }
}
```

### TSP 问题

**问题描述： 给定 n 个城市及其距离，寻找一条经过每个城市一次并回到起点的最短回路。**

```python
visited = [0]*(n+1)
def dfs(dis):
	if sum(visited) == n:
		ans = min(ans, dis)
		return
	for i in range(1, n+1):
		if visited[i]: continue
		visited[i] = True
		dfs(dis+cost[i])
		visited[i] = False
```

## 第十七章 平摊分析

### 聚合分析

对于一个栈，假设有 pop, push, multi-pop(n) 三个操作，时间复杂度分别为 O(1), O(1) 和 O(n)。进行 n 次操作，每次操作任选一个，那么时间复杂度是多少？如果按最坏情况，每次操作都是 multi-pop，那么总时间复杂度为 $O(n^2)$ 。但实际情况是，push 一次才能 pop一次，所以 pop 和 multi-pop 的次数和 push 次数相关。经过一系列证明可以得到实际的时间复杂度是 O(n)，所以平均到每个操作，他们的摊还代价是 O(1)。

同样对于一个 n 位二进制串 0000000，每次进行 Increment 操作，第一次只需要改变一个值，第二次需要改变两个 01 -> 10，第四次需要根本三个 011 -> 100，最终需要改变 n 个值。所以进行 n 次操作，最坏情况是 $O(n^2)$。但实际上每 $2^i$ 次操作翻转位数才会加一，所以经过一系列证明，最坏情况的时间是 O(n)，他们的摊还代价是 O(1)。

### 核算法

假设我们实现一个动态数组，支持push_back，每次容量不够时翻倍扩容。

- 大多数插入代价1（只放一个元素）。
- 偶尔扩容时代价≈当前size（要复制）。

用核算法：

- 每个插入操作我们收取摊还费用 â = 2（单位代价）。
- 使用规则：
    - 1单位支付当前插入本身（放新元素）。
    - 1单位存入银行（为这个新元素“预存”未来的复制费用）。

当发生扩容（从k到2k）时：

- 需要复制k个旧元素，每个旧元素在之前插入时已经为它预存了1单位。
- 总共可以从银行取出 k 单位。
- 实际扩容复制代价 ≈ k

容易归纳证明：银行余额始终 ≥ 0（实际上总是正的）。

因此：

- 每个插入的摊还代价 = 2 = O(1)
- n次插入的总摊还费用 = 2n = O(n)
- 总实际代价 ≤ 2n = O(n)（严格上界）

### 势能法

首先，我们定义状态 S 为当前某一数据结构的状态。该状态反应出该数据结构的元素值、元素个数等信息。然后，我们定义一个势能函数 $\Phi(S)$，表示当该数据结构处于状态 S 下的势能。和物理中的定义类似，你需要保证你定义的这个势能涵函数在初始时值为 0，且在算法执行的任意过程中值非负。并且势能的变化量在一定程度上可以反应出该对象的形态改变程度。

对于每个操作，我们定义摊还代价 $c'=c+\Phi(S')一\Phi(S)$。其中，c 表示该操作的实际成本，S和S'表示该操作前后数据结构的状态。直观地，均摊成本等于**实际成本加上势能变化量**。

对于动态数组扩容的例子，我们定义势能函数为 $\Phi(S)=2n-m$，n 是动态数组实际长度，m 是总长度。这么定义是因为能保证大于 0 ，并且插入元素会导致 n 和 m 改变然后势能变化。

|       | 实际成本 | 势能变化量                                   | 摊还代价      |
| ----- | ---- | --------------------------------------- | --------- |
| 不触发扩容 | 1    | $\Phi(S')-\Phi(S)=2(n+1)-m-(2n-m)=2$    | 1+2=3     |
| 触发扩容  | n+1  | $\Phi(S')-\Phi(S)=2(n+1)-2n-(2n-n)=2-n$ | n+1+2-n=3 |

摊还代价都是 3 也就是 O(1) 的时间。

### 课后题

>感觉不会考，全是数学证明

## 第十九章 二项堆

### 为什么需要二项堆

优先队列通常是使用二叉堆这个数据结构来实现，在某些应用中，**合并两个优先队列**是核心需求，而二叉堆的 Union 操作需要把另一个二叉堆的元素逐个插入，所以时间复杂度是 $O(n\lg n)$。二叉堆的目的就是维持优先队列的性质，并且降低合并操作的时间复杂度。

| 操作                | 二叉堆      | 二项堆     |
| ----------------- | -------- | ------- |
| make-heap         | O(1)     | O(1)    |
| insert            | O(logn)  | O(logn) |
| minimum           | O(1)     | O(logn) |
| extract-min       | O(logn)  | O(logn) |
| increase/decrease | O(logn)  | O(logn) |
| union             | O(nlogn) | O(logn) |
可以看到二项堆相对于二叉堆，合并操作的时间复杂度下降了，但是取最值的时间复杂度上升了。

### 定义和存储结构

先说二项树：

![](http://img.xilyfe.top/img/20260111113243544.png)


- 假设树高为 k，有 $2^k$ 个节点
- 第 i 层有 $C_k^i$ 个节点
- 根的度最大且为 $C_k^1=k$ 
- 二项树 $B_k$ 是有两颗 $B_{k-1}$ 合并而成，将一棵 $B_{k-1}$ 的根作为另一棵 $B_{k-1}$ 根的最左孩子。

二项堆是满足下述条件的二项树的集合：

- H 中的每棵二项树满足最小堆性质（根小于任意子节点）
- 对任意的非负整数 k，H 中至**多有一棵二项树根的度为 k**

---

二项堆中的所有二项树的根节点由一个单链表连接，**按照度（或者说二项树高度）增序**。二项树的节点定义如下：

```python
@dataclass
class Node:
	p: Node # 父节点
	key: int # 关键字
	degree: int # 度：子节点个数，就是C_k^i
	child: Node # 左孩子
	sibling: Node # 右兄弟
```

![image.png](http://img.xilyfe.top/img/20260111113807004.png)

先说一下 Minimum 操作，前面说过“二项堆中的所有二项树的根节点由一个单链表连接”，所以需要用一个线性查找，遍历所有根节点。假设二项堆总节点数为 n，最多 ⌊log₂ n⌋ + 1 棵树，所以时间复杂度为 $O(\lg n)$。

>二项堆每棵树都是某个 Bₖ，任意两棵树的 degree 不同，所以 1+2+4+8+...=n。

### 合并操作

- 把两个二项堆链表合成一个链表，仍然按照度排序根节点，复杂度是 $O(\lg n)$（因为根节点最多 ⌊log₂ n⌋ + 1 个）
- 对所有度相同的二项树进行合并，假设待合并的两棵树为 $B_{k-1}$
	- 将关键字较大的根挂到关键字较小的根下（第一个左孩子），得到一个 $B_k$ 的新二项树
	- degree+1，树高也+1
- 每次二项树合并时间复杂度为 O(1)，最坏情况每棵树度都相同，合并 $O(\lg n)$ 次，时间复杂度 $O(\lg n)$。

## 第二十一章 不相交集数据结构

### 概念

不相交集用于维护一组**动态变化的集合划分**，要求这些集合两两不相交。其核心目标不是存储集合中的元素内容，而是**高效维护“元素属于哪个集合”以及“集合是否需要合并”**。

形式化定义如下：给定一个全集 $S = {x₁, x₂, …, xₙ}$，不相交集维护的是 S 的一个划分 ${S₁, S₂, …, Sₖ}$，满足：

- Sᵢ ≠ ∅
- Sᵢ ∩ Sⱼ = ∅（i ≠ j）
- ⋃ Sᵢ = S

支持三种基本操作：

1. MAKE-SET(x)：创建一个新集合，仅包含元素 x。
2. FIND-SET(x)：返回包含 x 的集合的一个代表（representative）。
3. UNION(x, y)：将包含 x 和 y 的两个集合合并为一个集合。

>注意 find-set 返回的不是集合，而是集合的代表
>FIND-SET(x) == FIND-SET(y) 可以判断两个元素是不是在同一个集合

### 实现方法

#### 链表

每个集合用一个链表表示：

- 链表中存储该集合的所有元素：head → a → b → c
- 每个节点存一个指向“集合头节点”的指针：`a.head = b.head = c.head = head`
- 头节点代表该集合（作为 representative）

- make-set(x)：创建一个只含 x 的链表，O(1)
- find-set(x)：返回 x.head，O(1)
- union(x, y)：把 y 去掉 head 连到 x 尾部，O(n)

>FIND 非常快，但合并代价高
#### 森林(就是并查集)

- make-set(x)：`parent[x] = x`，O(1)
- find-set(x)：`while x = parent[x]: return x`，O(logn)，树高
- union(x, y)：`root_x = find_set(x), root_y = find_set(y), parent[root_y]=root_x`，O(logn)，时间复杂度取决于 find-set，不控制会退化为链表

采用路径压缩在find_set时候把路径上所有节点直接指向根，可以让 find-set 和 union 是常数时间。

### 应用

1. kruskal：用并查集判断 u 和 v是否连通，不连通则选择该边
2. 无向图连通分量：对每条边的两个节点 u 和 v 进行 union，最后几个根节点就是几个连通分量

### 课后题

![image.png](http://img.xilyfe.top/img/20260115152527918.png)

先找到根，然后自底向上更新全部 parent 为根。
```python
def find_set(T, x):
	root = x
	while T[root] != root:
		root = T[root]
	while x != root:
		p = T[x]	
		T[x] = root
		x = p
	return root
```

## 第二十二章 图论算法

### DFS 和 BFS

- 白色节点是未访问过的节点
- 灰色节点是已访问，但是还没有搜索周围节点的节点
- 黑色节点表示该结点的所有邻接结点均已被检查完毕

>颜色机制的本质作用是保证 **每个结点最多被发现一次**，用 visited 数组一样效果

|     | 邻接表    | 邻接矩阵   |
| --- | ------ | ------ |
| DFS | O(V+E) | O(V^2) |
| BFS | O(V+E) | O(V^2) |

### 最小生成树

安全边：假设 A 是最小生成树的一个子集，假如把一条边加入 A 后它仍然是最小生成树的子集，那么它就是安全边。

最小生成树的算法就是不断找到一条安全边，把它加入集合中。最终集合包含所有边，那么他就是最小生成树。

- Prim：
	- 加点法：然后 S 是已选择点的集和，那么不断挑选离 S 最近的点加入，连上那条边
	- 数组表示的话O(V^2)，用最小优先队列(二叉堆)是O(ElogV)。二叉堆 Extract-minimum 时间复杂度是 logV，总开销 VlogV。但是每次加点之后，需要更新其他点离 S 的最近距离，也就是二叉堆的 decrease-key 操作，时间复杂度是 O(logV)，最坏情况是 E 次，所以 O(ElogV)。
- Kruskal：
	- 加边法：不断选择权值最小并且不形成连通分量的边
	- 时间复杂度O(ElogV)，首先给边排序的时间复杂度是 $O(E\lg E)=O(E\lg V)$，然后一共 E 次循环，每次循环最多需要 2 次 find 1次 union，如果经过路径压缩，那么摊还代价是 $\alpha(V)$ ，所以总时间复杂度 $O(E*\alpha(V))=O(E)$，排序占主导所以是 O(ElgV) 

### 单源最短路

δ(s, v) 与 d\[v] 是什么？

- δ(s, v)：从源点 s 到 v 的真实最短路径长度（理论值）；
- d\[v]：算法运行过程中维护的上界估计，始终满足 d\[v] ≥ δ(s, v)。

---

边松弛：

若 d\[v] > d\[u] + w(u, v)：
- d\[v] = d\[u] + w(u, v)
- π\[v] = u

---

Bellman-Ford 算法：
1. 设置 dis\[起点]=0，dis\[其他点]=inf
2. 进行 v-1 轮遍历
	1. 遍历每一条边 <u,v >，对 dis\[v] 进行更新，看看能不能 dis\[v] = dis\[u]+w\[u,v] 松弛
3. 最后遍历一轮，如果还有更新，说明不能收敛，有负权环，return False。
4. 时间复杂度 O(EV)
5. 可以用于负权图，不能负权回路

---

Dijkstra 算法：
1. 设置 dis\[起点]=0，其他为inf
2. 更新起点相连节点的dis
3. 按照dis，遍历未访问的所有节点
	1. 将节点设为 visited
	2. 更新这个节点相连节点的dis
4. 和 prim 算法一样，时间开销受限于二叉堆的 decrease-key 操作，时间复杂度是 O(logV)，最坏情况是 E 次，所以 O(ElogV)。
5. 如果用斐波那契堆进行优化（二叉堆优化），那么开销是 O(VlogV+E)。
6. 用于非负权、有向图，**可以有环**


{{< admonition tip "为什么不是$O(V*(\lg V + E))$ 呢？" true >}}
这个式子隐含用了这样一个假设：**每一轮 Extract-Min 之后，最多会发生 EEE 次 Decrease-Key** 这是不成立的。
因为 每条边只在其起点被 Extract-Min 时被扫描一次，所以内层循环 **所有轮次加起来，总共只跑 E 次**。
 {{< /admonition >}}

---

DAG 算法
1. 只能有向无环图 -> 这样才保证存在拓扑排序，拓扑排序保证在处理 u 之前，所有可能到达 u 的前驱顶点 x 的最短路径都已经被正确计算
2. 先进行一次拓扑排序
3. 按拓扑排序扫描顶点，对每个顶点 u：
    - 对每条出边 (u, v) 进行松弛
4. 时间复杂度 O(E+V)，受限于拓扑排序开销


### 多源最短路

floyd 的思路就是依次增加中转顶点，看 i 和 j 之间的距离能不能通过中转顶点缩减。

```python
def floyd():
	for k in range(V):
		for i in range(V):
			for j in range(V):
				new_dis = dis[i][k] + dis[k][j]
				if (new_dis < dis[i][j]):
					dis[i][j] = new_dis
					path[i][j] = path[k][j]
```

>`path[i][j]` 表示：“在从 i 到 j 的最短路径上，j 的**前一个顶点是谁**”
>如果 `d[i][j] > d[i][k] + d[k][j]` ，说明 i → … → k → … → j。所以 j 之前一个顶点应该是 `d[k][j]`
>找 i - j 路径的方法就是，不断递归找 `d[i][j]=k` 然后找 `d[i][k]`

时间开销 $O(V^3)$，适合稠密图。

---

Johnson 算法

- 那么对每个节点进行 Dijsktra，用斐波那契堆开销是 O(VlogV+E)，总时间O(V^2logV+VE)。如果用二叉堆是O(VElogv)
- 但是 Dijsktra 只能用于非负权图，所以需要用 bellman-ford 计算一个势能函数，得到新的权值，代替原先可能为负数的权值
	- 假如一个新节点s，对其他所有节点的dis=0
	- 然后 bellman-ford 得到 s 对其他节点 v 的距离，作为 h(v)
	- 然后每条边 <u,v> 新的权重就是 w\[u,v] +h(u)-h(v)
- 适用于稀疏图

## 第三十一章 数论算法

### 最大公约数

#### 欧几里得算法

##### 代码

```python
def gcd(a, b):
	if b == 0: return a
	else: return gcd(b, a % b)
```

时间复杂度为：**O(log(min(a, b)))** 或 **O(log(max(a, b)))**

##### 证明

我们首先假设有两个数 $a$ 和 $b$，其中 $a$ 是不小于 $b$ 的数，记 $a$ 被 $b$ 除的余数为 $r$，那么 $a$ 可以写成这样的形式：

$$
a = bq + r
$$

其中 $q$ 是整数。现在假设 $a$ 和 $b$ 的一个约数为 $u$，那么 $a$ 和 $b$ 都能被 $u$ 整除，即

$$
\begin{align}
a&=su\\
b&=tu
\end{align}
$$


$s$ 和 $t$ 都是整数。这样可以得出：

$$
r = a - bq = su - (tu)q = (s - tq)u
$$

所以 $r$ 也能被 $u$ 整除，我们能得到一般规律如下：

> $a$ 和 $b$ 的约数也整除它们的余数 $r$，所以 $a$ 和 $b$ 的任一约数同时也是 $b$ 和 $r$ 的约数。 

反过来可以得出：

> $b$ 和 $r$ 的任一约数同时也是 $a$ 和 $b$ 的约数。

因此，我们可以推出：$a$ 和 $b$ 的约数的集合，全等于 $b$ 和 $r$ 的约数的集合，所以 $a$ 和 $b$ 的最大公约数，就是 $b$ 和 $r$ 的最大公约数。

$$
\text{gcd}(a,b) = \text{gcd}(b, r)
$$

根据递推性质，我们可以不断减小 $b$ 使得公式变为 $gcd(x,0)$，结果就是 x。

#### 扩展欧几里得算法

##### 代码

```python
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a / b) * y1
    return gcd, x, y
```

##### 证明

欧几里得算法产生如下除法序列： 

1. $a = q₁ b + r₁ (0 ≤ r₁ < b)$ 
2. $b = q₂ r₁ + r₂ (0 ≤ r₂ < r₁)$
3. $r₁ = q₃ r₂ + r₃ (0 ≤ r₃ < r₂)$
4. ... 
5. $r_{k-2} = q_k r_{k-1} + r_k$
6. $r_{k-1} = q_{k+1} r_k + 0$

则 $gcd(a, b) = r_k$ 最后一个非零余数。现在**从后往前回代**，证明 $r_k$ 可以表示为 $a$ 和 $b$ 的线性组合。

1. 从倒数第二步： $r_k = r_{k-2} - q_k r_{k-1}$ （这是 $r_{k-2}$ 和 $r_{k-1}$ 的线性组合）
2. 将 $r_{k-1}$ 代入上一式： $r_{k-1} = r_{k-3} - q_{k-1} r_{k-2}$ → $r_k = r_{k-2} - q_k (r_{k-3} - q_{k-1} r_{k-2}) = (1 + q_k q_{k-1}) r_{k-2} - q_k r_{k-3}$
3. 继续回代，最终所有余数都会被表达为更早的余数，直至：
    - $r₁$ 用 $a$ 和 $b$ 表示：$r₁ = a - q₁ b$
    - $b$ 用 $b$ 表示（自身）

最终，$r_k$（即 gcd）将被表达为 $a$ 和 $b$ 的整数系数线性组合： **gcd(a, b) = r_k = a x + b y**

### 线性模方程

#### 问题背景

在模 n 的世界里，只剩下 n 个“基本元素” ${0, 1, 2, …, n−1}$，每一个元素实际上代表一个无限集合 $[k] = { k + tn | t ∈ ℤ }$。例如在模 7 的世界里：

- 0 代表 $\{…, −14, −7, 0, 7, 14, …\}$
- 3 代表 $\{…, −11, −4, 3, 10, 17, …\}$

因此我们可以写出 $7 \equiv 12 (\text{mod} 5)$，因为它们对 5 的余数相同。而线性模方程解决的就是：

$$
ax \equiv b (mod \,n)
$$

其中 a, b, n 是已知整数，n > 0，x 是未知整数，我们要求的就是 **x 在模 n 意义下的解**。但是在模 n 的世界里，**除法并不天然存在**。所以线性模方程解决的是：

> **在模运算体系中，什么时候可以做“除法”，以及怎么做。**

#### 除法存在的条件

假设 $d=\text{gcd}(a, n)$，当且仅当 $d  |  b$（整除） 时，方程 $ax \equiv b (mod \, n)$ 有解，并且在模意义下存在 d 个不同的解。

#### 怎么做

```python
def linear_mod(a, b, n):
	d, x, y = gcd_extened(a, b)
	if d % b: return -1
	x0 = (x * b / d) % n
	for i in range(n):
		print(x0 + i * n / d) % n
```

>感觉不考，具体原理看不懂

#### 手算

**例： 求 $35x=10(mod 50)$ 的所有解**

1. 先判断有没有解：$d=gcd(35, 50)=5$ ，5 整除 10 所以有唯一解，解的数量为 5
2. 等式两边同除 d 得到 $7x \equiv 2(mod 10)$
3. 计算模 50 空间下 7 的逆元，就可以把 x 的系数消掉，经过计算 $7 \times 3 = 1 (mod 10)$
4. 等式两边同乘 3，得到 $x \equiv 6(mod 10)$
5. 所以解为 $x=6 + 10t$

>逆元其实就是普通意义上的相反数，我们需要求 $kx=b$ 里面 k 的相反数，这样两边同乘就可以去掉 x 的系数。

### 中国余数定理

中国余数定理主要解决**一组线性同余方程组的求解问题**，即： 给定多个模数和对应的余数，求一个整数 x 使得它同时满足所有这些同余条件。简单地说就是**线性模方程组求解**。

$$
\begin{cases}
x \equiv a_1 \pmod{m_1} \\
x \equiv a_2 \pmod{m_2} \\
\vdots \\
x \equiv a_k \pmod{m_k}
\end{cases}
$$

假设 $m_1,m_2,...,m_k$ 两两互质，那么方程组在模 $M=m_1m_2...m_k$ 空间下有唯一解。

{{< admonition tip "唯一解" true >}}
注意：这里的唯一解并不是说 x 只有唯一值 ，而是说在模 M 空间下 x 唯一。
 {{< /admonition >}}

解方程组步骤：
1. 计算 $M$ 和 $M_i$
	1. $M=m_1m_2...m_k$
	2. $M_i=M / m_i$
2. 计算 $M_i$ 在模 $m_i$ 下的乘法逆元 $M_i^{-1}$ ，$M_i · M_i^{-1} ≡ 1 (mod \, m_i)$ 
3. 计算 $c_i=M_i \times (M_i^{-1} \mod m_i)$ 
4. $x = \sum{c_i \times a_i (mod \, M)}$ 
5. 结果为 M\*k+ x，k 为任意整数

---

**例：找出被9,8,7除时，余数分别为1,2,3的 x**


$$
\begin{cases}
x \equiv 1 \pmod{9} \\
x \equiv 2 \pmod{8} \\
x \equiv 3 \pmod{7}
\end{cases}
$$

1. $M=9\times8\times7=504$
2. $M_1=56,M_2=63,M_3=72$
3. 计算逆元
	1. $56\times M_1^{-1}=9N+1 \rightarrow M_1^{-1}=5$
	2. $63\times M_2^{-1}=8N+1 \rightarrow M_2^{-1}=7$
	3. $72\times M_3^{-1}=7N+1 \rightarrow M_3^{-1}=4$
4. 计算 c
	1. $c_1=56 \times (5 \mod 9)=280$
	2. $c_2=63 \times (7 \mod 8)=441$
	3. $c_3=72 \times (4 \mod 7)=288$
5. 求和 $x=1*280+2*441+3*288=2026 \mod 504 = 10$
6. 结果为 504k+10

### RSA算法

1. 随机挑两个大素数 p 和 q
2. $n=pq$ 且 $\phi(n)=(p-1)(q-1)$ 
3. 找一个 e 使得 $gcd(e, \phi(n)) = 1$ ，也就是找一个和 $\phi(n)$ 互质的数
4. 计算模 $\phi(n)$ 空间下 e 的逆元 d 使得：$ed \equiv 1 (mod \, \phi(n))$ 
5. 公钥 (n, e)，私钥 (n, d)。知道了 e 但是不知道 p 和 q 没法推出逆元

> [!NOTE]
> RSA 加密系统的安全性主要来源于对大整数进行因式分解的困难性。

### 素数算法

#### 简单素数测试

判断 n 是不是素数，只要用 $2,...,\sqrt{n}$ 和它进行除法就知道了。

#### 伪素数测试

根据 Fermat 小定理：若 p 是素数，则对任意整数 a 满足 p 不整除 a，有 $a^{p−1} ≡ 1 (mod \, p)$。所以我们就想到它的逆否命题，对任意整数 a，且 p 整除 a，假如 $a^{p−1} \not\equiv 1 (mod \, p)$，那么 p 不是素数。如果等式成立，那么则称 p 是一个基为 a 的伪素数。

>整除的概念是：如果**存在整数 k**，使得 $a = b \cdot k$，那么称 **b 整除 a**。

那么称 **bbb 整除 aaa**，记作：

```python
def pseudo_prime(n):
	return math.pow(2, n-1) % n == 1
```

>错判合数为素数：Carmichael 数（561、1105、1729……）
### MR算法

Fermat 测试的问题在于：**存在 Carmichael 数，使所有 a 都通过测试**。MR 的改进在于：对任意合数 n，**不会对所有 a 都“伪装成功”**。若 n 是合数，则至少 3/4 的 a 能在 MR 测试中暴露 n 是合数。因此 MR 给出了一个**统一的错误概率上界**，而 Fermat 测试没有。

1. 根据整数分解定理，任意整数都可以**唯一**的分解为 **2 的若干次幂 × 一个奇数**，所以令 $n-1 = 2^s \times d$
2. 检查 $\{a^d, a^{2d}, a^{4d}, …, a^{2^{s−1}d}\}$，如果 **出现 −1（mod n）**，就符合素数应有的行为；

但是 **仍然会出现错判素数**，时间复杂度为 $O(T \times \lg{N})$，T 是检测轮次。
## 第三十二章 串匹配

{{< admonition tip "什么是串匹配问题" true >}}
给定一个文本串 $T = t_1t_2 \dots t_n$ ，长度为 n； 给定一个模式串 $P = p_1p_2 \dots p_m$，长度为 m； 要求找出所有满足 $T[s+1 … s+m] = P[1 … m]$ 的位置 s。
 {{< /admonition >}}

| 算法         | 预处理                    | 匹配(最坏情况) |
| ---------- | ---------------------- | -------- |
| 暴力         | 0                      | $O(nm)$  |
| Rabin-Karp | $O(m)$ 模式串算哈希          | $O(nm)$  |
| 有限自动机      | $O(m\vert \sum \vert)$ | $O(n)$   |
| KMP        | $O(m)$                 | $O(n)$   |

### 朴素

```python
for i in range(n):
	for j in range(m):
		if T[i+j] != P[j]:
			break
```

- 最坏情况下时间复杂度 $O(nm)$
- 最后情况下时间复杂度 $O(n)$，每次第一个字符就失配

>[!Note]
>最坏情况时间复杂度不是 $O(m)$，因为假如第一次就匹配成功，后面还需要匹配，串匹配问题要求找到**所有**满足条件的位置。

### Rabin-Karp

Rabin-Karp 算法先是比较模式串 P 与文本子串 T\[s+1 … s+m] 的哈希值，只有当哈希值相等时，才进行一次逐字符验证，这是一种“先粗筛、再精查”的思想。哈希函数为：

$$
Hash(T) = (T[0]*p^{n-1}+T[1]*p^{n-2}+ \dots + T[n-2]*p + T[n-1]) \% q
$$

- q 是一个大素数用于取模
- d 要求大于字母表大小，如 ASCII 可以取 256

```python
for i in range(n):
	if not hash(T[i;i+m], P): continue
	for j in range(m):
		if T[i+j] != P[j]: break
```

- 最坏情况时间复杂度 $O(nm)$，每个哈希值都符合
- 最好情况时间复杂度 $O(n+m)$，哈希时间+匹配时间

### 有限自动机

将模式串 P 构造成一个确定有限自动机（DFA），在扫描文本时： •每读入一个字符，只进行一次状态转移，不回退文本指针。

具体例子：模式串 P = "aab" 字母表 Σ = {a, b}

| 当前状态 q | 输入 a                                                  | 输入 b                                      |
| ------ | ----------------------------------------------------- | ----------------------------------------- |
| 0      | " "(空) + a = "a" → 最长公共前后缀 "a" → **1**                | " " + b = "b" → 无 → **0**                 |
| 1      | "a"(P\[0:0]) + a = "aa" → 最长公共前后缀 "aa" → **2**        | "a" + b = "ab" → 无 → **0**                |
| 2      | "aa"(P\[0:1]) + a = "aaa" → 最长公共前后缀 "aa" 是前缀 → **2**  | "aa" + b = "aab" → 最长后缀 "aab" 是前缀 → **3** |
| 3      | "aab"(P\[0:2]) + a = "aaba" → 最长公共前后缀 "a" 是前缀 → **1** | "aab" + b = "aabb" → 无 → **0**            |
>这里最长公共前后缀指的是，"aaba" 的后缀和模式串 "aab" 的前缀最长匹配

最终转移表：

| 状态 \ 输入 | a   | b   |
| ------- | --- | --- |
| 0       | 1   | 0   |
| 1       | 2   | 0   |
| 2       | 2   | 3   |
| 3       | 1   | 0   |

**匹配示例**： 文本 T = "aabaab"（长度 6）

| 位置 i | 读入字符 | 当前状态 | 新状态 | 是否匹配                 |
| ---- | ---- | ---- | --- | -------------------- |
| 0    | a    | 0    | 1   |                      |
| 1    | a    | 1    | 2   |                      |
| 2    | b    | 2    | 3   | **是**（位置 0-2: "aab"） |
| 3    | a    | 3    | 1   |                      |
| 4    | a    | 1    | 2   |                      |
| 5    | b    | 2    | 3   | **是**（位置 3-5: "aab"） |

如何画图：

![](http://img.xilyfe.top/img/20260113144632934.png)


- 预处理时间：$O(m\vert \sum \vert)$   
- 处理时间：$O(n)$

### KMP

KMP 的本质是：在发生失配时，不回退文本指针， 而是根据已经匹配的信息，决定模式串应当移动到哪里。

首先来看 next 数组的作用：它告诉我们当主串和模式串失配时候，主串应该退回到什么问题

![image.png](http://img.xilyfe.top/img/20260113150109425.png)

如图，当主串和子串在 `i=4` 失配时候，我们看前一位的 next数组 `next[3]=2` 就知道，它标志着我们应该跳过几个元素，例如这里告诉我们跳过两个元素，所以模式串回到第三位 `T[2]`，主串和模式串的前 2 位都相同直接跳过。

```python
def kmp(string, pattern, next):
	i, j  = 0, 0
	n, m = len(string), len(pattern)
	while i < n:
		if string[i] == pattern[j]:
			i += 1
			j += 1
		elif j > 0:
			j = next[j-1]
		else:
			i += 1
		if j == len(pattern):
			return i - j
```

那 next 数组怎么得到呢？
我们先思考一下为什么可以用 next 数组在失配时回退呢？因为在 `i=4` 失配时候，模式串失配处 C 前两位（也就是主串失配处 A 的前两位）AB 和模式串前两位 AB 相同。换句话说，假如模式串的某个位置失配的时候，失配处前 i 位（也就是主串失配处前 i 位）如果和模式串前 i 位一样就能跳过了。那我们只需要看模式串的最长公共前后缀就好了。

![image.png](http://img.xilyfe.top/img/20260113152224891.png)

对于 next 数组，我们固定第一位是 0：
1. 从第二位 B 开始，由于 A、B 没有公共前后缀，所以是 0
2. 第三位 A，A、B、A 的最长公共前后缀是 A，所以可以跳过 1 个
3. ...

那代码应该怎么写呢？不可能暴力求公共前后缀吧？

![image.png](http://img.xilyfe.top/img/20260113154848400.png)

如图当我们需要求 `next[6]` 时候，已经知道前 6 位的最长公共前后缀是 2 了，所以如果 `pattern[6]` 和 `pattern[2]` 相同，那我们就可以继续向后走了。那如果不相同呢？

![image.png](http://img.xilyfe.top/img/20260113155629769.png)

这里可以看到 `pattern[3]` 和 `pattern[7]` 不同，既然 ABA 没办法和下一个字符组成最长公共前后缀，那我们看看有没有更短的，比如前缀 AB 和后缀 AB 相同。这时候我们又不得不暴力搜索了吗？

这里我们还知道一个信息，就是前面最长公共前后缀是 3。既然我们只能找比 3 更短的公共前后缀，或者说在 ABA 里面找 `pattern[7]=B` 的最长刚刚前后缀，那么把 B 放在 C 的位置上来看就行了。这时候我们看到 C 前面 A 的 next 值为 1，也就是 ABA 的最长公共前后缀是 1，这时候加上 B，`pattern[1]` 也是 B 所以匹配了。

```python
def get_next(pattern):
	next = [0]
	prefix_len = 0
	i = 1
	while i < len(pattern):
		if pattern[i] == pattern[prefix_len]:
			prefix_len += 1
			i += 1
			next.append(prefix_len)
		else:
			if prefix_len == 0:
				next.append(prefix_len)
				i += 1
			else:
				prefix_len = next[prefix_len - 1]  # 只需要改一下prefix_len就可以到C的位置
	return next			
```

## 第三十四章 模型和NPC

### 图灵机模型

图灵机是一种抽象的计算模型，它就像一台最简单的“电脑原型”，用来描述“什么问题是可以用算法解决的”。机器启动后，就按照指令表一步步执行：
- 读 → 写 → 移动 → 换状态 → 重复……
- 直到进入“停机状态”，就停下来，纸带上剩下的内容就是输出结果。

- **确定性图灵机（DTM）**：每一步都只有唯一的选择，像普通程序，死板但可靠。
- **非确定性图灵机（NDTM）**：每一步可以有多个选择，它会“同时尝试所有可能”（像平行宇宙），只要有一条路成功就算成功。实际电脑要模拟它会慢很多。

### 语言识别能力

在计算理论里，我们常把问题转化为“识别一种语言”的问题。这里的“语言”不是中文英语，而是**一堆字符串的集合**。
- 比如：“所有由等数量的a和b组成的字符串”，像：abba、aabb、ababbbab 等，这就是一种“语言”。
- 问题就是：给一个字符串，机器能不能判断它是否属于这个语言？（是→接受，否→拒绝）

1. **有限自动机** 只能识别很简单、没有嵌套的模式。 例子：所有以“abc”开头的字符串，或者“全是0和1，且以1结尾”的二进制数。 不能处理括号匹配那种需要“记忆”的东西。
2. **下推自动机（加了个栈内存）** 能处理括号嵌套、回文串这类。 经典例子：{ a^n b^n } → aaabbb 这种“前半a和后半b数量相等”。 能检查代码里的括号是否匹配。
3. **线性有界自动机** 更强，能处理 a^n b^n c^n 这种“三部分数量相等”的。
4. **图灵机** 几乎什么都能识别，包括上面所有，还能处理更复杂的（甚至能模拟其他所有机器）。 但有些问题连图灵机都判断不了（比如著名的“停机问题”：给一段程序和输入，能不能预测它会不会死循环？）

### P、NP、NP 完全

- P 问题：图灵机可以在多项式时间内**解决**的问题
- NP 问题：图灵机可以在多项式时间内**验证**的问题

很显然，所有的 P 类问题都是 NP 问题。也就是说，能多项式地解决一个问题，必然能多项式地验证一个问题的解——既然正解都出来了，验证任意给定的解也只需要比较一下就可以了。关键是，人们想知道，是否所有的 NP 问题都是 P 类问题。我们可以再用集合的观点来说明。如果把所有 P 类问题归为一个集合 P 中，把所有 NP 问题划进另一个集合 NP 中，那么，显然有 P 属于 NP。现在，所有对 NP 问题的研究都集中在一个问题上，即究竟是否有 P=NP？通常所谓的“NP问题”，其实就一句话：证明或推翻 P=NP。

为了说明NPC问题，我们先引入一个概念——约化(Reducibility，有的资料上叫“归约”)。简单地说，一个问题A可以约化为问题B的含义即是，可以用问题B的解法解决问题A，或者说，问题A可以“变成”问题B。例如：求解一个一元一次方程可以约化为求解一个一元二次方程，因为只要把二次型系数固定为 0 就可以了。从约化的定义中我们看到，一个问题约化为另一个问题，时间复杂度增加了，问题的应用范围也增大了。通过对某些问题的不断约化，我们能够不断寻找复杂度更高，但应用范围更广的算法来代替复杂度虽然低，但只能用于很小的一类问题的算法。自然地，我们会想问，如果不断地约化上去，不断找到能“通吃”若干小NP问题的一个稍复杂的大NP问题，那么最后是否有可能找到一个时间复杂度最高，并且能“通吃”所有的 NP问题的这样一个超级NP问题？答案居然是肯定的。也就是说，存在这样一个NP问题，所有的NP问题都可以约化成它。换句话说，只要解决了这个问题，那么所有的NP问题都解决了。这种问题的存在难以置信，并且更加不可思议的是，这种问题不只一个，它有很多个，它是一类问题。这一类问题就是传说中的NPC 问题。

所以 NPC 问题的条件如下：
1. 是 NP 问题
2. 所有 NP 问题都可以约化为这个问题

最后 NP-Hard 问题指的是，不一定是 NP 问题的 NPC问题，例如 ”停机问题“ 这种没办法验证的问题。

### SAT 问题

SAT（布尔可满足性问题）：给定一个布尔公式，问：是否存在一种变量赋值，使整个公式为真？例如：

$$
(x_1 \lor \lnot x_3 \lor x_5)\ \land\ (\lnot x_2 \lor x_4)
$$

2-SAT 指的是每一个子句中，**最多只有 2 个文字** 的 SAT 问题，例如：$(x_1 \lor x_2)\ \land\ (\lnot x_2 \lor x_3)\ \land\ (\lnot x_1 \lor \lnot x_3)$。2-SAT 可以用强联通分量在线性时间解决，所以是 P 问题，不属于 NP 或者 NPC 问题。

>[!Note]
>NP 完全问题意味着 L ∈ NP 并且所有 NP 问题都能多项式时间归约到 L，假如 L 本身能多项式时间解决，那么 L ∈ P，与 P ≠ NP 矛盾了。

- 而 3-SAT 无法在多项式时间内解决，属于 NP 和 NPC 问题。
- CIRCUIT-SAT 也是 NP/NPC 问题。

## 第三十五章 近似问题

### 多项式时间近似模式

很多优化问题：不是 P 问题 ，甚至是 NP-hard ，但“稍微差一点的解”在工程上是可以接受的。PTAS 的目的就是在**多项式时间**内，保证解“离最优解不太远”。

可以自己指定一个误差参数 ε > 0（比如 ε = 0.01 表示 1% 误差）。 算法保证给出的解和最优解的差距不超过 ε：
- 对于最大化问题（如最大收益）：解 ≥ (1 - ε) × 最优值
- 对于最小化问题（如最小成本）：解 ≤ (1 + ε) × 最优值

对任意固定的 ε，算法运行时间是输入规模 n 的多项式（比如 O(n³) 或 O(n^{1/ε}) 之类的）。

>[!Note]
>1. 时间可以随着 1/ε 增长得很快（甚至是指数级的，比如 2^{1/ε} × n²），但只要 ε 固定下来，时间就是 n 的多项式，不会失控。
>2. FPTAS 和 PTAS 的区别是：FPTAS 要求对 1/ε 都是多项式，不能像什么一样是指数。

## 真题

### 2018 期中考

[真题PDF](https://github.com/XilyFeAAAA/ImgRepository/blob/main/file/18%E5%B9%B4%E6%9C%9F%E4%B8%AD.pdf)

1. 动态规划的优势在于它可以自底向上的解决重叠子问题，不用向 DFS 一样需要计算多次。
2. 1
3. 上界是 $O(nlg^2{n})$，可以用递归树求解：每一层是 nlgn，一共 lgn 层。
4. 根据主定理，这个递归式的时间复杂度是 $O(n^{log^a_4})$，而 Strassen 定理的时间复杂度是 $O(n^{log_2^7})$，所以要求 $log_4^a \lt log_2^7$，根据换底公式我们可以得到 $log_2^a \lt 2log_2^7$，所以 $a \lt 49$ 。
5. 

| 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0   | 0   | 1   | 1   | 1   | 1   | 1   | 1   |
| 0   | 1   | 1   | 1   | 1   | 2   | 2   | 2   |
| 0   | 1   | 1   | 1   | 1   | 2   | 2   | 3   |
| 0   | 1   | 1   | 2   | 2   | 2   | 2   | 3   |
| 1   | 1   | 1   | 2   | 3   | 3   | 3   | 3   |
| 0   | 1   | 2   | 2   | 3   | 3   | 4   | 4   |

6. $O(n\lg n + m) \ge O(mn)$ 得到 $m \ge \frac{n}{n-1}\lg n$ 此时性能优于 m 次线性时间的代价。如果元素经常变动，可以用一个顺序查找树，它是在红黑树基础上每个节点加了一个 size，表示以自己为根的子树个数。这样子查询和插入的时间复杂度是 $O(\lg n)$。
7. 用两次二分查找：
```python
lower_bound(A, n, x):
    l = 1, r = n + 1
    while l < r:
        mid = (l + r) // 2
        if A[mid] < x:
            l = mid + 1
        else:
            r = mid
    return l

upper_bound(A, n, x):
    l = 1, r = n + 1
    while l < r:
        mid = (l + r) // 2
        if A[mid] <= x:
            l = mid + 1
        else:
            r = mid
    return l

L = lower_bound(A, n, x)
R = upper_bound(A, n, x)

if L > n or A[L] != x:
    return "x 不存在"
else:
    return A[L:R-1]

```
8. 递推式为 $dp[i][j]=dp[i][j-1]+dp[i-1][j-1]$，优化方式同 LCS。
```
def combinator(m, n):
	dp[m+1][n+1] = 0
	for i in range(m+1):
		dp[i][0] = 1
	for i in range(1, m+1):
		for j in range(1, n+1):
			if i == j: dp[i][j] = 1
			else: dp[i][j] = dp[i][j-1] + dp[i-1][j-1]
	return dp
```
9. 这个问题可以结合 **最长公共子序列** 和 **最长连续子串** 两个问题。令 `dp[i][j]` 为 以 X\[i] 和 Y\[j] 结尾的最长公共子串长度，若字符相等：`dp[i][j] = dp[i−1][j−1] + 1` ，若字符不等：`dp[i][j] = 0`。优化思路同 LCS，`dp[i][j]` **只依赖左上角 `dp[i−1][j−1]`** ，因此，不需要保存完整二维表，只需保存**上一行的结果**。
```
LongestCommonSubstring(X, Y):
    n = length(X)
    m = length(Y)
    create array dp[0..n][0..m]
    maxLen = 0
    endPos = 0

    for i = 0 to n:
        dp[i][0] = 0
    for j = 0 to m:
        dp[0][j] = 0

    for i = 1 to n:
        for j = 1 to m:
            if X[i] == Y[j]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > maxLen:
                    maxLen = dp[i][j]
                    endPos = i
            else:
                dp[i][j] = 0

    return maxLen
```
### 2018 期末考

[真题 PDF](https://github.com/XilyFeAAAA/ImgRepository/blob/main/file/18%E5%B9%B4%E6%9C%9F%E6%9C%AB.pdf)

1. 快排最坏时间复杂度是 $O(n^2)$，归并是 $O(n\lg n)$。因为（1）空间开销显著更小：归并排序在数组实现下需要额外的 O(n) 辅助空间，而快排是原地排序；（2）常数因子更小，实际运行更快：归并排序在“合并”阶段需要频繁地进行数组拷贝与写入辅助数组，常数开销明显更大；（3）缓存局部性更好：快速排序在划分过程中，主要在当前子数组上进行顺序访问和局部交换，具有良好的空间局部性，能充分利用 CPU cache。归并排序在合并阶段需要在多个数组之间来回读写，缓存命中率较低。
2. 二叉堆 Minimum 操作更快 $O(1)$，二叉堆是 $O(\lg n)$；合并操作二项堆快 $O(\lg n)$，二叉堆是 $O(n\lg n)$
3. 动态规划步骤是：写出递推式，确定遍历顺序，确定初始条件
4. MR算法的改进包括：通过整数分解定理，将 p-1 分解为 $2^s\times d$ ，然后对 ${a^d, a^{2d}, a^{4d}, …, a^{2^{s−1}d}}$ 都进行检查；除了检查余数是否为 1，还检查是否为 -1。
5. $T(n)= T( \frac{2}{3} n ) + 1= T( (\frac{2}{3})² n ) + 1 + 1 = T( (\frac{2}{3})³ n ) + 1 + 1 + 1$ ，$(\frac{2}{3})^k · n = 1$ 得到递推深度为 $Θ(log n)$，所以时间复杂度 $Θ(log n)$。
6. ![image.png](http://img.xilyfe.top/img/20260117120124658.png)
7. 就是 DAG 最短路径问题：

| dp      |                                     |
| ------- | ----------------------------------- |
| dp\[1]  | 0                                   |
| dp\[2]  | 9                                   |
| dp\[3]  | 3                                   |
| dp\[4]  | 7                                   |
| dp\[5]  | 2                                   |
| dp\[6]  | min(dp\[2]+4,dp\[3]+2)=5            |
| dp\[7]  | min(dp\[2]+2,dp\[3]+7,dp\[5]+11)=10 |
| dp\[8]  | min(dp\[2]+1,dp\[4]+11,dp\[5]+8)=10 |
| dp\[9]  | min(dp\[6]+6,dp\[7]+4)=11           |
| dp\[10] | min(dp\[7]+3,dp\[8]+5)=13           |
| dp\[11] | min(dp\[6]+5,dp\[8]+6)=10           |
| dp\[12] | 15                                  |

8. 算法思路：按照重量排序，每次选重量最小的
	- 设 $S^*$ 是任意一个最优解，$S^*$ 选择的第一个元素下标是 i
	- 假设按照贪心算法选择的第一个元素下标为 0
	- 假如 `i==0`，那么最优解第一个元素就是贪心
	- 假如 `i != 0`，那么最优解的第一个元素肯定比 0 重价值低，那么替换为 0 就可以得到更优解。
9. 贪心策略是不断选择权重最小的两条边，在保证不形成环的情况下合并。用并查集来实现。
10. 代码简单，复杂度 $O(\lg n)$
11. 有序的情况，二分查找。在没有缺失的理想情况下，应有 `A[i] = i`。缺失一个数后，在缺失点右侧会出现  
`A[i] > i` 。`A[i] = i`，说明缺失的数在右半区，若 `A[i] > i`，说明缺失的数在左半区；无序的情况，遍历一遍数组把全部元素加起来，然后用 1 到 n+1 的和减去数组元素和，就是缺少的数。
12. KMP 和 有限自动机。


### 2021 期中考


### 2021 期末考


[真题 PDF](https://github.com/XilyFeAAAA/ImgRepository/blob/main/file/21%E5%B9%B4%E6%9C%9F%E6%9C%AB.pdf)

1. 正确，直接主定理
2. 正确，用 Order Statistic 算法在线性时间得到第 k1 个元素和第 k2 个元素，然后遍历一遍找到之间的值求和
3. 正确，用斐波那契堆+Johnson 算法
4. 错误，2-SAT 是 P 问题，不属于 NP 或者 NPC 问题。3-PAT 剩余 NPC问题。
5. 正确，PTAS 就是可以在接受 ep 误差的情况下，得到关于 n 多项式时间复杂的的算法。FPTAS 是更进一步，得到高于 ep 和 n 的多项式时间复杂度的算法。
6. 分治问题就是不断把大问题分解为小的子问题，然后解决子问题并且合并来解决原问题。方法包括：递归树，代入法，主定理
7. 
8. `dp[i][j]` 表示对于前 i 个物品和 j 重量，最高价值是多少。递推式是 `dp[i][j]=max(dp[i-1][j], dp[i-1][j-w]+v)`。称为伪多项式时间复杂度，因为时间复杂度是 $O(n · W)$，在多项式理论中输入规模是输入位数，所以应该是 $O(n \cdot 2^{logW})$ ，对于 W 是指数级。
9. π 就是看最长公共前后缀，第一位固定 0。
10. 和矩阵链乘法一样，需要三次循环，for 区间长度 for 左端点 for 分割位置，时间复杂度是 $O(n^3)$。
11. 用哈夫曼树的方法，排序之后不断选择最小两个合并，$O(n\lg n)$


### 2024 期中考


### 2025 期中考

已知：$T(n) = T(n/2) + T(\sqrt n) + n， T(1) = 1$ 求渐进上界。

我们考察每一层递归的总代价。

- 第 0 层代价 = n
- 第 1 层本层总代价：  n/2 + √n ≤ n/2 + n/2 = n（对 n ≥ 1 恒成立）
- 第 2 层：  所有非递归项之和仍然 小于 n。

所以可知：每一层递归的总代价 ≤ n。
观察最长递归路径 n → n/2 → n/4 → … → 1 ，深度是 O(log n)。  而 √n 分支下降更快，不会增加层数。每层代价$O(n)$，层数 $O(log n)$，因此：$T(n) = O(n \lg n)$。