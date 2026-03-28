---
title: Autograd from scratch
date: 2026-03-04T23:53:11+08:00
featuredImage: http://img.xilyfe.top/img/20260304235416410.png
authors:
  - Xilyfe
series:
  - DeepLearning
tags: []
lastmod: 2026-03-05T04:31:44+08:00
---
>用了这么久的 PyTorch 框架，发现居然不知道它是怎么实现自动计算梯度的，今天来学习一下

## 1. 前向传播&反向传播

我们回忆一下深度学习中，模型训练的流程。以简单的深度神经网络为例，我们不断把 mini-batch 送入网络进行前向传播，然后把输出的预测值同真实值对比，用 criterion 计算出此次迭代的 loss。之后把 loss 进行反向传播，送入神经网络模型中之前的每一层，以更新 weight 矩阵和 bias。其中前向传播就是一系列的矩阵+激活函数的组合运算，比较简单直观；反向传播就稍显复杂了，如果我们不用深度学习框架，单纯的使用 numpy 也是可以完成大多数的模型训练，因为反向传播本质上也就是一系列的矩阵运算，不过我们需要自己写方法以进行复杂的梯度计算和更新，而深度学习框架帮助我们解决的核心问题就是<mark>反向传播时的梯度计算和更新</mark>。

## 2. 链式求导

反向传播就是一个链式求导的过程，我们举一个例子：求函数 $f(x)=[w_2(w_1x^{2}+b)]^{3}$ 中参数 $w_1$，$w_2$ 和 $b$ 的偏导数。我们一种方法是可以直接暴力求导：

$$
\begin{align}
\frac{\partial f}{\partial w_1} &= 3 w_{2}^{3} x^{2} \left(b + w_{1} x^{2}\right)^{2}\\
\frac{\partial f}{\partial w_2} &= 3 w_{2}^{2} \left(b + w_{1} x^{2}\right)^{3} \\
\frac{\partial f}{\partial b} &= 3 w_{2}^{3} \left(b + w_{1} x^{2}\right)^{2} \\
\end{align}
$$

这种方式会导致大量的重复计算，开销很大。第二种方式我们可以考虑有限查分对偏导进行估计，基于导数的定义我们有：

$$
\frac{\partial f}{\partial \theta} = \lim_{\epsilon \to 0} \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}
$$

这种方式计算一个梯度需要两次评估函数 f，并且计算机浮点数有精度限制，如果 $\epsilon$ 太小，$f(\theta + \epsilon) - f(\theta)$ 会因舍入误差而失真；如果 $\epsilon$ 太大，近似又不准。我们可以通过中心查分来改进：

$$
\frac{\partial f}{\partial \theta}\approx \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon} + O(\epsilon^2)
$$

这比有限查分更准（误差是 $O(\epsilon^2)$ 而非 $O(\epsilon)$），但还是需要两次评估，且有相同的问题：慢和浮点误差，所以我们考虑通过链式求导来计算每个参数的偏导。对于上面的例子，我们设 $p=w_1x^2$，$q=p+b$，$k=w_2*q$，$f=k^3$。那我们从损失开始反向求导：

$$
\begin{align} 
\frac{\partial f}{\partial k}&=3k^2\\
\frac{\partial f}{\partial w_2}&=\frac{\partial f}{\partial k} \cdot \frac{\partial k}{\partial w_2}=3k^2 \cdot q\\
\frac{\partial f}{\partial q}&=\frac{\partial f}{\partial k} \cdot \frac{\partial k}{\partial q}=3k^2 \cdot w_2\\
\frac{\partial f}{\partial p}&=\frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial p}=3k^2 w_2 \cdot 1=3k^2 w_2\\
\frac{\partial f}{\partial b}&=\frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial b}=3k^2 w_2 \cdot 1=3k^2 w_2\\
\frac{\partial f}{\partial w_1}&=\frac{\partial f}{\partial p} \cdot \frac{\partial p}{\partial w_1}=3k^2 w_2 \cdot x^2 
\end{align}
$$

这种链式求导的方式避免了重复计算，每个局部导数只需计算一次，然后通过乘积链式传递到上游参数，从而高效地得到所有偏导数。在神经网络中，反向传播就是将梯度从输出层逐步向输入层传播的过程。

## 3. 计算图

为了实现上述提到的链式求导，我们利用计算图的拓扑结构来复用中间结果。在 PyTorch 和 Tensorflow 中，底层结构都是由 tensor 组成的计算图，虽然框架代码在实际 autograd 自动求梯度的过程中并没有显示地构造和展示出计算图，不过其计算路径确实是沿着计算图的路径来进行的。我们举个例子：假设我们有输入 $x$，然后对它进行一个线性变换，最后用均方差 MSE 得到损失 loss 可以表示为：

![image.png](http://img.xilyfe.top/img/20260305001325443.png)

在上图例子中，我们将前向传播的每一个中间变量都存储为 tensor，然后这些 tensor 节点形成一个前向传播的有向无环图 DAG。反向传播 autograd计算梯度时，我们从根节点开始遍历这些 tensors 来构造一个反向传播梯度的计算图模型，将计算得到的梯度值更新到上一层的节点，并重复此过程直至所有 `required=True` 的 tensor 都得到更新。这一层层的求导过程，隐式地利用了链式法则，最终各个变量的梯度值得以更新，故此过程形象地称为 autograd。

## 4. Torch 张量

```python
class Tensor:
    requires_grad: _bool = ...
    grad: Optional[Tensor] = ...
    data: Tensor = ...
    names: List[str] = ...
    @property
    def dtype(self) -> _dtype: ...
    @property
    def shape(self) -> Size: ...
    @property
    def device(self) -> _device: ...
    @property
    def T(self) -> Tensor: ...
    @property
    def grad_fn(self) -> Optional[Any]: ...
    @property
    def ndim(self) -> _int: ...
    @property
    def layout(self) -> _layout: ...

    def __abs__(self) -> Tensor: ...
    def __add__(self, other: Any) -> Tensor: ...
    @overload
    def __and__(self, other: Number) -> Tensor: ...
    @overload
    def __and__(self, other: Tensor) -> Tensor: ...
    @overload
    def __and__(self, other: Any) -> Tensor: ...
    def __bool__(self) -> builtins.bool: ...
    def __div__(self, other: Any) -> Tensor: ...
        ...
        ...
```

### 4.1 requires_grad

`requires_grad` 是一个布尔值，表示 autograd 时是否需要计算此 tensor 的梯度，默认False；用官方文档上的话描述：requires_grad允许从梯度计算中细粒度地排除子图，并可以提高计算效率。这里需要注意一点，如果某个 tensor 存在一个输入 `requires_grad=True` 那么这个 tensor 也必须记录梯度。当且仅当所有输入都无需记录梯度时，输出才可以不记录梯度，设置为 `requires_grad=False`。

```python
>>> x = torch.tensor(1)
>>> y = torch.tensor(2)
>>> z = torch.tensor(3.1, requires_grad=True)
>>> u = x+y
>>> u.requires_grad
>>> False
>>> v = u+z
>>> v.requires_grad
>>> True
```

### 4.2 grad

Tensor类变量，该变量表示梯度，初始为None；当self第一次调用backward()计算梯度时，生成新tensor节点，存储该属性存放梯度值，且当下次调用backward()时，梯度值可累积。(也可以设置清空)

### 4.3 grad_fn

反向传播时，用来计算梯度的函数。

### 4.4 is_leaf

标记该tensor是否为叶子节点：
- 按照惯例，所有requires_grad=False的Tensors都为叶子节点；
- 所有用户显示初始化的Tensors也为叶子节点；


## 5. 代码实现

这里先给出代码，再讲解具体的实现逻辑：

```python
@dataclass
class Dependency:
    op: Callable
    input: "Tensor"
    grad_fn: Callable


class Tensor:
    def __init__(
            self,
            data: Union[np.ndarray, int, float],
            grad: Optional["Tensor"] = None,
            depends: list[Dependency] = [],
            requires_grad: bool = False
    ) -> None:
        self.data = np.array(data)
        self.grad = grad
        self.depends = depends
        self.requires_grad = requires_grad

        if requires_grad:
            self.grad_zero()

    def grad_zero(self) -> None:
        assert self.requires_grad, "Cannot zero grad of non-requires_grad tensor"
        self.grad = Tensor(data=np.zeros_like(self.data))

    def backward(self, grad: Optional["Tensor"] = None):
        assert self.requires_grad, "Cannot backward on non-requires_grad tensor"
        if grad is None:
            grad = Tensor(np.ones_like(self.data))
        self.grad.data += grad.data
        for depend in self.depends:
            bp_grad = depend.grad_fn(grad.data)
            depend.input.backward(bp_grad)

    def __add__(self, other: "Tensor") -> "Tensor":
        return add(self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return sub(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        return mul(self, other)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        return div(self, other)

    def __str__(self) -> str:
        return str(self.data)


def add(input_1: Tensor, input_2: Tensor):
    data = input_1.data + input_2.data
    requires_grad = input_1.requires_grad or input_2.requires_grad

    if requires_grad:
        depends = []
        if input_1.requires_grad:
            depends.append(
                Dependency(
                    op=add,
                    input=input_1,
                    grad_fn=lambda grad: grad
                )
            )
        if input_2.requires_grad:
            depends.append(
                Dependency(
                    op=add,
                    input=input_2,
                    grad_fn=lambda grad: grad
                )
            )
        return Tensor(data=data, depends=depends, requires_grad=True)
    else:
        return Tensor(data=data)
```

我们用一个详细的例子理解一下 `Tensor` 为什么要这么设计。

![image.png](http://img.xilyfe.top/img/20260305152949964.png)


当我们计算 $v_2$ 梯度的时候，需要对它的 <mark>下游梯度*局部梯度</mark> 进行求和，例如 $\frac{\partial y}{\partial v_2}=\frac{\partial y}{\partial v_5} \cdot \frac{\partial v_4}{\partial v_2} + \frac{\partial y}{\partial v_4} \cdot \frac{\partial v_4}{\partial v_2}$，所以在计算得到新 Tensor 的时候，我们需要记录它的上游 Tensor，这样当我们处理完 $v_4$ 的梯度后就可以把它向上传给 $v_1$ 和 $v_2$。在代码中由于 $v_1 \times v_2=v_4$，所以我们生成新张量它的 `depends` 是 `Dependency(mul, v1, lambda grad: v2*grad)` 还有 `Dependency(mul, v2, lambda grad: v1*grad)`。这样当我们反向传播到 $v_4$ 时候，就会以此把 $v_4$ 的梯度代入 `grad_fn` 计算得到 $v_1$ 和 $v_2$ 一部分的梯度传给他。

前面说到我们需要对下游梯度\*局部梯度进行求和，这里我们就把它称做 **部分梯度**，由于它用到的下游梯度在反向传播时候才能得到，所以我们把它写在函数参数里，让下游的张量在调用是传进去：

```python
depends.append(
    Dependency(
        op=add,
        input=input_1,
        grad_fn=lambda grad: grad
    )
)
```


>我们举一个简单的例子算一下 $a$ 的梯度：
>![image.png](http://img.xilyfe.top/img/20260305162033069.png)
>- 首先我们沿着 $L \rightarrow d \rightarrow c \rightarrow a$ 计算：
>	1. 调用 `L.backward(None)`，默认 `grad=1`。
>	2. 循环到 `depend = d` 
>	3. 计算得到 `backward_grad = d.grad_fn()`，即 $\frac{\partial L}{\partial d}$。
>	4. 调用 `d.backward(backward_grad)`。
>	5. 在 `d.backward` 内部，循环到 `dep = c`：
>	6. `backward_grad = c.grad_fn()`，即 $\frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial c}$。
>	7. 然后再到 $a$， $a$ 的梯度累加了 $\frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial c} \cdot \frac{\partial c}{\partial a}$。
>- 然后我们沿着 $L \rightarrow e \rightarrow c \rightarrow a$ 计算：
>	1. $L$ 循环到 `depend = e` 
>	2. 计算得到 `backward_grad = e.grad_fn()`，即 $\frac{\partial L}{\partial e}$。
>	3. 调用 `e.backward(backward_grad)`。
>	4. 在 `e.backward` 内部，循环到 `dep = c`：
>	5. `backward_grad = c.grad_fn()`，即 $\frac{\partial L}{\partial e} \cdot \frac{\partial e}{\partial c}$。
>	6. 然后再到 $a$， $a$ 的梯度累加了 $\frac{\partial L}{\partial e} \cdot \frac{\partial e}{\partial c} \cdot \frac{\partial c}{\partial a}$，这时候 $a$ 的梯度就计算完成了。
