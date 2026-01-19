---
title: "Lecture 12: Neural Network"
date: '2025-12-02T11:24:11+08:00'
authors: [Xilyfe]
series: ["CS224N"]
tags: ["深度学习"]
--- 

# Mixed Precision Training

介绍混合精度训练之前先回顾计算机组成原理的两个知识点：

![](http://img.xilyfe.top/img/20260119120906809.png)

浮点数在计算机上面是以 "sign + exp + digits" 的格式存储的，exp 的大小决定了浮点数范围，digits 的大小决定了浮点数的精度。

---

在训练大型 DNN 的时候，如果采用 FP32 很可能遇到 CUDA 内存溢出的问题，这时候可以尝试把参数的精度从 FP32 调成 FP16，但是这个方法也存在问题：

1. 参数精度下降导致模型性能下降（实际上精度损失影响很小）
2. 如果参数小于 FP16 的表示范围就会变成 0，大于就会变成 NAN。

![](http://img.xilyfe.top/img/20260119120918931.jpg)

第二个问题是最严重的，NVIDIA 博客中的一幅图表示在 FP16 中这些梯度接近一半都会直接设为 0。

---

一个朴素的想法就是：

1. 拷贝模型的 FP32 参数为 Master Weight 保持不动
2. 将 FP32 的参数转为 FP16 用于前向计算和梯度计算
3. 将梯度转为 FP32，用来更新 Master Weight
4. 循环 1-3 步

这个想法没有解决根本问题，Forward 和 Backward 确实采用 FP16 减少内存占用了，但是精度变小还是可能导致梯度变成 0。

假设有：

- 模型：单参数 `w`（用 FP32 存储主参数，避免累积误差），初始值 `w=0.0000001`；
- 输入 `x=1.0`，真实标签 `y=0.000000103`（故意让预测值和真实值接近，制造微小梯度）；
- 损失函数：`L = 0.5*(w*x - y)^2`（MSE 加 0.5 是为了导数简洁）；
- 真实梯度：`g_true = (w*x - y)*x`（；

那么计算梯度 $g=(wx-y)*x=(0.0000001*1-000000103)*0.0000001=-3e-8$，超过 FP16 的表示范围，会直接表示为 0，就没法更新模型了。

所以新的想法就是，在计算梯度之前给他乘一个缩放因子，这样计算出来就在 FP16 的表示范围只能，然后变成 FP32 之后再除以缩放因子变回去：

1. 拷贝模型的 FP32 参数为 Master Weight 保持不动
2. 将 FP32 的参数转为 FP16 用于前向计算
3. 乘上缩放因子，然后计算梯度
4. 将梯度转为 FP32，除以缩放因子，再用来更新 Master Weight
5. 循环 1-4 步

```python
for epoch in epochs:
	for inp, tgt in data:
		optimizer.zero_grad()
		
		with auto_cast(device="cuda", dtype=torch.float16):
			out = model(inp)
			loss = loss_fn(out, tgt)
		
		scaler(loss).backward()
		scaler.step(optimizer)
		scaler.update()
```

---

还有一种解决方法就是用 BF16，BF16 也是以 2 字节存储，但是它将 digits 的长度减少让位给 exp，也就是说它牺牲了精度提高了表示范围。

![](http://img.xilyfe.top/img/20260119120927981.png)


# PEFT

PEFT 是 Parameters Efficiently Finetune，参数高效微调，指的是只更新参数集的一部分。


## LoRA

LoRA 的实现思路就是用低秩矩阵表示增量矩阵，类似自注意力机制里面的 Q、K、V。

假设存在线性层 $y=Wx$ 并且 $W \in R^{m \times n}$ ，那么可以将其分解为 $B \in m \times r$ 和 $A \in r \times n$ ：

$$
W =  BA
$$
因此我们可以冻结原模型参数不变，仅微调(训练)低秩矩阵的参数：

$$
Y = Wx + BAx \times alpha
$$

alpha 的取值一般为 1，它取决于是否需要大幅度改变模型：如果需要新学习的知识原模型没见过可以设为大于 1 的值。

> 实验证明将 LoRA 应用于 Q、V 矩阵效果最好。

---

SVD 分解：

任意矩阵 ΔW 都可以写成：

$$
\Delta W = U\Sigma V^T
$$

- $U \in n \times n$ 
- $V \in m \times n$ 
- $\Sigma$ 奇异值对角矩阵

通过低秩近似，我们可以得到：

$$
\Delta W \approx U_r \Sigma_r V_r^T
$$

和 LoRA 公式对比就有：

$$
BA \longleftrightarrow U_r \Sigma_r V_r^T
$$

所以 LoRA 的核心思路是：**利用 “SVD 说明低秩近似存在” 这个事实，用 SGD 直接在“低秩空间”里找最优解。**

Q：既然可以通过 SVD 分解得到低秩矩阵为什么还需要用 B 和 A 去学习呢？
A：因为还得通过 Fulltune 得到 $\Delta W$ 然后再分解，虽然参数占用减少了，但是计算量还增加了。

```python
class LoRALinear(nn.Module):
    
    def __init__(
        self, 
        base_layer: nn.Linear,
        r: int,
        alpha: float,
        dropout: float
    ):
        self.rank = r
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        # 注意数学公式中 x 都是列向量
        # 但是这里 x.shape = (bsize, len, in_features)
        # 所以 y=x*(BA)^T BA.shape = (out_feature, in_feature)
        if r > 0:
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.scaling = alpha / r
            
            # 初始化
            nn.init.kaiming_normal_(self.lora_A, a=0.001)
        
        self.dropout = nn.Dropout(dropout) if r > 0 else nn.Identity()
        
        # freeze parameters
        self.base_layer.weight.requires_grad = False
        self.base_layer.bias.requires_grad = False
        
    
    def forward(self, x):
        output = self.base_layer(x)
        
        if self.rank > 0:
            output = output + self.scaling * (x @ (self.lora_B @ self.lora_A).T)
            
        return self.dropout(output)
```

**LoRA 的初始化**

从代码中可以看到 LoRA 对 A、B 矩阵进行初始化时，对 A 才用 kaiming 初始化，对 B 矩阵直接初始化为 0。

- 让LoRA的更新 $\Delta W=BA$ 接近于 0 矩阵，从而不破坏预训练权重的行为。如果 B 初始化为 0 → $\Delta W=0 \times A$ 矩阵, 训练一开始，LoRA相当于没起作用，模型行为和原始预训练模型完全一致，非常稳定。但是如果 A 也初始化为 0 → 永远都是0，梯度更新不了，所以A必须有随机初始值，让梯度能正常回传到 B。
- 在前向传播中，低秩更新实际走的路径是：x → A → (scale) → B

**两种 LoRA 方式**

1. merge：训练完，把 $\Delta W=BA$ 直接加到原始权重 W 上。
2. adapter：训练完后仍然把A、B矩阵单独保存，推理时先加载大模型，再动态把 LoRA adapter 加载进来。