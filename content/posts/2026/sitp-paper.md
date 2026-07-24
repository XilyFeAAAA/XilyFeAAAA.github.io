---
title: 论文泛读
date: 2026-07-16T13:57:17+08:00
featuredImage: http://img.xilyfe.top/img/20260716140005073.png
authors:
  - Xilyfe
series:
  - 论文阅读
tags: []
lastmod: 2026-07-04T15:13:17+08:00
hidden: true
---
## Paper-1

- **题目**：Fast Forward Prediction of Metasurface Transmission Spectra Using Deep Learning
- **概要**：用一个 **ResNet-18 改的神经网络**替代传统电磁仿真器，从超表面单元结构直接预测透射光谱
- **数据**：用仿真器得到 1w 条数据，输入是一个**几何图像**，输出是光谱
- **模型**：单纯用 resnet-18 也就是 CNN 对图像进行卷积，然后加了一个 MLP 的分类头
- **总结**：一坨屎，没有借鉴意义

## Paper-2

- **题目**：Inverse design of polarization-insensitive all-dielectric BIC metasurface with dual Fano-resonances by deep learning
- **概要**：解决的是一个 **inverse design** 的问题，也就是从透射光谱曲线 → 器件参数。paper 创新点在于**把 1D 的连续曲线通过 GAF 编码进了 2D矩阵，保留了完整光谱形状**。然后通过双通道 CNN 和 MHA 做投影，得到器件参数。
- **数据**：300 维光谱曲线 → 3 维的器件参数（S/R/H）
- **模型**：用 GAF 编码把 300 维向量变成两个 56\*56 的灰度图，然后各自卷积+池化，之后把两个合并到一起做 MHA，最后 MLP 降维到 3 维。**单纯的用 MSE Loss 进行训练**
- **问题**：
	1. 论文解决的是 inverse design 不能照搬结构
	2. 它的参数空间比较小，大规模下不一定 work
- **借鉴**：
	1. GAF 的核心思想是"**给孤立的序列点显式注入两两关联的归纳偏置**"。你的 AutoEncoder 在压缩高维光谱时,可以借鉴这个思路——**在 AE 的 latent 或 loss 里加入'光谱点间自相关'的约束**,而不是把 1000 维当独立通道压。比如 loss 里加一项 Gram 矩阵重建误差,逼 AE 保住曲线的全局形态而非逐点数值。
	2. 用 MHA 建模多个 Fano 峰位置，单纯的 mlp 或者 cnn 不一定能捕捉长距离关系
	3. 余弦相似度给光谱形态打分

## Paper-3

- **题目**：**Itakura-Saito distance based autoencoder**
- **概要**：用 auto-encoder 给质谱数据降维时候，数值很小的峰容易被忽略，但实际很重要。创新点在于 loss 从 mse 改成了 is 距离，他可以让数值接近 0 的峰梯度也很大，强迫自编码器把小峰也能重建。
- **数据**：
- **模型**：auto-encoder+is loss
- **问题**：trade-off 是大峰的误差变大了
- **借鉴**：如果碰到小峰不识别问题，可能可以加一个 is 的 loss项？把系数设小一点。**加权关键区域**

>![image.png](http://img.xilyfe.top/img/20260716151833921.png)质谱数据类似这样，大部分是 0，少数位置突然一根尖峰。

## Paper-4

- **题目**：Nanophotonic Particle Simulation and Inverse Design Using Artificial Neural Networks

>暂时没用


## Paper-5

- **题目**：Deep learning the electromagnetic properties of metamaterials – a comprehensive review
- **借鉴**：
	1. 正向预测主流都是：全连接 DNN(FCNN)，有局部相关性用 CNN
	2. s 相邻频率点相关(因果线性响应)→ 1D-CNN 有效
	3. g 是像素化几何图 → 2D-CNN

## Paper-6

- **题目**：Deep learning in nano-photonics: inverse design and beyond
- **借鉴**：
	1. 迭代数据生成：对于难以预测的 Fano 峰样本，在器件参数附件多生成一些数据来训练，针对性的补充样本（有尖锐共振峰的样本、峰位变化很快的样本、峰宽很窄的样本）
	2. loss 里面加导数损失：一阶导 loss 可以约束 fano 峰的斜率，二阶导 loss 可以约束峰宽和谷的弯曲程度
	3. 通过数据集的光谱曲线，计算峰区位置，然后给 mse/导数权重 加权重。推理时不加权重，这不算数据泄露
	4. 更好的思路（maybe），是把背景和尖峰分开
		1. 用不同尺度的卷积找不同范围的特征信息，然后拼接起来
		2. 先预处理数据集，用平滑方法找到背景，和峰谷残差，然后训练时候用两个 head 一个预测背景，一个预测峰谷差值，最后加起来就是完整曲线了。每个 head 单独关注一部分，就不会让 fano峰被影响

## Paper-7

- **题目**：Optimizing Spectral Prediction in MXene-Based Metasurfaces Through Multi-Channel Spectral Refinement and Savitzky-Golay Smoothing

>暂时没用


## Paper-8

- **题目**：Deep learning for the design of photonic structures
- **概要**：一篇 DL 在光子结构应用的综述
- **借鉴**：
	1. MLP 应该作为 baseline 测试，然后做消融实验
	2. RNN/LSTM/GRU 可用于建模连续谱序列及其由多个共振模式产生的线形
	3. CNN+RNN 的组合已用于图像结构 → 光谱预测
	4. CNN 适合局部相关性的数据，可以试着给光谱。用多个尺寸的卷积核进行 CNN
	5. 评估时候除了 mse、r² 等指标，还需要考虑峰谷数量时候预测正确，余弦相似度等等

```text
E0: MLP + MSE
E1: MLP + derivative loss
E2: AE + MLP latent regression
E3: AE + 1D CNN decoder
E4: AE + multi-scale/dilated 1D ResNet decoder
E5: E4 + derivative + curvature loss
```


## Paper-9

- **题目**：Prediction of metasurface spectral response based on a deep neural network
- **概要**：一篇简单的 MLP 正向预测，结论就是 Adagrad 比 adam 什么效果好，就当测 baseline 了


## Paper-10

- **题目**：BIDIRECTIONAL GRU BASED AUTOENCODER FOR DIMENSIONALITY REDUCTION IN HYPERSPECTRAL IMAGES
- **概要**：创新性的用 BiGRU AE 把高维光谱曲线降维到 latent，核心思想是 **光谱波段不是互相独立的特征，而是有前后连续关系的序列；因此应该用双向 GRU**。但是项目的目标是把一个高维光谱做一个下游分类任务，所以对于局部的细节不是非常在意
	- **idea**：RNN 容易出现长序列梯度消失的问题，然后光谱数据每个点都和前后状态相关，所以用双向 GRU
- **模型**：auto-encoder+is loss
- **问题**：这篇论文的目标是下游分类，所以和我 参数→光谱的目标不一样
- **借鉴**：在 AE 中用双向 GRU 替代普通 FC/单向 RNN，让每个波段表示同时利用其前后谱段上下文。


目前的一些技术路线：
1. 直接 regression：MLP
2. 2 stage latent regression：
	1. ae reconstruction
	2. regression
3. end-to-end：parameter->latent->curve
4. 背景+峰谷分离：
	1. parameter→背景
	2. parameter→峰谷
	3. 相加
5. 混合专家
	1. 参数 → 判断谱型/峰模式
	2. 参数 → 对应专家预测光谱






## 对比方案

1. 直接 10 维器件参数 MLP 到 2000 维光谱曲线作为 baseline
2. auto-encoder 降维 + mlp regression：主要改进集中在 ae 方面，可以尝试的方案包括：
		1. 先用一个最普通的做baseline
		2. 在 ae 的 loss 里面加东西，一阶导，二阶导，系数默认0.01慢慢调
		3. 根据光谱曲线，计算得到峰谷位置，然后再训练时候给这些位置加mse系数：使用 `scipy.signal.find_peaks` 预先找出每条曲线的峰谷索引，生成一个与光谱同维度的权重向量 $W$（峰谷处赋予 5~10 倍权重，背景为 1）。
		4. CNN、multi-scal CNN
		5. 补充针对性样本训练
3. 背景+峰谷分离：
    1. parameter→背景
    2. parameter→峰谷
    3. 相加
>- **数据预处理**：利用传统信号处理算法（如 **Savitzky-Golay 滤波器**、大窗口中值滤波或多项式拟合），把 2000 维的原始光谱平滑化，提取出“纯背景曲线” $y_{\text{bg}}$。原始曲线减去背景，得到“纯峰谷残差” $y_{\text{peak}}$。 
>- **模型设计**：   
    - **Network A (MLP)**：10D 参数 $\to$ 2000D 背景（非常好拟合，普通 MLP 即可轻松搞定）。
    - **Network B (AE/CNN)**：10D 参数 $\to$ 2000D 峰谷残差（让模型集中全部注意力去捕捉零点附近的尖锐突变）。       
    - **最终输出**：$\hat{y} = \hat{y}_{\text{bg}} + \hat{y}_{\text{peak}}$。