# Segmentation-Based-Surface-Defect-Detection
This is a deep learning application project in the industrial field, intended to detect defects on the workpiece surface. The code is based on keras and runs on GPU.

这是一个应用深度学习方法解决工业问题的项目：基于分割神经网络的表面缺陷检测。代码基于Keras编写，支持GPU加速。

论文：[Segmentation-Based Deep-Learning Approach for Surface-Defect Detection](https://arxiv.org/abs/1903.08536)。

## 研究目标<br>
工件疲劳、损坏的现象广泛存在于工业界，鉴于其潜在的安全隐患，一种低成本高效率的表面缺陷检测方法亟待开发。

基于卷积神经网络的深度学习方法天然适合于解决此类问题，如下是研究目标。

<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%BC%BA%E9%99%B7%E6%A3%80%E6%B5%8B1.png" alt="Sample"  width="700">
</p>

工件表面的高清图像经预处理后使用神经网络做预测，输出两个信息：1.表面是否有缺陷；2.如有缺陷则使用分割方法将其标出。

## 数据集<br>
本项目使用的是公开的[KolektorSDD 数据集](http://www.vicos.si/Downloads/KolektorSDD)。

该数据集样本较少，共有 399 个样本，正样本与负样本的数量分别为 52 和 347，图片形状为 500 x 1267 (1267 为典型值，还有其他尺寸)。

以下是数据集中的一些样本图像：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%BC%BA%E9%99%B7%E6%A3%80%E6%B5%8B2.png" alt="Sample"  width="500">
</p>

每一张工件的图像都有一个对应的等尺寸的二值化 mask 图像，为人工标注的表面缺陷。无缺陷的样本，其 mask 是全黑的图像。

考虑到样本少且比例不均，需要对数据进行样本增强与扩充（未完全按照原文实现）。

我的做法是将原图裁减为 500 x 500 的正方形图像，并随机做旋转、翻转、缩放等，同时调整图像亮度并添加噪声，这样可以有效增加样本数量。

为了使样本比例均衡，我控制了正负样本生成的数量比为 1:1。详见 data_manager.py。

以下是一些正样本和对应的 mask 的示意图。

<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%BC%BA%E9%99%B7%E6%A3%80%E6%B5%8B51.png" alt="Sample"  width="500">
</p>

<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%BC%BA%E9%99%B7%E6%A3%80%E6%B5%8B52.png" alt="Sample"  width="500">
</p>


## 网络结构<br>
下图为原文作者设计的网络结构：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%BC%BA%E9%99%B7%E6%A3%80%E6%B5%8B3.png" alt="Sample"  width="700">
</p>

从结构图可以看出几个明显特征：有两个网络主体：分割网络和决策网络；有两个输出：分割输出和分类输出。同时，区别于一般做法，该结构采用了尺寸相对较大的卷积核：5x5 和 15x15；只有下采样层而没有上采样层。

原文的解释是，采用较大的卷积核可以获得更大的感受野；省去上采样层可以节省很多参数与计算量，更适合于工业落地。

我在复现论文是做了一些改进：将 15x15 的卷积核改为 5x5 的卷积核省去了一半的参数；且在分类输出层增加了 softmax 激活层（原文采用线性激活函数）。

## 训练<br>
由于网络结构中没有设计上采样层，其输出将小于输入尺寸。因此在准备喂入数据时，要缩放处理 mask 图片。

考虑到表面缺陷形状较窄，缩放后可能分辨率不够，需要在缩放之前对 mask 图像做膨胀处理，使缺陷更明显，这样再经过缩放可以保留更多有效信息。

原文将两个主体网络分开、分步训练。原因是网络参数较大难以训练，同时两部分网络的损失权值不好确定。

我设计的训练步骤如下图所示：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%BC%BA%E9%99%B7%E6%A3%80%E6%B5%8B4.png" alt="Sample"  width="600">
</p>

这两步训练主要更改了分割网络的目标函数：mse-->binary_crossentropy；同时减小了学习率。分割网络与决策网络的误差权重比为 1:0.5；分割网络与决策网络均采用 softmax 激活函数。

## 推理<br>
推理的过程和训练的主要区别在于，推理前后要保证图片尺寸不改变。

一个自然而然的想法就是将待检测图剪裁成若干份子图，分别输入网络做检测，再对输出做进一步处理。

剪裁的目标形状为 500x500，形状不匹配的再做缩放处理。

对于分类网络的输出，如果所有子图的分类都是无缺陷，则整体无缺陷；如果有一张子图有缺陷，则此样本为缺陷样本。

对于分割网络的输出，其形状均为 62x62，需将其缩放为原先的形状再逐个拼接，最终得到与原图相同形状的 mask。

上述步骤可由下图来表示：

<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%BC%BA%E9%99%B7%E6%A3%80%E6%B5%8B10.png" alt="Sample"  width="600">
</p>

## 结果<br>
这是在 80 个测试样本上的检测统计结果：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%BC%BA%E9%99%B7%E6%A3%80%E6%B5%8B7.png" alt="Sample"  width="600">
</p>

这是在数据集所有样本上的检测统计结果：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%BC%BA%E9%99%B7%E6%A3%80%E6%B5%8B8.png" alt="Sample"  width="600">
</p>

这是一些输出结果展示：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%BC%BA%E9%99%B7%E6%A3%80%E6%B5%8B9.png" alt="Sample"  width="600">
</p>

## 结论<br>
* 1.查全率较高，大部分缺陷样本均能正确分类，漏检少
* 2.查准率待提升，需减少对负样本的错误分类，有误检
* 3.该神经网络对噪声不敏感，对块状、条状斑纹敏感
* 4.分割结果大部分较好，对很细的缺陷分割效果不佳
