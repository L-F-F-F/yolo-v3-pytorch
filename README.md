# yolo-v3-pytorch
---
## 简介
AlexeyAB的[yolo-v3](https://github.com/AlexeyAB/darknet)基于C语言的darknet框架。yolo-v3的pytorch搭建英文教程[How to implement a YOLO (v3) object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)，作者Ayoosh Kathuria讲的很详细，网上其他部分教程都是基于这篇博客改写的（有些地方英文原版比中文翻译版好理解），教程分为五部分：

*	1、YOLO 的工作原理
*	2、创建 YOLO 网络层级
*	3、实现网络的前向传播
*	4、objectness 置信度阈值和非极大值抑制
*	5、设计输入和输出管道

[机器之心](https://www.jiqizhixin.com/users/7f316f0c-8f72-4231-bb30-0eb1dd5a5660)有两篇不错的中文翻译教程：[从零开始PyTorch项目：YOLO v3目标检测实现](https://www.jiqizhixin.com/articles/2018-04-23-3)，和[下篇](https://www.jiqizhixin.com/articles/042602)。Ayoosh Kathuria也在github上传了源码[ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)，与博客的代码在细节上略有不同。可根据自己需要自行修改。

---
## 权重存储方式
![](https://i.imgur.com/BsBDCaA.png)

上图展示了权重存储方式。首先，权重只属于两种类型的层，即批归一化层（batch norm layer）和卷积层。这些层的权重储存顺序和配置文件中定义层级的顺序完全相同。所以，如果一个 convolutional 后面跟随着 shortcut 块，而 shortcut 连接了另一个 convolutional 块，则你会期望文件包含了先前 convolutional 块的权重，其后则是后者的权重。当批归一化层出现在卷积模块中时，它是不带有偏置项的。然而，当卷积模块不存在批归一化，则偏置项的「权重」就会从文件中读取。
