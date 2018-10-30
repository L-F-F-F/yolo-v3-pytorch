# yolo-v3-pytorch
---
AlexeyAB的[yolo-v3](https://github.com/AlexeyAB/darknet)基于C语言的darknet框架。yolo-v3的pytorch搭建英文教程[How to implement a YOLO (v3) object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)，原作者讲的很详细，网上其他部分教程都是基于这篇博客改写的（有些地方英文原版比中文翻译版好理解），教程分为五部分：

	1、YOLO 的工作原理
	2、创建 YOLO 网络层级
	3、实现网络的前向传播
	4、objectness 置信度阈值和非极大值抑制
	5、设计输入和输出管道
[机器之心](https://www.jiqizhixin.com/users/7f316f0c-8f72-4231-bb30-0eb1dd5a5660)有两篇不错的中文翻译教程：[从零开始PyTorch项目：YOLO v3目标检测实现](https://www.jiqizhixin.com/articles/2018-04-23-3)，和[下篇](https://www.jiqizhixin.com/articles/042602)。相关github上源码[ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)，基于英文教程，与教程的代码略有不同。