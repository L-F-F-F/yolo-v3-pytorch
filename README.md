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

权重读取这块比较容易出错，按照原作者的方法在加载官方文件 `yolov3.cfg` 和 `yolov3.weights` 时不会出错，但是再使用自己的cfg文件以及训练好的weights文件时，可能会出现类似的错误

>RuntimeError: invalid argument 2: size '[72 x 256 x 1 x 1]' is invalid for input with 18431 elements at ..\src\TH\THStorage.c:41

原因是权重函数的 Header 位数读取出错。从yolov3开始，subversion数的类型是 `size_t` 而不是int32。同时注意，对于Joseph Redmon提供的官方YOLO v3权重文件，Header长度为5 x 32位，而在YOLO v2中，它是4x32位长。 **解决方法** 将 `darknet.py` 中的函数 `load_weights`修改 `count`为4，即修改 `header = np.fromfile(fp, dtype=np.int32, count=5)` 为 `header = np.fromfile(fp, dtype=np.int32, count=4)`

或者将 `load_weights()` 函数的开头修改为如下代码，这样可兼容不同版本的weights文件：

	def load_weights(self, weightfile):
	
	    #Open the weights file
	    fp = open(weightfile, "rb")
	
	    #The first 4 values are header information 
	    # 1. Major version number
	    # 2. Minor Version Number
	    # 3. Subversion number 
	    # 4. IMages seen 
	    header = np.fromfile(fp, dtype = np.int32, count = 3)        
	    if (header[0]*10+header[1] >=2) and (header[0] < 1000) and (header[1] < 1000):
	        sub_header = np.fromfile(fp, dtype = np.int32, count = 2)
	    else:
	        sub_header = np.fromfile(fp, dtype = np.int32, count = 1)
	    
	    header = np.append(header,sub_header)
	
	    self.header = torch.from_numpy(header)
	    
	    self.seen = self.header[3]
	    
	    #The rest of the values are the weights
	    # Let's load them up
	    weights = np.fromfile(fp, dtype = np.float32)

在 Ayoosh Kathuria 的 github 上有个 [issue](https://github.com/ayooshkathuria/pytorch-yolo-v3/issues/19) 专门讨论这个问题。

---
## 检测
权重文件上传至[百度云](https://pan.baidu.com/s/1KT8voQAUz8bOIHdmM59_1A)，密码275z，适用于检测较高分辨率的军用飞机，不再适用COCO等，包含19类军机，输入图片 768*768 `F18 B1 C130 F15E KC135 F16 A10 Fighter F4 T38 F22 C17 F15 E3 KC10 B52 B2 T41 Boeing`

运行`python detector.py`默认检测 testPics/14-9.jpg，`python detector.py --images testPics/` 检测 testPics 目录下所有图片，检测结果保存在 det 目录下。

	命令行
	--images		修改待检测图片路径，默认 testPics/14-9.jpg
	--det			保存检测结果的目录，默认 det
	--bs			batch大小，默认 1
	--confidence	objectness置信度，默认 0.5
	--nms_thresh	NMS阈值，默认 0.4
	--cfg			cfg文件，默认 cfg/yolo-obj_4_416.cfg
	--weights		权重文件，默认 yolo-obj_4_416_28000.weights
	--reso			输入图像的分辨率，默认 768
![](https://i.imgur.com/iRGorvp.jpg) ![](https://i.imgur.com/MWpoN3D.jpg)
