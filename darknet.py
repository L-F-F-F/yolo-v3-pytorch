from __future__ import division

from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def parse_cfg(cfgfile):
    '''这里的思路是解析 cfg，将每个块存储为词典。
    这些块的属性和值都以键值对的形式存储在词典中。'''
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]  # 删去空行
    lines = [x for x in lines if x[0] != '#']  # 删去注释行
    lines = [x.rstrip().lstrip() for x in lines]  # 删去结尾、开头的空白符等

    blocks = []  # 返回值，每个块是一个字典，再构成一个列表
    block = {}
    for line in lines:
        if line[0] == "[":  # 新的块
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()  # index从1到倒数第2个作为type，删去结尾空白符
        else:  # 数字赋值行
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


def create_modules(blocks):
    '''我们的函数将会返回一个 nn.ModuleList。这个类几乎等同于一个包含 nn.Module 对象的普通列表。
    然而，当添加 nn.ModuleList 作为 nn.Module 对象的一个成员时（即当我们添加模块到我们的网络时），
    所有 nn.ModuleList 内部的 nn.Module 对象（模块）的 parameter 也被添加作为 nn.Module 对象
    （即我们的网络，添加 nn.ModuleList 作为其成员）的 parameter。
    当我们定义一个新的卷积层时，我们必须定义它的卷积核维度。虽然卷积核的高度和宽度由 cfg 文件提供，
    但卷积核的深度是由上一层的卷积核数量（或特征图深度）决定的。需要持续追踪被应用卷积层的卷积核数量。
    变量 prev_filter 来做这件事。初始化为 3，因为图像有对应 RGB 通道的 3 个通道。
    路由层（route layer）从前面层得到特征图（可能是拼接的）。如果在路由层之后有一个卷积层，
    那么卷积核将被应用路由层得到的特征图。不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。
    随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上。'''

    net_info = blocks[0]  # [net]块
    module_list = nn.ModuleList()
    prev_filters = 3  # 前一层的卷积核深度
    output_filters = []  # 输出卷积核数量 列表

    # 思路是迭代模块的列表，并为每个模块创建一个 PyTorch 模块
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()  # 每个模块可能包含多个层，Sequential顺序串联起来

        # 卷积层
        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])  # 卷积通道数
            padding = int(x["pad"])  # 填充数量
            kernel_size = int(x["size"])  # 卷积核大小
            stride = int(x["stride"])  # 步长

            if padding:
                pad = (kernel_size - 1) // 2  # 整数除法
            else:
                pad = 0

            # 添加卷积层
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # batch norm 层，标准化操作，训练时计算均值方差，验证时用方差验证
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            if activation == "leaky":
                acti = nn.LeakyReLU(0.1, inplace=True)  # 负斜率0.1的relu，是否选择覆盖运算
                module.add_module("leak_{0}".format(index), acti)

        # 上采样层，函数由interpolate替代，放缩倍数scale_factor和size只能给一个，扩展输入矩阵
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            # 矩阵扩为2倍，最近邻nearest，线性linear，双线性bilinear 和 三线性trilinear
            upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            # 自0.4.0版本Upsample函数虚指定align_corners，对齐角落
            module.add_module("upsample_{}".format(index), upsample)

        # 路由层，获取之前层的拼接，有两种：1个数字，2个数字的
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])  # 第一个数字
            try:
                end = int(x["layers"][1])  # 如果有第二个数字
            except:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()  # 空层，连接作用 保证前向传播
            module.add_module("route_{0}".format(index), route)

            # 在路由层之后的卷积层会把它的卷积核应用到之前层的特征图（可能是拼接的）上。
            # filters 变量以保存路由层输出的卷积核数量，即两层叠加
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # 捷径层（跳过连接），将前一层的特征图添加到后面的层上
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            # 捷径层也使用空的层，因为它还要执行一个非常简单的操作（加）。
            # 没必要更新 filters 变量，因为它只是将前一层的特征图添加到后面的层上而已。

        # yolo 检测层
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")  # 锚点框
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]  # 取mask中提到的几对锚点框

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        # 统计
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    # net_info就是net层信息，再返回各层list
    return (net_info, module_list)


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


# 测试 parse_cfg 和 create_modules
# cfg = parse_cfg('yolov3.cfg')
# [net,mod] = create_modules(cfg)
# print(mod)

class Darknet(nn.Module):
    def __init__(self, cfgfile):  # 初始化解析了cfg文件
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA=False):  # 前向传播，x是输入
        modules = self.blocks[1:]  # blocks[0:]是net层
        outputs = {}  # 由于路由层和捷径层需要之前层的输出特征图，outputs 中缓存每个层的输出特征图。
        # 关键在于层的索引，且值对应特征图。

        write = 0  # flag：是否遇到第一个检测图,如果 write 是 0，则收集器尚未初始化。
        # 如果 write 是 1，则收集器已经初始化，我们只需要将检测图与收集器级联起来即可

        for i, module in enumerate(modules):
            module_type = (module["type"])

            # 卷积层、上采样层
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            # 路由层，获取之前层的连接
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:  # 正数的话，减当前层index
                    layers[0] = layers[0] - i

                if len(layers) == 1:  # 路由层如果只有1个数字
                    x = outputs[i + (layers[0])]

                else:  # 路由层两个数字，特征图连接
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    # print("map1{}",map1.shape)
                    # print(map2.shape)
                    x = torch.cat((map1, map2), 1)  # cat函数将两个特征图沿深度方向连起来

            # 捷径层（跳过连接），将前一层的特征图添加到from的层上
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            # 完成util的predict_transform函数之后
            # 检测层
            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors

                inp_dim = int(self.net_info["height"])  # 输入维度

                num_classes = int(module["classes"])  # 类别数

                x = x.data
                # print(x)
                if CUDA:
                    x = x.cuda()
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:  # 收集器尚未初始化
                    detections = x
                    write = 1

                else:  # 将检测图与收集器级联起来即可
                    detections = torch.cat((detections, x), 1)
            outputs[i] = x

        return detections

    def load_weights(self, weightfile):

        fp = open(weightfile, "rb")
        # 第一个 160 比特的权重文件保存了 5 个 int32 值，它们构成了文件的标头
        # 主版本数，次版本数，子版本数，4、5是训练期间网络看到的图像
        header = np.fromfile(fp, dtype=np.int32, count=3)
        if (header[0] * 10 + header[1] >= 2) and (header[0] < 1000) and (header[1] < 1000):
            sub_header = np.fromfile(fp, dtype=np.int32, count=2)
        else:
            sub_header = np.fromfile(fp, dtype=np.int32, count=1)
        header = np.append(header, sub_header)
        #上面几行与原博客不同，以兼容YOLO v2 v3的weights文件

        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0  # 追踪权重数组位置指针
        for i in range(len(self.module_list)):  # 迭代地加载权重文件到网络的模块上
            # 块包含第一块，模块不包含第一块，net块
            module_type = self.blocks[i + 1]["type"]

            # 卷积层
            if module_type == "convolutional":
                model = self.module_list[i]
                try:  # 是否有batch_normalize 批量标准化
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]
                # print(batch_normalize)
                if (batch_normalize):  # 有标准化
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()  # Batch Norm Layer 的权重的数量

                    # 加载权重
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # 根据模型权重的维度调整重塑加载的权重
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # 将数据复制到模型中
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else: # 如果没有标准化，只需要加载卷积层的偏置项

                    num_biases = conv.bias.numel()# 偏置数量

                    # 加载权重
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # 根据模型权重的维度调整重塑加载的权重
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # 将数据复制到模型中
                    conv.bias.data.copy_(conv_biases)

                # 最后，加载卷积层的权重
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


def get_test_input():  # 测试输入图片
    img = cv2.imread("testPics/14-9.jpg")
    # img = cv2.imread("testPics/dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # 输入维度416
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis, :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  # ndarray 转 Tensor 转float
    img_ = Variable(img_)  # 变量
    return img_


# 测试读cfg文件，输出的张量size为torch.Size([1, 10647, 24])，第一个维度为批量batch大小，单张图像
# 24行，包括4个边界框属性(bx,by,bh,bw)、1个objectness分数和19个类别分数
#10647 是每个图像中所预测的边界框的数量
# # modeltest = Darknet("cfg/yolov3.cfg")
# modeltest = Darknet("cfg/yolo-obj_4_416.cfg")
# inp = get_test_input()
# pred = modeltest(inp, torch.cuda.is_available())
# # print (pred)
# print(pred.size())

model = Darknet("cfg/yolo-obj_4_416.cfg")
model.load_weights("yolo-obj_4_416_28000.weights")
