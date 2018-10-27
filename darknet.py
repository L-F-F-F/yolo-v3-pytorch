from __future__ import division

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


def creat_modules(blocks):
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
        module = nn.Sequential  # 每个模块可能包含多个层，Sequential顺序串联起来

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
                pad = (kernel_size - 1) // 2 #整数除法
            else:
                pad = 0

            #卷积层
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(index),conv)

            #batch norm 层，标准化操作，训练时计算均值方差，验证时用方差验证
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),bn)

            if activation == "leaky":
                acti = nn.LeakyReLU(0.1,inplace=True)   #负斜率0.1的relu，是否选择覆盖运算
                module.add_module("leak_{0}".format(index),acti)
# cfg = parse_cfg('yolov3.cfg')
# creat_modules(cfg)
# print(cfg)
