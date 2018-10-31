from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

#predict_transform 函数把检测特征图转换成二维张量
#5个参数：prediction 输出、inp_dim 输入图像的维度、anchors 锚点、num_classes 类、CUDA。
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes#每个边界框有5+C个属性：坐标tx ty tw th Obj 类别共C类
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    #锚点的维度与net块的h和w属性一致，输入图像的维度和检测图的维度之商就是步长，
    # 所以用检测特征图的步长分割锚点
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    #对 (x,y) 坐标和 objectness 分数执行 Sigmoid 函数操作
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    #将网格偏移添加到中心坐标预测中
    grid = np.arange(grid_size)#0到grid_size-1，步长为1数列
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset