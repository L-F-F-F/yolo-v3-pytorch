from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


# predict_transform 函数把检测特征图转换成二维张量
# 5个参数：prediction 输出、inp_dim 输入图像的维度、anchors 锚点、num_classes 类、CUDA。
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    # grid_size = inp_dim // stride
    grid_size = prediction.size(2)
    bbox_attrs = 5 + num_classes  # 每个边界框有5+C个属性：坐标tx ty tw th Obj 类别共C类
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # 锚点的维度与net块的h和w属性一致，输入图像的维度和检测图的维度之商就是步长，
    # 所以用检测特征图的步长分割锚点
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # 对 (x,y) 坐标和 objectness 分数执行 Sigmoid 函数操作
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # 将网格偏移添加到中心坐标预测中
    grid = np.arange(grid_size)  # 0到grid_size-1，步长为1数列
    a, b = np.meshgrid(grid, grid)  # 网格

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # repeat
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset  # (x,y) 坐标加上偏移

    # 将锚点应用到边界框维度中
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
    # 2:4 是tw th

    # 将 sigmoid 激活函数应用到类别分数中：
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # 将检测图的大小调整到与输入图像大小一致，乘以stride变量（边界框属性根据特征图大小而定）
    # 如果特征图13x13 输入图像大小是 416 x 416，那么我们将属性乘 32，或乘 stride 变量
    prediction[:, :, :4] *= stride

    return prediction


# 输入为预测结果、置信度（objectness 分数阈值）、种类数 和 nms_conf（NMS IoU 阈值）
def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    # prediction中 低于objectness分数的每个边界框，其每个属性值都置0，即一整行
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)  # 保留了大于置信度的部分
    prediction = prediction * conf_mask

    # 框的中心坐标 转 左上角右下角，以计算IOU
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    write = False  # 标识尚未初始化输出

    for ind in range(batch_size):
        image_pred = prediction[ind]  # 每张图像 Tensor，[10647 x (5+类别数)]

        # 获取最高类别分数 的分数 以及 索引
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)

        # 不需要所有类别分数，仅保留最高分数的索引以及分数，替换 image_pred
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)  # cat 1 是列方向连接成tensor
        # 所以[10647 x 7]

        # 删除objectness置信度小于阈值的置0条目
        non_zero_index = (torch.nonzero(image_pred[:, 4]))
        try:  # 处理无检测结果的情况
            image_pred_ = image_pred[non_zero_index.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:  # PyTorch 0.4兼容
            continue

        # -1是 类别 的 index，获得一个图像的所有种类
        img_classes = unique(image_pred_[:, -1])

        # 按类别执行NMS
        for cls in img_classes:
            # 得到一个类别的所有检测
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # 对所有检测排序,按照objectness置信度
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            # 对于每一个检测,执行NMS 非极大值抑制
            for i in range(idx):

                try:  # 当前box之后所有boxes的IOU
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:  # image_pred_class[i+1,:]返回空张量
                    break

                except IndexError:  # image_pred_class移除部分后，idx索引越界
                    break

                # 清除IoU>阈值的检测
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # 移除0条目
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            #对图像中的类cls进行多次检测，重复batch_id
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            #一旦其被初始化，就将后续的检测结果与它连接起来。
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    #输出为 Dx8 的张量；其中 D 是所有图像中的「真实」检测结果，每个都用一行表示。
    # 每一个检测结果都有 8 个属性，即：该检测结果所属的 batch 中图像的索引、
    # 4 个角的坐标、objectness 分数、有最大置信度的类别的分数、该类别的索引。
    try:
        return output
    except:
        return 0


# 因为同一类别可能会有多个「真实」检测结果,unique获取任意给定图像中存在的类别
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)  # 去除重复数字,排序
    unique_tensor = torch.from_numpy(unique_np)  # 转tensor

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


# IOU计算 交并比
def bbox_iou(box1, box2):
    '''

    :param box1:边界框行，这是由循环中的变量 i 索引
    :param box2:多个边界框行构成的张量
    :return:边界框与第二个输入中的每个边界框的 IoU
    '''
    # 边框的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 获取交叉矩形的坐标
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # 交集部分面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # 并集部分面积
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou
