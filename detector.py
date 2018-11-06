from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


# 命令行
def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    # images（用于指定输入图像或图像目录）
    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)

    # det（保存检测结果的目录）
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)

    # batch大小
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)

    # objectness置信度
    parser.add_argument("--confidence", dest="confidence", help=
    "Object Confidence to filter predictions", default=0.5)

    # NMS阈值
    parser.add_argument("--nms_thresh", dest="nms_thresh", help=
    "NMS Threshhold", default=0.4)

    # cfg（替代配置文件
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolo-obj_4_416_small.cfg", type=str)

    # weights
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolo-obj_24000_4_416_small.weights", type=str)

    # reso（输入图像的分辨率，可用于在速度与准确度之间的权衡）
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

classes = load_classes("data/obj.names")
num_classes = len(classes)
# print(len(classes))
# print(classes)
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()

# 模型评估
model.eval()

