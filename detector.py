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
                        default="testPics/14-9.jpg", type=str)

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
                        default="cfg/yolo-obj_4_416.cfg", type=str)

    # weights
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolo-obj_4_416_28000.weights", type=str)

    # reso（输入图像的分辨率，可用于在速度与准确度之间的权衡）
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="768", type=str)

    return parser.parse_args()

colors = pkl.load(open("pallete", "rb")) # 可供选择的颜色列表
# 绘制边界框
def writeBox(x, results,color = random.choice(colors)):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)  # -1表示填充的矩形
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img



# -------------------
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

read_dir = time.time()  # 记录时间的检查点
# 从磁盘读取图像或从目录读取多张图像
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

if not os.path.exists(args.det):  # 保存检查结果路径
    os.makedirs(args.det)

# 用OpenCV加载多张图片图像
load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

# 转成PyTorch图像格式
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

# 包含原始图像的维度的列表
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

if CUDA:
    im_dim_list = im_dim_list.cuda()

# 创建batch
leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                            len(im_batches))])) for i in range(num_batches)]

write = 0
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    # 载入图片
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    # prediction = model(Variable(batch, volatile=True), CUDA)
    prediction = model(Variable(batch), CUDA)

    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num

            print("{0:20s} 检测用时 {1:6.3f} 秒".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("检测到对象:", " "))
            print("----------------------------------------------------------")
        continue

    prediction[:, 0] += i * batch_size  # 将batch索引转换成imlist索引

    if not write:  # 初始化output
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
        im_id = i * batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} 检测用时 {1:6.3f} 秒".format(image.split("/")[-1], (end - start) / batch_size))
        print("{0:20s} {1:s}".format("检测到对象:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()

# 在图像上绘制边界框
try:
    output
except NameError:
    print("不存在检测结果")
    exit()

# 输出边界框对应网络输入大小，需要将边界框属性转换到图像的原始尺寸
im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
output[:, 1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

output_recast = time.time()

# 绘制边界框
class_load = time.time()

draw = time.time()

# 原地修改 loaded_ims 之中的图像
list(map(lambda x: writeBox(x, loaded_ims), output))

# 保存检测结果图像,det_图像名
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

# 将带有检测结果的图像写入det_names中的地址
list(map(cv2.imwrite, det_names, loaded_ims))
end = time.time()

# 显示输出时间的总结
print("总结")
print("----------------------------------------------------------------")
print("{:25s} {}".format("任务", "所用时间(s)"))
print()
print("{:25s} {:.3f}".format("读入目录", load_batch - read_dir))
print("{:25s} {:.3f}".format("加载batch", start_det_loop - load_batch))
print("{:25s} {:.3f}".format("检测(" + str(len(imlist)) + "张图)", output_recast - start_det_loop))
print("{:25s} {:.3f}".format("输出处理", class_load - output_recast))
print("{:25s} {:.3f}".format("绘制边界框", end - draw))
print("{:25s} {:.3f}".format("平均检测时间", (end - load_batch) / len(imlist)))
print("----------------------------------------------------------------")

torch.cuda.empty_cache()


