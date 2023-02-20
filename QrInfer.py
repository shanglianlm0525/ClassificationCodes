# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/3/18 14:45
# @Author : liumin
# @File : QrInfer.py

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import os
import torch.nn.functional as F
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def infer():
    device = torch.device("cpu")
    output_num = 2
    weight_path = 'weights/3_16_24_1.000.pth'
    img_path = '000252.jpg'
    input_size = (64, 64)

    model_ft = models.shufflenet_v2_x0_5(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, output_num)
    # print(model_ft)
    model_ft = model_ft.to(device)

    model_ft.load_state_dict(torch.load(weight_path))
    model_ft.eval()

    image = cv2.imread(img_path)
    image = cv2.resize(image, input_size)
    image = image.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image.astype(np.float32)).div_(255.0)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.set_grad_enabled(False):
        outputs = model_ft(image)
        _, preds = torch.max(outputs, 1)
        print(outputs, F.softmax(outputs, 1), preds)


if __name__ == '__main__':
    infer()
    print('finished!')
