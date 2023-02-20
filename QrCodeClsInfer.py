# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/6/6 20:20
# @Author : liumin
# @File : QrCodeClsInfer.py

from __future__ import print_function, division

from random import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import cv2
import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def random_crop(img, dst_hw=(72, 444)):
    h, w = img.shape[:2]
    if h < dst_hw[0] or w < dst_hw[1]:
        return img

    padh, padw = h - dst_hw[0], w - dst_hw[1]  # hw padding
    padh /= 2
    padw /= 2  # divide padding into 2 sides
    top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
    left, right = int(round(padw - 0.1)), int(round(padw + 0.1))

    top = random.randint(0, top)
    bottom = random.randint(0, bottom)
    left = random.randint(0, left)
    right = random.randint(0, right)

    img = img[top:(h-bottom), left:(w-right), :]
    return img

def padding_resize(img, dst_hw=(72, 444), fill=0):
    h, w = img.shape[:2]
    padh, padw = dst_hw[0] - h, dst_hw[1] - w  # hw padding
    if padh < 0:
        padh_cut = (-padh) // 2
        img = img[padh_cut:(padh_cut+dst_hw[0]), :, :]
    if padw < 0:
        padw_cut = (-padw) // 2
        img = img[:, padw_cut:(padw_cut+dst_hw[1]), :]

    h, w = img.shape[:2]
    padh, padw = dst_hw[0] - h, dst_hw[1] - w  # hw padding
    padh /= 2
    padw /= 2  # divide padding into 2 sides
    top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
    left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
    img_paded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill)  # add border
    return img_paded

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


class DataFactory(Dataset):
    def __init__(self, root='', stage='train', input_size=(32, 32), padding_resize=False):
        super(DataFactory, self).__init__()
        self.stage = stage
        self._imgs = []
        self._imgLabel = []
        self._img_name = []
        self.input_size = input_size
        self.padding_resize = padding_resize

        if self.stage == 'test':
            paths = glob.glob(os.path.join(root, '*.jpg'))
            for imgPath in tqdm(paths):
                cvimg = cv2.imread(imgPath)
                self._imgs.append(cvimg)
                self._img_name.append(imgPath)
        else:
            classes = [d.name for d in os.scandir(root) if d.is_dir()]
            for root, fnames, _ in sorted(os.walk(root)):
                for fname in sorted(fnames):
                    paths = glob.glob(os.path.join(root, fname, '*.jpg'))
                    for imgPath in paths:
                        cvimg = cv2.imread(imgPath)
                        self._img_name.append(imgPath)
                        self._imgs.append(cvimg)
                        self._imgLabel.append(classes.index(fname))
            print(f'{self.stage}: {classes}:datasets:', len(self._imgs))

    def __getitem__(self, idx):
        image = self._imgs[idx]
        file_name = self._img_name[idx]

        if self.stage == "train":
            if self.padding_resize:
                image = padding_resize(image, self.input_size)
            else:
                image = cv2.resize(image, self.input_size)

            if random.random() < 0.5:
                image = cv2.flip(image, 1)  # #这里用到的是水平翻转
            if random.random() < 0.5:
                image = cv2.flip(image, 0)  # 这里用到的是垂直翻转
            if random.random() < 0.2:
                image = cv2.flip(image, -1)  # 这里用到的是水平垂直翻转
            '''
            # Albumentations
            if random.random() < 0.01:
                ksize = random.choice([3, 5, 7])
                image = cv2.GaussianBlur(image, (ksize, ksize), 0)
            if random.random() < 0.01:
                image = cv2.medianBlur(image, random.choice([3, 5, 7]))
            if random.random() < 0.01:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            '''
            '''
            angle = random.uniform(-2, 2)
            image = imutils.rotate_bound(image, angle)
            if image.shape[0] != self.input_size[1] or image.shape[1] != self.input_size[0]:
                image = cv2.resize(image, self.input_size)
            '''

            # HSV color-space
            augment_hsv(image, hgain=0.015, sgain=0.7, vgain=0.4)

            image = image.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image.astype(np.float32)).div_(255.0)

            return image, self._imgLabel[idx]
        else:
            if self.padding_resize:
                image = padding_resize(image, self.input_size)
            else:
                image = cv2.resize(image, self.input_size)
            image = image.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image.astype(np.float32)).div_(255.0)

            if self.stage == "test":
                return image, self._imgs[idx], self._img_name[idx]
            else:
                return image, self._imgLabel[idx]


    def __len__(self):
        return self._imgs.__len__()


class InferData():
    def __init__(self):
        super(InferData, self).__init__()

        self.model_path_ft = '/home/lmin/pythonCode/ClassificationCodes/weights/qrcode_all.pth'

        self.model_ft = models.shufflenet_v2_x1_0(pretrained=False)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, 2)
        self.model_ft.load_state_dict(torch.load(self.model_path_ft), strict=False)


        self.batch_size = 1
        self.num_workers = 8
        self.input_size = (288, 288)  # (width,height)

        self.device = torch.device("cuda:0")
        self.model_ft = self.model_ft.to(self.device)
        self.model_ft.eval()

    def run(self, img_dir):
        test_datasets = DataFactory(img_dir, stage='test', input_size=self.input_size)
        test_dataloader = DataLoader(test_datasets, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers)

        preds = []
        names = []
        for inputs, _, name in test_dataloader:
            inputs = inputs.to(self.device)
            outputs = self.model_ft(inputs)
            _, pred = torch.max(outputs, 1)
            pred_array = pred.cpu().numpy().tolist()
            preds.extend(pred_array)
            names.extend(name)
        return preds, names

if __name__ == '__main__':
    inferData = InferData()

    img_dir = '/home/lmin/data/classifications/qrcode_all/train/OK'

    import time

    since = time.time()
    preds, names = inferData.run(img_dir)
    time_elapsed = time.time() - since
    print("Time used:", time_elapsed)
    if 'NG' in img_dir:
        print(1 - np.sum(preds)/len(preds))
    else:
        print(np.sum(preds) / len(preds))
    print('*' * 50)

    preds = np.array(preds)
    names = np.array(names)
    if 'NG' in img_dir:
        idx = np.where(preds == 1)
    else:
        idx = np.where(preds == 0)
    print(names[idx])

    print('finished!')
