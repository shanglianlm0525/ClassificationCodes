# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/3/16 10:04
# @Author : liumin
# @File : QrCodeCls.py

from __future__ import print_function, division

import math
import random

import imutils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import os
import copy
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib
from torch.utils.data import Dataset
import cv2
import glob

from ClassificationCodes.ema import ModelEMA

matplotlib.use('pdf')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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

            # angle = random.uniform(-1, 1)
            # image = imutils.rotate_bound(image, angle)
            # if image.shape[0] != self.input_size[1] or image.shape[1] != self.input_size[0]:
            #     image = cv2.resize(image, self.input_size)

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
            # image = image.half()

            if self.stage == "test":
                return image, self._imgs[idx], self._img_name[idx]
            else:
                return image, self._imgLabel[idx]

    def __len__(self):
        return self._imgs.__len__()


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, device):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda(device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class TrainQrcodeData():
    def __init__(self):
        super(TrainQrcodeData, self).__init__()
        self.data_dir = '/home/lmin/data/classifications/qrcode'
        localtime = time.localtime(time.time())
        self.save_model_path = f'weights/qrcode.pth'
        self.input_size = (336, 416) # (width,height)

        self.batch_size = 32
        self.output_num = 2
        self.num_epochs = 100  #
        self.multi_scale = False
        self.use_ema = True
        # self.weight = torch.from_numpy(np.array([1.0,7.5])).float()  # [10.0, 1.0]
        self.init_lr = 0.01  # [0.1 0.01 0.001 0.0001 0.00001 0.000001]0.01 0.001 0.0001
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def SGD_CosineAnnealingLR(self, model, T_max, SGD_lr, SGD_m=0.9, weight_decay=0, eta_min=0.):
        """
        :param model: torch model
        :param T_max: CosineAnnealingLR Maximum number of iterations (0~π steps).
        :param SGD_lr: SGD lr
        :param SGD_m: SGD momentum
        :param eta_min: CosineAnnealingLR Minimum learning rate.
        :param weight_decay: SGD weight decay.
        :return: optimizer and scheduler
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=SGD_lr, momentum=SGD_m, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=eta_min, last_epoch=-1)
        return optimizer, scheduler

    def run(self):
        image_datasets = {x: DataFactory(os.path.join(self.data_dir, x), stage=x, input_size=self.input_size) for x in ['train', 'val']}  # , x
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.batch_size if x == 'train' else 1, shuffle=True if x == 'train' else False,
                          num_workers=8) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = models.shufflenet_v2_x1_0(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, self.output_num)
        ema = ModelEMA(model_ft)
        # print(model_ft)
        model_ft = model_ft.to(self.device)

        # m = torch.load('./models/yw5/10_27.pth')
        # model_ft.load_state_dict(m)
        # criterion = nn.CrossEntropyLoss(weight=self.weight)
        criterion = FocalLoss(class_num=self.output_num)
        criterion.cuda(self.device)

        # Observe that all parameters are being optimized
        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = None
        # optimizer_ft, scheduler = self.SGD_CosineAnnealingLR(model_ft, 64, 0.07, eta_min=0.001)

        # Decay LR by a factor of 0.1 every 7 epochs
        # lr_scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

        since = time.time()
        best_model_wts = copy.deepcopy(model_ft.state_dict())
        best_acc = 0.0

        preds_list = []
        gt_labels_list = []
        # all_size_ = [(128, 128), (96, 96), (64, 64)]
        # all_size_ = [(96, 96)]
        for epoch in range(self.num_epochs):
            # t = image_datasets['train']
            # t.input_size = all_size_[epoch // (self.num_epochs//len(all_size_))]
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 50)
            preds_list.clear()
            gt_labels_list.clear()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model_ft.train()  # Set model to training mode
                else:
                    model_ft.eval()  # Set model to evaluate mode
                    preds_list = []
                    gt_labels_list = []

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    # t = image_datasets['train']
                    # rd = random.randint(0, len(all_size_)-1)
                    # t.input_size = all_size_[rd]
                    if self.multi_scale:
                        short_size = min(self.input_size)
                        gs = 8
                        sz = random.randrange(short_size * 0.5, short_size * 1.5 + gs) // gs * gs  # size
                        sf = sz / short_size  # scale factor
                        if sf != 1:
                            ns = [math.ceil(x * sf / gs) * gs for x in
                                  self.input_size]  # new shape (stretched to gs-multiple)
                            inputs = nn.functional.interpolate(inputs, size=ns, mode='bilinear', align_corners=False)

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # print(labels)

                    # zero the parameter gradients
                    optimizer_ft.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model_ft(inputs)
                        _, preds = torch.max(outputs, 1)

                        # loss = criterion(outputs, labels)
                        loss = criterion(outputs, labels, self.device)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer_ft.step()
                            # scheduler.step()
                            if epoch > 20:
                                if scheduler is not None:
                                    scheduler.step()
                                # print(optimizer_ft.param_groups[0]['lr'])
                    preds_list.extend(preds.cpu().detach().numpy().tolist())
                    gt_labels_list.extend(labels.data.cpu().detach().numpy().tolist())
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                # if phase == 'train':
                #     lr_scheduler_ft.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model_ft.state_dict())
            print('')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model_ft.load_state_dict(best_model_wts)
        torch.save(model_ft.state_dict(), self.save_model_path)
        # torch.save(model_ft, self.save_model_path)
        return model_ft


if __name__ == '__main__':
    finetuneData = TrainQrcodeData()
    finetuneData.run()
    print('finished!')

