# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/3/16 10:52
# @Author : liumin
# @File : torch2onnx.py

import os
from torch import nn
from torchvision import models
import torch.onnx
import onnx
import onnxsim

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def torch2onnx(half = False, simplify = True):
    # init parameter
    device = torch.device("cpu")
    input = torch.randn(1, 3, 64, 64).to(device)  # BCHW
    output_num = 2
    weight_path = 'weights/3_16_24_1.000.pth'
    onnx_path = 'onnxs/qrcode_cls.onnx'
    onnx_sim_path = 'onnxs/qrcode_cls_sim.onnx'

    model_ft = models.shufflenet_v2_x0_5(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, output_num)
    print(model_ft)
    model_ft = model_ft.to(device)

    model_ft.load_state_dict(torch.load(weight_path))
    model_ft.eval()

    # Input to the model
    if half:
        input, model_ft = input.half(), model_ft.half()  # to FP16

    # Export the model
    torch.onnx.export(model_ft,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      verbose=False)

    # Checks
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # simplify
    if simplify:
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, 'assert check failed'
        onnx.save(model_onnx, onnx_sim_path)


if __name__ == "__main__":
    torch2onnx()


