# ClassificationCodes


## install

```bash
# onnx
pip install onnx
pip install onnx-simplifier
```

## train

```bash
python QrCodeCls.py
```

## pytorch->onnx

```bash
python torch2onnx.py
```

## onnx->ncnn

```bash
.\onnx2ncnn onnxs/qrcode_cls_sim.onnx ncnns/qrcode_cls.param ncnns/qrcode_cls.bin
```
