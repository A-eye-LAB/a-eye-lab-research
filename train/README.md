# Swin model
## Train model
```bash
python main.py --cfg configs/train-swin.yaml
```

## Convert script to onnx
```bash
python -m src.onnx.convert-swin --model_path {torch model path} --image_path {image path}
```

## Result

```bash
Predict script model
    predict result : [[-9.94594   9.986399]]
    predict time   : 0.24417448043823242
Predict onnx model
    predict result : [[-9.945941  9.986402]]
    predict time   : 0.22106051445007324
❯ git status
```