# Yolov8 을 이용한 Iris detection

## Setup
원하는 환경 셋팅
```bash
conda create -n yolo python=3.11 -y
conda activate yolo
pip install -r requirements.txt
```

## Download pretrain model
```bash
python load_model.py
```

## Train
1. data.yaml 파일에서 데이터셋 경로 변경
2. ```python train.py```

## Make soft mask
1. 파일 제일 아래에서 적용할 이미지 경로 입력
2. output 경로 입력
3. ```python soft_mask.py```

## Yolov8n 모델 테스트
1. ```main.py``` 파일 내 이미지 경로 변경
```bash
python main.py
```