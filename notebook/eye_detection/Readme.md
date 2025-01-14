# Eye Detection Code
Eye Detection Repository

## Requirements
- python 3.9.20

## Installation
```bash
$ git clone git@github.com:A-eye-LAB/a-eye-lab-research.git
$ cd notebook/eye_detection
$ pip install -r requirements_eye.txt
```

## Features
전처리 방법 
    - 백내장 탐지 모델에 적용되는 전처리 방식 그대로 사용 
    

## Usage
### 0. mobilnetv3_large 모델 다운로드 받기 
- 구글드라이브 a-eye-lab > model > MobileNetV3 Large > model.pt
    - https://drive.google.com/file/d/18xe1u2QU1zogBRxF457zoTi5bYmS0qm7/view?usp=drive_link
- eye_detection/models/ 폴더에 저장 
- model.pt 파일명 변경 > mobilenetv3_large.pt

### 1. 평균 임베딩 계산
튜닝된 모델  
```bash
$ python mean_embedding.py --image_dir /home/yujin/data/C001 /home/yujin/data/C003 --embedding_file embedding_pkl/mean_embedding_tuning.pkl --tuning --model_path models/mobilenetv3_large.pt
```

튜닝 안된 모델
```bash
$ python mean_embedding.py --image_dir /home/yujin/data/C001 /home/yujin/data/C003 --embedding_file embedding_pkl/mean_embedding.pkl
```

### 2. 눈 분류 추론
튜닝 모델
```bash
$ python infer.py --mean_embedding_path embedding_pkl/mean_embedding_tuning.pkl --test_image testset/1/eye6.jpeg --tuning --model_path models/mobilenetv3_large.pt
```

튜닝 안된 모델 
```bash
$ python infer.py --mean_embedding_path embedding_pkl/mean_embedding.pkl --test_image testset/1/eye6.jpeg
```

### 3. 모델 평가 
튜닝 모델
```bash
$ python eval.py --data_dir testset --embedding_file embedding_pkl/mean_embedding_tuning.pkl --threshold 0.65 --tuning --model_path models/mobilenetv3_large.pt
```
```text
# 평가 결과
{
    'valid/accuracy': 0.40974730253219604, 
    'valid/f1_score': 0.4429301619529724, 
    'valid/precision': 0.4193548262119293, 
    'valid/recall': 0.4693140685558319, 
    'valid/specificity': 0.3501805067062378
}
```

튜닝 안된 모델
```bash
$ python eval.py --data_dir testset --embedding_file embedding_pkl/mean_embedding.pkl --threshold 0.65
```
```text
# 평가 결과
{
    'valid/accuracy': 0.8050541281700134, 
    'valid/f1_score': 0.7578475475311279, 
    'valid/precision': 1.0, 
    'valid/recall': 0.6101083159446716, 
    'valid/specificity': 1.0
}
```

## File Structure
```text
eye_detection/
├── models/                  # 모델 파일 저장
│   └── mobilenetv3_large.pt # 튜닝된 모델
│   └── mv3.py
├── utils.py                 # 유틸리티 
├── eval.py                  # 모델 평가 스크립트
├── infer.py                 # 이미지 추론 스크립트
├── mean_embedding.py        # 평균 임베딩 계산 스크립트
├── requirements_eye.txt     # 프로젝트의 필수 라이브러리 목록
└── README.md                
```