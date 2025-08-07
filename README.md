# 👀 A-Eye Lab Research

## 📁 프로젝트 구조

```bash
a-eye-lab-research/
├── docker/          # Docker 관련 설정 및 코드
├── dataset/         # 데이터셋 처리 및 관리 코드
├── train/           # 모델 학습 관련 코드
├── eval/            # 모델 평가 및 검증 코드
├── export/          # ONNX 변환 및 모델 내보내기 코드
└── notebook/        # 개인 연구 노트북
```

## 🚀 시작하기

### 환경 설정

```bash
# 프로젝트 루트 디렉토리로 이동
cd a-eye-lab-research/docs

# (선택) conda 환경 생성
conda create -n aeye-lab python=3.9
conda activate aeye-lab

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# 필요한 패키지 설치 (requirements.txt가 있다면)
pip install -r requirements.txt
```

## 📊 학습 데이터 다운로드
```bash
https://huggingface.co/a-eyelab
```

## 🏷️ 학습 데이터 설정 방법 및 학습 파라미터 설정
모델 학습 시에 학습 데이터 설정은 config/train.yaml에서 아래의 항목처럼 기재하시면 됩니다.

```bash
DATASET:
  TRAIN_DATA_DIR: [
    "your/data/path"
  ]
```

데이터 형식은 아래와 같이 맞춰주시면 datasets.py 모듈에서 라벨을 인식하여 학습을 진행할 수 있습니다.

```bash
a-eye-lab-research/
├── 0/ # 정상군
└── 1/ # 환자군
```


## 🎯 모델 학습

```bash
# 학습 디렉토리로 이동
cd a-eye-lab-research/train

# 모델 학습 실행
python3 main.py --cfg configs/train.yaml
```


## 📊 모델 평가

테스트 데이터를 통해 최종적으로 모델 평가를 위해서 다음과 같이 진행하시면 됩니다.
```python
if __name__ == "__main__":
    main(
        dataset_path="path/to/data",    # 평가할 데이터셋 경로
        model_path="path/to/model"      # 학습된 모델 파일 경로 (.onnx, .pt 형식 지원)
    )
```
코드 실행은 다음과 같습니다.

```bash
# 평가 디렉토리로 이동
cd a-eye-lab-research/eval

# 모델 평가 실행
python3 main.py
```
## 📦 ONNX 변환

```bash
# 내보내기 디렉토리로 이동
cd a-eye-lab-research/export

# ONNX 변환 실행
python3 export.py
```

## 📝 사용법

1. **환경 설정**: 먼저 필요한 패키지들을 설치합니다
2. **데이터 준비**: `dataset/` 디렉토리에서 데이터를 준비합니다
3. **모델 학습**: `train/` 디렉토리에서 모델을 학습합니다
4. **모델 평가**: `eval/` 디렉토리에서 학습된 모델을 평가합니다
5. **모델 내보내기**: `export/` 디렉토리에서 ONNX 형식으로 변환합니다

## 🔧 설정 파일

- `configs/train.yaml`: 학습 설정 파일

## 📚 추가 정보

- 개인 연구 노트북은 `notebook/` 디렉토리에서 확인할 수 있습니다.
