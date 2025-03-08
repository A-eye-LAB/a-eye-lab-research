# A-Eye-Lab-Research

## 프로젝트 개요
카메라로 촬영한 눈 이미지를 이용하여 백내장 유무를 분류하는 AI 모델 관련 코드입니다. 데이터 전처리, 모델 학습, 평가 및 ONNX 변환과 관련된 코드가 포함되어 있습니다.

## 설치 방법
```bash
git clone https://github.com/your-username/A-Eye-Lab-Research.git
cd A-Eye-Lab-Research

pip install -r docs/requirements.txt
```

## 사용 예시
```bash
# 데이터셋 다운로드
./download_dataset.bash <download_path>

# 모델 학습
python train/main.py --cfg train/configs/train_mv3.yaml

# ONNX 변환
python export/export.py --model_path models/best_model.pth # TODO

# 모델 평가
python eval/eval.py --model_path models/best_model.pth --dataset_path data/test # TODO

# ONNX 모델 평가
python eval/onnx_eval.py --onnx_path models/best_model.onnx --dataset_path data/test # TODO
```

## 기술 스택
- **프로그래밍 언어**: Python
- **프레임워크**: PyTorch
- **모델 변환**: ONNX


## 폴더 구조
```
A-Eye-Lab-Research/
├── dataset/       # 데이터셋 관련 코드
├── train/         # 학습 관련 코드
│   ├── config/    # 학습 관련 config
│   ├── models/    # 모델 코드
│   ├── modules/   # 모델 학습 관련 코드 (optimizer, scheduler 등)
├── eval/          # 평가 관련 코드
├── export/        # ONNX 변환 코드
├── requirements.txt  # 필수 패키지 목록
├── README.md      # 프로젝트 설명 파일
```