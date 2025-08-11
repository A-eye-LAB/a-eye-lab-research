# 백내장 데이터 전처리

이 프로젝트는 백내장 데이터셋을 전처리하는 파이프라인을 구현합니다. 전체 과정은 다음과 같은 단계로 구성됩니다:

1. 데이터 중복 제거 (Step 1)
2. 수동 필터링 (Step 2)
3. 홍채 영역 검출 및 크롭 (Step 3)

## Step 1: 중복 제거

**개요**: 원본 데이터셋에서 중복 이미지를 제거하여 데이터 품질을 향상시킵니다.

**주요 기능**:
- MD5 해시 기반 중복 제거
- 이미지 리사이즈 후 퍼셉추얼 해싱
- 히스토그램 유사도 기반 그룹화
- 설정 가능한 유사도 임계값

**사용법**:
```bash
python preprocess_deduplicate.py \
    --source raw_data \
    --target preprocessed/step1_dedup_0.7 \
    --threshold 0.7
```

**인자 설명**:
- `--source`: 원본 데이터셋 경로 (기본값: `raw_data`)
- `--target`: 중복 제거된 데이터 저장 경로 (기본값: `preprocessed/step1_dedup_0.7`)
- `--threshold`: 유사도 임계값 (기본값: `0.7`)

### 출력 구조
```
preprocessed/step1_dedup_0.75/
├── {데이터셋_이름}/                    # 원본 데이터셋 구조
│   ├── 0/
│   │   └── [중복이_없는_이미지들]      # 중복이 없는 단일 이미지들
│   └── 1/
│       └── [중복이_없는_이미지들]      # 중복이 없는 단일 이미지들
├── duplicated_removed/                # 중복 그룹
│   └── {데이터셋_이름}/
│       ├── 0/
│       │   ├── group_001/
│       │   │   ├── representative_image1.jpg  # 대표 이미지
│       │   │   ├── duplicate1.jpg            # 유사 이미지
│       │   │   └── duplicate2.jpg            # 유사 이미지
│       │   └── group_002/
│       │       └── ...
│       └── 1/
│           └── ...
└── log/                               # 로그 파일 디렉토리
    ├── {데이터셋_이름}_deduplication_log.txt  # 상세 중복 제거 로그
    └── OVERALL_SUMMARY_LOG.txt           # 전체 요약 로그
```

## Step 2: 수동 필터링

### 개요
이 단계에서는 Step 1에서 중복이 제거된 데이터셋을 수동으로 전처리된 데이터셋과 비교하여 필터링합니다. 수동 전처리된 데이터셋에 존재하는 이미지만 유지하고, 나머지는 filter_out 디렉토리로 분리합니다.

### 주요 기능
- **해시 기반 매칭**: MD5 해시를 사용하여 정확한 파일 매칭
- **자동 분류**: 매칭된 파일은 target 디렉토리로, 매칭되지 않은 파일은 filter_out으로 분리
- **구조 보존**: 원본 디렉토리 구조 유지

### 사용 방법
```bash
python preprocess_filtering_manually.py --source preprocessed/step1_dedup_0.75 --target preprocessed/step2_manually_filtered
```

#### 인자 설명
- `--source`: Step 1의 결과 디렉토리 (기본값: "preprocessed/step1_dedup_0.75")
- `--target`: 필터링된 데이터가 저장될 디렉토리 (기본값: "preprocessed/step2_manually_filtered")
- `--manual`: 수동 전처리된 데이터셋 디렉토리 (기본값: "manually_preprocessed")

### 출력 구조
```
preprocessed/step2_manually_filtered/
├── {데이터셋_이름}/                    # 매칭된 이미지들
│   ├── 0/
│   │   └── [매칭된_이미지들]
│   └── 1/
│       └── [매칭된_이미지들]
└── filter_out/                        # 매칭되지 않은 이미지들
    └── {데이터셋_이름}/
        ├── 0/
        │   └── [필터링된_이미지들]
        └── 1/
            └── [필터링된_이미지들]
```

## Step 3: 홍채 영역 검출 및 크롭

### 개요
이 단계에서는 YOLO 모델을 사용하여 이미지에서 홍채 영역을 검출하고, 검출된 영역을 중심으로 이미지를 크롭합니다. Step 2에서 필터링된 데이터셋을 입력으로 사용하며, 검출 실패한 이미지들은 filter_out으로 분리합니다.

### 사전 준비
1. YOLO 환경 활성화:
```bash
conda activate yolo
```

2. huggingface-hub 설치:
```bash
pip install huggingface-hub
```

3. YOLO 모델 다운로드:
```bash
# models 디렉토리 생성
mkdir -p models

# 모델 다운로드
hf download a-eyelab/yolo11s_iris --local-dir models
```

다운로드된 `iris.pt` 파일이 자동으로 사용됩니다.

### 주요 기능
- **홍채 검출**:
  - YOLO 모델을 사용한 정확한 홍채 영역 검출
  - 신뢰도 기반 최적 영역 선택
  - 검출된 영역 주변 컨텍스트 포함 크롭

- **자동 필터링**:
  - 홍채가 검출되지 않은 이미지는 filter_out으로 분리
  - 홍채가 두 개 이상 검출된 이미지는 filter_out으로 분리
  - 검출 성공한 이미지만 크롭하여 저장

- **데이터 구조 보존**:
  - Step 2의 디렉토리 구조 유지
  - 원본 파일명 유지

### 사용 방법
```bash
# 기본 사용
python preprocess_crop_iris.py --source preprocessed/step2_manually_filtered --target preprocessed/step3_crop

# 전체 옵션
python preprocess_crop_iris.py \
    --source preprocessed/step2_manually_filtered \
    --target preprocessed/step3_crop \
    --model models/iris.pt \
    --conf 0.25 \
    [--box] \
    [--save-label]
```

#### 인자 설명
- `--source`: Step 2의 결과 디렉토리 (기본값: "preprocessed/step2_manually_filtered")
- `--target`: 크롭된 이미지 저장 디렉토리 (기본값: "preprocessed/step3_crop")
- `--model`: YOLO 모델 파일 경로 (기본값: "models/iris.pt")
- `--conf`: 홍채 검출 신뢰도 임계값 (기본값: 0.25, 25% 이상의 확률로 검출된 경우만 처리)
- `--box`: 검출된 박스 표시 여부
- `--save-label`: 라벨 파일 저장 여부

### 출력 구조
```
step3_crop/
├── 01_pub_alexandra/     # 홍채 검출 및 크롭 완료
│   ├── 0/ (크롭된 이미지들)
│   └── 1/ (크롭된 이미지들)
├── 02_pub_hemooredaoo/   # 홍채 검출 및 크롭 완료
│   ├── 0/ (크롭된 이미지들)
│   └── 1/ (크롭된 이미지들)
├── ...
├── 08_pub_DIRL/          # 홍채 검출 및 크롭 완료
│   ├── 0/ (크롭된 이미지들)
│   └── 1/ (크롭된 이미지들)
├── 09_collected_india_data_app/  # 원본 그대로 복사
│   ├── 0/ (원본 이미지들)
│   └── 1/ (원본 이미지들)
├── 10_collected_india_field_trip/ # 원본 그대로 복사
│   ├── 0/ (원본 이미지들)
│   └── 1/ (원본 이미지들)
├── 11_collected_kor_data/        # 원본 그대로 복사
│   ├── 0/ (원본 이미지들)
│   └── 1/ (원본 이미지들)
└── filter_out/            # 검출 실패한 이미지들
    ├── 01_pub_alexandra/
    ├── 02_pub_hemooredaoo/
    └── ...
```

## Step 4: Train-Test Split

**개요**: 크롭된 데이터셋을 train과 test로 분할하여 머신러닝 모델 학습을 위한 데이터를 준비합니다.

**주요 기능**:
- **01-06번 데이터셋**: Random split (파일 단위 랜덤 분할)
- **07-11번 데이터셋**: Person-aware split (개인 단위 분할, 동일인 분리 방지)
- 각 데이터셋별로 다른 train/test 비율 적용
- 파일명에 데이터소스 정보 포함

**사용법**:
```bash
python preprocess_train_test_split.py \
    --source preprocessed/step3_crop \
    --target preprocessed/step4_train_test_split \
    --config train_test_split_config.yaml \
    --seed 42
```

**인자 설명**:
- `--source`: 크롭된 이미지가 있는 디렉토리 (기본값: `preprocessed/step3_crop`)
- `--target`: train-test 분할 결과 저장 경로 (기본값: `preprocessed/step4_train_test_split`)
- `--config`: 분할 설정 YAML 파일 (기본값: `train_test_split_config.yaml`)
- `--seed`: 랜덤 시드 (기본값: `42`)

**출력 구조**:
```
step4_train_test_split/
├── train/
│   ├── 0/ (01_pub_alexandra_C (20).jpg, 02_pub_hemooredaoo_train_223.jpg, ...)
│   └── 1/ (01_pub_alexandra_C (31).jpg, 02_pub_hemooredaoo_train_158.jpg, ...)
└── test/
    ├── 0/ (01_pub_alexandra_C (6).jpg, 02_pub_hemooredaoo_train_141.jpg, ...)
    └── 1/ (01_pub_alexandra_C (8).jpg, 02_pub_hemooredaoo_train_221.jpg, ...)
```

## 전체 파이프라인 실행

```bash
# Step 1: 초기 중복 제거
python preprocess_deduplicate.py \
    --source raw_data \
    --target preprocessed/step1_dedup_0.7 \
    --threshold 0.7

# Step 2: 수동 필터링 (01-06번만)
python preprocess_filtering_manually.py \
    --source preprocessed/step1_dedup_0.7 \
    --target preprocessed/step2_manually_filtered

# Step 3: 홍채 검출 및 크롭 (01-08번만)
python preprocess_crop_iris.py \
    --source preprocessed/step2_manually_filtered \
    --target preprocessed/step3_crop \
    --model models/iris.pt \
    --conf 0.25

# Step 4: Train-Test Split
python preprocess_train_test_split.py \
    --source preprocessed/step3_crop \
    --target preprocessed/step4_train_test_split \
    --config train_test_split_config.yaml \
    --seed 42
```

## 참고사항

- **데이터셋 범위**: 01-11번 데이터셋 처리 (01-06번: 필터링 + 홍채 크롭, 07-08번: 홍채 크롭, 09-11번: 원본 복사)
- **권장 임계값**: Step 1 중복 제거 시 0.7 사용 (데이터 품질과 양의 균형)
- **filter_out**: 홍채 검출 실패한 이미지들을 별도 폴더에 보관
- **단일 검출**: 홍채가 1개만 검출된 이미지만 처리 (다중 검출 시 filter_out으로 이동)
- **Person-aware Split**: 07-11번 데이터셋에서 동일인의 눈이 train/test에 분리되지 않도록 보장
- **파일명 규칙**: train/test 폴더의 파일명에 `{데이터셋명}_{원본파일명}` 형식으로 데이터소스 정보 포함