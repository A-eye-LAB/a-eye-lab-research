import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("best.pt")

def crop_with_margin(img, box, margin_ratio=0.1):
    """
    Detection box에 마진을 추가하여 크롭합니다.

    Args:
        img: 원본 이미지
        box: YOLO detection box
        margin_ratio: 마진 비율 (기본값: 0.1 = 10%)

    Returns:
        크롭된 이미지
    """
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 박스의 너비와 높이 계산
    width = x2 - x1
    height = y2 - y1

    # 마진 계산
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)

    # 마진을 적용한 새로운 좌표 계산
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img.shape[1], x2 + margin_x)
    y2 = min(img.shape[0], y2 + margin_y)

    return img[y1:y2, x1:x2]

def detect_iris(img_path, score_threshold=0.5):
    # 이미지 불러오기
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # YOLO로 홍채 검출
    results = model(img)[0]

    # Use only the box with the highest score
    boxes = results.boxes
    confidences = boxes.conf
    best_box_idx = confidences.argmax()
    box = boxes[best_box_idx]

    # 마진을 포함한 크롭 적용
    return crop_with_margin(img, box, margin_ratio=0.1)  # 마진 비율 조정 가능


def process_all_images(base_dir, output_dir, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    # 결과 저장 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 모든 이미지 파일 찾기
    for root, dirs, files in os.walk(base_dir):
        print(root)
        # 제외할 디렉토리 건너뛰기
        if any(exclude_dir in root for exclude_dir in exclude_dirs):
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 입력 파일 경로
                input_path = os.path.join(root, file)

                # 상대 경로 계산
                rel_path = os.path.relpath(root, base_dir)
                # 출력 디렉토리 생성
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)

                # 출력 파일 경로
                output_path = os.path.join(output_subdir, f'highlight_{file}')

                try:
                    # 이미지 처리
                    res_array = detect_iris(input_path)
                    if res_array is False:
                        continue
                    result_image = Image.fromarray(res_array)
                    result_image.save(output_path)
                    print(f"Processed: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")


if __name__ == "__main__":
    base_dir = "../../data"
    output_dir = "../../margin_data"

    # 제외할 디렉토리 목록
    exclude_dirs = [
        # "real_data",  # 예시: 이 디렉토리는 제외
        # "kaggle_cataract_nand",
        "C001",
        "C002",
        "C003",
        "kaggle_cataract_kersh",
        "Cataract_Detection-using-CNN",
        "Cataract-Detection-and-Classification"
    ]

    process_all_images(base_dir, output_dir, exclude_dirs)