import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("best.pt")

def detect_iris(img_path, score_threshold=0.5):
    # 이미지 불러오기
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # YOLO로 홍채 검출
    results = model(img)

    # 원 안쪽은 1, 바깥쪽은 0.2를 곱하는 마스크 생성
    mask = np.ones_like(img, dtype=np.float32) * 0.2  # 기본값 0.2로 설정

    if len(results) > 0:
        result = results[0]  # 첫 번째 결과
        boxes = result.boxes
        if len(boxes) > 0:
            # confidence score가 가장 높은 박스 찾기
            confidences = boxes.conf
            best_box_idx = confidences.argmax()
            box = boxes[best_box_idx]
            score = float(confidences[best_box_idx])

            if score < score_threshold:
                return False  # 기준 미달이면 False 반환

            # 바운딩 박스 좌표 가져오기
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 사각형의 중심점과 반지름 계산
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = min((x2 - x1), (y2 - y1)) // 2  # 사각형에 내접하는 원의 반지름

            # 원 마스크 생성
            cv2.circle(mask, (center_x, center_y), radius, (1.0, 1.0, 1.0), -1)

        else: return False

    # 원본 이미지에 마스크 적용
    result = (img.astype(np.float32) * mask).astype(np.uint8)

    return result


def process_all_images(base_dir, output_dir, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    # 결과 저장 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 모든 이미지 파일 찾기
    for root, dirs, files in os.walk(base_dir):
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
    # 데이터셋 경로를 입력하면 하위 경로의 모든 이미지에 대해 처리
    # 결과 데이터셋도 원본과 같은 경로로 저장
    base_dir = "./dataset/data"
    output_dir = "./results/yolo/2265-softmask-2"

    # 제외할 디렉토리 목록
    exclude_dirs = [
        "real_data",  # 예시: 이 디렉토리는 제외
        "kaggle_cataract_nand",
    ]

    process_all_images(base_dir, output_dir, exclude_dirs)