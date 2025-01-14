"""평균 임베딩 계산 코드
"""

import os
import torch
import pickle
import argparse
import numpy as np

from tqdm.auto import tqdm
from utils import preprocess_image, load_model


def get_args():
    parser = argparse.ArgumentParser(
        description="눈 이미지 분류를 위한 평균 임베딩 값 계산"
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        nargs="+",
        required=True,
        help="참조 이미지 디렉토리 경로",
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        required=True,
        default="mean_embedding.pkl",
        help="평균 임베딩 파일 저장 경로",
    )
    parser.add_argument(
        "--tuning", action="store_true", help="튜닝된 Mobilenetv3 사용 여부"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="mobilenetv3_large.pt",
        help="튜닝된 모델 경로(MobileNetv3_Large.pt)",
    )

    args = parser.parse_args()
    return args


def extract_embedding(image_path):
    """임베딩 추출 함수"""

    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        embedding = model(input_tensor).squeeze().numpy()

    return embedding


def calculate_mean_embedding(image_dirs, batch_size=128, save_path=None):
    """평균 임베딩 계산 함수"""

    embeddings = []
    files = []

    for image_dir in image_dirs:
        files.extend(
            [
                os.path.join(root, file)
                for root, _, file_list in os.walk(image_dir)
                for file in file_list
            ]
        )

    if not files:
        raise ValueError(f"'{image_dirs}'에 이미지 파일이 없습니다.")

    print(
        f"{image_dirs} 데이터, 총 {len(files)}개의 데이터 평균 임베딩 계산, batch size : {batch_size}"
    )

    for i in tqdm(range(0, len(files), batch_size), desc=f"눈 데이터 평균 임베딩 계산"):
        batch_files = files[i : i + batch_size]
        batch_embeddings = []

        for file_path in batch_files:
            embedding = extract_embedding(file_path)
            batch_embeddings.append(embedding)

        embeddings.extend(batch_embeddings)

    if not embeddings:
        raise ValueError("유효한 임베딩을 계산할 수 없습니다.")

    mean_embedding = np.mean(embeddings, axis=0)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(mean_embedding, f)
            print(f"평균 임베딩이 '{save_path}'에 저장되었습니다.")

    return mean_embedding


if __name__ == "__main__":

    args = get_args()

    try:
        model = load_model(args.tuning, args.model_path)

        print("=== 평균 임베딩 계산 시작 ===")
        mean_embedding = calculate_mean_embedding(
            args.image_dir, save_path=args.embedding_file
        )
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")


# 평균 임베딩 계산 (튜닝모델) : python mean_embedding.py --image_dir /home/yujin/data/C001 /home/yujin/data/C003 --embedding_file embedding_pkl/mean_embedding_tuning.pkl --tuning --model_path models/mobilenetv3_large.pt
# 평균 임베딩 계산 (튜닝 안된 모델) : python mean_embedding.py --image_dir /home/yujin/data/C001 /home/yujin/data/C003 --embedding_file embedding_pkl/mean_embedding.pkl
