"""평균 임베딩 값으로 눈 탐지하는 코드
"""

import time
import argparse
import numpy as np
from utils import load_mean_embedding, is_eye_image, load_model


def get_args():
    parser = argparse.ArgumentParser(description="눈 분류 추론")

    parser.add_argument(
        "--mean_embedding_path",
        type=str,
        required=True,
        help="계산된 평균 임베딩 파일 경로(pkl)",
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
    parser.add_argument(
        "--test_image", type=str, required=True, help="테스트할 이미지 경로"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65, help="유사도 임계값 (0~1)"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    try:
        mean_emb = load_mean_embedding(args.mean_embedding_path)
        model = load_model(args.tuning, args.model_path)

        print("=== 이미지 유사성 판단 ===")
        st = time.time()
        result, similarity = is_eye_image(
            model, args.test_image, mean_emb, threshold=args.threshold
        )
        et = time.time()

        print(
            f"눈 이미지, {np.round(similarity,2)}"
            if result
            else f"눈 이미지 아님, {np.round(similarity,2)}"
        )
        print("infer time : ", et - st)

    except Exception as e:
        print(f"실행 중 오류 발생: {e}")


# 추론 코드(튜닝 모델 사용) : python infer.py --mean_embedding_path embedding_pkl/mean_embedding_tuning.pkl --test_image testset/1/eye6.jpeg --tuning --model_path models/mobilenetv3_large.pt
# 추론 코드(튜닝 안된 모델 사용) : python infer.py --mean_embedding_path embedding_pkl/mean_embedding.pkl --test_image testset/1/eye6.jpeg
