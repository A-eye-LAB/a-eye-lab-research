import os
import argparse
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils import load_model, is_eye_image, evaluate_model, load_mean_embedding


def get_args():
    parser = argparse.ArgumentParser(description="Eye Image Classification")

    parser.add_argument(
        "--data_dir", type=str, required=True, help="테스트 이미지 폴더 경로"
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        default="mean_embedding.pkl",
        help="평균 임베딩 파일 경로",
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
        "--threshold", type=float, default=0.65, help="유사도 임계값 (0~1)"
    )
    parser.add_argument("--num_threads", type=int, default=4, help="사용할 스레드 수")

    args = parser.parse_args()
    return args


def process_file(model, file_path, label, mean_embedding, threshold):
    """
    각 파일에 대한 is_eye_image 처리 함수.
    """
    result, similarity = is_eye_image(model, file_path, mean_embedding, threshold)
    return file_path, label, result * 1


def eval(model, data_dir, mean_embedding, threshold, num_threads):
    classes = {"eye": 1, "no eye": 0}

    y_true = []
    y_pred = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []

        for label_dir, label in classes.items():
            class_path = os.path.join(data_dir, str(label))

            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                futures.append(
                    executor.submit(
                        process_file, model, file_path, label, mean_embedding, threshold
                    )
                )

        for future in tqdm(futures, desc="Evaluating", total=len(futures)):
            file_path, label, prediction = future.result()
            y_true.append(label)
            y_pred.append(prediction)

    metrics = evaluate_model(y_true, y_pred)
    return metrics


if __name__ == "__main__":
    args = get_args()
    mean_embedding = load_mean_embedding(args.embedding_file)
    model = load_model(args.tuning, args.model_path)
    metrics = eval(
        model, args.data_dir, mean_embedding, args.threshold, args.num_threads
    )
    print(metrics)

# 검증코드(튜닝 모델) : python eval.py --data_dir testset --embedding_file embedding_pkl/mean_embedding_tuning.pkl --threshold 0.7 --tuning --model_path models/mobilenetv3_large.pt
# 검증코드(튜닝 안된 모델) : python eval.py --data_dir testset --embedding_file embedding_pkl/mean_embedding.pkl --threshold 0.65
