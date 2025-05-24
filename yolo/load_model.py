from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import os

def load_hf_model(repo_id="a-eyelab/yolov8-iris", filename="yolov8m-iris.pt", cache_dir=None):
    """
    Hugging Face에서 YOLO 모델을 다운로드하고 로드합니다.

    Args:
        repo_id (str): Hugging Face 저장소 ID
        filename (str): 다운로드할 파일 이름
        cache_dir (str, optional): 캐시 디렉토리 경로. 기본값은 None (기본 캐시 디렉토리 사용)

    Returns:
        YOLO: 로드된 YOLO 모델
    """
    try:
        # 모델 파일 다운로드
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )

        # YOLO 모델 로드
        model = YOLO(model_path)
        print(f"Successfully loaded model from {model_path}")

        return model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

if __name__ == "__main__":
    # 사용 예시
    model = load_hf_model()
    if model:
        print("Model loaded successfully!")