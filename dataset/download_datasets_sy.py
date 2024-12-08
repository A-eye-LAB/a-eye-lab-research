import os
import shutil
from pathlib import Path
import logging
from datetime import datetime
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi


HOME = str(Path.home())
DATA_DIR = Path(HOME) / "datasets"
LOG_DIR = Path(HOME) / "logs"


LABEL_MAPPINGS = {
    "cataract": "1",
    "normal": "0",
}


def setup_logging():
    """
    로그 파일과 콘솔 출력 설정
    """
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"dataset_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

# Kaggle API 사용하면 데이테셋을 사용할 수 있음 
# Kaggle json 파일이 다운로드 되어있다는 것을 가정해야함 

def download_kaggle_dataset(dataset_dir: Path, kaggle_dataset_name: str):
    """
    Kaggle API를 사용하여 데이터셋 
    """
    try:
        logging.info(f"Downloading dataset: {kaggle_dataset_name}")
        api = KaggleApi()
        api.authenticate()

    
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Kaggle 데이터 다운로드 및 압축 해제
        api.dataset_download_files(kaggle_dataset_name, path=dataset_dir, unzip=True)
        logging.info(f"Dataset {kaggle_dataset_name} downloaded and extracted successfully.")
    except Exception as e:
        logging.error(f"Error downloading dataset {kaggle_dataset_name}: {e}")
        raise


def organize_images(src_dir: Path, dst_dir: Path):
    """
    라벨 0 또는 1
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    processed_count, error_count = 0, 0

    for class_name, label in LABEL_MAPPINGS.items():
        src_class_dir = src_dir / class_name
        dst_label_dir = dst_dir / label
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        if not src_class_dir.exists():
            logging.warning(f"Source class directory does not exist: {src_class_dir}")
            continue

        for img_path in src_class_dir.glob("*"):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                try:
                    shutil.copy2(img_path, dst_label_dir / img_path.name)
                    processed_count += 1
                except Exception as e:
                    error_count += 1
                    logging.error(f"Failed to copy image {img_path}: {e}")

    logging.info(f"Processed {processed_count} images with {error_count} errors.")


def main():
    setup_logging()

   
    parser = argparse.ArgumentParser(description="Kaggle Dataset Downloader and Organizer")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name to save under ./dataset/data/") # 내가 데이터 설정할 수 있게 
    parser.add_argument("--kaggle_dataset", type=str, required=True, help="Kaggle dataset path (e.g., hemooredaoo/cataract)") # kaggle url 입력하면 됨! 
    args = parser.parse_args()

    dataset_name = args.dataset_name
    kaggle_dataset = args.kaggle_dataset

  
    dataset_dir = DATA_DIR / dataset_name  
    processed_images_dir = dataset_dir / "cataract-image-dataset/processed_images/test" 
    output_dir = Path("./dataset/data") / dataset_name  

    try:
        logging.info(f"Starting dataset download and organization for {dataset_name}")

        
        if not processed_images_dir.exists():
            download_kaggle_dataset(dataset_dir, kaggle_dataset)
        else:
            logging.info(f"Dataset already exists at {processed_images_dir}. Skipping download.")

       
        organize_images(processed_images_dir, output_dir)
        logging.info(f"Dataset {dataset_name} processing completed successfully.")
    except Exception as e:
        logging.error(f"Error processing dataset {dataset_name}: {e}")

# 실행
if __name__ == "__main__":
    main()
