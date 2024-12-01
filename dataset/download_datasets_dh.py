import logging
import os
import shutil
import zipfile
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict

import requests
from PIL import Image

LABEL_MAPPINGS = {
    "cataract": "1",
    "processed_images": {"cataract": "1", "normal": "0"},
    "eye_cataract": {"Mature": "1", "Immature": "1", "Normal": "0"},
}

SPLIT_SETTINGS = {"cataract": [""], "processed_images": ["train", "test"], "eye_cataract": ["train", "test", "valid"]}

DATASET_URLS = {
    "mohammedgamal37l30": "https://www.kaggle.com/api/v1/datasets/download/mohammedgamal37l30/eye-cataractmature-immature-normal",
    "hemooredaoo": "https://www.kaggle.com/api/v1/datasets/download/hemooredaoo/cataract",
    "alexandramohammed": "https://www.kaggle.com/api/v1/datasets/download/alexandramohammed/cataract-image",
}


def create_label_directories(dataset_paths, dataset_type: str) -> None:
    base_path = dataset_paths[dataset_type]["dst"]
    for label in ["0", "1"]:
        (Path(base_path) / label).mkdir(parents=True, exist_ok=True)


def setup_logging():
    log_file = f'dataset_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def download_and_unzip(data_dir, dataset_name: str, url: str) -> None:
    try:
        data_dir.mkdir(exist_ok=True)
        os.chdir(data_dir)
        zip_path = f"{dataset_name}.zip"

        logging.info(f"다운로드 중: {dataset_name}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # HTTP 에러 체크

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"압축 해제 중: {dataset_name}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_name)

        os.remove(zip_path)
    except Exception as e:
        logging.error(f"{dataset_name} 처리 중 오류 발생: {str(e)}")
        raise


def process_single_label_dataset(src_path: Path, dst_path: Path, label: str) -> tuple[int, int]:
    processed_count = 0
    error_count = 0
    for img_path in src_path.glob("*"):
        if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            try:
                shutil.copy2(img_path, dst_path / label / img_path.name)
                processed_count += 1
            except Exception as e:
                error_count += 1
                logging.error(f"이미지 복사 실패 {img_path}: {str(e)}")
    return processed_count, error_count


def remove_icc_profile(img_path: Path) -> None:
    try:
        with Image.open(img_path) as img:
            if img.info.get("icc_profile"):
                logging.info(f"Removing ICC profile: {img_path}")
                img.save(str(img_path), icc_profile=None)
    except Exception as e:
        logging.error(f"ICC 프로파일 제거 실패 {img_path}: {str(e)}")


def process_multi_label_dataset(
    src_path: Path, dst_path: Path, label_mapping: Dict[str, str], split: str
) -> tuple[int, int]:
    processed_count = 0
    error_count = 0
    split_path = src_path / split if split else src_path

    for class_name, label in label_mapping.items():
        class_path = split_path / class_name
        if not class_path.exists():
            logging.warning(f"경로를 찾을 수 없음: {class_path}")
            continue

        for img_path in class_path.glob("*"):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                try:
                    # 파일 이름에 split 정보 추가
                    new_filename = f"{split}_{img_path.stem}" if split else img_path.stem
                    new_filename = f"{new_filename}{img_path.suffix}"
                    dst_file = dst_path / label / new_filename
                    shutil.copy2(img_path, dst_file)
                    # processed_images 데이터셋의 경우 ICC 프로파일 제거
                    if "processed_images" in str(src_path):
                        remove_icc_profile(dst_file)
                    processed_count += 1
                except Exception as e:
                    error_count += 1
                    logging.error(f"이미지 복사 실패 {img_path}: {str(e)}")

    return processed_count, error_count


def move_and_label_images(dataset_paths, dataset_type: str) -> None:
    src_path = Path(dataset_paths[dataset_type]["src"])
    dst_path = Path(dataset_paths[dataset_type]["dst"])
    label_mapping = LABEL_MAPPINGS[dataset_type]
    total_processed = 0
    total_errors = 0

    if isinstance(label_mapping, str):
        processed, errors = process_single_label_dataset(src_path, dst_path, label_mapping)
        total_processed += processed
        total_errors += errors
        logging.info(f"처리된 이미지 수: {processed} (에러: {errors})")
        return

    for split in SPLIT_SETTINGS[dataset_type]:
        split_processed, split_errors = process_multi_label_dataset(src_path, dst_path, label_mapping, split)
        total_processed += split_processed
        total_errors += split_errors
        if split:
            logging.info(f"{split} 분할 처리된 이미지 수: {split_processed} (에러: {split_errors})")

    logging.info(f"총 처리된 이미지 수: {total_processed} (총 에러: {total_errors})")


def main(data_dir, dataset_paths):
    setup_logging()
    logging.info("데이터셋 다운로드 및 압축 해제 시작")

    for dataset_name, url in DATASET_URLS.items():
        try:
            download_and_unzip(data_dir, dataset_name, url)
        except Exception as e:
            logging.error(f"{dataset_name} 처리 실패: {str(e)}")
            continue

    logging.info("이미지 파일 처리 시작")
    for dataset in dataset_paths:
        logging.info(f"처리 중: {dataset}")
        create_label_directories(dataset_paths, dataset)
        move_and_label_images(dataset_paths, dataset)

        # 원본 데이터 폴더 삭제
        src_folder = dataset_paths[dataset]["src"].parent
        if dataset == "processed_images":
            src_folder = src_folder.parent
        if src_folder.exists():
            try:
                shutil.rmtree(src_folder)
                logging.info(f"원본 폴더 삭제 완료: {src_folder}")
            except Exception as e:
                logging.error(f"원본 폴더 삭제 실패 {src_folder}: {str(e)}")

        logging.info(f"완료: {dataset}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--download_path", type=str, default="dataset/data", help="Path to download datasets (default: dataset/data)"
    )

    args = parser.parse_args()

    data_dir = Path(args.download_path).resolve()

    dataset_paths = {
        "cataract": {"src": data_dir / "alexandramohammed/cataract", "dst": data_dir / "C001"},
        "processed_images": {
            "src": data_dir / "hemooredaoo/cataract-image-dataset/processed_images",
            "dst": data_dir / "C002",
        },
        "eye_cataract": {"src": data_dir / "mohammedgamal37l30/eye_cataracwithlabels", "dst": data_dir / "C003"},
    }

    main(data_dir, dataset_paths)
