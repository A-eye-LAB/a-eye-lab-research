"""
    - DOWNLOAD_FUNCTIONS에 있는 데이터셋들을 전부 다운로드
    - 다운로드가 완료된 데이터셋들은 제외하고 다운로드 (1.complete 파일 존재유무로 검사)

Add dataset:
    - dataset.utils.download_functions에 해당 데이터셋 다운로드 함수 추가
    - DOWNLOAD_FUNCTIONS에 데이터셋 이름과 다운로드 함수를 추가
"""

import os
import shutil
from argparse import ArgumentParser

from download_utils import (
    arrange_dataset,
    clone_git,
    download_completed,
    download_kaggle_dataset,
    mark_as_completed,
)


def download_cataract_detection_using_cnn(dataset_dir):
    clone_git("https://github.com/krishnabojha/Cataract_Detection-using-CNN.git", "tmp/")

    arrange_dataset(
        "tmp/",
        dataset_dir,
        {
            "1": ["Dataset/Test/Cataract", "Dataset/Train/Cataract"],
            "0": ["Dataset/Test/Normal", "Dataset/Train/Normal"],
        },
    )


def download_cataract_detection_and_classification(dataset_dir):
    clone_git("https://github.com/piygot5/Cataract-Detection-and-Classification.git", "tmp/")

    arrange_dataset(
        "tmp/",
        dataset_dir,
        {
            "1": [
                "phase 1 Binary/Binary classification/dataset/phase1/cataract",
                "phase 2 Types/Sift_statistics/Dataset/mild cataract",
                "phase 2 Types/Sift_statistics/Dataset/severe cataract",
            ],
            "0": [
                "phase 1 Binary/Binary classification/dataset/phase1/normal",
                "phase 2 Types/Sift_statistics/Dataset/healthy eyes",
            ],
        },
    )


def download_kaggle_cataract_aksh(dataset_dir):
    download_dir = download_kaggle_dataset("akshayramakrishnan28/cataract-classification-dataset")

    arrange_dataset(download_dir, dataset_dir, {"1": ["train/immature", "train/mature"]})


def download_kaggle_cataract_kersh(dataset_dir):
    download_dir = download_kaggle_dataset("kershrita/cataract")

    arrange_dataset(download_dir, dataset_dir, {"1": ["Cataract"], "0": ["Normal"]})


def download_kaggle_cataract_nand(dataset_dir):
    download_dir = download_kaggle_dataset("nandanp6/cataract-image-dataset")

    arrange_dataset(
        download_dir,
        dataset_dir,
        {
            "1": ["processed_images/train/cataract", "processed_images/test/cataract"],
            "0": ["processed_images/train/normal", "processed_images/test/normal"],
        },
        allow_duplications=True,
    )


DOWNLOAD_FUNCTIONS = {
    "Cataract_Detection-using-CNN": download_cataract_detection_using_cnn,
    "Cataract-Detection-and-Classification": download_cataract_detection_and_classification,
    # "kaggle_cataract_aksh": download_kaggle_cataract_aksh, # salt-n-pepper noise 적용으로 버림
    "kaggle_cataract_kersh": download_kaggle_cataract_kersh,
    "kaggle_cataract_nand": download_kaggle_cataract_nand,
}


if __name__ == "__main__":
    parser = ArgumentParser(description="Download all datasets")
    parser.add_argument(
        "--download_path", type=str, default="dataset/data", help="Path to download datasets (default: dataset/data)"
    )

    args = parser.parse_args()

    for dataset_name, download_f in DOWNLOAD_FUNCTIONS.items():
        print(f">>> Downloading dataset {dataset_name}...")
        dataset_dir = os.path.join(args.download_path, dataset_name)
        if download_completed(dataset_dir):
            print(f"{dataset_dir} already exists.. skip download")
        else:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            download_f(dataset_dir)
            mark_as_completed(dataset_dir)
