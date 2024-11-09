"""
Usage: python datset/download_all_datasets.py
    - DOWNLOAD_FUNCTIONS에 있는 데이터셋들을 전부 다운로드
    - 다운로드가 완료된 데이터셋들은 제외하고 다운로드 (1.complete 파일 존재유무로 검사)

Add dataset:
    - dataset.utils.download_functions에 해당 데이터셋 다운로드 함수 추가
    - DOWNLOAD_FUNCTIONS에 데이터셋 이름과 다운로드 함수를 추가
"""

import shutil

from utils.download_functions import *
from utils.download_utils import download_completed, mark_as_completed

URLS = {
    "kaggle_cataract_aksh": "akshayramakrishnan28/cataract-classification-dataset",
    "kaggle_cataract_kersh": "kershrita/cataract",
    "kaggle_cataract_nand": "nandanp6/cataract-image-dataset",
}

DOWNLOAD_FUNCTIONS = {
    "Cataract_Detection-using-CNN": download_cataract_detection_using_cnn,
    "Cataract-Detection-and-Classification": download_cataract_detection_and_classification,
    "kaggle_cataract_aksh": download_kaggle_cataract_aksh,
    "kaggle_cataract_kersh": download_kaggle_cataract_kersh,
    "kaggle_cataract_nand": download_kaggle_cataract_nand,
}

DATASET_DIR_FORMAT = "dataset/data/{dataset_name}/"


for dataset_name, download_f in DOWNLOAD_FUNCTIONS.items():
    print(f">>> Downloading dataset {dataset_name}...")
    dataset_dir = DATASET_DIR_FORMAT.format(dataset_name=dataset_name)
    if download_completed(dataset_dir):
        print(f"{dataset_dir} already exists.. skip download")
    else:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        download_f(dataset_dir)
        mark_as_completed(dataset_dir)
