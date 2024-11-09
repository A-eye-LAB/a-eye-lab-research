from utils.download_utils import download_kaggle_dataset, clone_git, arrange_dataset


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
