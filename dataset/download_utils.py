import os
import shutil

import kagglehub
from git import Repo

COMPLETE_FILE = "1.complete"


def download_kaggle_dataset(kaggle_dir, download_dir=None):
    """
    kaggle dataset을 다운로드한다.
    download_dir이 주어지면 해당 위치로 다운로드, None이면 default dir로 다운로드됨
    다운로드 받은 path를 return
    """
    tmp_dir = kagglehub.dataset_download(kaggle_dir, force_download=True)

    if download_dir is not None:
        shutil.rmtree(download_dir, ignore_errors=True)
        subdirs = os.listdir(tmp_dir)
        for _dir in subdirs:
            shutil.move(os.path.join(tmp_dir, _dir), os.path.join(download_dir, _dir))
        return download_dir
    else:
        return tmp_dir


def clone_git(repo_url, download_dir):
    shutil.rmtree(download_dir, ignore_errors=True)
    Repo.clone_from(repo_url, download_dir)


def arrange_dataset(source_dir, dest_dir, label_dir_dict, allow_duplications=False):
    """
    source_dir에 있는 데이터셋을 label_dir_dict에 명시된 대로 dest_dir로 옮긴다.
    label_dir_dict: dictionary
        - key[str]: dest_dir에 저장될 라벨명
        - value[List]: 해당 key 하위에 들어갈 source_dir의 디렉토리 이름)
    """
    for label, dirs in label_dir_dict.items():
        os.makedirs(os.path.join(dest_dir, label))
        for dir in dirs:
            _mv_dir(os.path.join(source_dir, dir), os.path.join(dest_dir, label), allow_duplications)
    shutil.rmtree(source_dir)


def _mv_dir(source_dir, dest_dir, allow_duplications):
    """폴더 내의 모든 파일을 옮긴다."""
    for f in os.listdir(source_dir):
        source_file = os.path.join(source_dir, f)
        dest_file = os.path.join(dest_dir, f)
        try:
            shutil.move(source_file, dest_file)
        except Exception as e:
            if allow_duplications and os.path.exists(dest_file):
                f_ = _find_unique_filename(dest_dir, f)
                shutil.move(source_file, os.path.join(dest_dir, f_))
            else:
                raise e


def _find_unique_filename(dir, filename):
    n = 1
    while os.path.exists(os.path.join(dir, filename + f" ({n})")):
        n += 1
    return filename + f" ({n})"


def download_completed(download_dir):
    if os.path.exists(os.path.join(download_dir, COMPLETE_FILE)):
        return True
    else:
        return False


def mark_as_completed(download_dir):
    with open(os.path.join(download_dir, COMPLETE_FILE), "w") as f:
        f.write("Download complete")
