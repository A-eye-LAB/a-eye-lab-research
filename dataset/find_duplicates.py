"""
Hash 값 비교를 통해 중복 이미지 검사를 시행하고 중복된 이미지들을 제거합니다.
"""

import hashlib
import os
from argparse import ArgumentParser
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


def find_duplicate_images(dataset_path):
    hash_dict = defaultdict(list)
    for root, _, files in os.walk(dataset_path):
        for img_file in tqdm(files, desc=root):
            if not img_file.endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            img_path = os.path.join(root, img_file)
            img = Image.open(img_path).resize((100, 100)).convert("RGB")
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            hash_dict[img_hash].append(img_path)
    duplicates = [files for files in hash_dict.values() if len(files) > 1]
    return duplicates


if __name__ == "__main__":
    parser = ArgumentParser(description="Find duplicate images")
    parser.add_argument("--dataset_path", type=str, help="Path to download datasets")
    parser.add_argument("--remove", action="store_true", help="Remove duplicates")

    args = parser.parse_args()

    duplicates = find_duplicate_images(args.dataset_path)
    print(f"Found {len(duplicates)} duplicate pairs (total {sum(len(files) for files in duplicates)} images)")
    for files in duplicates:
        print(files)

    if args.remove:
        for files in duplicates:
            for file in files[1:]:
                os.remove(file)
        print("Duplicates removed")
