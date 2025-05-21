import os
from argparse import ArgumentParser

import cv2
import numpy as np
from tqdm import tqdm


def _resize_with_ratio(raw_img, h_):
    h, w = raw_img.shape
    resize_ratio = h_ / h
    w_ = int(w * resize_ratio)
    img = cv2.resize(raw_img, (w_, h_))
    return img, resize_ratio


def _detect_iris(image_path, blur, h_, param1, param2):
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img, resize_ratio = _resize_with_ratio(raw_img, h_)
    img = cv2.medianBlur(img, blur)

    hough_circle = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 3, param1=param1, param2=param2)

    if hough_circle is None:
        return img, (None, None)

    x_, y_, radius = np.round(hough_circle[0, :]).astype("int")[0]
    cv2.circle(img, (x_, y_), radius, (255, 0, 0), 4)

    x, y, radius = int(x_ / resize_ratio), int(y_ / resize_ratio), int(radius / resize_ratio)
    return img, (x, y)


def _localize_image(image_path, x, y):
    img = cv2.imread(image_path)
    if x is None or y is None:
        return img
    new_img = np.zeros_like(img)
    h, w, _ = img.shape
    dx, dy = w // 2 - x, h // 2 - y
    new_img[max(dy, 0) : min(dy + h, h), max(dx, 0) : min(dx + w, w)] = img[
        max(-dy, 0) : min(-dy + h, h), max(-dx, 0) : min(-dx + w, w)
    ]
    return new_img


def process(image_path, out_path):
    img, center = _detect_iris(image_path, blur=19, h_=512, param1=30, param2=50)
    new_img = _localize_image(image_path, *center)

    root_dir, mid_dir = image_path.split("/", 1)
    mid_dir, filename = mid_dir.rsplit("/", 1)
    new_dir = os.path.join(out_path, mid_dir)
    os.makedirs(new_dir, exist_ok=True)

    cv2.imwrite(os.path.join(new_dir, filename), new_img)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_path", type=str, default="data")
    parser.add_argument("--out_path", type=str, default="data_localized")
    args = parser.parse_args()

    all_files = []
    for dataset in os.listdir(args.in_path):
        for label in os.listdir(os.path.join(args.in_path, dataset)):
            if not os.path.isdir(os.path.join(args.in_path, dataset, label)):
                continue
            for filename in os.listdir(os.path.join(args.in_path, dataset, label)):
                all_files.append(os.path.join(args.in_path, dataset, label, filename))

    for image_path in tqdm(all_files):
        process(image_path, args.out_path)
