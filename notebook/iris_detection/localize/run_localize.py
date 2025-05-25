import os
from argparse import ArgumentParser

import cv2
import numpy as np


def _localize_image_with_bbox(img, bbox):
    """
    bounding box 중심을 기준으로 이미지 중심으로 이동
    """
    x_min, y_min, x_max, y_max = bbox
    bbox_center_x = (x_min + x_max) // 2
    bbox_center_y = (y_min + y_max) // 2

    h, w = img.shape[:2]
    dx = w // 2 - bbox_center_x
    dy = h // 2 - bbox_center_y

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    localized_img = cv2.warpAffine(
        img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    ) 
    new_bbox = (x_min + dx, y_min + dy, x_max + dx, y_max + dy)

    return localized_img, new_bbox


def _draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    """
    이미지에 bbox 그리기
    """
    x_min, y_min, x_max, y_max = bbox
    return cv2.rectangle(img.copy(), (x_min, y_min), (x_max, y_max), color, thickness)


def process(image_path: str, out_path: str, bbox: tuple):

    img = cv2.imread(image_path) 
    img_with_bbox = _draw_bbox(img, bbox) 
    localized_img, new_bbox = _localize_image_with_bbox(img, bbox)
    localized_with_bbox = _draw_bbox(localized_img, new_bbox)
 
    # 생략가능
    root_dir, mid_dir = image_path.split("/", 1)
    mid_dir, filename = mid_dir.rsplit("/", 1)
    name, ext = os.path.splitext(filename)
    new_dir = os.path.join(out_path, mid_dir)
    os.makedirs(new_dir, exist_ok=True)

    cv2.imwrite(os.path.join(new_dir, f"{name}_orig_with_bbox{ext}"), img_with_bbox)
    cv2.imwrite(
        os.path.join(new_dir, f"{name}_aligned_with_bbox{ext}"), localized_with_bbox
    )


if __name__ == "__main__":
    image_path = "/Users/iyujin/Desktop/eye_detection/resize_img/test_Immature (21).jpg"

    # yolo bbox
    bbox = (50, 130, 160, 190)

    # run process
    process(image_path, "data", bbox)
