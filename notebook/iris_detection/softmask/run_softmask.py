import cv2
import numpy as np
from PIL import Image


def detect_iris(img_path):
    # 이미지 불러오기
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_eq = cv2.equalizeHist(gray)
    gray = cv2.medianBlur(gray_eq, 9)

    # 이미지 크기에 따라 매개변수 조정
    height, width = gray.shape
    minDist = 1000  # height // 8  
    minRadius = height // 15  # 20  
    maxRadius = height // 2  # 4  

    # 허프 원 검출 : iris(홍채) 영역 찾기
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=minDist,
        param1=30,
        param2=30,
        minRadius=minRadius,
        maxRadius=maxRadius,
    )

    mask = np.zeros_like(gray, dtype=np.uint8)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(mask, (i[0], i[1]), i[2], 255, -1)

    soft_mask = cv2.GaussianBlur(mask, (101, 101), 0)

    soft_mask_3ch = cv2.merge([soft_mask] * 3)
    eye_segment_soft = (img.astype(np.float32) * (soft_mask_3ch / 255.0)).astype(
        np.uint8
    )

    return eye_segment_soft


if __name__ == "__main__":
    img_path = "/home/yujin/data/Cataract_Detection-using-CNN/0/cat_0_1644.jpg"
    res_array = detect_iris(img_path)
    eye_segment_soft = Image.fromarray(res_array)
    eye_segment_soft.save("eye_segment_soft.png")
