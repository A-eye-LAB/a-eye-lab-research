# 파일명: crop_eyes_haar.py
import os
import cv2
from tqdm import tqdm


input_root = "/home/mia/a-eye-lab-research/data_original2"
output_root = "/home/mia/a-eye-lab-research/cropped_eyes_haar"
os.makedirs(output_root, exist_ok=True)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def detect_and_crop_eye(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(eyes) == 0:
        return None

   
    eyes = sorted(eyes, key=lambda b: b[0])
    x, y, w, h = eyes[0]
    eye_img = image[y:y+h, x:x+w]
    return eye_img


for dataset in os.listdir(input_root):
    dataset_path = os.path.join(input_root, dataset)
    for label in ['0', '1']:
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        save_label_path = os.path.join(output_root, label)
        os.makedirs(save_label_path, exist_ok=True)

        for fname in tqdm(os.listdir(label_path), desc=f" {dataset}/{label}"):
            if not fname.lower().endswith(('.jpg', '.png')):
                continue

            fpath = os.path.join(label_path, fname)
            img = cv2.imread(fpath)

            if img is None:
                continue

            eye_crop = detect_and_crop_eye(img)
            if eye_crop is None or eye_crop.size == 0:
                continue

            eye_crop = cv2.resize(eye_crop, (224, 224))
            save_path = os.path.join(save_label_path, f"{dataset}_{label}_{fname}")
            cv2.imwrite(save_path, eye_crop)

