import cv2
from ultralytics import YOLO


def predict(model, image_path, threshold=0.5):
    image = cv2.imread(image_path)
    result = model.predict(image, conf=threshold)
    return result


if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    predict(model, "test.jpg", threshold=0.5)