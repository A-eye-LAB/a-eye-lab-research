from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob

# 입력한 디렉토리 내부에 있는 모든 이미지들을 predict 하고 결과를 저장.
# 결과는 이미지와 라벨
# 제외할 디렉토리 지정 가능

def predict_image(model, image_path, conf_threshold=0.25, output_dir=None, input_dir=None):

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Make prediction
    results = model(image, conf=conf_threshold)[0]

    print(f"Detected {len(results.boxes)} objects")

    # Get all predicti
    boxes = results.boxes.data.tolist()

    if boxes:
        # Find the box with highest confidence score
        best_box = max(boxes, key=lambda x: x[4])
        x1, y1, x2, y2, score, class_id = best_box

        # Convert to integers for drawing
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])


        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if x2-x1 > y2-y1:
            rect_size = x2-x1
        else:
            rect_size = y2-y1

        crop_size = 0
        crop_size = min(img_width, img_height, rect_size *3)

        x1 = center_x - (crop_size / 2)
        x2 = center_x + (crop_size / 2)
        y1 = center_y - (crop_size / 2)
        y2 = center_y + (crop_size / 2)

        if x1 < 0:
            x2 += (0 - x1)
            x1 = 0
        if y1 < 0:
            y2 += (0 - y1)
            y1 = 0
        if x2 > img_width:
            x1 -= (x2 - img_width)
            x2 = img_width
        if y2 > img_height:
            y1 -= (y2 - img_height)
            y2 = img_height

        # Convert to integers for drawing
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        assert abs(x2-x1) == abs(y2-y1), f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}"

        # crop image
        crop_image = image[y1:y2, x1:x2]

        # save image
        rel_path = os.path.relpath(image_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, crop_image)

def process_directory(model, input_dir, exclude_dirs=None, output_dir=None, conf_threshold=0.25):
    if exclude_dirs is None:
        exclude_dirs = []

    # Get all image files, excluding specified directories
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Process each image
    for image_path in image_files:
        print(f"\nProcessing {os.path.basename(image_path)}...")
        predict_image(model, image_path, conf_threshold, output_dir, input_dir)

    print(f"\nAll predictions completed! Results saved in '{output_dir}' directory")

if __name__ == '__main__':
    # Path to your trained model
    model_path = 'best.pt'  # or 'runs/train/exp/weights/best.pt'

    # Directory containing images to process
    input_dir = '/hdd/data/eval_test'  # replace with your input directory
    # input_dir = './edge_case'
    output_dir = '/hdd/data/cropted_eval'
    # output_dir = 'prediction_results11'

    model = YOLO(model_path)
    # List of directories to exclude
    # exclude_dirs = ['C001', 'C002', 'C003']  # Add directories you want to exclude
    exclude_dirs = []

    # Process all images in the directory
    process_directory(model, input_dir, exclude_dirs, output_dir)