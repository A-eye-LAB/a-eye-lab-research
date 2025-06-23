import os
import cv2
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

none = 0
one = 0
two = 0

# For binary classification evaluation
y_true = []  # Ground truth labels (based on directory: 0 or 1)
y_pred = []  # Predicted labels (0: no detection, 1: detection)

def main( model, image_path, output_dir, input_dir, conf_threshold=0.7):
    global none, one, two, y_true, y_pred
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    results = model(image)[0]
    boxes = results.boxes.data.tolist()

    # Get ground truth from directory name (0 or 1)
    dir_name = os.path.basename(os.path.dirname(image_path))
    ground_truth = int(dir_name)  # 0 or 1

    # Filter boxes by confidence threshold
    filtered_boxes = [box for box in boxes if box[4] >= conf_threshold]

    # Get prediction (0: no detection, 1: detection) based on filtered boxes
    prediction = 1 if len(filtered_boxes) > 0 else 0

    # Store for evaluation
    y_true.append(ground_truth)
    y_pred.append(prediction)

    if filtered_boxes:
        for box in filtered_boxes:
            x1, y1, x2, y2, score, class_id = box
            if score < conf_threshold: continue

            # Convert to integers for drawing
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add confidence score to the label
            label = f"eye {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Create output directory if it doesn't exist
        rel_path = os.path.relpath(image_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, image)

    if len(filtered_boxes) == 0:
        none += 1
    elif len(filtered_boxes) == 1:
        one += 1
    else:
        two += 1

def process_directory(model, input_dir, exclude_dirs=None, output_dir=None, conf_threshold=0.7):
    if exclude_dirs is None:
        exclude_dirs = []

    # Get all image files, excluding specified directories
    image_files = []
    dir_counts = {}  # Dictionary to store image count per directory

    for root, dirs, files in os.walk(input_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                image_files.append(image_path)

                # Get directory name (0, 1, etc.)
                dir_name = os.path.basename(root)
                if dir_name not in dir_counts:
                    dir_counts[dir_name] = 0
                dir_counts[dir_name] += 1

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    # Process each image
    for image_path in image_files:
        print(f"\nProcessing {os.path.basename(image_path)}...")
        main(model, image_path, output_dir, input_dir, conf_threshold)


    print(f"Found {len(image_files)} images")
    print("Images per directory:")
    for dir_name, count in sorted(dir_counts.items()):
        print(f"  {dir_name}: {count} images")
    print(f"\nAll predictions completed! Results saved in '{output_dir}' directory")
    print(f"none: {none}, one: {one}, two: {two}")

    # Calculate evaluation metrics
    print("\n" + "="*50)
    print("BINARY CLASSIFICATION EVALUATION")
    print(f"Confidence Threshold: {conf_threshold}")
    print("="*50)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("Predicted:  0    1")
    print(f"Actual 0: [{cm[0,0]:4d} {cm[0,1]:4d}]")
    print(f"Actual 1: [{cm[1,0]:4d} {cm[1,1]:4d}]")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nMetrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\nDetailed Metrics:")
    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Specificity:     {specificity:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Detection (0)', 'Detection (1)']))

if __name__ == "__main__":
    input_dir = '/hdd/eye_detect/testset'
    output_dir = 'test_results'
    model_path = 'best.pt'

    model = YOLO(model_path)
    process_directory(model, input_dir, exclude_dirs=None, output_dir=output_dir)