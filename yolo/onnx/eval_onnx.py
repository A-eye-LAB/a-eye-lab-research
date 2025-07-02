import os
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

none = 0
one = 0
two = 0

# For binary classification evaluation
y_true = []  # Ground truth labels (based on directory: 0 or 1)
y_pred = []  # Predicted labels (0: no detection, 1: detection)

def preprocess_image(image_path, input_size=(640, 640)):
    """
    이미지 로드 및 모델 입력 전처리
    """
    image = cv2.imread(image_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)

    orig_h, orig_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, input_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # HWC → CHW
    input_tensor = np.expand_dims(image_transposed, axis=0).astype(np.float32)

    return input_tensor, (orig_w, orig_h), image



def main(session, image_path, input_size=(640, 640), conf_threshold=0.6):
    global none, one, two, y_true, y_pred

    image = cv2.imread(image_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    input_tensor, (orig_w, orig_h), original_image = preprocess_image(
        image_path, input_size
    )

    # Get input details
    input_names = [i.name for i in session.get_inputs()]

    input_feed = {
        input_names[0]: input_tensor,  # "images"
        input_names[1]: np.array([orig_w, orig_h], dtype=np.float32),  # "image_size"
    }

        # Run inference
    outputs = session.run(None, input_feed)

        # Parse results - only use confidence for detection
    # Count detections based on confidence threshold
    detection_count = 0

    # Handle different output formats
    try:
        for i in range(len(outputs[2])):
            if outputs[2][i] >= conf_threshold:
                detection_count += 1

    except Exception as e:
        print(f"Error processing detection results: {e}")
        detection_count = 0

    # Get ground truth from directory name
    # 0: 눈이 없는 이미지 (검출되지 않아야 함)
    # 1: 눈이 있는 이미지 (검출되어야 함)
    dir_name = os.path.basename(os.path.dirname(image_path))
    ground_truth = int(dir_name)  # 0 or 1

    # Get prediction based on detection results
    # 0: 검출되지 않음 (no detection)
    # 1: 하나 이상 검출됨 (one or more detections)
    prediction = 1 if detection_count > 0 else 0

    # Store for evaluation
    y_true.append(ground_truth)
    y_pred.append(prediction)

    # Uncomment below for detailed per-image results
    # print(f"  GT: {ground_truth} | Pred: {prediction} | Detections: {detection_count}")

    # Count detections for statistics
    if detection_count == 0:
        none += 1
    elif detection_count == 1:
        one += 1
    else:
        two += 1

def process_directory(session, input_dir, exclude_dirs=None, conf_threshold=0.7):
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
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
        main(session, image_path, (640, 640), conf_threshold)

    print(f"Found {len(image_files)} images")
    print("Images per directory:")
    for dir_name, count in sorted(dir_counts.items()):
        print(f"  {dir_name}: {count} images")
    print(f"\nAll predictions completed!")
    print(f"Detection count distribution - none: {none}, one: {one}, two+: {two}")

    # Calculate evaluation metrics
    print("\n" + "="*60)
    print("EYE DETECTION BINARY CLASSIFICATION EVALUATION (ONNX)")
    print(f"Confidence Threshold: {conf_threshold}")
    print("="*60)

    print("Ground Truth Labels:")
    print("  0: 눈이 없는 이미지 (검출되지 않아야 함)")
    print("  1: 눈이 있는 이미지 (검출되어야 함)")
    print("\nPrediction Labels:")
    print("  0: 검출되지 않음")
    print("  1: 하나 이상 검출됨")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print("                Predicted")
    print("                No Det(0)  Det(1)")
    print(f"Actual No Eye(0):  {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"Actual Eye(1):     {cm[1,0]:4d}     {cm[1,1]:4d}")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nPerformance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\nDetailed Results:")
    print(f"True Positives (TP):  {tp:4d} - 눈이 있고 올바르게 검출됨")
    print(f"True Negatives (TN):  {tn:4d} - 눈이 없고 올바르게 검출 안됨")
    print(f"False Positives (FP): {fp:4d} - 눈이 없는데 잘못 검출됨")
    print(f"False Negatives (FN): {fn:4d} - 눈이 있는데 검출 못함")
    print(f"Specificity:          {specificity:.4f} ({specificity*100:.1f}%)")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Eye (0)', 'Eye Detected (1)']))

if __name__ == "__main__":
    input_dir = '/hdd/eye_detect/testset'
    model_path = 'iris.onnx'  # ONNX model path

    # Load ONNX model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # GPU first, then CPU fallback
    session = ort.InferenceSession(model_path, providers=providers)

    # Print model info
    print(f"Model input: {session.get_inputs()[0].name}")
    print(f"Model input shape: {session.get_inputs()[0].shape}")
    print(f"Model output: {session.get_outputs()[0].name}")
    print(f"Model output shape: {session.get_outputs()[0].shape}")

    process_directory(session, input_dir, exclude_dirs=None)