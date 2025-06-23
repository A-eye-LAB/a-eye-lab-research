import cv2
import numpy as np
import onnxruntime as ort


def preprocess_image(image_path, input_size=(640, 640)):
    """
    이미지 로드 및 모델 입력 전처리
    """
    image = cv2.imread(image_path)

    orig_h, orig_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, input_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # HWC → CHW
    input_tensor = np.expand_dims(image_transposed, axis=0).astype(np.float32)

    return input_tensor, (orig_w, orig_h), image


def run_onnx_inference(onnx_path, image_path, input_size=(640, 640)):
    input_tensor, (orig_w, orig_h), original_image = preprocess_image(
        image_path, input_size
    )

    session = ort.InferenceSession(onnx_path)
    input_names = [i.name for i in session.get_inputs()]

    input_feed = {
        input_names[0]: input_tensor,  # "images"
        input_names[1]: np.array([orig_w, orig_h], dtype=np.float32),  # "image_size"
    }

    outputs = session.run(None, input_feed)

    result = {
        "crop_bbox": outputs[0].tolist(),
        "iris_bbox": outputs[1].tolist(),
        "confidence": outputs[2].tolist(),
    }
    return result, original_image


if __name__ == "__main__":

    onnx_model_path = "iris_nms.onnx"
    image_path = "/home/yujin/a-eye-lab/yolo/onnx/testset/one_eye.jpg"

    detections, image = run_onnx_inference(onnx_model_path, image_path)

    print(detections)

    ### 시각화
    from utils import draw_boxes

    save_path = "iris_nms_result.jpg"
    draw_boxes(image, detections, save_path)
