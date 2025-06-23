import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
from ultralytics import YOLO


class YOLOWithNMS(nn.Module):
    def __init__(self, model_path, conf_threshold=0.8, iou_threshold=0.6):
        super().__init__()
        self.model = YOLO(model_path).model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def forward(self, x, image_size):
        pred = self.model(x)[0]
        pred = pred[0].transpose(0, 1)

        cx, cy, w, h, conf = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]
        mask = conf >= self.conf_threshold
        cx, cy, w, h, conf = cx[mask], cy[mask], w[mask], h[mask], conf[mask]

        w_scale, h_scale = image_size.unbind(0)
        x1 = (cx - w / 2) * (w_scale / 640.0)
        y1 = (cy - h / 2) * (h_scale / 640.0)
        x2 = (cx + w / 2) * (w_scale / 640.0)
        y2 = (cy + h / 2) * (h_scale / 640.0)

        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        if boxes.shape[0] == 0:
            crop = torch.zeros(1, 4, dtype=torch.float32, device=x.device)
            empty_box = torch.zeros(1, 4, dtype=torch.float32, device=x.device)
            empty_score = torch.zeros(1, 1, dtype=torch.float32, device=x.device)
            return crop, empty_box, empty_score

        else:
            keep = torchvision.ops.nms(boxes, conf, self.iou_threshold)
            if keep.numel() == 0:
                crop = torch.zeros(1, 4, dtype=torch.float32, device=x.device)
                empty_box = torch.zeros(1, 4, dtype=torch.float32, device=x.device)
                empty_score = torch.zeros(1, 1, dtype=torch.float32, device=x.device)
                return crop, empty_box, empty_score

            boxes = boxes[keep]
            scores = conf[keep].unsqueeze(1)

            crop = self.crop_bbox(boxes, w_scale, h_scale)

            return crop, boxes, scores

    def crop_bbox(self, boxes, img_width, img_height):

        center_x = (boxes[:, 0] + boxes[:, 2]) / 2
        center_y = (boxes[:, 1] + boxes[:, 3]) / 2
        rect_size = torch.max(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        crop_size = torch.min(
            torch.stack(
                [
                    img_width.repeat(rect_size.shape[0]),
                    img_height.repeat(rect_size.shape[0]),
                    rect_size * 3,
                ],
                dim=1,
            ),
            dim=1,
        ).values

        x1 = center_x - crop_size / 2
        x2 = center_x + crop_size / 2
        y1 = center_y - crop_size / 2
        y2 = center_y + crop_size / 2

        x_shift = torch.clamp(-x1, min=0.0) - torch.clamp(x2 - img_width, min=0.0)
        y_shift = torch.clamp(-y1, min=0.0) - torch.clamp(y2 - img_height, min=0.0)

        # 조정된 좌표 계산 및 클램핑
        x1 = torch.clamp(x1 + x_shift, 0, img_width)
        y1 = torch.clamp(y1 + y_shift, 0, img_height)
        x2 = torch.clamp(x2 + x_shift, 0, img_width)
        y2 = torch.clamp(y2 + y_shift, 0, img_height)

        crops = torch.stack([x1, y1, x2, y2], dim=1)  # (N,4)
        return crops


def preprocess_image(image_path, input_size=(640, 640)):
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, input_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(image_transposed, axis=0).astype(np.float32)

    return torch.tensor(input_tensor), (orig_w, orig_h), image


def export_onnx(image_path, model_path="iris.pt", output_path="iris_nms.onnx"):

    model = YOLOWithNMS(model_path)
    model.eval()

    dummy_input, (orig_w, orig_h), _ = preprocess_image(image_path)
    dummy_image_size = torch.tensor([orig_w, orig_h], dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_input, dummy_image_size),
        output_path,
        input_names=["images", "image_size"],
        output_names=["crop_bbox", "iris_bbox", "confidence"],
        opset_version=12,
        dynamic_axes={
            "images": {0: "batch"},
            "crop_bbox": {0: "num_detections"},
            "iris_bbox": {0: "num_detections"},
            "confidence": {0: "num_detections"},
        },
    )

    print(f"Export 완료: {output_path}")


if __name__ == "__main__":
    image_path = "../assets/test.jpeg"
    export_onnx(image_path, "iris.pt", "iris_nms.onnx")
