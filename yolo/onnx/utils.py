import cv2


def draw_boxes(image, detections, output_path="result.jpg", color=(0, 255, 0)):
    """탐지된 iris와 crop을 위한 box 그리는 함수
    시각화용 함수이기 때문에 배포시에는 사용되지 않습니다!!
    """
    bboxes = detections["iris_bbox"]
    crop_bbox = detections["crop_bbox"]
    scores = detections["confidence"]

    for box, score in zip(bboxes, scores):
        x1, y1, x2, y2 = map(int, box)  # float → int 변환
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            (f"{score[0]:.2f}" if isinstance(score, list) else f"{score:.2f}"),
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    for box in crop_bbox:
        x1, y1, x2, y2 = map(int, box)  # float → int 변환
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imwrite(output_path, image)

    return image
