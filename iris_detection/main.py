"""Iris Detection Algorithm (Main)"""

from pupli_detection import Pupil
from hough_transform import CircleDetection


class IrisDetectionParams:
    """홍채 감지에 필요한 매개변수 정의"""

    def __init__(
        self,
        image_path,
        threshold_value,
        min_dist,
        param1,
        param2,
        min_radius,
        max_radius,
        canny_thr1,
        canny_thr2,
        save=False,
    ):
        self.image_path = image_path
        self.threshold_value = threshold_value
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.canny_thr1 = canny_thr1
        self.canny_thr2 = canny_thr2
        self.save = save


def iris_detect(params: IrisDetectionParams):
    """동공 중심을 기준으로 홍채 탐지

    Args:
        params (IrisDetectionParams): 매개변수 객체

    Returns:
        tuple: 탐지된 동공 중심 좌표에 가장 가까운 원 (x, y, 반지름).
    """
    # Detect the pupil center
    pupil_detector = Pupil(params.image_path, params.threshold_value)

    # Target coordinates for the closest circle search
    target_x, target_y = (
        pupil_detector.x,
        pupil_detector.y,
    )

    # save image
    # pupil_detector.save_position_on_image("res.png")

    # Detect iris circles around the pupil
    circle_detector = CircleDetection(
        image_path=params.image_path,
        min_dist=params.min_dist,
        param1=params.param1,
        param2=params.param2,
        min_radius=params.min_radius,
        max_radius=params.max_radius,
        canny_thr1=params.canny_thr1,
        canny_thr2=params.canny_thr2,
        target_x=target_x,
        target_y=target_y,
    )

    res_circle = circle_detector.process_image(save=params.save)

    return res_circle


# Usage example
if __name__ == "__main__":

    image_path = "example/cataract2.jpg"

    # Set parameters using the IrisDetectionParams class
    params = IrisDetectionParams(
        image_path=image_path,
        threshold_value=50,
        min_dist=1,  # 원 사이 최소 거리
        param1=30,  # Canny 엣지 검출기 상한값
        param2=30,  # 중심 검출 임계값
        min_radius=0,  # 최소 반지름 (홍채 추정치에 맞게 조정)
        max_radius=150,  # 최대 반지름 (홍채 추정치에 맞게 조정)
        canny_thr1=30,  # Canny threshold1
        canny_thr2=30,  # Canny threshold2
        save=True,  # Save the result image
    )

    # Run the iris detection
    res_circle = iris_detect(params)

    if res_circle:
        print(f"x, y, radius = {res_circle[0]}, {res_circle[1]}, {res_circle[2]}")
    else:
        print("No circle detected.")
