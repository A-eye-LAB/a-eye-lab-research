import cv2
import numpy as np


class CircleDetection:
    def __init__(
        self,
        params,
        target_x,
        target_y,
    ):
        self.image_path = params.image_path
        self.min_dist = params.min_dist  # 원 사이 최소 거리
        self.param1 = params.param1  # Canny 엣지 검출기 상한값
        self.param2 = params.param2  # 중심 검출 임계값
        self.min_radius = params.min_radius  # 최소 반지름 (홍채 추정치에 맞게 조정)
        self.max_radius = params.max_radius  # 최대 반지름 (홍채 추정치에 맞게 조정)
        self.canny_thr1 = params.canny_thr1  # Canny threshold1
        self.canny_thr2 = params.canny_thr2  # Canny threshold2
        self.target_x = target_x
        self.target_y = target_y

    def preprocess_image(self):
        """이미지 전처리: 그레이스케일 변환, 블러링, Canny 에지 검출"""

        src = cv2.imread(self.image_path)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_thr1, self.canny_thr2)
        return src, edges

    def detect_circle_candidates(self, edges, top_n=10):
        """허프 변환을 사용하여 상위 N개의 원 반환"""

        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            return circles[:top_n]

        else:
            return None

    def find_closest_circle(self, circles):
        """주어진 좌표에서 가장 가까운 원 찾기"""

        closest_circle = None
        min_distance = float("inf")

        for x, y, radius in circles:
            distance = np.hypot(
                float(x) - float(self.target_x), float(y) - float(self.target_y)
            )
            if distance < min_distance:
                min_distance = distance
                closest_circle = (x, y, radius)

        if closest_circle is None:
            print("가장 가까운 원을 찾을 수 없습니다.")

        return closest_circle

    @staticmethod
    def draw_circles(src, circles, color=(0, 255, 0), center_color=(0, 0, 255)):
        """이미지에 원 그리기"""

        for x, y, radius in circles:
            cv2.circle(src, (x, y), radius, color, 2)
            cv2.circle(src, (x, y), 2, center_color, 3)

    def process_image(self, save=False):
        """이미지에서 10개의 원을 검출하고 타겟 좌표와 가장 가까운 원을 찾고 그리기"""

        src, edges = self.preprocess_image()
        circles = self.detect_circle_candidates(edges)

        if circles is not None:
            closest_circle = self.find_closest_circle(circles)

            if save:
                # 원 후보들 그리기
                # self.draw_circles(src, circles)

                # 가장 가까운 원 그리기
                if closest_circle is not None:
                    self.draw_circles(
                        src,
                        [closest_circle],
                        color=(255, 0, 0),
                        center_color=(0, 100, 255),
                    )
                cv2.imwrite("result.png", src)

            return list(closest_circle)

        else:
            return []
