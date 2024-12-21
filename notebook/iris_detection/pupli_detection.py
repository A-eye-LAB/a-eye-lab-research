"""Pupil Center Coordinate Detection Algorithm
- reference code : https://github.com/antoinelame/GazeTracking
"""

import numpy as np
import cv2


class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil from an image file.
    """

    def __init__(self, image_path, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None
        # self.radius = None
        self.original_image = None

        # Load the image from the given path
        self.detect_iris(image_path)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris"""

        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = np.clip(new_frame, 0, 255).astype(np.uint8)

        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = np.clip(new_frame, 0, 255).astype(np.uint8)

        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    def detect_iris(self, image_path):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid from the loaded image.
        """

        # Load the image
        self.original_image = cv2.imread(image_path)
        print(f"input image size : {self.original_image.shape}")
        if self.original_image is None:
            raise ValueError(f"Could not load image from path: {image_path}")

        # Convert the image to grayscale
        eye_frame_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        self.iris_frame = self.image_processing(eye_frame_gray, self.threshold)

        contours, _ = cv2.findContours(
            self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments["m10"] / moments["m00"])
            self.y = int(moments["m01"] / moments["m00"])

        except (IndexError, ZeroDivisionError):
            pass

    def save_position_on_image(self, output_path):
        """Draws the pupil position and radius on the original image and saves it."""

        if self.x is not None and self.y is not None:
            # Draw a circle at the detected pupil position
            cv2.circle(
                self.original_image, (self.x, self.y), 4, (0, 255, 0), -1
            )  # Green circle for the center
            cv2.imwrite(output_path, self.original_image)

        else:
            print("Pupil position not detected; nothing to save.")
