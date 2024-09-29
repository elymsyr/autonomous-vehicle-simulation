import numpy as np
import cv2
import math

class BirdEyeViewMapping():
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
    
    def pipeline(self, image):
        region_of_interest_vertices = [
            (0, self.height),
            (self.width / 2, self.height / 2),
            (self.width, self.height),
        ]
        
        polygon = np.array(region_of_interest_vertices, np.int32)
        polygon = polygon.reshape((-1, 1, 2))  # Reshape for polylines

        # Draw the polygon outline
        cv2.polylines(image, [polygon], isClosed=True, color=(255, 120, 0), thickness=2)

        # Convert to grayscale and apply Canny edge detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cannyed_image = cv2.Canny(gray_image, 100, 200)

        # Mask out the region of interest
        cropped_image = self.region_of_interest(
            cannyed_image,
            np.array([region_of_interest_vertices], np.int32)
        )

        # Perform Hough Line Transformation to detect lines
        lines = cv2.HoughLinesP(
            cropped_image,
            rho=6,
            theta=np.pi / 60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=25
        )

        # Separating left and right lines based on slope
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []

        if lines is None:
            return image

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                if math.fabs(slope) < 0.5:  # Ignore nearly horizontal lines
                    continue
                if slope <= 0:  # Left lane
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:  # Right lane
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

        # Fit a linear polynomial to the left and right lines
        min_y = int(image.shape[0] * (3 / 5))  # Slightly below the middle of the image
        max_y = image.shape[0]  # Bottom of the image

        if left_line_x and left_line_y:
            poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
        else:
            left_x_start, left_x_end = 0, 0  # Defaults self, if no lines detected

        if right_line_x and right_line_y:
            poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))
        else:
            right_x_start, right_x_end = 0, 0  # Defaults self, if no lines detected

        # Create the filled polygon between the left and right lane lines
        lane_image = self.draw_lane_lines(
            image,
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y]
        )

        return lane_image
    