import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class BirdEyeViewMapping():
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.interest_vertices = [
            (int(0.1*self.width), int(0.58*self.height)), # left top
            (int(-6*self.width), int(self.height)),     # left bottom (adjusted to image edge)
            (int(0.9*self.width), int(0.58*self.height)),   # right top
            (int(7*self.width), int(self.height))       # right bottom (adjusted to image edge)
        ]
        self.desired_points = np.float32(self.interest_vertices)
        self.image_corners = np.float32([[0, 0], [0, self.height], [self.width, 0], [self.width, self.height]])
        self.matrix = cv2.getPerspectiveTransform(self.desired_points, self.image_corners)
        self.inv_matrix = cv2.getPerspectiveTransform(self.image_corners, self.desired_points)        

    def perspective_transform_point(self, point):
        point_homogeneous = np.array([point[0], point[1], 1.0])
        transformed_point = np.dot(self.matrix, point_homogeneous)
        transformed_point /= transformed_point[2]
        return tuple(map(int, transformed_point[:2]))

    def perspective_transform(self, image):
        # for point in self.interest_vertices:
        #     cv2.circle(image, point, 5, (255,0,0), 3)
        transformed_frame = cv2.warpPerspective(image, self.matrix, (self.width, self.height))
        return transformed_frame


# img = cv2.imread("media/city-car-example-4.png")

# asa = BirdEyeViewMapping(img.shape[1], img.shape[0])

# perspective_transform = asa.perspective_transform(image=img)

# # cv2.imwrite('media/segmented_image.png', region_image)
# asa.show_image(perspective_transform)

import cv2
import numpy as np

class SmoothCircle():
    def __init__(self, center, track_id, radius = 5):
        self.track_id = track_id
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.point = np.array(center, dtype=np.float32)

    def update_point(self, new_point):
        """ Update the point's position while keeping it within the circle. """
        self.point = new_point
        self.push_circle_if_needed()

        # # If the point is outside the circle, move it back to the edge
        # if distance > 3self.radius:
        #     # Normalize the direction vector
        #     direction = (self.point - self.center) / distance
        #     self.point = self.center + direction * self.radius

    def push_circle_if_needed(self):
        """ Push the circle if the point reaches the edge and continues to move. """
        distance = np.linalg.norm(self.point - self.center)
        if distance >= self.radius:
            direction = (self.point - self.center) / distance
            self.center += direction * (distance - self.radius)

    def draw(self, canvas):
        cv2.circle(canvas, tuple(map(int, self.center)), int(self.radius), (0, 255, 0), 2)
        cv2.circle(canvas, tuple(map(int, self.point)), 5, (255, 0, 0), -1)  # Red point

    def reset_circle(self):
        point = self.point
        self.center = point

class ImageOperations():
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
    
    @staticmethod
    def adjust_contrast(self, image, alpha = 2.5, beta = -60):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)        

    @staticmethod
    def canny_image(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        adjusted_image = self.adjust_contrast(gray_image)
        cannyed_image = cv2.Canny(adjusted_image, 100, 200)
        return cannyed_image

    @staticmethod
    def show_image(self, image, image2 = None):
        cv2.imshow('test', image)
        if image2 is not None: cv2.imshow('test2', image2)
        if cv2.waitKey(100000) == 27: return

    @staticmethod
    def region_by_channel(self, image, cluster=3):
        adjusted_image = self.adjust_contrast(image, alpha=2.5, beta=-60)
        dark_channel = self.get_dark_channel(adjusted_image, patch_size=30)
        img_2D = dark_channel.reshape((-1, 1))
        
        kmeans = KMeans(cluster, init='k-means++', max_iter=250, n_init=10, random_state=35).fit(img_2D)
        values = kmeans.predict(img_2D)
        mask1 = values.reshape((image.shape[0], image.shape[1]))
        mask1 = np.expand_dims(mask1, axis=-1)
        centers = kmeans.cluster_centers_.astype(int)

        # Create an empty image with the same height and width as the original
        clustered_image = np.zeros_like(image)

        # Assign colors based on the predicted values
        for i in range(cluster):
            clustered_image[mask1[:, :, 0] == i] = centers[i]

        # Convert clustered_image to uint8 if not already
        clustered_image = np.clip(clustered_image, 0, 255).astype(np.uint8)
        return clustered_image

    @staticmethod
    def get_dark_channel(self, image, patch_size=15):
        # Min value across the RGB channels
        min_channel = np.min(image, axis=2)
        # Apply a min filter with the given patch size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
