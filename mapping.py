import numpy as np
import cv2
from sklearn.cluster import KMeans

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
        self.view_point = self.perspective_transform_point((self.width//2, self.height))
        self.alert_area = np.array(self.alert_area_calc(), dtype=np.int32)

    def alert_area_calc(self):
        return [
            self.perspective_transform_point((int(0.42 * self.width),int(0.64 * self.height))),  # left top
            self.perspective_transform_point((int(0.17 * self.width),int(0.95 * self.height))),  # left bottom
            self.perspective_transform_point((int(0.74 * self.width),int(0.95 * self.height))),   # right bottom
            self.perspective_transform_point((int(0.56 * self.width),int(0.64 * self.height))),  # right top
            ]

    def perspective_transform_point(self, point, track_id = None):
        point_homogeneous = np.array([point[0], point[1], 1.0])
        transformed_point = np.dot(self.matrix, point_homogeneous)
        transformed_point /= transformed_point[2]
        return tuple(map(int, transformed_point[:2]))

    def perspective_transform_point_w_distance_estimation(self, point, track_id, a=0.1, b=1.00012):
        point_homogeneous = np.array([point[0], point[1], 1.0])
        transformed_point = np.dot(self.matrix, point_homogeneous)
        transformed_point /= transformed_point[2]
        x, y = tuple(map(int, transformed_point[:2]))
        if track_id == 6:
            print(track_id)
            print(f"Before: {x},{y}")
            distance = self.distance_between_points((x,y), self.view_point)
            reducer = self.f_polynomial(distance)
            new_distance = int(distance//reducer)
            print(f"distance: {distance:.0f} - {new_distance:.0f} - {reducer:.1f}")
            x, y = self.new_point_on_line((x,y), new_distance)
            print(f"Before: {x},{y}")
        return (int(x), int(y))

    def perspective_transform(self, image): 
        # for point in self.interest_vertices:
        #     cv2.circle(image, point, 5, (255,0,0), 3)
        transformed_frame = cv2.warpPerspective(image, self.matrix, (self.width, self.height))
        return transformed_frame

    def f(self, x, a, b):
        # return a + b * np.log(x)
        return x - ((x*a) * (b**x))

    def f_polynomial(self, x, a = 0.000005, b = 0, c = 1):
        return a * x**2 + b * x + c

    def new_point_on_line(self, target_point, new_distance):
        # Unpack the view point and target point
        xs, ys = self.view_point
        xt, yt = target_point

        # Calculate the distance between the two points
        d = self.distance_between_points(self.view_point, target_point)

        # Calculate the direction vector
        direction_vector = (xt - xs, yt - ys)
        
        # Normalize the direction vector
        norm_dir_vec = (direction_vector[0] / d, direction_vector[1] / d)

        # Calculate the new point based on the new distance
        xn = xs + new_distance * norm_dir_vec[0]
        yn = ys + new_distance * norm_dir_vec[1]

        return (xn, yn)

    def distance_between_points(self, point1, point2):
        # Unpack the points
        x1, y1 = point1
        x2, y2 = point2

        # Calculate the distance using the distance formula
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return distance

    def area_check(self, point):
        return True if cv2.pointPolygonTest(self.alert_area, point, measureDist=False) > 0 else False

class SmoothCircle():
    def __init__(self, center, track_id, radius=5):
        self.track_id = track_id
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.point = np.array(center, dtype=np.float32)
        self.positions = [np.array(center, dtype=np.float32)]  # Start with initial position
        self.directions = []

    def update_positions(self, new_point):
        # Add the new point to positions
        if not self.positions:
            self.positions.append(new_point)
        else:
            self.positions.append(new_point)
            self.reset_history_on_opposite_direction()

    def detect_opposite_direction(self, overall_vector, new_vector):
        """Detect if the new_vector indicates a direction opposite to the overall movement."""
        # Compute the dot product to find the angle between the vectors
        dot_product = np.dot(overall_vector, new_vector)
        if dot_product < 0:
            return True  # The vectors are moving in opposite directions
        return False
    
    def reset_history_on_opposite_direction(self):
        """Reset history if a significant change in direction is detected."""
        if len(self.positions) >= 3:
            # Calculate overall movement vector (from the first to the last position)
            overall_vector = self.positions[-1] - self.positions[0]
            
            for i in range(1, len(self.positions) - 1):
                current_vector = self.positions[i + 1] - self.positions[i]
                
                # Check if the new vector is opposite to the overall trend
                if self.detect_opposite_direction(overall_vector, current_vector):
                    # Reset history starting from the point of change
                    self.positions = self.positions[i:]
                    break

    def predict_next_n_moves(self, n=3):
        # Reset history if opposite direction is detected
        self.reset_history_on_opposite_direction()

        # Check if there are enough self.positions for prediction
        if len(self.positions) < 2:
            raise ValueError("Not enough data points for prediction.")
        
        # Get the last 5 positions or fewer if not enough points
        last_positions = self.positions[-5:]

        # Calculate the average difference in positions
        dx = last_positions[-1][0] - last_positions[-2][0]
        dy = last_positions[-1][1] - last_positions[-2][1]

        for i in range(len(last_positions) - 1):
            dx += last_positions[i + 1][0] - last_positions[i][0]
            dy += last_positions[i + 1][1] - last_positions[i][1]

        dx /= (len(last_positions) - 1)
        dy /= (len(last_positions) - 1)
        
        # Initialize list to store predicted positions
        predicted_positions = []
        current_position = last_positions[-1]

        # Predict the next 'n' moves
        for i in range(n):
            dx = dx * (5 if dx < .4 and dx > -.4 else 1)
            dy = dy * (5 if dy < .4 and dy > -.4 else 1)
            next_position = current_position + np.array([dx, dy])
            predicted_positions.append(next_position)
            current_position = next_position  # Update current position for the next iteration
        
        return np.array(predicted_positions)

    def update_point(self, new_point):
        """ Update the point's position while keeping it within the circle. """
        self.point = new_point
        self.push_circle_if_needed()

        self.update_positions(new_point)  # Update the position history
        next_path = self.predict_next_n_moves()
        # if self.track_id in [30,5]: 
        #     print(f"Center: {self.center} -> ", end=" ")
        #     for point in next_path:
        #         print(tuple(map(int, point)), end=' - ')                
        #     for point in self.positions:
        #         print(tuple(map(int, point)), end=', ')
        #     print("\n")        
        
        return next_path

    def push_circle_if_needed(self):
        """ Push the circle if the point reaches the edge and continues to move. """
        distance = np.linalg.norm(self.point - self.center)
        if distance >= self.radius:
            direction = (self.point - self.center) / distance
            self.center += direction * (distance - self.radius)
            self.positions.append(self.center)

    def draw(self, canvas):
        cv2.circle(canvas, tuple(map(int, self.center)), int(self.radius), (0, 255, 0), 2)
        cv2.circle(canvas, tuple(map(int, self.point)), 5, (255, 0, 0), -1)  # Red point

    def reset_circle(self):
        point = self.point
        self.center = point

class ImageOperations():
    @staticmethod
    def adjust_contrast(image, alpha = 2.5, beta = -60):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)        

    @staticmethod
    def canny_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        adjusted_image = ImageOperations.adjust_contrast(gray_image)
        cannyed_image = cv2.Canny(adjusted_image, 100, 200)
        return cannyed_image

    @staticmethod
    def show_image(image, image2 = None):
        cv2.imshow('test', image)
        if image2 is not None: cv2.imshow('test2', image2)
        if cv2.waitKey(100000) == 27: return

    @staticmethod
    def region_by_channel(image, cluster=3):
        adjusted_image = ImageOperations.adjust_contrast(image, alpha=2.5, beta=-60)
        dark_channel = ImageOperations.get_dark_channel(adjusted_image, patch_size=30)
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
    def get_dark_channel(image, patch_size=15):
        # Min value across the RGB channels
        min_channel = np.min(image, axis=2)
        # Apply a min filter with the given patch size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
