import numpy as np
import cv2
from sklearn.cluster import KMeans

class BirdEyeViewMapping():
    """
    A class for generating a bird's-eye view mapping of an area from a given perspective.
    This class handles perspective transformations to create a top-down view of the scene 
    and provides methods for transforming points, checking distances, and detecting areas of interest.

    Attributes:
        width (int): The width of the image or frame to be transformed.
        height (int): The height of the image or frame to be transformed.
        interest_vertices (list of tuples): The vertices of interest in the original image, representing polly.
        desired_points (np.float32): Transformed coordinates of the points of interest.
        image_corners (np.float32): The coordinates of the image corners.
        matrix (np.ndarray): The perspective transformation matrix for transforming the view.
        inv_matrix (np.ndarray): The inverse perspective transformation matrix.
        view_point (tuple): The transformed view point based on the center of the image's bottom edge.
        alert_area (np.ndarray): The polygon defining the alert area after perspective transformation.
    """    
    def __init__(self, width : int, height : int, interest_vertices : list[tuple[int, int]] = None) -> None:
        """
        Initializes the BirdEyeViewMapping with specified width and height.

        Args:
            width (int): The width of the image or frame.
            height (int): The height of the image or frame.
            interest_vertices : (list[tuple[int, int]]) = Default (None). The vertices of interest in the original image, representing polly.
        """        
        self.width : int = width
        self.height : int = height
        self.interest_vertices : list[tuple[int,int]] = [
            (int(0.1*self.width), int(0.58*self.height)), # left top
            (int(-6*self.width), int(self.height)),       # left bottom
            (int(0.9*self.width), int(0.58*self.height)), # right top
            (int(7*self.width), int(self.height))         # right bottom
        ] if not interest_vertices else interest_vertices
        self.desired_points : np.ndarray = np.float32(self.interest_vertices)
        self.image_corners : np.ndarray = np.float32([[0, 0], [0, self.height], [self.width, 0], [self.width, self.height]])
        self.matrix : np.ndarray = cv2.getPerspectiveTransform(self.desired_points, self.image_corners)
        self.inv_matrix : np.ndarray = cv2.getPerspectiveTransform(self.image_corners, self.desired_points)
        self.view_point : np.ndarray = self.perspective_transform_point((self.width//2, self.height))
        self.alert_area : np.ndarray = np.array(self.alert_area_calc(), dtype=np.int32)

    def alert_area_calc(self) -> list[np.ndarray]:
        """
        Calculates the alert area coordinates in the transformed perspective.

        Returns:
            list[np.ndarray]: The vertices of the alert area polygon.
        """        
        return [
            self.perspective_transform_point((int(0.42 * self.width),int(0.64 * self.height))),  # left top
            self.perspective_transform_point((int(0.17 * self.width),int(0.95 * self.height))),  # left bottom
            self.perspective_transform_point((int(0.74 * self.width),int(0.95 * self.height))),  # right bottom
            self.perspective_transform_point((int(0.56 * self.width),int(0.64 * self.height))),  # right top
            ]

    def perspective_transform_point(self, point: tuple[int,int], track_id: int = None) -> tuple[int, int]:
        """
        Applies perspective transformation to a given point.

        Args:
            point (tuple): A tuple representing the x, y coordinates of the point.
            track_id (optional): An identifier for tracking purposes (default is None).

        Returns:
            tuple[int, int]: The transformed point's coordinates as a tuple of integers.
        """        
        point_homogeneous = np.array([point[0], point[1], 1.0])
        transformed_point = np.dot(self.matrix, point_homogeneous)
        transformed_point /= transformed_point[2]
        return tuple(map(int, transformed_point[:2]))

    def perspective_transform_point_w_distance_estimation(self, point: tuple[int,int], track_id: int, a=0.1, b=1.00012):
        """
        Applies perspective transformation to a given point with distance estimation by the function.

        Args:
            point (tuple): A tuple representing the x, y coordinates of the point.
            track_id (int): An identifier for tracking purposes (default is None).
            a, b (float, optional): Function parameters.

        Returns:
            tuple[int, int]: The transformed point's coordinates as a tuple of integers.
        """          
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

    def perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a perspective transformation to the entire image/frame.

        Args:
            image (np.ndarray): The image or frame to be transformed.

        Returns:
            np.ndarray: The transformed image with a bird's-eye view.
        """         
        # for point in self.interest_vertices:
        #     cv2.circle(image, point, 5, (255,0,0), 3)
        transformed_frame = cv2.warpPerspective(image, self.matrix, (self.width, self.height))
        return transformed_frame

    def new_point_on_line(self, target_point: tuple[int,int], new_distance: float) -> tuple[int,int]:
        """
        Calculates a new point along the line extending from the view point to a target point,
        at a specified distance from the view point.

        Args:
            target_point (tuple): A tuple representing the coordinates (x, y) of the target point.
            new_distance (float): The distance from the view point to the new point.

        Returns:
            tuple: The coordinates (x, y) of the new point on the line.
        """        
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

    def distance_between_points(self, point1: tuple[int,int], point2: tuple[int,int]) -> float:
        """
        Calculates the Euclidean distance between two points.

        Args:
            point1 (tuple): The first point as a tuple of (x, y).
            point2 (tuple): The second point as a tuple of (x, y).

        Returns:
            float: The distance between the two points.
        """        
        # Unpack the points
        x1, y1 = point1
        x2, y2 = point2

        # Calculate the distance using the distance formula
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return distance

    def area_check(self, point: tuple[int,int]) -> bool:
        """
        Checks whether a given point lies within the predefined alert area.

        Args:
            point (tuple): The point to check, represented as (x, y) coordinates.

        Returns:
            bool: True if the point is within the alert area, False otherwise.
        """        
        return True if cv2.pointPolygonTest(self.alert_area, point, measureDist=False) > 0 else False

    def f(self, x, a, b):
        # return a + b * np.log(x)
        return x - ((x*a) * (b**x))

    def f_polynomial(self, x, a = 0.000005, b = 0, c = 1):
        return a * x**2 + b * x + c

class SmoothCircle:
    """
    A class that represents a smooth-moving circle, tracking its positions and predicting future movements.
    The circle's behavior is designed to reset when an opposite direction in movement is detected.

    Attributes:
        center (np.ndarray): The current center of the circle.
        track_id (int): An identifier for tracking purposes.
        radius (int): The radius of the circle.
        point (np.ndarray): The current point being tracked within the circle.
        positions (list): A history of the positions the circle has passed through.
        directions (list): A list of direction vectors for movement analysis.
    """

    def __init__(self, center: list[int, int], track_id: int, radius: float = 5) -> None:
        """
        Initializes the SmoothCircle with a specified center, track ID, and radius.

        Args:
            center (list[int, int]): The initial position of the circle's center.
            track_id (int): An identifier for the circle object.
            radius (float, optional): The radius of the circle. Default is 5.
        """
        self.track_id = track_id
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.point = np.array(center, dtype=np.float32)
        self.positions = [np.array(center, dtype=np.float32)]  # Start with initial position
        self.directions = []

    def update_positions(self, new_point: np.ndarray) -> None:
        """
        Updates the list of positions with a new point, and checks for changes in direction.

        Args:
            new_point (np.ndarray): The new point to add to the positions history.
        """
        if not self.positions:
            self.positions.append(new_point)
        else:
            self.positions.append(new_point)
            self.reset_history_on_opposite_direction()

    def detect_opposite_direction(self, overall_vector: np.ndarray, new_vector: np.ndarray) -> bool:
        """
        Detects if the new vector indicates a direction opposite to the overall movement.

        Args:
            overall_vector (np.ndarray): The overall direction vector of the circle.
            new_vector (np.ndarray): The current movement vector.

        Returns:
            bool: True if the new vector indicates an opposite direction, False otherwise.
        """
        dot_product = np.dot(overall_vector, new_vector)
        return dot_product < 20

    def reset_history_on_opposite_direction(self) -> None:
        """
        Resets the movement history if a significant change in direction is detected.
        It keeps only the recent points after the point where the direction changed.
        """
        if len(self.positions) >= 3:
            overall_vector = self.positions[-1] - self.positions[0]
            for i in range(1, len(self.positions) - 1):
                current_vector = self.positions[i + 1] - self.positions[i]
                if self.detect_opposite_direction(overall_vector, current_vector):
                    self.positions = self.positions[i + 1:] if len(self.positions) > i else self.positions[i:]
                    break

    def predict_next_n_moves(self, n: int = 3) -> np.ndarray:
        """
        Predicts the next 'n' positions of the circle based on its movement history.

        Args:
            n (int): The number of future positions to predict. Default is 3.

        Returns:
            np.ndarray: An array of predicted positions.
        """
        self.reset_history_on_opposite_direction()

        if len(self.positions) < 2:
            return np.array([])

        last_positions = self.positions[-5:]
        dx = last_positions[-1][0] - last_positions[-2][0]
        dy = last_positions[-1][1] - last_positions[-2][1]

        for i in range(len(last_positions) - 1):
            dx += last_positions[i + 1][0] - last_positions[i][0]
            dy += last_positions[i + 1][1] - last_positions[i][1]

        dx /= (len(last_positions) - 1)
        dy /= (len(last_positions) - 1)

        predicted_positions = []
        current_position = last_positions[-1]

        for i in range(n):
            dx = dx * (5 if -0.4 < dx < 0.4 else 1)
            dy = dy * (5 if -0.4 < dy < 0.4 else 1)
            next_position = current_position + np.array([dx, dy])
            predicted_positions.append(next_position)
            current_position = next_position

        return np.array(predicted_positions)

    def update_point(self, new_point: np.ndarray) -> np.ndarray:
        """
        Updates the circle's current point position and checks if it needs to be pushed.

        Args:
            new_point (np.ndarray): The new point to update.

        Returns:
            np.ndarray: Predicted future positions of the circle.
        """
        self.point = new_point
        self.push_circle_if_needed()
        self.update_positions(new_point)
        return self.predict_next_n_moves()

    def push_circle_if_needed(self) -> None:
        """
        Adjusts the circle's center if the point reaches the edge and continues to move.
        Ensures the point stays within the defined radius of the circle.
        """
        distance = np.linalg.norm(self.point - self.center)
        if distance >= self.radius:
            direction = (self.point - self.center) / distance
            self.center += direction * (distance - self.radius)
            self.positions.append(self.center)

    def draw(self, canvas: np.ndarray) -> None:
        """
        Draws the circle and its current point on the given canvas.

        Args:
            canvas (np.ndarray): The image or canvas where the circle is drawn.
        """
        cv2.circle(canvas, tuple(map(int, self.center)), int(self.radius), (0, 255, 0), 2)
        cv2.circle(canvas, tuple(map(int, self.point)), 5, (255, 0, 0), -1)

    def reset_circle(self) -> None:
        """
        Resets the circle's center to its current point position.
        """
        self.center = self.point

class ImageOperations:
    """
    A utility class that contains various image processing operations, such as contrast adjustment,
    edge detection, and segmentation.

    Methods:
        adjust_contrast(image, alpha=2.5, beta=-60): Adjusts the contrast and brightness of an image.
        canny_image(image): Applies Canny edge detection to an image.
        show_image(image, image2=None): Displays one or two images in a window.
        region_by_channel(image, cluster=3): Segments an image using k-means clustering based on its dark channel.
        get_dark_channel(image, patch_size=15): Computes the dark channel of an image.
    """

    @staticmethod
    def adjust_contrast(image: np.ndarray, alpha: float = 2.5, beta: float = -60) -> np.ndarray:
        """
        Adjusts the contrast and brightness of an image.

        Args:
            image (np.ndarray): The input image.
            alpha (float, optional): The contrast factor. Default is 2.5.
            beta (float, optional): The brightness offset. Default is -60.

        Returns:
            np.ndarray: The contrast-adjusted image.
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def canny_image(image: np.ndarray) -> np.ndarray:
        """
        Converts the image to grayscale, adjusts the contrast, and applies Canny edge detection.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The image after applying Canny edge detection.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        adjusted_image = ImageOperations.adjust_contrast(gray_image)
        cannyed_image = cv2.Canny(adjusted_image, 100, 200)
        return cannyed_image

    @staticmethod
    def show_image(image: np.ndarray, image2: np.ndarray = None) -> None:
        """
        Displays one or two images in separate windows.

        Args:
            image (np.ndarray): The first image to be displayed.
            image2 (np.ndarray, optional): The second image to be displayed. Default is None.
        """
        cv2.imshow('test', image)
        if image2 is not None:
            cv2.imshow('test2', image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def region_by_channel(image: np.ndarray, cluster: int = 3) -> np.ndarray:
        """
        Segments an image using k-means clustering based on its dark channel.

        Args:
            image (np.ndarray): The input image.
            cluster (int, optional): The number of clusters for k-means. Default is 3.

        Returns:
            np.ndarray: The segmented image with clustered regions.
        """
        adjusted_image = ImageOperations.adjust_contrast(image, alpha=2.5, beta=-60)
        dark_channel = ImageOperations.get_dark_channel(adjusted_image, patch_size=30)
        img_2D = dark_channel.reshape((-1, 1))

        kmeans = KMeans(n_clusters=cluster, init='k-means++', max_iter=250, n_init=10, random_state=35).fit(img_2D)
        values = kmeans.predict(img_2D)
        mask1 = values.reshape((image.shape[0], image.shape[1]))
        mask1 = np.expand_dims(mask1, axis=-1)
        centers = kmeans.cluster_centers_.astype(int)

        clustered_image = np.zeros_like(image)
        for i in range(cluster):
            clustered_image[mask1[:, :, 0] == i] = centers[i]

        clustered_image = np.clip(clustered_image, 0, 255).astype(np.uint8)
        return clustered_image

    @staticmethod
    def get_dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
        """
        Computes the dark channel of an image, which is the minimum value in a local patch.

        Args:
            image (np.ndarray): The input image.
            patch_size (int, optional): The size of the patch to compute the dark channel. Default is 15.

        Returns:
            np.ndarray: The dark channel of the image.
        """
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
