from ultralytics import YOLO
import numpy as np
from Xlib import X, display
import cv2
import mss
from time import perf_counter
from lane import *
from mapping import BirdEyeViewMapping, SmoothCircle
from collections import defaultdict
from direction import *

class CityDriveCapture:
    def __init__(self, model_path: str, window_title: str = 'City Car Driving Home Edition Steam', video_output: str = None, fps: float = 30.0, video_input: str = None) -> None:
        """
        Initialize the CityDriveCapture object with model path and other parameters.

        Args:
            model_path (str): Path to the YOLO model.
            window_title (str, optional): Title of the window. Defaults to 'City Car Driving Home Edition Steam'.
            video_output (str, optional): Path for the video output file. Defaults to None.
            fps (float, optional): Frames per second for video processing. Defaults to 30.0.
            video_input (str, optional): Path for the video input file. Defaults to None.
        """
        self.window_title = window_title
        self.CLICKED = False

        self.model: YOLO = YOLO(model_path)
        self.classes = ['car', 'person']

        self.path = None
        self.sliding_windows = None
        self.track_history = defaultdict(lambda: [])

        self.display = display.Display()
        self.root = self.display.screen().root
        self.window, self.monitor = self.find_window()
        self.fps_list = []

        # Video writer setup
        self.video_output = video_output
        self.video_input = video_input
        self.fps = fps
        self.video_writer = None
        self.cap = None
        
        self.turn: float = 0
        self.speed: int = 0
        self.direction_detector = DirectionDetection()

        self.move_circles: dict[int, SmoothCircle] = {}

    def capture(self, sct) -> np.ndarray:
        """
        Capture an image from the screen.

        Args:
            sct: The screen capture object.

        Returns:
            np.ndarray: Captured image in BGR format.
        """
        img = sct.grab(self.monitor)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

    def create_monitor(self, window) -> dict:
        """
        Create a monitor object to define the capture region.

        Args:
            window: The window object to monitor.

        Returns:
            dict: A dictionary containing the monitor dimensions.
        """
        geometry = window.query_tree().parent.get_geometry()
        return {
            "top": int(geometry.y) + 50,
            "left": geometry.x + 1,
            "width": geometry.width - 1,
            "height": int(geometry.height) - 108
        }

    def find_window(self) -> tuple:
        """
        Find the window by its title.

        Returns:
            tuple: The window object and its monitor dimensions.
        """
        window = None
        window_id = None
        window_ids = self.root.get_full_property(self.display.intern_atom('_NET_CLIENT_LIST'), X.AnyPropertyType).value
        for window_id in window_ids:
            window = self.display.create_resource_object('window', window_id)
            window_name_str = window.get_wm_name()
            if window_name_str and self.window_title in window_name_str:
                window = window
                window_id = window_id
                break
        monitor = self.create_monitor(window=window) if window else None
        return window, monitor

    def bev(self, transformed_point: tuple, id: int, canvas: np.ndarray, predicted_points: list = None) -> np.ndarray:
        """
        Draw a circle and text for a detected object in the canvas.

        Args:
            transformed_point (tuple): The point to be transformed and drawn.
            id (int): The ID of the detected object.
            canvas (np.ndarray): The canvas on which to draw.
            predicted_points (list, optional): Predicted points for the object. Defaults to None.

        Returns:
            np.ndarray: Updated canvas with the drawn circle and text.
        """
        radius = 16
        canvas = cv2.circle(canvas, transformed_point, radius, (255, 0, 0), -1)
        text = str(id)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = transformed_point[0] - text_size[0] // 2
        text_y = transformed_point[1] + text_size[1] // 2
        canvas = cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 0), 2)
        if predicted_points is not None:
            canvas = self.draw_lines_and_circle(predicted_points, canvas, transformed_point)

        return canvas

    def check_alert_area(self, points: list) -> list:
        """
        Check if any of the given points are within the alert area.

        Args:
            points (list): List of points to check.

        Returns:
            list: List of boolean values indicating if each point is in the alert area.
        """
        point_checked = []
        for point in points:
            point_checked.append(True if self.bev_transformer.area_check(point=point) else False)
        return point_checked

    def draw_lines_and_circle(self, predicted_points: list, canvas: np.ndarray, start_point: tuple) -> np.ndarray:
        """
        Draw lines and a circle based on the predicted points.

        Args:
            predicted_points (list): List of predicted points.
            canvas (np.ndarray): The canvas on which to draw.
            start_point (tuple): The starting point for the lines.

        Returns:
            np.ndarray: Updated canvas with lines and a circle drawn.
        """
        start_point = tuple(map(int, start_point))
        points_to_draw = [start_point] + predicted_points
        
        for point_index in range(len(points_to_draw) - 1):
            point1 = points_to_draw[point_index]
            point2 = points_to_draw[point_index + 1]
            canvas = cv2.line(canvas, point1, point2, color=(0, 0, 255), thickness=2)  # Red lines
        last_point = points_to_draw[-1]
        canvas = cv2.circle(canvas, last_point, radius=16, color=(0, 0, 255), thickness=1)  # Red outline
        
        distance = self.bev_transformer.distance_between_points(start_point, self.bev_transformer.view_point)
        distance_text = f"{distance:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(distance_text, font, font_scale, thickness)[0]
        text_x = last_point[0] - text_size[0] // 2
        text_y = last_point[1] + 30

        canvas = cv2.putText(canvas, distance_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
        return canvas

    def process_frame(self, show_result: bool, canvas: np.ndarray) -> None:
        """
        Process a frame for object detection and display results.

        Args:
            show_result (bool): Flag indicating whether to display the results.
            canvas (np.ndarray): The canvas on which to draw results.
        """
        self.direction_detector.load_direction(self.window_image)
        self.turn = self.direction_detector.turn_on_screen.get()
        self.speed = self.direction_detector.speed_on_screen.get()
        
        # Run YOLOv8 to detect cars in the current self.window_image
        results = self.model.track(\
            cv2.rectangle(self.window_image, (int(0.15*self.width), int(0.82*self.height)),\
                (int(0.85*self.width), int(self.height)),\
                    (0,0,0), -1),\
                        persist=True, verbose=False, tracker = 'botsort.yaml')

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.int().cpu().tolist()
            class_ids = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            names = results[0].names

            # self.window_image, self.sliding_windows, *_ = self.lane_detector.detect_lane(self.window_image)
            self.sliding_windows, *_ = self.lane_detector.detect_lane(self.window_image)
            self.window_image = results[0].plot(img=self.window_image, line_width=2)

            if self.window_image is not None:
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if True: # names[class_id] in self.classes
                        x, y, _, h = box
                        alert = False
                        if track_id not in self.move_circles.keys():
                            self.move_circles[track_id] = SmoothCircle(center=[x, y+h//2], radius=6, track_id=track_id)
                            transformed_point = self.bev_transformer.perspective_transform_point((x, y+h//2), track_id)
                            canvas = self.bev(transformed_point, track_id, canvas)
                        else:
                            predicted_points = self.move_circles[track_id].update_point(np.array([x, y+h//2]))
                            predicted_points = [self.bev_transformer.perspective_transform_point(tuple(map(int, p)), track_id) for p in predicted_points]
                            for point in predicted_points:
                                if self.bev_transformer.area_check(point): alert = True
                            new_x, new_y = tuple(self.move_circles[track_id].center.astype(int))
                            transformed_point = self.bev_transformer.perspective_transform_point((int(new_x), int(new_y)), track_id)
                            canvas = self.bev(transformed_point, track_id, canvas, predicted_points)
                        track = self.track_history[track_id]
                        track.append((float(x), float(y)))
                        if self.bev_transformer.area_check(transformed_point): alert = True

                        if alert:
                            self.window_image = cv2.circle(self.window_image, (x,y), 14, (0, 0, 255), 3)

                        if len(track) > 30:
                            track.pop(0)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(self.window_image, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # FPS calculation
        currTime = perf_counter()
        fps = 1 / (currTime - self.prevTime)
        self.fps_list.append(fps)
        self.prevTime = currTime

        # Display results and FPS
        if show_result:
            cv2.putText(self.window_image, f"FPS: {int(fps)}", (5, self.height-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(self.window_image, f"Speed: {self.speed}", (5, self.height-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(self.window_image, f"Direction: {self.turn:.2f}", (5, self.height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # if self.window_image is not None and self.sliding_windows is not None:
            #     merged_image = np.vstack((cv2.cvtColor(cv2.resize(self.sliding_windows, (811, 414)), cv2.COLOR_GRAY2BGR), cv2.resize(self.window_image, (811, 414))))
            #     cv2.imshow(f'{self.window_title} Results', merged_image)
            # else:
            if canvas is not None: cv2.imshow(f'{self.window_title} YOLOV8 IMAGEaa', cv2.resize(canvas, (int(canvas.shape[1]//1.6), int(canvas.shape[0]//1.6))))
            if self.window_image is not None: cv2.imshow(f'{self.window_title} YOLOV8 IMAGE', cv2.resize(self.window_image, (self.window_image.shape[1]//2, self.window_image.shape[0]//2)))
            if self.sliding_windows is not None: cv2.imshow(f'{self.window_title} SLIDING WINDOWS IMAGE', cv2.resize(self.sliding_windows, (self.sliding_windows.shape[1]//2, self.sliding_windows.shape[0]//2)))

        # Write self.window_image to video if enabled
        if self.video_writer:
            self.video_writer.write(self.window_image)

    def capture_from_video(self, show_result: bool = True) -> None:
        """
        Captures frames from the specified video file, processes each frame, 
        and optionally displays the results in a window.

        Parameters:
            show_result (bool): If True, displays the processed frames in a window.
                                Default is True.
        
        Returns:
            None: The method processes the video frames and manages display/output.
        """        
        self.cap = cv2.VideoCapture(self.video_input)

        if not self.cap.isOpened():
            print(f"Error opening video file {self.video_input}")
            return

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.lane_detector = LaneDetectorU(self.width, self.height)
        self.bev_transformer = BirdEyeViewMapping(self.width, self.height)
        canvas = np.ones((self.height, self.width, 3))
        canvas = self.bev_transformer.perspective_transform(canvas)
        points = np.array(self.bev_transformer.alert_area).reshape((-1, 1, 2)).astype(np.int32)
        canvas = cv2.polylines(canvas, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        if self.video_output:
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for AVI format
            self.video_writer = cv2.VideoWriter(self.video_output, fourcc, self.fps, (self.width, self.height))

        self.prevTime = perf_counter()

        while self.cap.isOpened():
            ret, self.window_image = self.cap.read()
            if ret:
                self.process_frame(show_result, canvas.copy())

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()

        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()

    def window_linux(self, show_result: bool = True) -> None:
        """
        Captures video either from a file or from the screen, processes each frame, 
        and displays the results in a window.

        Parameters:
            show_result (bool): If True, displays the processed frames in a window.
                                Default is True.

        Returns:
            None: The method processes the video frames and manages display/output.
        """        
        if self.video_input:  # Run from video if provided
            self.capture_from_video(show_result)
        else:  # Run from window capture
            with mss.mss() as sct:
                self.window_image = self.capture(sct)
                self.width, self.height = int(self.window_image.shape[1]), int(self.window_image.shape[0])
                self.lane_detector = LaneDetectorU(self.width, self.height)
                self.bev_transformer = BirdEyeViewMapping(self.width, self.height)
                canvas = np.ones((self.height, self.width, 3))
                canvas = self.bev_transformer.perspective_transform(canvas)
                points = np.array(self.bev_transformer.alert_area).reshape((-1, 1, 2)).astype(np.int32)
                canvas = cv2.polylines(canvas, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                self.fps_list = []

                if self.video_output:
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for AVI format
                    self.video_writer = cv2.VideoWriter(self.video_output, fourcc, self.fps, (self.width, self.height))

                self.prevTime = perf_counter()

                while True:
                    self.window_image = self.capture(sct)
                    if self.window_image is not None:
                        self.process_frame(show_result, canvas.copy())

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    if len(self.fps_list) > 10000:
                        break

            print(sum(self.fps_list) / len(self.fps_list), len(self.fps_list))

            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()

if '__main__' == __name__:
    model_path = f'weights/yolo11n.pt'
    agent = CityDriveCapture(model_path=model_path, video_input='test_media/clip0.mp4')
    agent.window_linux()
