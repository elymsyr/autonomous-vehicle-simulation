from ultralytics import YOLO
import numpy as np
from Xlib import X, display
import cv2
import mss
from time import perf_counter
from lane import *
from collections import defaultdict

class CityDriveCapture():
    def __init__(self, model_path, window_title='City Car Driving Home Edition Steam', video_output=None, fps=30.0, video_input=None):
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

    def capture(self, sct):
        img = sct.grab(self.monitor)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

    def create_monitor(self, window):
        geometry = window.query_tree().parent.get_geometry()
        return {
            "top": int(geometry.y) + 50,
            "left": geometry.x + 1,
            "width": geometry.width - 1,
            "height": int(geometry.height) - 108
        }

    def find_window(self):
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

    def estimate_distance(self, bbox_width, bbox_height, label):
        # For simplicity, assume the distance is inversely proportional to the box size
        # This is a basic estimation, you may use camera calibration for more accuracy
        objects = {
            'car': {
                'focal_length': 800,
                'known_width': 1.9,
            },
            'person': {
                'focal_length': 700,
                'known_width': 1.6,
            }
        }
        if label in objects.keys():
            distance = (objects[label]['known_width'] * objects[label]['focal_length']) / (bbox_width if label == 'car' else bbox_height)  # Basic distance estimation
            return distance
        else: 0

    def process_frame(self, show_result):
        canvas = np.ones((600, self.window_image.shape[1], 3), dtype=np.uint8)
        # Run YOLOv8 to detect cars in the current self.window_image
        results = self.model.track(\
            cv2.rectangle(self.window_image, (int(0.15*self.width), int(0.82*self.height)),\
                (int(0.85*self.width), int(self.height)),\
                    (0,0,0), -1),\
                        persist=True, verbose=False)

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.int().cpu().tolist()
            class_ids = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            names = results[0].names
            
            # Visualize the results on the self.window_image
            
            # self.window_image, self.sliding_windows, *_ = self.lane_detector.detect_lane(self.window_image)
            # self.sliding_windows, *_ = self.lane_detector.detect_lane(self.window_image)
            self.window_image = results[0].plot(img=self.window_image)

            if self.window_image is not None:
                detected_points = []
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if names[class_id] in self.classes:
                        x, y, w, h = box
                        track = self.track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 90 tracks for 90 self.window_images
                            track.pop(0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(self.window_image, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                        # Draw the tracking lines
                        distance = self.estimate_distance(w, h, names[class_id].strip())
                        # Display the estimated distance
                        distance_label = f'{distance:.2f}m'
                        cv2.putText(self.window_image, distance_label, (x, y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(self.window_image, "0", (x+w//2, y+h//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        if distance <= 99:
                            detected_points.append((class_id, (self.window_image.shape[1]//2+w)*2, distance*10, track_id))

                for point in detected_points:
                    print(point)
                    class_id, x, y, track_id = point
                    cv2.circle(canvas, (int(x), int(y)), radius=18 if names[class_id] == 'car' else 11, color=(0, 255, 0) if names[class_id] == 'car' else (255, 0, 0), thickness=-1)
                    # Calculate the size of the text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2
                    font_thickness = 2
                    text_size = cv2.getTextSize(f"{track_id}", font, font_scale, font_thickness)[0]
                    
                    # Find the bottom-left corner of the text inside the circle
                    text_x = int(x - text_size[0] // 2)  # Center the text horizontally
                    text_y = int(y + text_size[1] // 2)  # Center the text vertically
                    
                    # Put the text inside the circle
                    cv2.putText(canvas, f"{track_id}", (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)


        # FPS calculation
        currTime = perf_counter()
        fps = 1 / (currTime - self.prevTime)
        self.fps_list.append(fps)
        self.prevTime = currTime

        # Display results and FPS
        if show_result:
            cv2.putText(self.window_image, f"FPS: {int(fps)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if self.window_image is not None and self.sliding_windows is not None:
                merged_image = np.vstack((cv2.cvtColor(cv2.resize(self.sliding_windows, (811, 414)), cv2.COLOR_GRAY2BGR), cv2.resize(self.window_image, (811, 414))))
                cv2.imshow(f'{self.window_title} Results', merged_image)
            else:
                if self.window_image is not None: cv2.imshow(f'{self.window_title} YOLOV8 IMAGE', cv2.resize(self.window_image, (self.window_image.shape[1]//2, self.window_image.shape[0]//2)))
                if canvas is not None: cv2.imshow(f'{self.window_title}', cv2.resize(canvas, (canvas.shape[1]//2, canvas.shape[0]//2)))
                if self.sliding_windows is not None: cv2.imshow(f'{self.window_title} SLIDING WINDOWS IMAGE', cv2.resize(self.sliding_windows, (self.sliding_windows.shape[1]//2, self.sliding_windows.shape[0]//2)))

        # Write self.window_image to video if enabled
        if self.video_writer:
            self.video_writer.write(self.window_image)

    def capture_from_video(self, show_result=True):
        self.cap = cv2.VideoCapture(self.video_input)

        if not self.cap.isOpened():
            print(f"Error opening video file {self.video_input}")
            return

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.lane_detector = LaneDetectorU(self.width, self.height)

        if self.video_output:
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for AVI format
            self.video_writer = cv2.VideoWriter(self.video_output, fourcc, self.fps, (self.width, self.height))

        self.prevTime = perf_counter()

        while self.cap.isOpened():
            ret, self.window_image = self.cap.read()
            if ret:
                self.process_frame(show_result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()

        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()

    def window_linux(self, show_result: bool = True):
        if self.video_input:  # Run from video if provided
            self.capture_from_video(show_result)
        else:  # Run from window capture
            with mss.mss() as sct:
                self.window_image = self.capture(sct)
                self.width, self.height = int(self.window_image.shape[1]), int(self.window_image.shape[0])
                self.lane_detector = LaneDetectorU(self.width, self.height)

                prevTime = 0
                fps = 0
                self.fps_list = []

                if self.video_output:
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for AVI format
                    self.video_writer = cv2.VideoWriter(self.video_output, fourcc, self.fps, (self.width, self.height))

                self.prevTime = perf_counter()

                while True:
                    self.window_image = self.capture(sct)
                    if self.window_image is not None:
                        self.process_frame(self.window_image, show_result)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    if len(self.fps_list) > 10000:
                        break

            print(sum(self.fps_list) / len(self.fps_list), len(self.fps_list))

            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()


if '__main__' == __name__:
    model_path = f'weights/yolov8n.pt'
    agent = CityDriveCapture(model_path=model_path, video_input='media/city_car.mp4')
    agent.window_linux()
