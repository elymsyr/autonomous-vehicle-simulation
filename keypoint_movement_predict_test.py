import numpy as np
import cv2
from time import perf_counter, sleep
from keypoints import KeypointTrack

class CityDriveCaptureTest():
    def __init__(self, video_output=None, fps=100.0, video_input=None):
        self.fps_list = []
        self.video_output = video_output
        self.video_input = video_input
        self.fps = fps
        self.video_writer = None
        self.cap = None
        self.keypoint_area = np.array(
            [
                [0,0],
                [1080,0],
                [1080, 270],
                [0, 270]
            ]
        )        
        

    def process_frame(self, show_result):
        currTime = perf_counter()
        fps = 1 / (currTime - self.prevTime)
        self.fps_list.append(fps)
        self.prevTime = currTime

        x, y, w, h = cv2.boundingRect(self.keypoint_area)
        roi = self.window_image[y:y+h, x:x+w]
        mask = np.zeros((h, w), dtype=np.uint8)
        keypoint_area_in_roi = self.keypoint_area - [x, y]
        cv2.fillPoly(mask, [keypoint_area_in_roi], 255)
        cropped_image = cv2.bitwise_and(roi, roi, mask=mask)

        cropped_image = self.tracker.load(image=cropped_image)

        self.window_image = cv2.polylines(self.window_image, [self.keypoint_area], isClosed=True, color=(255, 0, 0), thickness=3)

        if show_result:
            cv2.putText(self.window_image, f"FPS: {int(fps)}", (8, self.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            if self.window_image is not None:
                cv2.imshow('YOLOV8 IMAGE', cv2.resize(self.window_image, (self.window_image.shape[1] // 2, self.window_image.shape[0] // 2)))
                
            if cropped_image is not None:
                cv2.imshow('Clipped YOLOV8 IMAGE', cv2.resize(cropped_image, (cropped_image.shape[1] // 2, cropped_image.shape[0] // 2)))
        if self.video_writer:
            self.video_writer.write(self.window_image)

    def capture_from_video(self, show_result=True):
        self.cap = cv2.VideoCapture(self.video_input)

        if not self.cap.isOpened():
            print(f"Error opening video file {self.video_input}")
            return

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.video_output:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for MP4 format
            self.video_writer = cv2.VideoWriter(self.video_output, fourcc, self.fps, (self.width, self.height))

        self.prevTime = perf_counter()
        time_per_frame = 1.0 / self.fps  # Calculate time per frame for the desired fps

        ret, self.window_image = self.cap.read()

        self.tracker = KeypointTrack(self.window_image)

        while self.cap.isOpened():
            ret, self.window_image = self.cap.read()
            if ret:
                frame_start_time = perf_counter()  # Capture the start time of frame processing
                self.process_frame(show_result)

                # Introduce a delay to maintain the FPS
                elapsed_time = perf_counter() - frame_start_time
                time_to_sleep = time_per_frame - elapsed_time
                if time_to_sleep > 0:
                    sleep(time_to_sleep)  # Sleep for the remaining time to maintain consistent FPS

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()

        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()

    def window_linux(self, show_result: bool = True):
        self.capture_from_video(show_result)

if '__main__' == __name__:
    agent = CityDriveCaptureTest(video_input='media/city_car_test.mp4')
    agent.window_linux()
