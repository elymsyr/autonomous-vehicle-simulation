import numpy as np
import cv2
from time import perf_counter, sleep

DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 0, 0, 0, 0, 0): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

class Digits():
    def __init__(self) -> None:
        self.digits: dict[str, int] = {'0': 0, '1': 0, '2': 0}
        
    def update(self, number: int, idx = None):
        if idx and isinstance(idx, int) and idx >= 0 and idx <= 2: self.digits[f"{idx}"] = number
        else:
            if len(str(number)) <= 3:
                number = str(number)
                while len(number) != 3: number = f"0{number}"
                for idx, value in enumerate(number):
                    self.digits[f"{idx}"] = int(number)
    def get(self):
        return int(str(self.digits["0"])+str(self.digits["1"])+str(self.digits["2"]))

class CityDriveCaptureTest():
    def __init__(self, fps=100.0, video_input=None):
        self.fps_list = []
        self.video_input = video_input
        self.fps = fps
        self.cap = None
        self.digits_on_screen = Digits()
        self.turn_on_screen: int = 20
        self.areas = {
            'speed': 2*np.array([
                [74, 44],
                [106.5, 44],
                [106.5, 60.5],
                [74, 60.5]
            ],dtype=np.int32),
            'turn': 2*np.array([
                [133, 290],
                [160, 290],
                [160, 292],
                [133, 292]
            ],dtype=np.int32)
        }
        self.digits = [
            np.array([
                [0, 0],
                [20, 0],
                [20, 33],
                [0, 33]],dtype=np.int32),
            np.array([
                [23, 0],
                [43, 0],
                [43, 33],
                [23, 33]],dtype=np.int32),
            np.array([
                [45, 0],
                [65, 0],
                [65, 33],
                [45, 33]],dtype=np.int32)
        ]
        # modify_src_points = lambda width, height, margin : np.array([[0, x], [y, 0], [19, 33], [0, 33]], dtype=np.int32)
        self.src_points = np.array([[3, 0],[20, 0],[17, 33],[0, 33]],dtype=np.float32)
        self.dst_points = np.array([[0,0], [20,0], [20,33], [0,33]], dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def process_frame(self, show_result):
        currTime = perf_counter()
        fps = 1 / (currTime - self.prevTime)
        self.fps_list.append(fps)
        self.prevTime = currTime

        cropped_parts = {}

        for name, area in self.areas.items():
            x, y, w, h = cv2.boundingRect(area)
            roi = self.window_image[y:y+h, x:x+w]
            mask = np.zeros((h, w), dtype=np.uint8)
            keypoint_area_in_roi = area - [x, y]
            cv2.fillPoly(mask, [keypoint_area_in_roi], 255)
            
            cropped_image = cv2.bitwise_and(roi, roi, mask=mask)

            if name == 'turn': cropped_image = self.turn_detection(cropped_image)
            if name == 'speed': cropped_image = self.speed_detection(cropped_image)
            
            cropped_parts[name] = cropped_image

            self.window_image = cv2.polylines(self.window_image, [area], isClosed=True, color=(255, 0, 0), thickness=1)

        if show_result:
            cv2.putText(self.window_image, f"FPS: {int(fps)}", (8, self.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            if self.window_image is not None:
                cv2.imshow('YOLOV8 IMAGE', cv2.resize(self.window_image, (self.window_image.shape[1] // 2, self.window_image.shape[0] // 2)))
            
            for item, value in cropped_parts.items():
                cv2.imshow(f'{item.upper()}', cv2.resize(value, (value.shape[1] * 2, value.shape[0] * 2)))

    def capture_from_video(self, show_result=True):
        self.cap = cv2.VideoCapture(self.video_input)

        if not self.cap.isOpened():
            print(f"Error opening video file {self.video_input}")
            return

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.prevTime = perf_counter()
        time_per_frame = 1.0 / self.fps  # Calculate time per frame for the desired fps

        ret, self.window_image = self.cap.read()

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

        cv2.destroyAllWindows()

    def window_linux(self, show_result: bool = True):
        self.capture_from_video(show_result)
        
    def turn_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_x = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if x > biggest_x: biggest_x = x
        self.turn_on_screen = biggest_x
        print("Turn: ", self.turn_on_screen)
        return image

    def speed_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        for idx, digit in enumerate(self.digits):
            x, y, w, h = cv2.boundingRect(digit)
            roi = thresholded[y:y+h, x:x+w]

            roi = cv2.warpPerspective(roi, self.matrix, (20,33))

            roiH, roiW, *_ = roi.shape
            (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
            dHC = int(roiH * 0.05)

            segments = [
                ((0, 0), (w, dH)),	# top
                ((0, 0), (dW, h // 2)),	# top-left
                ((w - dW, 0), (w, h // 2)),	# top-right
                ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
                ((0, h // 2), (dW, h)),	# bottom-left
                ((w - dW, h // 2), (w, h)),	# bottom-right
                ((0, h - dH), (w, h))	# bottom
            ]
            on = [0] * len(segments)
            
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                if total / float(area) > 0.5:
                    on[i]= 1
            try:
                digit = DIGITS_LOOKUP[tuple(on)]
                self.digits_on_screen.update(digit, idx)
            except: pass
            
            cv2.imshow(f"{idx}", cv2.resize(roi, (roi.shape[1]*2, roi.shape[0]*2)))
        
        print("Speed: ", self.digits_on_screen.get())
        
        return thresholded

if '__main__' == __name__:
    agent = CityDriveCaptureTest(video_input='media/clip0.mp4')
    agent.window_linux()
