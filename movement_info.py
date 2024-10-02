import numpy as np
import cv2
from time import perf_counter, sleep

DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 0, 0, 0, 0, 0): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

class CityDriveCaptureTest():
    def __init__(self, fps=100.0, video_input=None):
        self.fps_list = []
        self.video_input = video_input
        self.fps = fps
        self.cap = None
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
                [19, 0],
                [19, 33],
                [0, 33]],dtype=np.int32),
            np.array([
                [21, 0],
                [42, 0],
                [42, 33],
                [21, 33]],dtype=np.int32),
            np.array([
                [44, 0],
                [65, 0],
                [65, 33],
                [44, 33]],dtype=np.int32)            
        ]      

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
        # Find contours of the moving part
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours to find the bounding box of the green part
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # You may add criteria to filter the right part based on size, aspect ratio, etc.
            
            # Draw bounding box
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Print or track the horizontal position (x)
            print("Horizontal position (x):", x)
        return image

    def speed_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # # Find contours of the moving part
        # contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        #     x, y, w, h = cv2.boundingRect(contour)

        #     # You may add criteria to filter the right part based on size, aspect ratio, etc.

        #     # Draw bounding box
        #     image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        i = 0
        digits = []
        for digit in self.digits:
            x, y, w, h = cv2.boundingRect(digit)
            roi = thresholded[y:y+h, x:x+w]        

            # roiH, roiW, *_ = roi.shape
            # (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
            # dHC = int(roiH * 0.05)
            # # define the set of 7 segments
            # segments = [
            #     ((0, 0), (w, dH)),	# top
            #     ((0, 0), (dW, h // 2)),	# top-left
            #     ((w - dW, 0), (w, h // 2)),	# top-right
            #     ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
            #     ((0, h // 2), (dW, h)),	# bottom-left
            #     ((w - dW, h // 2), (w, h)),	# bottom-right
            #     ((0, h - dH), (w, h))	# bottom
            # ]
            # on = [0] * len(segments)
            
            # for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            #     # extract the segment ROI, count the total number of
            #     # thresholded pixels in the segment, and then compute
            #     # the area of the segment
            #     segROI = roi[yA:yB, xA:xB]
            #     total = cv2.countNonZero(segROI)
            #     area = (xB - xA) * (yB - yA)
            #     # if the total number of non-zero pixels is greater than
            #     # 50% of the area, mark the segment as "on"
            #     if total / float(area) > 0.5:
            #         on[i]= 1
            # lookup the digit and draw it on the image
            try:
                # digit = DIGITS_LOOKUP[tuple(on)]
                # digits.append(digit)
                cv2.imshow(f"{i}", roi)
                i += 1
            except: pass
            
            # thresholded = cv2.polylines(thresholded, [digit], isClosed=True, color=(255, 0, 0), thickness=1)

        return thresholded

if '__main__' == __name__:
    agent = CityDriveCaptureTest(video_input='media/clip0.mp4')
    agent.window_linux()
