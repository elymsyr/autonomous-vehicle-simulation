import numpy as np
import cv2

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

class Turn():
    def __init__(self) -> None:
        self.turn: float = 0

    def update(self, number: int):
        if number > 45: number = 44
        self.turn = (number - 22)/22
    def get(self):
        return self.turn

class DirectionDetection():
    def __init__(self):
        self.speed_on_screen = Digits()
        self.turn_on_screen = Turn()
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
        self.turn_src_points = np.array([[3, 0],[20, 0],[17, 33],[0, 33]],dtype=np.float32)
        self.turn_dst_points = np.array([[0,0], [20,0], [20,33], [0,33]], dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.turn_src_points, self.turn_dst_points)

    def load_direction(self, image):

        cropped_parts = {}

        for name, area in self.areas.items():
            x, y, w, h = cv2.boundingRect(area)
            roi = image[y:y+h, x:x+w]
            mask = np.zeros((h, w), dtype=np.uint8)
            area_in_roi = area - [x, y]
            cv2.fillPoly(mask, [area_in_roi], 255)

            cropped_image = cv2.bitwise_and(roi, roi, mask=mask)

            if name == 'turn': cropped_image = self.turn_detection(cropped_image)
            if name == 'speed': cropped_image = self.speed_detection(cropped_image)

            cropped_parts[name] = cropped_image

    def turn_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_x = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if x > biggest_x: biggest_x = x
        self.turn_on_screen.update(biggest_x)
        # print("Turn: ", self.turn_on_screen.get())
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
                ((0, 0), (w, dH)),
                ((0, 0), (dW, h // 2)),
                ((w - dW, 0), (w, h // 2)),
                ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)),
                ((0, h // 2), (dW, h)),
                ((w - dW, h // 2), (w, h)),
                ((0, h - dH), (w, h))
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
                self.speed_on_screen.update(digit, idx)
            except: pass

            # cv2.imshow(f"{idx}", cv2.resize(roi, (roi.shape[1]*2, roi.shape[0]*2)))
        # print("Speed: ", self.digits_on_screen.get())

        return thresholded
