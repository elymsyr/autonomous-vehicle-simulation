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

class Digits:
    def __init__(self) -> None:
        """
        Initializes the Digits class with a dictionary to hold digit values.
        """
        self.digits: dict[str, int] = {'0': 0, '1': 0, '2': 0}

    def update(self, number: int, idx: int = None) -> None:
        """
        Updates the digit dictionary with a new number.

        Args:
            number (int): The digit or number to be updated.
            idx (int, optional): Specific index to update if provided.
        """
        if idx is not None and isinstance(idx, int) and 0 <= idx <= 2:
            self.digits[f"{idx}"] = number
        else:
            number_str = str(number).zfill(3)  # Ensure the number is 3 digits long
            for i, digit in enumerate(number_str):
                self.digits[f"{i}"] = int(digit)

    def get(self) -> int:
        """
        Returns the combined integer value from the digit dictionary.

        Returns:
            int: The integer value represented by the digits.
        """
        return int(''.join(str(self.digits[str(i)]) for i in range(3)))

class Turn:
    def __init__(self) -> None:
        """
        Initializes the Turn class to track the turn value.
        """
        self.turn: float = 0.0

    def update(self, number: int) -> None:
        """
        Updates the turn value based on the input number.

        Args:
            number (int): The raw position value to calculate the turn.
        """
        number = min(max(number, 0), 44)  # Ensure number is within range
        self.turn = (number - 22) / 22  # Normalize the value to a range of [-1, 1]

    def get(self) -> float:
        """
        Returns the normalized turn value.

        Returns:
            float: The normalized turn value.
        """
        return self.turn

class DirectionDetection:
    def __init__(self) -> None:
        """
        Initializes the DirectionDetection class with areas of interest for speed and turn detection.
        """
        self.speed_on_screen = Digits()
        self.turn_on_screen = Turn()
        self.areas = {
            'speed': 2 * np.array([[74, 44], [106.5, 44], [106.5, 60.5], [74, 60.5]], dtype=np.int32),
            'turn': 2 * np.array([[133, 290], [160, 290], [160, 292], [133, 292]], dtype=np.int32)
        }
        self.digits = [
            np.array([[0, 0], [20, 0], [20, 33], [0, 33]], dtype=np.int32),
            np.array([[23, 0], [43, 0], [43, 33], [23, 33]], dtype=np.int32),
            np.array([[45, 0], [65, 0], [65, 33], [45, 33]], dtype=np.int32)
        ]
        self.turn_src_points = np.array([[3, 0], [20, 0], [17, 33], [0, 33]], dtype=np.float32)
        self.turn_dst_points = np.array([[0, 0], [20, 0], [20, 33], [0, 33]], dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.turn_src_points, self.turn_dst_points)

    def load_direction(self, image: np.ndarray) -> None:
        """
        Processes the input image to detect and extract the turn and speed values.

        Args:
            image (np.ndarray): The input image containing the dashboard.
        """
        cropped_parts = {}
        for name, area in self.areas.items():
            x, y, w, h = cv2.boundingRect(area)
            roi = image[y:y + h, x:x + w]
            mask = np.zeros((h, w), dtype=np.uint8)
            area_in_roi = area - [x, y]
            cv2.fillPoly(mask, [area_in_roi], 255)
            cropped_image = cv2.bitwise_and(roi, roi, mask=mask)

            if name == 'turn':
                cropped_image = self.turn_detection(cropped_image)
            elif name == 'speed':
                cropped_image = self.speed_detection(cropped_image)

            cropped_parts[name] = cropped_image

    def turn_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Detects the turn indicator from the image.

        Args:
            image (np.ndarray): The input image for turn detection.

        Returns:
            np.ndarray: The processed image with detected turn indicators highlighted.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_x = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if x > biggest_x:
                biggest_x = x
        self.turn_on_screen.update(biggest_x)
        return image

    def speed_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Detects the speed digits from the image.

        Args:
            image (np.ndarray): The input image for speed digit detection.

        Returns:
            np.ndarray: The thresholded image with digit segments highlighted.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        for idx, digit in enumerate(self.digits):
            x, y, w, h = cv2.boundingRect(digit)
            roi = thresholded[y:y + h, x:x + w]
            roi = cv2.warpPerspective(roi, self.matrix, (20, 33))

            segments = [
                ((0, 0), (w, int(h * 0.15))),
                ((0, 0), (int(w * 0.25), h // 2)),
                ((w - int(w * 0.25), 0), (w, h // 2)),
                ((0, (h // 2) - int(h * 0.05)), (w, (h // 2) + int(h * 0.05))),
                ((0, h // 2), (int(w * 0.25), h)),
                ((w - int(w * 0.25), h // 2), (w, h)),
                ((0, h - int(h * 0.15)), (w, h))
            ]
            on = [0] * len(segments)

            for i, ((xA, yA), (xB, yB)) in enumerate(segments):
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                if total / float(area) > 0.5:
                    on[i] = 1

            try:
                digit_value = DIGITS_LOOKUP[tuple(on)]
                self.speed_on_screen.update(digit_value, idx)
            except KeyError:
                pass

        return thresholded