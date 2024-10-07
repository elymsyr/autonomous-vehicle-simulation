import numpy as np
from Xlib import X, display
import cv2, mss
from time import perf_counter

class CityDriveCaptureTest:
    """
    A class designed to capture and record video from a specific window in Linux environments,
    such as the 'City Car Driving Home Edition Steam' game.

    Attributes:
        window_title (str): The title of the window to capture.
        display (display.Display): The display object representing the screen.
        root (window): The root window of the display.
        window (window): The target window for capturing.
        monitor (dict): The monitor geometry details for screen capture.
        fps_list (list[float]): List to store frame rates.
        video_output (str): The file path for video output.
        fps (int): Frames per second for recording.
        video_writer (cv2.VideoWriter): The object responsible for writing video files.
        cap (object): Placeholder for additional capture functionality.

    Methods:
        create_monitor(window): Creates the monitor configuration for screen capture.
        find_window(): Finds the target window based on the window title.
        capture(sct): Captures an image of the specified monitor area.
        window_linux(show_result=True): Captures and records video from the specified window.
    """

    def __init__(self, video_output: str, window_title: str = 'City Car Driving Home Edition Steam', fps: int = 30) -> None:
        """
        Initializes the CityDriveCaptureTest object with the specified parameters.

        Args:
            video_output (str): The path to save the recorded video.
            window_title (str, optional): The title of the target window. Default is 'City Car Driving Home Edition Steam'.
            fps (int, optional): The frames per second for the video recording. Default is 30.
        """
        self.window_title = window_title
        self.display = display.Display()
        self.root = self.display.screen().root
        self.window, self.monitor = self.find_window()

        self.fps_list = []
        self.video_output = video_output
        self.fps = fps
        self.video_writer = None
        self.cap = None

    def create_monitor(self, window) -> dict:
        """
        Creates a monitor configuration for screen capture.

        Args:
            window: The window object to define the monitor geometry.

        Returns:
            dict: A dictionary containing the monitor's top, left, width, and height.
        """
        geometry = window.query_tree().parent.get_geometry()
        return {
            "top": int(geometry.y) + 50,
            "left": geometry.x + 1,
            "width": geometry.width - 1,
            "height": int(geometry.height) - 108
        }

    def find_window(self):
        """
        Finds the target window based on the specified window title.

        Returns:
            tuple: A tuple containing the window object and the monitor configuration.
        """
        window = None
        window_ids = self.root.get_full_property(self.display.intern_atom('_NET_CLIENT_LIST'), X.AnyPropertyType).value
        for window_id in window_ids:
            window = self.display.create_resource_object('window', window_id)
            window_name_str = window.get_wm_name()
            if window_name_str and self.window_title in window_name_str:
                break
        monitor = self.create_monitor(window=window) if window else None
        return window, monitor

    def capture(self, sct: mss.mss) -> np.ndarray:
        """
        Captures an image of the specified monitor area.

        Args:
            sct (mss.mss): The MSS object for capturing the screen.

        Returns:
            np.ndarray: The captured image converted to BGR format.
        """
        img = sct.grab(self.monitor)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

    def window_linux(self, show_result: bool = True) -> None:
        """
        Captures and records video from the specified window, displaying the FPS in real-time.

        Args:
            show_result (bool, optional): Whether to show the captured frames in a window. Default is True.
        """
        with mss.mss() as sct:
            self.window_image = self.capture(sct)
            self.width, self.height = int(self.window_image.shape[1]), int(self.window_image.shape[0])

            self.fps_list = []

            if self.video_output:
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for AVI format
                self.video_writer = cv2.VideoWriter(self.video_output, fourcc, self.fps, (self.width, self.height))

            self.prevTime = perf_counter()

            while True:
                self.window_image = self.capture(sct)
                if self.window_image is not None:
                    currTime = perf_counter()
                    fps = 1 / (currTime - self.prevTime)
                    self.fps_list.append(fps)
                    self.prevTime = currTime

                    if show_result:
                        cv2.putText(self.window_image, f"FPS: {int(fps)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        if self.window_image is not None: cv2.imshow(f'YOLOV8 IMAGE', cv2.resize(self.window_image, (self.window_image.shape[1]//2, self.window_image.shape[0]//2)))

                    if self.video_writer:
                        self.video_writer.write(self.window_image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if len(self.fps_list) > 10000:
                    break

if '__main__' == __name__:
    agent = CityDriveCaptureTest(video_output='media/clip0.mp4')
    agent.window_linux()
