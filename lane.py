# Global variables
prev_leftx = None
prev_lefty = None
prev_rightx = None
prev_righty = None
prev_left_fit = []
prev_right_fit = []

prev_leftx2 = None
prev_lefty2 = None
prev_rightx2 = None
prev_righty2 = None
prev_left_fit2 = []
prev_right_fit2 = []


import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt

class LaneDetector:
    def __init__(self, width: int, height: int) -> None:
        """
        Initialize the LaneDetector object with image dimensions.

        Args:
            width (int): Width of the image.
            height (int): Height of the image.
        """
        self.width = width
        self.height = height

    def region_of_interest(self, img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """
        Apply a mask to the image to focus on the region of interest.

        Args:
            img (np.ndarray): Input image.
            vertices (np.ndarray): Array of vertices defining the region of interest.

        Returns:
            np.ndarray: Masked image with only the region of interest.
        """
        mask = np.zeros_like(img)
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lane_lines(self, img: np.ndarray, left_line: list, right_line: list, color: list = [0, 255, 0], thickness: int = 10) -> np.ndarray:
        """
        Draw a filled polygon between the detected lane lines on the image.

        Args:
            img (np.ndarray): Original image.
            left_line (list): Coordinates of the left lane line.
            right_line (list): Coordinates of the right lane line.
            color (list, optional): Color of the polygon. Defaults to [0, 255, 0].
            thickness (int, optional): Thickness of the lines. Defaults to 10.

        Returns:
            np.ndarray: Image with the polygon overlay.
        """
        line_img = np.zeros_like(img)
        poly_pts = np.array([[
            (left_line[0], left_line[1]),
            (left_line[2], left_line[3]),
            (right_line[2], right_line[3]),
            (right_line[0], right_line[1])
        ]], dtype=np.int32)

        cv2.fillPoly(line_img, poly_pts, color)
        img = cv2.addWeighted(img, 0.8, line_img, 0.5, 0.0)
        return img

    def pipeline(self, image: np.ndarray) -> np.ndarray:
        """
        The lane detection pipeline for processing the input image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with detected lanes.
        """
        region_of_interest_vertices = [
            (0, self.height),
            (self.width / 2, self.height / 2),
            (self.width, self.height),
        ]

        polygon = np.array(region_of_interest_vertices, np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        cv2.polylines(image, [polygon], isClosed=True, color=(255, 120, 0), thickness=2)

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cannyed_image = cv2.Canny(gray_image, 100, 200)
        cropped_image = self.region_of_interest(
            cannyed_image,
            np.array([region_of_interest_vertices], np.int32)
        )

        lines = cv2.HoughLinesP(
            cropped_image,
            rho=6,
            theta=np.pi / 60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=25
        )

        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []

        if lines is None:
            return image

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                if math.fabs(slope) < 0.5:
                    continue
                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

        min_y = int(image.shape[0] * (3 / 5))
        max_y = image.shape[0]

        if left_line_x and left_line_y:
            poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
        else:
            left_x_start, left_x_end = 0, 0

        if right_line_x and right_line_y:
            poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))
        else:
            right_x_start, right_x_end = 0, 0

        lane_image = self.draw_lane_lines(
            image,
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y]
        )

        return lane_image


class LaneDetectorU:
    def __init__(self, width: int, height: int, window_count: int = 8, margin: int = None) -> None:
        """
        Initialize the LaneDetectorU object with image dimensions and other parameters.

        Args:
            width (int): Width of the image.
            height (int): Height of the image.
            window_count (int, optional): Number of sliding windows for lane detection. Defaults to 8.
            margin (int, optional): Margin size for the sliding windows. Defaults to calculated value if None.
        """
        self.width = width
        self.height = height
        self.window_count = window_count
        self.interest_vertices = [
            (int(0.33 * self.width), int(0.64 * self.height)),
            (int(0.05 * self.width), int(0.95 * self.height)),
            (int(0.64 * self.width), int(0.64 * self.height)),
            (int(0.9 * self.width), int(0.95 * self.height)),
        ]
        self.desired_points = np.float32(self.interest_vertices)
        self.image_corners = np.float32([[0, 0], [0, self.height], [self.width, 0], [self.width, self.height]])
        self.matrix = cv2.getPerspectiveTransform(self.desired_points, self.image_corners)
        self.inv_matrix = cv2.getPerspectiveTransform(self.image_corners, self.desired_points)

        if margin is None:
            self.margin = int((1 / 12) * self.width)
        self.minpix = int((1 / 24) * self.width)

        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("L - H", "Trackbars", 0, 255, lambda x: None)
        cv2.createTrackbar("L - S", "Trackbars", 0, 255, lambda x: None)
        cv2.createTrackbar("L - V", "Trackbars", 150, 255, lambda x: None)
        cv2.createTrackbar("U - H", "Trackbars", 255, 255, lambda x: None)
        cv2.createTrackbar("U - S", "Trackbars", 255, 255, lambda x: None)
        cv2.createTrackbar("U - V", "Trackbars", 255, 255, lambda x: None)

    def detect_lane(self, img: np.ndarray) -> tuple:
        transformed_frame = cv2.warpPerspective(img, self.matrix, (self.width, self.height))
        hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv_transformed_frame, lower, upper)
    
        histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
        midpoint = np.int32(histogram.shape[0]/2)

        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint    


        frame_sliding_window = mask.copy()
        window_height = np.int32(mask.shape[0]/self.window_count)		
        nonzero = mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = []
        right_lane_inds = []
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        leftx_current = leftx_base
        rightx_current = rightx_base    

        for window in range(self.window_count):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = mask.shape[0] - (window + 1) * window_height
            win_y_high = mask.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),(
            win_xleft_high,win_y_high), (255,255,255), 2)
            cv2.rectangle(frame_sliding_window,(win_xright_low,win_y_low),(
            win_xright_high,win_y_high), (255,255,255), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xleft_low) & (
                                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xright_low) & (
                                nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
                
            # If you found > minpix pixels, recenter next window on mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
					
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract the pixel coordinates for the left and right lane lines
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds] 
        righty = nonzeroy[right_lane_inds]
    
        # Fit a second order polynomial curve to the pixel coordinates for
        # the left and right lane lines
        left_fit = None
        right_fit = None

        global prev_leftx
        global prev_lefty 
        global prev_rightx
        global prev_righty
        global prev_left_fit
        global prev_right_fit


        # Make sure we have nonzero pixels		
        if len(leftx)==0 or len(lefty)==0 or len(rightx)==0 or len(righty)==0:
            leftx = prev_leftx
            lefty = prev_lefty
            rightx = prev_rightx
            righty = prev_righty
		
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2) 

        # Add the latest polynomial coefficients		
        prev_left_fit.append(left_fit)
        prev_right_fit.append(right_fit)

        # Calculate the moving average	
        if len(prev_left_fit) > 10:
            prev_left_fit.pop(0)
            prev_right_fit.pop(0)
            left_fit = sum(prev_left_fit) / len(prev_left_fit)
            right_fit = sum(prev_right_fit) / len(prev_right_fit)
                
        prev_leftx = leftx
        prev_lefty = lefty 
        prev_rightx = rightx
        prev_righty = righty

        # ### Show Window

        # # Create the x and y values to plot on the image  
        # ploty = np.linspace(
        # 0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
        # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # # Generate an image to visualize the result
        # out_img = np.dstack((
        # frame_sliding_window, frame_sliding_window, (
        # frame_sliding_window))) * 255
            
        # # Add color to the left line pixels and right line pixels
        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
        # 0, 0, 255]
    
        # # Generate an image to draw the lane lines on 
        # warp_zero = np.zeros_like(mask).astype(np.uint8)
        # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))		
            
        # # Recast the x and y points into usable format for cv2.fillPoly()
        # pts_left = np.array([np.transpose(np.vstack([
        #                     left_fitx, ploty]))])
        # pts_right = np.array([np.flipud(np.transpose(np.vstack([
        #                     right_fitx, ploty])))])
        # pts = np.hstack((pts_left, pts_right))
        
        # # Draw lane on the warped blank image
        # cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # # Warp the blank back to original image space using inverse perspective 
        # # matrix (Minv)
        # newwarp = cv2.warpPerspective(color_warp, self.inv_matrix, (
        #                             img.shape[
        #                             1], img.shape[0]))
        
        # # Combine the result with the original image
        # result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        
        # for vertice in self.interest_vertices:
        #     cv2.circle(result, vertice, 5, (0,0,255), -1)    
        
        return frame_sliding_window, {"leftx": leftx,"lefty": lefty ,"rightx": rightx,"righty": righty}
        # return result, frame_sliding_window, out_img, pts, {"leftx": leftx,"lefty": lefty ,"rightx": rightx,"righty": righty}
