import numpy as np
import cv2
from time import perf_counter, sleep

class KeypointTrack():
    def __init__(self, image) -> None:
        self.keypoint_history = []
        self.keypoints = None
        self.descriptors = None
        self.detector = cv2.ORB_create()
        self.bf = cv2.BFMatcher()
        self.last_frame = None
    
    def load(self, image):
        keypoints, descriptors = self.load_points(image)

        # if len(self.keypoint_history) >= 5: self.keypoint_history.pop(0)
        # self.keypoint_history.append(self.keypoints)

        image = self.draw_keypoints(image, keypoints)

        matches = self.match(keypoints, descriptors)

        frames_image = self.draw_keypoints_movement_merged(image, matches)

        self.last_frame, self.keypoints, self.descriptors = image, keypoints, descriptors

        return frames_image
    
    def load_points(self, image):
        keypoints = self.detector.detect(image, None)
        if keypoints is not None: keypoints = [kp for kp in keypoints if kp.size > 60 and kp.response > 0.0001]
        keypoints, descriptors = self.detector.compute(image, keypoints)
        if keypoints is not None and descriptors is not None:
            len_before  = len(keypoints)
            # for kp in keypoints:
            #     print(kp.size, kp.response)
            ignored_idx = []
            matches = self.bf.match(descriptors, descriptors)
            for match in matches:
                if match.queryIdx != match.trainIdx: ignored_idx.append(match.queryIdx)
            keypoints = [value for index, value in enumerate(keypoints) if index not in ignored_idx]
            if len_before != len(keypoints): print(f"{len_before}  ->  {len(keypoints)}")
        return self.detector.compute(image, keypoints)

    def draw_keypoints(self, image, keypoints):
        return cv2.drawKeypoints(image, keypoints, color=(0,255,120), flags=0, outImage=None)

    def draw_keypoints_movement(self, image, matches):
        if self.last_frame is None: return image
        # Iterate over the matches to draw movement lines
        for (prev_point, curr_point) in matches:
            # Draw a line between previous and current keypoint locations
            cv2.line(image, 
                    (int(prev_point[0]), int(prev_point[1])), 
                    (int(curr_point[0]), int(curr_point[1])), 
                    color=(255, 0, 0), thickness=1)
            
            # Optionally, draw a circle at the current point to highlight it
            cv2.circle(image, 
                    (int(curr_point[0]), int(curr_point[1])), 
                    5, color=(0, 0, 255), thickness=-1)

        return image

    def draw_keypoints_movement_merged(self, image, matches):
        if self.last_frame is None: return image
        # Get the dimensions of the last frame
        height, width = self.last_frame.shape[:2]
        
        # Create a new image to merge the current image and last frame side by side
        merged_image = np.hstack((self.last_frame, image))

        # Iterate over the matches to draw movement lines
        for (prev_point, curr_point) in matches:
            # Calculate the coordinates for the merged image
            prev_x, prev_y = int(prev_point[0]), int(prev_point[1])
            curr_x, curr_y = int(curr_point[0]) + width, int(curr_point[1])  # Shift current point by the width of the last frame

            # Draw a line between previous and current keypoint locations in the merged image
            cv2.line(merged_image, 
                    (prev_x, prev_y), 
                    (curr_x, curr_y), 
                    color=(255, 0, 0), thickness=1)

            # Optionally, draw a circle at the current point to highlight it
            cv2.circle(merged_image, 
                    (curr_x, curr_y), 
                    5, color=(0, 0, 255), thickness=-1)

        return merged_image

    def match(self, keypoints, descriptors):
        matches_over_frames = []
        if keypoints is not None and descriptors is not None and self.keypoints is not None and self.descriptors is not None:
            matches = self.bf.match(self.descriptors, descriptors)
            for match in matches:
                # queryIdx gives keypoint index from target image
                query_idx = match.queryIdx
                # .trainIdx gives keypoint index from current frame 
                train_idx = match.trainIdx

                    # take coordinates that matches
                pt1 = self.keypoints[query_idx].pt

                # current frame keypoints coordinates
                pt2 = keypoints[train_idx].pt
                matches_over_frames.append((pt1, pt2))                
        return matches_over_frames

