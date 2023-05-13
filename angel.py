"""
Blink Detection Script

This script detects blinks in a live video stream or a video file by analyzing eye landmarks.
It uses the dlib library for face and landmark detection, OpenCV for video rendering, and
scipy for calculating the Eye Aspect Ratio (EAR).

Usage:
    1. Ensure that the required dependencies are installed: cv2, dlib, imutils, scipy, numpy.
    2. Place the video file in the same directory as this script, or provide the full path to the video file.
    3. Initialize the BlinkDetector class with the video path, blink threshold, and consecutive frames.
    4. Call the run() method to start the blink detection process.
    5. Press 'q' to exit the script.

Author: Ethan Lew 
"""
import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np


class BlinkDetector:
    def __init__(self, video_path, blink_threshold=0.35, consecutive_frames=2):
        """
        Initialize the BlinkDetector class.

        Args:
            video_path (str): Path to the video file.
            blink_threshold (float): Threshold value for detecting a blink (default: 0.35).
            consecutive_frames (int): Number of consecutive frames required to confirm a blink (default: 2).
        """
        self.video_path = video_path
        self.blink_threshold = blink_threshold
        self.consecutive_frames = consecutive_frames
        self.blink_count = 0
        self.viewer_window_name = "Weeping Angel"

        self.left_eye_start, self.left_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.right_eye_start, self.right_eye_end = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

        self.cam = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(self.video_path)

        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor('Model/shape_predictor_68_face_landmarks.dat')

    def calculate_EAR(self, eye):
        """
        Calculate the Eye Aspect Ratio (EAR) using the landmarks of the eye.

        Args:
            eye (list): List of (x, y) coordinates of the eye landmarks.

        Returns:
            float: Calculated EAR value.
        """
        y1 = dist.euclidean(eye[1], eye[5])
        y2 = dist.euclidean(eye[2], eye[4])
        x1 = dist.euclidean(eye[0], eye[3])
        ear = (y1 + y2) / x1
        return ear

    def run(self):
        """
        Run the blink detection process.
        """
        if self.cap.isOpened():
            ret, framev = self.cap.read()

        while True:
            _, frame = self.cam.read()
            frame = imutils.resize(frame, width=640)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(img_gray)

            for face in faces:
                shape = self.landmark_predictor(img_gray, face)
                shape = face_utils.shape_to_np(shape)
                left_eye = shape[self.left_eye_start:self.left_eye_end]
                right_eye = shape[self.right_eye_start:self.right_eye_end]

                left_ear = self.calculate_EAR(left_eye)
                right_ear = self.calculate_EAR(right_eye)
                avg_ear = (left_ear + right_ear) / 2

                if avg_ear < self.blink_threshold:
                    self.blink_count += 1
                    if self.cap.isOpened():
                        ret, _framev = self.cap.read()
                        if ret:
                            framev = _framev
                    cv2.putText(frame, 'Blink Detected', (30, 30),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                else:
                    if self.consecutive_frames > 0:
                        self.consecutive_frames -= 1
                    else:
                        self.consecutive_frames = 0

            frameh = np.hstack([frame, framev])
            cv2.imshow(self.viewer_window_name, frameh)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = BlinkDetector('Asset/weeping.mp4', blink_threshold=0.35, consecutive_frames=2)
    detector.run()
