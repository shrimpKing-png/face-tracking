"""
mediapipe detector implementation
Last Update: 26JUNE2025
Author: GPAULL & IWEBB
"""

import mediapipe as mp
import cv2 as cv
import numpy as np
from face_tracking.processing.landmark_processor import points_to_landmarks


class MediaPipeDetector:
    """
    MediaPipe-based face detector that mirrors the DlibDetector interface.
    Returns landmarks in a format compatible with dlib for interchangeability.
    """

    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                 min_detection_con=0.1, min_tracking_con=0.5):
        """
        Initialize MediaPipe FaceMesh detector.

        Args:
            static_image_mode (bool): Whether to treat input as static images
            max_num_faces (int): Maximum number of faces to detect (kept at 1 for dlib compatibility)
            refine_landmarks (bool): Whether to refine landmarks around eyes and lips
            min_detection_con (float): Minimum confidence for face detection
            min_tracking_con (float): Minimum confidence for face tracking
        """
        self.type = 'mediapipe'
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_con = min_detection_con
        self.min_tracking_con = min_tracking_con

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            self.refine_landmarks,
            self.min_detection_con,
            self.min_tracking_con
        )

        # MediaPipe to dlib landmark mapping (68 key facial landmarks)
        # This maps MediaPipe's 468 landmarks to dlib's 68 landmark format
        self.MEDIAPIPE_TO_DLIB_MAP = [
            # Jaw line (0-16)
            172, 176, 148, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67,
            # Right eyebrow (17-21)
            296, 334, 293, 300, 276,
            # Left eyebrow (22-26)
            70, 63, 105, 66, 107,
            # Nose bridge (27-30)
            9, 10, 151, 195,
            # Lower nose (31-35)
            236, 3, 51, 48, 115,
            # Right eye (36-41)
            33, 7, 163, 144, 145, 153,
            # Left eye (42-47)
            362, 398, 384, 385, 386, 387,
            # Outer lip (48-59)
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            # Inner lip (60-67)
            78, 95, 88, 178, 87, 14, 317, 402
        ]

    def extract_faces(self, frame):
        """
        Extract facial landmarks from a frame using MediaPipe.

        Args:
            frame (np.ndarray): Input frame (grayscale or BGR)

        Returns:
            list: List containing single landmark object (or None if no face detected)
                  Format matches dlib detector output for interchangeability
        """
        print('extracting face with mediapipe')
        # Convert frame to RGB if needed
        if len(frame.shape) == 3:
            if frame.shape[2] == 3:  # BGR
                imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            else:  # Already RGB
                imgRGB = frame
        else:  # Grayscale
            imgRGB = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)

        # Process frame with MediaPipe
        results = self.faceMesh.process(imgRGB)
        print(results)

        if results.multi_face_landmarks:
            # Get the first face (MediaPipe can detect multiple, but we use only first for dlib compatibility)
            face_landmarks = results.multi_face_landmarks[0]

            # Extract all landmark points
            h, w = frame.shape[:2]
            all_points = []

            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                all_points.append((x, y))

            # Map MediaPipe's 468 landmarks to dlib's 68 landmarks
            dlib_points = []
            for mp_idx in self.MEDIAPIPE_TO_DLIB_MAP:
                if mp_idx < len(all_points):
                    dlib_points.append(all_points[mp_idx])
                else:
                    # Fallback if index is out of range
                    dlib_points.append((0, 0))

            # Convert points to dlib-compatible landmark format
            points_array = np.array(dlib_points, dtype=np.float32).reshape(-1, 1, 2)
            landmarks = points_to_landmarks(points_array)

            return (landmarks, len(results.multi_face_landmarks), 1)

        # No face detected
        return [None]