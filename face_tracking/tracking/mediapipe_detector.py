"""
tracking/mediapipe_detector.py
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
                 min_detection_con=0.1, min_tracking_con=0.5, num_landmarks=468):
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
        self.num_landmarks = num_landmarks
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

    def extract_faces(self, frame):
        """
        Extract facial landmarks from a frame using MediaPipe.

        Args:
            frame (np.ndarray): Input frame (grayscale or BGR)

        Returns:
            list: List containing single landmark object (or None if no face detected)
                  Format matches dlib detector output for interchangeability
        """
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
            for mp_idx in range(self.num_landmarks):
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