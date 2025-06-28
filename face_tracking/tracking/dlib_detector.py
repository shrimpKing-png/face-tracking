# -*- coding: utf-8 -*-
"""
tracking/dlib_detector.py
Created on Wed Jun 25 20:10:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import dlib
import numpy as np
import cv2 as cv
from face_tracking.config import settings as cfg
import warnings
from importlib import resources


class DlibDetector:
    def __init__(self,
                 detector_name=cfg.DETECTOR_NAME,
                 predictor_name=cfg.PREDICTOR_NAME):
        model_dir = resources.files('face_tracking').joinpath('dlib-models')
        detector_path = str(model_dir.joinpath(detector_name))
        predictor_path = str(model_dir.joinpath(predictor_name))
        self.detector = dlib.simple_object_detector(detector_path)
        self.predictor = dlib.shape_predictor(predictor_path)
        self.type = 'dlib'

    def extract_faces(self, img):
        """
        :param img: the image that contains the face or faces you want to extract
        """
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)

        faces = self.detector(img)
        num_faces = len(faces)
        if num_faces == 0:
            warnings.warn('No faces detected')
            return None, faces, num_faces
        elif num_faces == 1:
            # Retrieve the landmarks for the face selected
            face = faces[0]
            landmarks = self.predictor(img, face)
            return landmarks, face, num_faces
        else:
            print(f"Multiple faces detected ({num_faces} faces found)")
            print("Displaying landmarks for each face: ")

            all_landmarks = []

            # Show landmarks for each face
            for i, face in enumerate(faces):
                landmarks = self.predictor(img, face)
                all_landmarks.append(landmarks)
            display_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            for j, landmarks in enumerate(all_landmarks):
                landmark_points = []
                for i in range(landmarks.num_parts):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    landmark_points.append((x, y))

                # Plot all landmark points on the image
                for i, (x, y) in enumerate(landmark_points):
                    cv.circle(display_img, (x, y), 3, (0, 255, 0), -1)  # Green circles
                    cv.putText(display_img, str(j + 1), (x + 5, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            cv.imshow("ref_img", display_img)
            print("Enter the correct number of the face you want to extract: ")
            key = cv.waitKey(10)
            try:
                correct_face_num = int(chr(key))
            except ValueError:
                print("No face selected, using face 1.")
                correct_face_num = 1
            cv.destroyAllWindows()
            return all_landmarks[correct_face_num - 1], faces[correct_face_num - 1], num_faces
