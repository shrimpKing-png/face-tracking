# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:10:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import numpy as np
from ..utils.data_structs import SmoothedLandmarks


def landmarks_to_points(landmarks) -> np.ndarray:
    """
    Convert dlib landmarks to a numpy points array.

    This format is required for OpenCV's optical flow function.

    Args:
        landmarks: A dlib landmarks object.

    Returns:
        A numpy array of shape (num_points, 1, 2).
    """
    if landmarks is None:
        return None

    points = np.array([[part.x, part.y] for part in landmarks.parts()], dtype=np.float32)
    return points.reshape(-1, 1, 2)


def points_to_landmarks(points: np.ndarray) -> object:
    """
    Convert a numpy array of points back to a dlib-like landmarks object.

    Args:
        points: A numpy array of landmark points.

    Returns:
        An object that mimics the dlib landmarks structure.
    """
    if points is None:
        return None

    return SmoothedLandmarks(points)