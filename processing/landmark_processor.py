# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:10:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import numpy as np


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
        raise ValueError('No landmarks provided')

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


class SmoothedLandmarks:
    """
    A data structure to hold smoothed landmark points, mimicking dlib's interface.
    Allows for creation of new landmarks to use with legacy tracking functions.
    """

    def __init__(self, points_array: np.ndarray):
        """
        Initializes the object with a numpy array of points.

        Args:
            points_array: A numpy array of shape (num_points, 1, 2) or (num_points, 2).
        """
        self.points = points_array.reshape(-1, 2)
        self.num_parts = len(self.points)

    def part(self, index: int):
        """
        Returns a point object with 'x' and 'y' attributes, similar to dlib.

        Args:
            index: The index of the landmark point.

        Returns:
            A simple object with integer x and y coordinates.
        """

        class Point:
            def __init__(self, x, y):
                self.x = int(x)
                self.y = int(y)

        return Point(self.points[index, 0], self.points[index, 1])
