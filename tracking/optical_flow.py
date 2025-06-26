# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:10:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import cv2 as cv
import numpy as np
from typing import Tuple, Optional, Dict


class OpticalFlowTracker:
    """
    Encapsulates the logic for Lucas-Kanade (LK) optical flow tracking.

    This class is responsible for tracking a set of points from one frame to
    the next. It manages the state required for frame-to-frame tracking,
    including the previous frame and the points being tracked, abstracting
    away the specifics of the OpenCV optical flow implementation.
    """

    def __init__(self, lk_params: Optional[Dict] = None):
        """
        Initializes the optical flow tracker.

        Args:
            lk_params (Optional[Dict]): Parameters for the LK optical flow
                algorithm (cv.calcOpticalFlowPyrLK). If None, default
                parameters from the original project will be used.
        """
        if lk_params is None:
            # Default parameters from the original FaceTracker class
            self.lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
            )
        else:
            self.lk_params = lk_params

        # State variables for tracking
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None

    def initialize(self, initial_frame: np.ndarray, initial_points: np.ndarray) -> None:
        """
        Initializes or resets the tracker with a new frame and a set of points.
        This must be called before the first call to `track`.

        Args:
            initial_frame (np.ndarray): The first grayscale frame for tracking.
            initial_points (np.ndarray): The initial set of points to track,
                with shape (num_points, 1, 2).
        """
        self.prev_gray = initial_frame
        self.prev_points = initial_points

    def track(
            self, current_frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Tracks points from the previous frame to the current frame.

        This method uses the stored previous frame and points to calculate the
        new positions in the provided current frame. After the calculation, it
        updates its internal previous frame state to prepare for the next call.

        Note: This method does not update the points to be tracked for the next
        iteration. Call `update_tracked_points` with the final (e.g., smoothed)
        points to complete the tracking cycle.

        Args:
            current_frame (np.ndarray): The current grayscale video frame.

        Returns:
            A tuple containing:
            - next_points (Optional[np.ndarray]): The predicted new positions.
            - status (Optional[np.ndarray]): Indicates if flow was found for each point.
            - error (Optional[np.ndarray]): The error for each point.
            Returns (None, None, None) if the tracker is not initialized.
        """
        if self.prev_gray is None or self.prev_points is None:
            print("Warning: OpticalFlowTracker is not initialized. Call .initialize() first.")
            return None, None, None
        next_points, status, error = cv.calcOpticalFlowPyrLK(
            self.prev_gray, current_frame, self.prev_points, None, **self.lk_params
        )

        # Update the previous frame for the next iteration
        self.prev_gray = current_frame.copy()

        return next_points, status, error

    def update_tracked_points(self, points: np.ndarray) -> None:
        """
        Updates the points to be tracked for the next call to `track`.

        This should be called after processing the results of a `track` call,
        for instance, after smoothing or combining the tracked points with
        new detections.

        Args:
            points (np.ndarray): The new set of points to track from the
                most recent frame. Shape (num_points, 1, 2).
        """
        self.prev_points = points

    @staticmethod
    def validate_tracking(
            next_points: np.ndarray, prev_points: np.ndarray, status: np.ndarray
    ) -> np.ndarray:
        """
        Validates the tracked points based on the status array.

        If tracking for a point failed (status is 0), its previous position
        is used instead of the new, likely incorrect, position. This is a
        utility function and does not depend on tracker state.

        Args:
            next_points (np.ndarray): The tracked points from optical flow.
            prev_points (np.ndarray): The points from the previous frame.
            status (np.ndarray): The status array returned by `cv.calcOpticalFlowPyrLK`.

        Returns:
            np.ndarray: An array of points where failed tracks have been
                replaced with their previous positions.
        """
        valid_points = next_points.copy()

        # Identify points where tracking failed
        failed_tracks = (status.flatten() == 0)

        # For failed tracks, revert to the previous point's position
        # We need to reshape prev_points to match valid_points if they differ
        if prev_points.shape != valid_points.shape:
            prev_points = prev_points.reshape(valid_points.shape)

        valid_points[failed_tracks] = prev_points[failed_tracks]

        return valid_points
