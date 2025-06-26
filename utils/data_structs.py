# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:10:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import numpy as np
import config.settings as cfg


class TrackingHistory:
    """
    A data structure to hold and manage the historical data for face tracking,
    including landmark positions and motion vectors.
    """

    def __init__(self, num_landmarks=cfg.NUM_LANDMARKS, smoothing_window=cfg.SMOOTHING_WINDOW):
        """
        Initializes the tracking history.

        Args:
            num_landmarks (int): The number of facial landmarks.
            smoothing_window (int): The number of frames for the smoothing window.
        """
        self.num_landmarks = num_landmarks
        self.smoothing_window = smoothing_window

        # History of dlib-detected landmark points
        self.dlib_points_history = []

        # History for weighted moving average
        self.position_history = np.zeros((smoothing_window, num_landmarks, 2))
        self.history_filled = 0

        # Motion vectors history
        self.motion_vectors = np.array([]).reshape(0, num_landmarks)

    def log_motion_vector(self, motion_magnitudes: np.ndarray):
        """
        Adds a new set of motion magnitudes to the history.

        Args:
            motion_magnitudes (np.ndarray): An array of motion magnitudes.
        """
        if motion_magnitudes.shape[0] != self.num_landmarks:
            # Handle cases where the number of landmarks might not match
            # This could be padding or raising an error. For now, we'll print a warning.
            print(
                f"Warning: Motion vector size mismatch. Expected {self.num_landmarks}, got {motion_magnitudes.shape[0]}")
            # We will pad with zeros to maintain consistency
            padded_magnitudes = np.zeros(self.num_landmarks)
            padded_magnitudes[:motion_magnitudes.shape[0]] = motion_magnitudes
            motion_magnitudes = padded_magnitudes

        motion_row = motion_magnitudes.reshape(1, -1)
        self.motion_vectors = np.vstack([self.motion_vectors, motion_row])

    def add_position_to_history(self, positions: np.ndarray):
        """
        Adds landmark positions to the circular history buffer.

        Args:
            positions (np.ndarray): Array of (x, y) positions.
        """
        if positions.shape != (self.num_landmarks, 2):
            print(f"Warning: Position data shape mismatch. Expected ({self.num_landmarks}, 2), got {positions.shape}")
            return

        # Use np.roll for a more efficient circular buffer
        self.position_history = np.roll(self.position_history, 1, axis=0)
        self.position_history[0, :, :] = positions

        if self.history_filled < self.smoothing_window:
            self.history_filled += 1

    def add_dlib_points(self, points: np.ndarray):
        """
        Adds the latest dlib-detected points to the history.

        Args:
            points (np.ndarray): The array of detected landmark points.
        """
        self.dlib_points_history.append(points)
