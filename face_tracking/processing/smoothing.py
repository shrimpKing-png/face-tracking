# -*- coding: utf-8 -*-
"""
processing/smoothing.py
Created on Wed Jun 25 20:10:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

from typing import Optional, TYPE_CHECKING
import numpy as np
from face_tracking.config import settings as cfg

if TYPE_CHECKING:
    pass


class SmoothingEngine:
    """
    Handles smoothing algorithms for landmark positions using TrackingHistory.

    This class provides methods for applying smoothing techniques, such as a
    weighted moving average, to a series of data points. It is optimized to work
    directly with TrackingHistory objects to avoid data duplication.
    """

    def __init__(
            self,
            decay_factor: float = cfg.DECAY_FACTOR,
            smoothing_window: int = cfg.SMOOTHING_WINDOW,
    ):
        """
        Initializes the SmoothingEngine with configurable parameters.

        Args:
            decay_factor (float): The factor by which the influence of
                older data points decreases. Defaults to the value in
                `config.settings`.
            smoothing_window (int): The number of previous frames to
                consider for the moving average. Defaults to the value in
                `config.settings`.
        """
        self.decay_factor = decay_factor
        self.smoothing_window = smoothing_window

    def calculate_weighted_moving_average(
            self,
            current_positions: np.ndarray,
            tracking_history: 'TrackingHistory',
            decay_factor: Optional[float] = None,
    ) -> np.ndarray:
        """
        Calculates the weighted moving average using TrackingHistory data.

        This method applies an exponential decay to the weights of the historical
        positions stored in the TrackingHistory object, giving more influence to
        more recent data.

        Args:
            current_positions (np.ndarray): The current positions of the landmarks,
                with shape (num_landmarks, 2).
            tracking_history (TrackingHistory): The tracking history object containing
                position history data.
            decay_factor (Optional[float]): If provided, this will override the
                default decay factor for this calculation.

        Returns:
            np.ndarray: The smoothed landmark positions, with shape
                (num_landmarks, 2).
        """
        active_decay_factor = decay_factor if decay_factor is not None else self.decay_factor

        # Use the actual filled history length, not the buffer size
        history_len = min(tracking_history.history_filled, self.smoothing_window)

        if history_len == 0:
            return current_positions

        # Generate weights with exponential decay
        weights = np.array([active_decay_factor ** i for i in range(history_len + 1)])
        weights /= np.sum(weights)  # Normalize weights

        # Get the relevant portion of history (only filled positions)
        relevant_history = tracking_history.position_history[:history_len]

        # Combine current and historical positions for calculation
        all_positions = np.concatenate(
            [current_positions[np.newaxis, :, :], relevant_history], axis=0
        )

        # Apply weights and sum to get the smoothed positions
        smoothed_positions = np.sum(
            all_positions * weights[:, np.newaxis, np.newaxis], axis=0
        )

        return smoothed_positions

    def apply_smoothing(
            self,
            current_positions: np.ndarray,
            tracking_history: 'TrackingHistory',
            method: str = 'weighted_moving_average',
            **kwargs,
    ) -> np.ndarray:
        """
        Applies a specified smoothing method using TrackingHistory data.

        This acts as a dispatcher to various smoothing algorithms that work
        directly with TrackingHistory objects. Currently supports
        'weighted_moving_average'.

        Args:
            current_positions (np.ndarray): The current positions of the landmarks.
            tracking_history (TrackingHistory): The tracking history object.
            method (str): The smoothing method to use.
            **kwargs: Additional keyword arguments to pass to the smoothing method.

        Returns:
            np.ndarray: The smoothed landmark positions.

        Raises:
            ValueError: If an unsupported smoothing method is specified.
        """
        if method == 'weighted_moving_average':
            return self.calculate_weighted_moving_average(
                current_positions, tracking_history, **kwargs
            )
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def smooth_and_update_history(
            self,
            current_positions: np.ndarray,
            tracking_history: 'TrackingHistory',
            method: str = 'weighted_moving_average',
            **kwargs,
    ) -> np.ndarray:
        """
        Convenience method that applies smoothing and updates the history in one call.

        This method applies the specified smoothing algorithm and then automatically
        adds the smoothed positions to the tracking history.

        Args:
            current_positions (np.ndarray): The current positions of the landmarks.
            tracking_history (TrackingHistory): The tracking history object.
            method (str): The smoothing method to use.
            **kwargs: Additional keyword arguments to pass to the smoothing method.

        Returns:
            np.ndarray: The smoothed landmark positions.
        """
        smoothed_positions = self.apply_smoothing(
            current_positions, tracking_history, method, **kwargs
        )

        # Update the history with the smoothed positions
        tracking_history.add_position_to_history(smoothed_positions)

        return smoothed_positions
