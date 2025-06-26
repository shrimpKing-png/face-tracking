# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:10:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

from typing import Optional
import numpy as np
import config.config as cfg


class SmoothingEngine:
    """
    Handles smoothing algorithms for landmark positions.

    This class provides methods for applying smoothing techniques, such as a
    weighted moving average, to a series of data points. It is designed to be
    stateless, with all necessary data and parameters passed into its methods.
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
            position_history: np.ndarray,
            decay_factor: Optional[float] = None,
    ) -> np.ndarray:
        """
        Calculates the weighted moving average of landmark positions.

        This method applies an exponential decay to the weights of the historical
        positions, giving more influence to more recent data.

        Args:
            current_positions (np.ndarray): The current positions of the landmarks,
                with shape (num_landmarks, 2).
            position_history (np.ndarray): A history of previous landmark
                positions, with a shape of (history_length, num_landmarks, 2).
            decay_factor (Optional[float]): If provided, this will override the
                default decay factor for this calculation.

        Returns:
            np.ndarray: The smoothed landmark positions, with shape
                (num_landmarks, 2).
        """
        active_decay_factor = decay_factor if decay_factor is not None else self.decay_factor

        history_len = min(len(position_history), self.smoothing_window)
        if history_len == 0:
            return current_positions

        # Generate weights with exponential decay
        weights = np.array([active_decay_factor ** i for i in range(history_len + 1)])
        weights /= np.sum(weights)  # Normalize weights

        # Combine current and historical positions for calculation
        all_positions = np.concatenate(
            [current_positions[np.newaxis, :, :], position_history[:history_len]], axis=0
        )

        # Apply weights and sum to get the smoothed positions
        smoothed_positions = np.sum(
            all_positions * weights[:, np.newaxis, np.newaxis], axis=0
        )

        return smoothed_positions

    def apply_smoothing(
            self,
            current_positions: np.ndarray,
            position_history: np.ndarray,
            method: str = 'weighted_moving_average',
            **kwargs,
    ) -> np.ndarray:
        """
        Applies a specified smoothing method to the landmark positions.

        This acts as a dispatcher to various smoothing algorithms. Currently,
        it supports 'weighted_moving_average'.

        Args:
            current_positions (np.ndarray): The current positions of the landmarks.
            position_history (np.ndarray): A history of previous landmark positions.
            method (str): The smoothing method to use.
            **kwargs: Additional keyword arguments to pass to the smoothing method.

        Returns:
            np.ndarray: The smoothed landmark positions.

        Raises:
            ValueError: If an unsupported smoothing method is specified.
        """
        if method == 'weighted_moving_average':
            return self.calculate_weighted_moving_average(
                current_positions, position_history, **kwargs
            )
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
