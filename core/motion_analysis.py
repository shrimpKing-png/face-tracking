import numpy as np
import pandas as pd


class MotionAnalyzer:
    """
    Handles motion vector analysis and z-score calculations.
    This class is stateless and all methods are static or class methods
    to reflect that.
    """

    @staticmethod
    def calculate_z_scores(
            current_magnitudes: np.ndarray,
            motion_history: np.ndarray,
            history_window: int,
    ) -> np.ndarray:
        """
        Calculate z-scores for current motion magnitudes based on history.

        Args:
            current_magnitudes (np.ndarray): Current frame motion magnitudes.
            motion_history (np.ndarray): The history of motion vectors.
            history_window (int): The number of previous frames to consider.

        Returns:
            np.ndarray: Z-scores for each landmark.
        """
        num_landmarks = current_magnitudes.shape[0]
        if motion_history.shape[0] < 2:
            return np.ones(num_landmarks) * 10.0

        start_idx = max(0, motion_history.shape[0] - history_window)
        history = motion_history[start_idx:, :]

        if history.shape[0] < 2:
            return np.ones(num_landmarks) * 10.0

        mean = np.mean(history, axis=0)
        std = np.std(history, axis=0)

        # Avoid division by zero
        std[std == 0] = 1.0

        z_scores = np.abs((current_magnitudes - mean) / std)
        return z_scores

    @staticmethod
    def build_motion_stat_df(motion_history: np.ndarray, num_landmarks: int) -> pd.DataFrame:
        """
        Builds a pandas DataFrame from the motion vectors history.

        Args:
            motion_history (np.ndarray): The history of motion vectors.
            num_landmarks (int): The number of landmarks.

        Returns:
            pd.DataFrame: A DataFrame containing motion vector statistics.
        """
        column_names = [f'landmark_{i}_motion_vector' for i in range(num_landmarks)]

        if motion_history.shape[0] == 0:
            print("Warning: No motion vectors available to build the DataFrame.")
            return pd.DataFrame(columns=column_names)

        df = pd.DataFrame(data=motion_history, columns=column_names)
        landmark_motion_sums = df.sum()
        sorted_landmark_motion = landmark_motion_sums.sort_values(ascending=True)

        print("\\n--- Landmark Motion Analysis ---")
        print("Total motion for each landmark, from least to greatest:")
        print("-" * 35)
        for landmark_name, total_motion in sorted_landmark_motion.items():
            print(f"{landmark_name}: {total_motion:.2f}")
        print("-" * 35)

        return df
