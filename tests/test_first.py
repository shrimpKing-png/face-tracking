# -*- coding: utf-8 -*-
"""
Unit tests for face tracking package components.
Created: June 26, 2025
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Mock the imports that might not be available in test environment
sys.modules['face_tracking.utils'] = Mock()
sys.modules['face_tracking.config'] = Mock()

# Mock settings
mock_settings = Mock()
mock_settings.DECAY_FACTOR = 0.9
mock_settings.SMOOTHING_WINDOW = 10
sys.modules['face_tracking.config'].settings = mock_settings

# Now import the modules under test
import face_tracking
from face_tracking import MotionAnalyzer
from face_tracking.processing.landmark_processor import landmarks_to_points, points_to_landmarks, SmoothedLandmarks
from face_tracking.processing.frame_processor import normalize_frame
from face_tracking.processing.smoothing import SmoothingEngine


class MockTrackingHistory:
    """Mock TrackingHistory class for testing."""

    def __init__(self, motion_vectors=None, num_landmarks=68, position_history=None, history_filled=0):
        self.motion_vectors = motion_vectors if motion_vectors is not None else np.array([])
        self.num_landmarks = num_landmarks
        self.position_history = position_history if position_history is not None else np.array([])
        self.history_filled = history_filled

    def add_position_to_history(self, positions):
        """Mock method to add positions to history."""
        pass


class MockDlibLandmarks:
    """Mock dlib landmarks object for testing."""

    def __init__(self, points):
        self.points = points
        self.num_parts = len(points)

    def part(self, index):
        """Mock part method."""

        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        return Point(self.points[index][0], self.points[index][1])


class TestMotionAnalyzer(unittest.TestCase):
    """Test cases for MotionAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.current_magnitudes = np.array([1.0, 2.0, 3.0, 4.0])
        self.motion_history = np.array([
            [0.5, 1.5, 2.5, 3.5],
            [1.0, 2.0, 3.0, 4.0],
            [1.5, 2.5, 3.5, 4.5],
            [2.0, 3.0, 4.0, 5.0]
        ])

    def test_calculate_z_scores_normal_case(self):
        """Test z-score calculation with normal data."""
        z_scores = MotionAnalyzer.calculate_z_scores(
            self.current_magnitudes,
            self.motion_history,
            history_window=10
        )

        self.assertEqual(len(z_scores), len(self.current_magnitudes))
        self.assertTrue(np.all(z_scores >= 0))  # Z-scores should be non-negative (absolute values)

    def test_calculate_z_scores_insufficient_history(self):
        """Test z-score calculation with insufficient history."""
        empty_history = np.array([])
        z_scores = MotionAnalyzer.calculate_z_scores(
            self.current_magnitudes,
            empty_history,
            history_window=10
        )

        expected = np.ones(len(self.current_magnitudes)) * 10.0
        np.testing.assert_array_equal(z_scores, expected)

    def test_calculate_z_scores_single_frame_history(self):
        """Test z-score calculation with single frame history."""
        single_frame_history = np.array([[1.0, 2.0, 3.0, 4.0]])
        z_scores = MotionAnalyzer.calculate_z_scores(
            self.current_magnitudes,
            single_frame_history,
            history_window=10
        )

        expected = np.ones(len(self.current_magnitudes)) * 10.0
        np.testing.assert_array_equal(z_scores, expected)

    def test_calculate_z_scores_zero_std(self):
        """Test z-score calculation when standard deviation is zero."""
        constant_history = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0]
        ])
        z_scores = MotionAnalyzer.calculate_z_scores(
            self.current_magnitudes,
            constant_history,
            history_window=10
        )

        # Should handle division by zero case
        self.assertEqual(len(z_scores), len(self.current_magnitudes))
        self.assertTrue(np.all(np.isfinite(z_scores)))

    def test_build_motion_stat_df_normal_case(self):
        """Test building motion statistics DataFrame with normal data."""
        mock_history = MockTrackingHistory(
            motion_vectors=self.motion_history,
            num_landmarks=4
        )

        with patch('builtins.print'):  # Suppress print output during testing
            df = MotionAnalyzer.build_motion_stat_df(mock_history)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], self.motion_history.shape[0])
        self.assertEqual(df.shape[1], self.motion_history.shape[1])

        expected_columns = [f'landmark_{i}_motion_vector' for i in range(4)]
        self.assertListEqual(list(df.columns), expected_columns)

    def test_build_motion_stat_df_empty_history(self):
        """Test building motion statistics DataFrame with empty history."""
        mock_history = MockTrackingHistory(
            motion_vectors=np.array([]).reshape(0, 4),
            num_landmarks=4
        )

        with patch('builtins.print'):  # Suppress print output during testing
            df = MotionAnalyzer.build_motion_stat_df(mock_history)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
        expected_columns = [f'landmark_{i}_motion_vector' for i in range(4)]
        self.assertListEqual(list(df.columns), expected_columns)


class TestLandmarkProcessor(unittest.TestCase):
    """Test cases for landmark processor functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_points = [(10, 20), (30, 40), (50, 60)]
        self.mock_landmarks = MockDlibLandmarks(self.test_points)

    def test_landmarks_to_points_normal_case(self):
        """Test converting landmarks to points array."""
        points = landmarks_to_points(self.mock_landmarks)

        self.assertEqual(points.shape, (3, 1, 2))
        self.assertEqual(points.dtype, np.float32)

        # Check values
        np.testing.assert_array_equal(points[0, 0], [10.0, 20.0])
        np.testing.assert_array_equal(points[1, 0], [30.0, 40.0])
        np.testing.assert_array_equal(points[2, 0], [50.0, 60.0])

    def test_landmarks_to_points_none_input(self):
        """Test landmarks_to_points with None input."""
        with self.assertRaises(ValueError) as context:
            landmarks_to_points(None)

        self.assertIn('No landmarks provided', str(context.exception))

    def test_points_to_landmarks_normal_case(self):
        """Test converting points array back to landmarks."""
        points_array = np.array([[[10, 20]], [[30, 40]], [[50, 60]]], dtype=np.float32)
        landmarks = points_to_landmarks(points_array)

        self.assertIsInstance(landmarks, SmoothedLandmarks)
        self.assertEqual(landmarks.num_parts, 3)

        # Test accessing points
        point0 = landmarks.part(0)
        self.assertEqual(point0.x, 10)
        self.assertEqual(point0.y, 20)

    def test_points_to_landmarks_none_input(self):
        """Test points_to_landmarks with None input."""
        result = points_to_landmarks(None)
        self.assertIsNone(result)

    def test_smoothed_landmarks_initialization(self):
        """Test SmoothedLandmarks initialization."""
        points_array = np.array([[10, 20], [30, 40], [50, 60]])
        landmarks = SmoothedLandmarks(points_array)

        self.assertEqual(landmarks.num_parts, 3)
        np.testing.assert_array_equal(landmarks.points, points_array)

    def test_smoothed_landmarks_reshaped_input(self):
        """Test SmoothedLandmarks with reshaped input."""
        points_array = np.array([[[10, 20]], [[30, 40]], [[50, 60]]])
        landmarks = SmoothedLandmarks(points_array)

        expected = np.array([[10, 20], [30, 40], [50, 60]])
        np.testing.assert_array_equal(landmarks.points, expected)

    def test_smoothed_landmarks_part_access(self):
        """Test accessing individual parts of SmoothedLandmarks."""
        points_array = np.array([[10.5, 20.7], [30.2, 40.8]])
        landmarks = SmoothedLandmarks(points_array)

        point0 = landmarks.part(0)
        self.assertEqual(point0.x, 10)  # Should be converted to int
        self.assertEqual(point0.y, 20)

        point1 = landmarks.part(1)
        self.assertEqual(point1.x, 30)
        self.assertEqual(point1.y, 40)


class TestFrameProcessor(unittest.TestCase):
    """Test cases for frame processor functions."""

    def test_normalize_frame_normal_case(self):
        """Test frame normalization with normal input."""
        # Create test image and mask
        arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])

        result = normalize_frame(arr, mask)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, arr.shape)

        # Check that values are in 0-255 range
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))

    def test_normalize_frame_all_same_values(self):
        """Test frame normalization when all values are the same."""
        arr = np.array([[50, 50, 50], [50, 50, 50]])
        mask = np.array([[1, 1, 1], [1, 1, 1]])

        result = normalize_frame(arr, mask)

        self.assertEqual(result.dtype, np.uint8)
        # When all values are the same, should result in zeros
        np.testing.assert_array_equal(result, np.zeros_like(arr, dtype=np.uint8))

    def test_normalize_frame_with_zeros(self):
        """Test frame normalization with zero values."""
        arr = np.array([[0, 10, 20], [30, 40, 50]])
        mask = np.array([[0, 1, 1], [1, 1, 1]])

        result = normalize_frame(arr, mask)

        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))


class TestSmoothingEngine(unittest.TestCase):
    """Test cases for SmoothingEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = face_tracking.processing.smoothing.SmoothingEngine(decay_factor=0.8, smoothing_window=5)
        self.current_positions = np.array([[10, 20], [30, 40], [50, 60]])

        # Create position history
        self.position_history = np.array([
            [[9, 19], [29, 39], [49, 59]],  # Most recent
            [[8, 18], [28, 38], [48, 58]],  # Second most recent
            [[7, 17], [27, 37], [47, 57]]  # Third most recent
        ])

        self.mock_history = MockTrackingHistory(
            position_history=self.position_history,
            history_filled=3
        )

    def test_initialization(self):
        """Test SmoothingEngine initialization."""
        engine = face_tracking.processing.smoothing.SmoothingEngine(decay_factor=0.5, smoothing_window=10)
        self.assertEqual(engine.decay_factor, 0.5)
        self.assertEqual(engine.smoothing_window, 10)

    def test_initialization_with_defaults(self):
        """Test SmoothingEngine initialization with default values."""
        engine = face_tracking.processing.smoothing.SmoothingEngine()
        self.assertEqual(engine.decay_factor, 0.9)  # From mock settings
        self.assertEqual(engine.smoothing_window, 10)

    def test_calculate_weighted_moving_average_normal_case(self):
        """Test weighted moving average calculation."""
        result = self.engine.calculate_weighted_moving_average(
            self.current_positions,
            self.mock_history
        )

        self.assertEqual(result.shape, self.current_positions.shape)
        self.assertEqual(result.dtype, np.float64)

        # Result should be different from current positions (smoothed)
        self.assertFalse(np.array_equal(result, self.current_positions))

    def test_calculate_weighted_moving_average_no_history(self):
        """Test weighted moving average with no history."""
        empty_history = MockTrackingHistory(history_filled=0)

        result = self.engine.calculate_weighted_moving_average(
            self.current_positions,
            empty_history
        )

        # Should return current positions unchanged
        np.testing.assert_array_equal(result, self.current_positions)

    def test_calculate_weighted_moving_average_custom_decay(self):
        """Test weighted moving average with custom decay factor."""
        result = self.engine.calculate_weighted_moving_average(
            self.current_positions,
            self.mock_history,
            decay_factor=0.5
        )

        self.assertEqual(result.shape, self.current_positions.shape)

    def test_apply_smoothing_weighted_moving_average(self):
        """Test apply_smoothing with weighted moving average method."""
        result = self.engine.apply_smoothing(
            self.current_positions,
            self.mock_history,
            method='weighted_moving_average'
        )

        self.assertEqual(result.shape, self.current_positions.shape)

    def test_apply_smoothing_unknown_method(self):
        """Test apply_smoothing with unknown method."""
        with self.assertRaises(ValueError) as context:
            self.engine.apply_smoothing(
                self.current_positions,
                self.mock_history,
                method='unknown_method'
            )

        self.assertIn('Unknown smoothing method: unknown_method', str(context.exception))

    def test_smooth_and_update_history(self):
        """Test smooth_and_update_history method."""
        # Mock the add_position_to_history method
        self.mock_history.add_position_to_history = Mock()

        result = self.engine.smooth_and_update_history(
            self.current_positions,
            self.mock_history
        )

        self.assertEqual(result.shape, self.current_positions.shape)

        # Verify that add_position_to_history was called
        self.mock_history.add_position_to_history.assert_called_once()

        # Check that it was called with the smoothed positions
        called_args = self.mock_history.add_position_to_history.call_args[0][0]
        np.testing.assert_array_equal(called_args, result)


class TestIntegration(unittest.TestCase):
    """Integration tests for component interaction."""

    def test_landmark_processing_pipeline(self):
        """Test the complete landmark processing pipeline."""
        # Create mock landmarks
        original_points = [(10, 20), (30, 40), (50, 60)]
        mock_landmarks = MockDlibLandmarks(original_points)

        # Convert to points
        points = landmarks_to_points(mock_landmarks)

        # Convert back to landmarks
        new_landmarks = points_to_landmarks(points)

        # Verify the round-trip conversion
        self.assertEqual(new_landmarks.num_parts, len(original_points))

        for i, (orig_x, orig_y) in enumerate(original_points):
            point = new_landmarks.part(i)
            self.assertEqual(point.x, orig_x)
            self.assertEqual(point.y, orig_y)

    def test_motion_analysis_with_smoothing_engine(self):
        """Test motion analysis with smoothing engine results."""
        # Create motion data
        motion_vectors = np.array([
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [2.0, 3.0, 4.0]
        ])

        mock_history = MockTrackingHistory(
            motion_vectors=motion_vectors,
            num_landmarks=3
        )

        # Test z-score calculation
        current_magnitudes = np.array([1.8, 2.8, 3.8])
        z_scores = MotionAnalyzer.calculate_z_scores(
            current_magnitudes,
            motion_vectors,
            history_window=10
        )

        self.assertEqual(len(z_scores), 3)
        self.assertTrue(np.all(z_scores >= 0))

        # Test DataFrame building
        with patch('builtins.print'):
            df = MotionAnalyzer.build_motion_stat_df(mock_history)

        self.assertEqual(df.shape, motion_vectors.shape)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
