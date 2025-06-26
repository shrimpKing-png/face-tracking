"""
processing/__init__.py
The 'processing' module provides tools for data transformation and enhancement.

This includes utilities for normalizing video frames, converting between
landmark formats, and applying smoothing algorithms to positional data.
"""

from .frame_processor import normalize_frame
from .landmark_processor import (
    landmarks_to_points,
    points_to_landmarks,
    SmoothedLandmarks,
)
from .smoothing import SmoothingEngine

# Define the public API for the 'processing' module
__all__ = [
    'normalize_frame',
    'landmarks_to_points',
    'points_to_landmarks',
    'SmoothedLandmarks',
    'SmoothingEngine'
]