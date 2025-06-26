"""
tracking/__init__.py
The 'tracking' module provides classes for detecting and tracking facial
features in video frames.

This package encapsulates different tracking strategies:
- DlibDetector: For robust, model-based face and landmark detection.
- OpticalFlowTracker: For efficient frame-to-frame point tracking.
"""

from .dlib_detector import DlibDetector
from .optical_flow import OpticalFlowTracker
from .mediapipe_detector import MediaPipeDetector

# __all__ defines the public API of the package
__all__ = [
    'DlibDetector',
    'OpticalFlowTracker',
    'MediaPipeDetector'
]
