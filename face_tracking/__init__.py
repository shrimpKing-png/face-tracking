# -*- coding: utf-8 -*-
"""
face_tracking package initialization
Last Update: 25JUNE2025
Author: GPAULL
"""

from face_tracking.core import MotionAnalyzer, FaceTracker
from face_tracking.tracking import DlibDetector, MediaPipeDetector, OpticalFlowTracker
from face_tracking.utils import general, visualizations
from face_tracking.utils import MaskGenerator
from face_tracking.processing import (normalize_frame,
                                      landmarks_to_points,
                                      points_to_landmarks,
                                      SmoothingEngine,
                                      SmoothedLandmarks)

__all__ = [
    "MaskGenerator",
    "MotionAnalyzer",
    'normalize_frame',
    'landmarks_to_points',
    'points_to_landmarks',
    'SmoothedLandmarks',
    'SmoothingEngine',
    'DlibDetector',
    'OpticalFlowTracker',
    'general',
    'visualizations',
    'utils',
    'config'
]
