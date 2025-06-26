"""
The 'core' module contains the central components that orchestrate the face
tracking process.

This includes the main FaceTracker class, which integrates various sub-systems,
along with high-level utilities for motion analysis and mask generation.
"""

# It's assumed the main FaceTracker class will reside in face_tracker.py
# from .face_tracker import FaceTracker
from utils.mask_operations import MaskGenerator
from .face_tracker import FaceTracker
from .motion_analysis import MotionAnalyzer

# Define the public API for the 'core' module
__all__ = [
    # 'FaceTracker',  # Uncomment when FaceTracker is implemented in this module
    'MaskGenerator',
    'MotionAnalyzer',
    'FaceTracker'
]
