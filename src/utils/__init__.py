"""
The 'utils' module contains helper classes and functions used throughout the
face_tracking library.

This includes custom data structures for managing tracking history and
utility functions for visualizing tracking results.
"""

from .data_structs import TrackingHistory
from general import filebrowser, list_to_video, video_to_list
from .mask_operations import MaskGenerator
import visualizations

# Define the public API for the 'utils' module
__all__ = [
    'TrackingHistory',
    'general',
    'MaskGenerator',
    'visualizations',
    'video_to_list',
    'list_to_video',
    'filebrowser'
]
