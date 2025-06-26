"""
The 'utils' module contains helper classes and functions used throughout the
face_tracking library.

This includes custom data structures for managing tracking history and
utility functions for visualizing tracking results.
"""

from .data_structs import TrackingHistory

# Define the public API for the 'utils' module
__all__ = [
    'TrackingHistory',
]