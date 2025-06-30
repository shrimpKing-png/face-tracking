# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:08:12 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import cv2 as cv
import numpy as np

# --- File Paths ---
# Paths to the dlib models required for face and landmark detection.
# It's good practice to manage these paths in a central config so they
# can be easily updated if the model files are moved.
PREDICTOR_NAME = 'dlib_landmark_predictor.dat'
DETECTOR_NAME = 'dlib_face_detector.svm'

# --- Model Parameters ---
# The number of facial landmarks the model is trained to detect.
# Your current model uses 54 points.
NUM_LANDMARKS = 54

# --- Smoothing and History Parameters ---
# Number of past frames to consider for z-score outlier detection. This helps
# determine if a landmark's movement is unusual compared to its recent behavior.
HISTORY_WINDOW = 30

# Number of past frames to include in the weighted moving average calculation.
# This directly controls the "smoothness" of the landmark positions.
SMOOTHING_WINDOW = 10
DECAY_FACTOR = 0.9

# --- Feature Toggles ---
# These booleans act as switches to enable or disable major features of the tracker.
# This is great for debugging or for users who may only want a subset of the functionality.
USE_OPTICAL_FLOW = True
USE_MOVING_AVERAGE = True
USE_NEIGHBORS_TRANSFORM = False
NUM_NEIGHBORS = 10

# --- Optical Flow Parameters ---
# Configuration for the Lucas-Kanade optical flow algorithm.
# These values control window size, pyramid level, and termination criteria.
# It's best to keep these together as they are all related to the same algorithm.
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

# --- Motion Analysis and Weighting ---
# The z-score threshold for detecting unusually small motion (jitter).
# A lower value makes the detection more sensitive.
Z_SCORE_THRESHOLD = 2.0

# Weights for blending dlib's raw detection with the optical flow prediction.
# Giving a higher weight to optical flow can result in smoother tracking, but
# might drift over time if dlib detection is lost.
LANDMARK_WEIGHT = 1
FLOW_WEIGHT = 0

# Special weights applied when motion is very small (below the z-score threshold).
# This is a smart trick to reduce jitter by relying more heavily on the smoother
# optical flow prediction when the face is relatively still.
LOW_MOTION_LANDMARK_WEIGHT = 1
LOW_MOTION_FLOW_WEIGHT = 0

# Colors for mask visualization.
MASK_COLORS = np.array([
    [0, 0, 255],  # Red
    [0, 255, 0],  # Green
    [255, 0, 0],  # Blue
    [0, 255, 255],  # Yellow
    [255, 0, 255],  # Magenta
    [255, 255, 0],  # Cyan
    [128, 0, 128],  # Purple
    [255, 165, 0],  # Orange
    [0, 128, 128],  # Teal
    [128, 128, 0],  # Olive
], dtype=np.uint8)
