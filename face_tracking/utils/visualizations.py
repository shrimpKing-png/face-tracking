# -*- coding: utf-8 -*-
"""
utils/visualizations.py
Created on Wed Jun 26 01:14:18 2025
Last Update: 25JUNE2025
@author: GPAULL
"""
from typing import Tuple, Optional, List
from face_tracking.processing import frame_processor
import cv2 as cv
import numpy as np
from face_tracking.config import settings as cfg
from face_tracking.processing.landmark_processor import landmarks_to_points
from face_tracking.utils import MaskGenerator


def visualize_landmarks(img: np.ndarray, landmarks) -> np.ndarray:
    """Optimized landmark visualization."""
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # Vectorized landmark extraction
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(landmarks.num_parts)]

    # Batch drawing operations
    for i, (x, y) in enumerate(points):
        cv.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv.putText(img, str(i), (x + 5, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    return img

def colored_mask_viseye(viseye_lst, frame, mask_colors=cfg.MASK_COLORS):
    """
    Creates a colored visualization of multiple masks with a color key.

    Args:
        viseye_lst (list): List of grayscale mask images
        frame (numpy.ndarray): Reference frame for dimensions (grayscale)

    Returns:
        numpy.ndarray: BGR image with colored masks and color key
    """
    # Fast path for empty mask list
    if not viseye_lst:
        return cv.cvtColor(frame, cv.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()

    frame_height, frame_width = frame.shape[:2]
    num_masks = min(len(viseye_lst), len(mask_colors))

    # Pre-allocate output image
    viseye_bgr = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Direct pixel assignment
    for i in range(num_masks):
        vmask = viseye_lst[i]
        if vmask is None:
            continue

        # Get color for this mask
        color = mask_colors[i]

        # mask processing - single operation
        if vmask.ndim == 3:
            mask_indices = cv.cvtColor(vmask, cv.COLOR_BGR2GRAY) > 0
        else:
            mask_indices = vmask > 0

        # VECTORIZED COLOR ASSIGNMENT
        if np.any(mask_indices):
            viseye_bgr[mask_indices] = color

    # KEY RENDERING
    _render_color_key_optimized(viseye_bgr, num_masks, frame_height)

    return viseye_bgr


def _render_color_key_optimized(viseye_bgr, num_masks, frame_height, mask_colors=cfg.MASK_COLORS):
    """
    Optimized color key rendering with minimal OpenCV operations.

    OPTIMIZATIONS:
    - Single overlay operation instead of multiple addWeighted calls
    - Pre-calculated dimensions
    - Batch text rendering
    - Direct pixel manipulation for background
    """
    if num_masks == 0:
        return

    # Calculate key dimensions
    key_height = min(30 * num_masks, frame_height // 3)
    key_width = 150

    bg_slice = viseye_bgr[5:key_height + 25, 5:key_width]
    bg_slice.fill(0)  # Black background

    # Pre-define commonly used values
    font = cv.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)

    # Title
    cv.putText(viseye_bgr, "Mask Key:", (10, 25), font, 0.6, white, 1)

    # BATCH RENDERING: Process all swatches in optimized loop
    for i in range(num_masks):
        y_pos = 45 + (i * 25)
        color = tuple(map(int, mask_colors[i]))  # Convert to tuple once

        # Color swatch - 2 rectangles
        cv.rectangle(viseye_bgr, (15, y_pos - 8), (35, y_pos + 8), color, -1)
        cv.rectangle(viseye_bgr, (15, y_pos - 8), (35, y_pos + 8), white, 1)

        # Label text
        cv.putText(viseye_bgr, f"Mask {i}", (45, y_pos + 5), font, 0.5, white, 1)

def render_visualization(
        frame: np.ndarray,
        landmarks,
        masks: List[List[int]],
        mask_generator: MaskGenerator
) -> Tuple[np.ndarray, Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """
    Renders tracking visualizations on a frame using the optimized MaskGenerator logic.

    Args:
        frame: The original video frame to draw on.
        landmarks: The landmark data (dlib or mediapipe object) for the frame.
        masks: A list of lists, where each inner list contains landmark indices.
        mask_generator: An instance of the MaskGenerator class.

    Returns:
        A tuple containing:
        - The frame with landmarks visualized.
        - A list of masked images.
        - A list of the corresponding mask arrays.
    """
    if landmarks is None:
        return frame, None, None

    img_normalized = frame_processor.normalize_frame(frame, np.ones_like(frame)) \
        if frame.dtype != np.uint8 else frame

    # Apply masks using the provided generator, as done in the old function
    masked_images, newmasks_list = mask_generator.apply_masks(img_normalized, landmarks, masks)

    # Create a copy for drawing to avoid modifying the array used in masking
    vis_img = img_normalized.copy()

    # Efficient landmark visualization
    vis_img = visualize_landmarks(vis_img, landmarks)

    return vis_img, masked_images, newmasks_list

# def plot_landmarks_on_frame(frame: np.ndarray, landmarks) -> np.ndarray:
#     """ Causes issues with framevalues for some odd reason. May add back later
#     Draws landmark points and their indices on a frame.
#
#     Args:
#         frame: The image to draw on.
#         landmarks: A landmarks object with a .num_parts and a .part(i) method.
#
#     Returns:
#         The frame with landmarks visualized.
#     """
#     vis_frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
#     for i in range(landmarks.num_parts):
#         point = landmarks.part(i)
#         cv.circle(vis_frame, (point.x, point.y), 3, (0, 255, 0), -1)
#         cv.putText(vis_frame, str(i), (point.x + 5, point.y - 5),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
#     return vis_frame
