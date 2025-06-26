# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 01:14:18 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import cv2 as cv
import numpy as np


def colored_mask_viseye(viseye_lst, frame):
    """
    Creates a colored visualization of multiple masks with a color key.

    Args:
        viseye_lst (list): List of grayscale mask images
        frame (numpy.ndarray): Reference frame for dimensions (grayscale)

    Returns:
        numpy.ndarray: BGR image with colored masks and color key
    """
    # Define colors for each mask (BGR format)
    mask_colors = [
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
    ]

    # Convert grayscale frame to BGR for color visualization
    if len(viseye_lst) > 0:
        # Start with a black BGR image
        viseye_bgr = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        # Add each mask with its unique color
        for i, vmask in enumerate(viseye_lst):
            if i < len(mask_colors):
                color = mask_colors[i]
                # Create a colored version of the mask
                colored_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

                # Apply the mask and color it
                mask_indices = vmask > 0
                colored_mask[mask_indices] = color

                # Add to the combined visualization
                viseye_bgr = cv.addWeighted(viseye_bgr, 1.0, colored_mask, 0.7, 0)

        # Add color key in top-left corner
        key_height = min(30 * min(len(viseye_lst), len(mask_colors)), frame.shape[0] // 3)
        key_width = 150

        # Create semi-transparent overlay for the key background
        overlay = viseye_bgr.copy()
        cv.rectangle(overlay, (5, 5), (key_width, key_height + 20), (0, 0, 0), -1)
        viseye_bgr = cv.addWeighted(viseye_bgr, 0.7, overlay, 0.3, 0)

        # Add title
        cv.putText(viseye_bgr, "Mask Key:", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Add color swatches and numbers for each mask
        for i in range(min(len(viseye_lst), len(mask_colors))):
            y_pos = 45 + (i * 25)
            color = mask_colors[i]

            # Draw color swatch
            cv.rectangle(viseye_bgr, (15, y_pos - 8), (35, y_pos + 8), color, -1)
            cv.rectangle(viseye_bgr, (15, y_pos - 8), (35, y_pos + 8), (255, 255, 255), 1)

            # Add mask number in white
            cv.putText(viseye_bgr, f"Mask {i}", (45, y_pos + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return viseye_bgr
    else:
        # If no masks, convert grayscale to BGR
        return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

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
