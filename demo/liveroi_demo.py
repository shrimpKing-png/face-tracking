# -*- coding: utf-8 -*-
"""
This file contains the standalone visualization logic and an example of
how to integrate it with the ThreadedTracker in a main application loop.
"""
import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional
import csv
import warnings

# Assuming these utilities are available in your project structure
from face_tracking.utils.mask_generator import MaskGenerator
from face_tracking.utils.visualizations import visualize_landmarks
from face_tracking.core.threading import ThreadedTracker
from face_tracking.processing import frame_processor
from face_tracking.utils.visualizations import colored_mask_viseye

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


# --- Example Usage in your main application (e.g., liveroi_demo.py) ---

def run_live_demo():
    """Example main loop for a live ROI demo."""
    # --- 1. Initialization ---
    try:
        print("Loading ROI masks from 'landmark_rois.csv'...")
        with open('landmark_rois.csv', 'r') as file:
            reader = csv.reader(file)
            masks = [list(map(int, row)) for row in reader if row]
        print(f"Loaded {len(masks)} pre-defined ROI masks")
        MASKS_TO_APPLY = masks
    except FileNotFoundError:
        warnings.warn("'landmark_rois.csv' not found. No masks will be applied.")
        MASKS_TO_APPLY = []

    # Instantiate the threaded tracker
    tracker = ThreadedTracker(use_optical_flow=False, use_moving_average=True, landmark_detector='mediapipe',
                              num_landmarks=468)
    tracker.start()

    # Instantiate the mask generator for the main thread
    mask_generator = MaskGenerator()

    # Video capture setup
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        tracker.stop()
        return

    latest_landmarks = None

    # --- 2. Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.flip(frame, 1)

        # --- A. Pass frame to background thread ---
        tracker.add_frame_to_process(frame.copy())

        # --- B. Get latest results from background thread (non-blocking) ---
        result = tracker.get_latest_result()
        if result is not None:
            _, frame_result_obj = result
            if frame_result_obj:
                # Correctly extract the landmark data from the result object
                latest_landmarks = frame_result_obj

        # --- C. Render visualization on the main thread ---
        vis_img, _, newmasks_list = render_visualization(
            frame,
            latest_landmarks,
            MASKS_TO_APPLY,
            mask_generator
        )

        # --- D. Generate and display the colored mask visualization ---
        if newmasks_list:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            colored_masks_display = colored_mask_viseye(newmasks_list, gray_frame)
            # Combine the two views side-by-side
            combined_view = np.hstack((vis_img, colored_masks_display))
            cv.imshow('Live Tracking and Masks', combined_view)
        else:
            # If no masks, just show the main tracking view
            cv.imshow('Live Tracking and Masks', vis_img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 3. Cleanup ---
    print("Exiting...")
    tracker.stop()
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    run_live_demo()
