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
import time
from face_tracking.utils.mask_generator import MaskGenerator
from face_tracking.core.threading import ThreadedTracker, ThreadedFrameProcessor
from face_tracking.utils.visualizations import render_visualization


# --- Example Usage in your main application (e.g., liveroi_demo.py) ---

def run_live_demo():
    frame_index = 0
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

    tracker = ThreadedTracker(use_optical_flow=False, use_moving_average=True, landmark_detector='mediapipe',
                              num_landmarks=468)
    tracker.start()
    frame_processor = ThreadedFrameProcessor()
    frame_processor.start()

    mask_generator = MaskGenerator()
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        tracker.stop()
        return

    # Store the latest valid visualization image
    latest_vis_img = None
    latest_colored_masks = None

    # --- 2. Main Loop ---
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.flip(frame, 1)

        # --- A. Pass the latest frame to the background thread ---
        tracker.add_frame_to_process(frame.copy())

        # --- B. Get the latest processed result ---
        result = tracker.get_latest_result()
        if result is not None:
            # **FIX:** Unpack the frame and its corresponding result object.
            # This ensures the landmarks are always matched with the correct frame.
            processed_frame, frame_result_obj = result

            if frame_result_obj:
                # --- C. Render visualization on the correctly matched frame ---
                vis_img, _, newmasks_list = render_visualization(
                    processed_frame,  # **FIX:** Use the frame from the tracker's result
                    frame_result_obj,  # Use the landmarks from the result
                    MASKS_TO_APPLY,
                    mask_generator
                )
                # Update the latest visualization image
                latest_vis_img = vis_img

                # --- D. Send the correctly matched frame and masks to the processor ---
                if newmasks_list:
                    # **FIX:** Pass the 'processed_frame' to ensure the mask visualization
                    # is generated on the same frame as the landmark visualization.
                    frame_processor.add_frame_to_process(processed_frame.copy(), newmasks_list)

        # --- E. Get latest colored mask result ---
        colored_result = frame_processor.get_latest_result()
        if colored_result is not None:
            frame_index += 1
            latest_colored_masks = colored_result
            colored_result = None

        # --- F. Display the most recently generated visualizations ---
        display_frame = frame  # Default to showing the live camera feed
        if latest_vis_img is not None:
            display_frame = latest_vis_img  # Show the latest tracking visualization if available
            if latest_colored_masks is not None:
                # Combine the two views side-by-side
                display_frame = np.hstack((latest_vis_img, latest_colored_masks))
                latest_vis_img = None

        cv.imshow('Live Tracking and Masks', display_frame)
        if frame_index % 60 == 0:
            print(f'FPS: {frame_index / (time.time() - start_time)}')

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 3. Cleanup ---
    print("Exiting...")
    tracker.stop()
    frame_processor.stop()  # Ensure the frame processor is also stopped
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    run_live_demo()
