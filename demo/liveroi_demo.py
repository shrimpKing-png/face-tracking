# -*- coding: utf-8 -*-
"""
Face Tracking Demo - Visual Output Only
Streamlined version for quick demonstration
"""
import os
import cv2 as cv
import numpy as np
import face_tracking as ft
import time


def demo_face_tracking(use_of=True, use_ma=True, landmark_detector='mediapipe'):
    """
    Demo version that just creates visual output with face tracking
    """
    start_time = time.time()
    # Setup output directory
    filename = input("Enter a file name: ")
    output_name = f"{filename}_demo"
    #setup videocap
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    print("Press 'q' to quit")

    # Initialize face tracker
    print("Initializing face tracker...")
    face_tracker = ft.FaceTracker(use_optical_flow=use_of, use_moving_average=use_ma, num_landmarks=468, landmark_detector=landmark_detector)
    # initialize the face_tracker with first frame
    ret, frame = cap.read()
    if landmark_detector == 'dlib':
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_tracker.process_frame(frame)

    # Get first frame for mask setup
    frame = np.array(frame)
    if len(frame.shape) != 2:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Use pre-defined ROI masks (you can modify this section)
    print("Loading ROI masks...")
    try:
        import csv
        with open('landmark_rois.csv', 'r') as file:
            reader = csv.reader(file)
            masks = list(reader)
        masks = [[int(landmark) for landmark in mask] for mask in masks]
        print(f"Loaded {len(masks)} pre-defined ROI masks")
    except FileNotFoundError:
        print("No pre-defined ROI file found. Creating default masks...")
        # Create some default facial ROI masks if file doesn't exist
        first_landmarks = face_tracker.get_original_landmarks(0)
        if first_landmarks is None:
            print("No face detected in first frame. Exiting.")
            return

        masks = [[67, 69, 108, 151, 337, 299, 297, 338, 10, 109, 67],
                 [70, 156, 143, 111, 35, 124, 70],
                 [276, 300, 383, 372, 340, 265, 353, 276]]

    masknum = len(masks)

    # Pre-allocate arrays for frame storage
    frame_height, frame_width = frame.shape[:2]
    mask_array = np.zeros((10000, frame_height, frame_width * 2, 3), dtype=np.uint8)

    frame_idx = 0
    print("Processing frames...")

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        elif frame.ndim == 3 and landmark_detector == 'dlib':
            frame_disp = frame.copy()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_tracker.process_frame(frame)

        # Use smoothed face tracking
        img, masked_images, newmasks = face_tracker.ft_lndmrk_outline(frame_idx, frame, masks)

        if masked_images is None:
            # No face detected, skip this frame
            continue

        # Create visual masks for each ROI
        viseye_lst = []
        for i in range(masknum):
            if i < len(newmasks):
                newmask = newmasks[i]
            else:
                newmask = np.zeros_like(frame)

            newmask = newmask.astype(bool)

            # Process image for visualization
            maskedimg = frame * newmask
            viseye_lst.append(maskedimg)

        # Create colored visualization
        if frame.dtype == np.float32:
            print('normalizin frame')
            frame_cv = ft.normalize_frame(frame, np.zeros_like(frame))
        else:
            frame_cv = frame
        if 'frame_disp' in locals():
            viseye = ft.visualizations.colored_mask_viseye(viseye_lst, frame_disp)
        else:
            viseye = ft.visualizations.colored_mask_viseye(viseye_lst, img)
        viseye = np.hstack((img, viseye))

        # Display frame
        cv.imshow("Face Tracking Demo", viseye)
        if frame_idx % 60 == 0:
            fps = frame_idx / int(time.time() - start_time)
            print(f'Elapsed time: {int(time.time() - start_time)}, FPS: {fps:.2f}fps')

        # Store frame
        if frame_idx < 10000:
            mask_array[frame_idx] = viseye
            frame_idx += 1

        # Check for quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

    # Save output video
    print("Saving demo video...")
    frame_count = frame_idx
    mask_lst = [mask_array[i] for i in range(frame_count)]
    ft.general.list_to_video(mask_lst, f'{output_name}_visual_demo')
    print(f"Demo complete! Output saved as: {output_name}_visual_demo")
    print(f"Time taken: {time.time() - start_time}s, estimated fps: {frame_count/time.time() - start_time}")
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    print("Face Tracking Visual Demo")
    print("This demo creates visual output showing face tracking with ROI masks")
    demo_face_tracking(use_of=True, use_ma=False, landmark_detector='dlib')