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


def demo_face_tracking(landmark_detector, num_landmarks=54, use_of=True, use_ma=False):
    """
    Demo version that just creates visual output with face tracking
    """
    videopath = ft.utils.filebrowser(True)
    if videopath == '':
        print("No video selected. Exiting.")
        return
    start_time = time.time()
    # Setup output directory
    filename = os.path.basename(videopath)
    output_dir = f'{filename}_OF{use_of}_MA{use_ma}'
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, f"{filename}_demo")
    print(f'Output name: {output_name}')

    # Load frames
    frames = ft.general.video_to_list(videopath)
    if not frames:
        print("No frames loaded. Exiting.")
        return

    print(f"Loaded {len(frames)} frames")

    # Initialize face tracker
    print("Initializing face tracker...")
    face_tracker = ft.FaceTracker(use_optical_flow=use_of, use_moving_average=use_ma, landmark_detector=landmark_detector, num_landmarks=num_landmarks)
    # initialize the face_tracker with first frame
    face_tracker.process_frame(frames[0])

    # Get first frame for mask setup
    frame = frames[0]
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

        # Create a simple mask around eye region (landmarks 36-47 typically)
        masks = [[36, 37, 38, 39, 40, 41]]  # Right eye region
        print("Using default eye region mask")

    masknum = len(masks)

    # Pre-allocate arrays for frame storage
    num_frames = len(frames)
    frame_height, frame_width = frame.shape[:2]
    mask_array = np.zeros((num_frames, frame_height, frame_width * 2, 3), dtype=np.uint8)

    frame_count = 0
    print("Processing frames...")
    mask_generator = ft.MaskGenerator()

    # Process each frame
    for frame_idx, frame in enumerate(frames):
        if frame is None:
            break
        face_tracker.process_frame(frame)

        # Use smoothed face tracking
        smoothed_landmarks = face_tracker.get_smoothed_landmarks(frame_idx)
        img, masked_images, newmasks = ft.visualizations.render_visualization(frame, smoothed_landmarks, masks, mask_generator)

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
        viseye = ft.visualizations.colored_mask_viseye(viseye_lst, frame)
        viseye = np.hstack((img, viseye))

        # Display frame
        cv.imshow("Face Tracking Demo", viseye)

        # Store frame
        if frame_count < num_frames:
            mask_array[frame_count] = viseye
            frame_count += 1

        # Check for quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Progress update
        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}/{len(frames)}")

    cv.destroyAllWindows()

    # Save output video
    print("Saving demo video...")
    mask_lst = [mask_array[i] for i in range(frame_count)]
    face_tracker.get_motion_stats().to_csv(f'{output_name}_motion_stats.csv')
    ft.general.list_to_video(mask_lst, f'{output_name}_visual_demo')
    print(f"Demo complete! Output saved as: {output_name}_visual_demo")
    print(f"Time taken: {time.time() - start_time}s, estimated fps: {frame_count/(time.time() - start_time)}")
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    print("Face Tracking Visual Demo")
    print("This demo creates visual output showing face tracking with ROI masks")
    demo_face_tracking(use_of=True, use_ma=True, landmark_detector='dlib', num_landmarks=54)
