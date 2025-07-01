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
import itertools
from multiprocessing import Pool

def process_video_permutation(args):
    """
    Worker function for multiprocessing.
    Processes one video with one permutation of settings.
    """
    videopath, use_of, use_ma, landmark_detector, num_landmarks = args
    print(f"Processing {videopath} with OF={use_of}, MA={use_ma}")

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
        print(f"No frames loaded for {videopath}. Skipping.")
        return

    print(f"Loaded {len(frames)} frames for {videopath}")

    # Initialize face tracker
    print(f"Initializing face tracker for {videopath}...")
    face_tracker = ft.FaceTracker(use_optical_flow=use_of, use_moving_average=use_ma, landmark_detector=landmark_detector, num_landmarks=num_landmarks)
    # initialize the face_tracker with first frame
    face_tracker.process_frame(frames[0])

    # Get first frame for mask setup
    frame = frames[0]
    if len(frame.shape) != 2:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Use pre-defined ROI masks
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
            print(f"No face detected in first frame of {videopath}. Skipping.")
            return

        # Create a simple mask around eye region
        masks = [[36, 37, 38, 39, 40, 41]]  # Right eye region
        print("Using default eye region mask")

    masknum = len(masks)

    # Pre-allocate arrays for frame storage
    num_frames = len(frames)
    frame_height, frame_width = frame.shape[:2]
    mask_array = np.zeros((num_frames, frame_height, frame_width * 2, 3), dtype=np.uint8)

    frame_count = 0
    print(f"Processing frames for {videopath}...")
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

        # Store frame
        if frame_count < num_frames:
            mask_array[frame_count] = viseye
            frame_count += 1

    # Save output video
    print(f"Saving demo video for {videopath}...")
    mask_lst = [mask_array[i] for i in range(frame_count)]
    face_tracker.get_motion_stats().to_csv(f'{output_name}_motion_stats.csv')
    ft.general.list_to_video(mask_lst, f'{output_name}_visual_demo')
    print(f"Demo complete for {videopath}! Output saved as: {output_name}_visual_demo")
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"Time taken for {videopath} (OF={use_of}, MA={use_ma}): {elapsed_time:.2f}s, estimated fps: {fps:.2f}")
    print(f"Processed {frame_count} frames for {videopath}")


def main():
    """
    Main function to run the face tracking demo with multiprocessing.
    """
    start_time = time.time()
    print("Face Tracking Visual Demo")
    print("This demo creates visual output showing face tracking with ROI masks.")
    print("Select one or more video files to process. Click 'Cancel' to stop.")

    videopaths = []
    while True:
        videopath = ft.utils.filebrowser(True)
        if videopath == '':
            break
        videopaths.append(videopath)

    if not videopaths:
        print("No videos selected. Exiting.")
        return

    landmark_detector = 'dlib'
    num_landmarks = 54

    # All possible permutations of use_of and use_ma
    permutations = list(itertools.product([True, False], repeat=2))

    # Create a list of arguments for the worker function
    tasks = []
    num_permutations = len(permutations) * len(videopaths)
    for videopath in videopaths:
        for use_of, use_ma in permutations:
            tasks.append((videopath, use_of, use_ma, landmark_detector, num_landmarks))

    # Use multiprocessing to run tasks in parallel
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_video_permutation, tasks)

    print("\nAll processing complete.")
    print(f"Total processing time: {time.time() - start_time:.2f}s. Num_permutations: {num_permutations}")


if __name__ == "__main__":
    main()