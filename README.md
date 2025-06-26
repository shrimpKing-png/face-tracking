Advanced Face Tracking Library

Introduction

This Python library is all about tracking faces in videos with high precision and in real-time. It's built to give you smooth, stable landmark tracking by smartly combining dlib's powerful face detection with optical flow and some advanced smoothing techniques. The goal is to get rid of that annoying jitter, even when the video gets a little wild.

Features

    Hybrid Tracking: Instead of just relying on one method, it uses both dlib for accuracy
    and optical flow for speed and consistency between frames. It's the best of both worlds.

    Advanced Smoothing: Utilizes optical flow and/or a weighted moving average + z-score 
    analysis to calm down any jittery landmarks, making them look natural and stable for 
    cleaner data extraction.

    Motion Analysis: Motion vectors are saved for each individual landmark so you can 
    actually dig into the data and see how different parts of the face are moving, or which 
    landmarks are problem spots, which is great for more detailed analysis.

    Modular Design: Everything is organized neatly into modules that are easy to work with. 
    If you want to tinker or add your own stuff, it won't be a headache.

    Mask Generation: Need to isolate the eyes or mouth? There are tools to quickly create 
    masks for specific facial areas using the landmarks.

Installation

You can install the library in one of two ways.

Option 1: Download and Install Manually

    Go to the Releases Page and download the latest .zip or .tar.gz file.

    Unzip the file and install it using pip:
    Bash

    # Navigate into the unzipped directory
    cd face-tracking-vx.x.x/

    # Install the package
    pip install .

Option 2: Install Directly from GitHub

    You can also install the latest release directly using pip and the link to the repository.
    
    Navigate to the releases page and download the latest face_tracking.whl file
    Now open you python terminal and run pip install face_tracking-x.x.x-py3-none-any.whl

Quick Start

Getting started is pretty straightforward. Here's a quick look at how you'd use the tracker on a list of video frames:
Python

    import cv2
    from face_tracking.core import FaceTracker
    
    Assume 'video_frames' is a list of frames (numpy arrays) from a video
    video_frames = load_your_video_frames()
    
    Initialize the tracker
    tracker = FaceTracker(use_optical_flow=True, use_moving_average=True, landmark_detector='dlib') #
    Current detector options are dlib or mediapipe. dlib is configured to num_landmarks of 54 so change that if you use other models#
    
    Process the frames
    tracker.process_frames(video_frames) #
    
    Get the smoothed landmarks for a specific frame
    frame_index = 10
    smoothed_landmarks = tracker.get_smoothed_landmarks(frame_index) #
    
    if smoothed_landmarks:
        # You can now use the landmarks for further processing
        print(f"Found {smoothed_landmarks.num_parts} landmarks in frame {frame_index}.") #

API Reference

The library's functionality is broken down into a few key areas.

Core Components

    FaceTracker (from face_tracking.core.face_tracker): This is the main brain of the operation. You'll create an instance of this class to manage the detection, tracking, and smoothing pipeline.

        __init__(self, use_optical_flow, use_moving_average): Creates the tracker and lets you turn optical flow and smoothing on or off.

        process_frames(self, frames): Feed it a list of your video frames, and it'll work its magic to find and track the landmarks.

        get_smoothed_landmarks(self, frame_index): After processing, use this to get the final, super-smooth landmarks for any frame.

        get_original_landmarks(self, frame_index): If you need them, this gets you the raw landmarks as they were first detected by dlib, before any smoothing.

        get_motion_stats(self): Returns a pandas DataFrame with cool stats about how much each landmark moved throughout the video.

        ft_lndmrk_outline(self, frame_index, frame, masks): A handy utility to draw the final landmarks and any masks you've created directly onto a frame.

Tracking

    DlibDetector (from face_tracking.tracking.dlib_detector): This is our go-to for finding faces and the initial set of landmarks in an image. It's a wrapper around dlib's powerful models.

    OpticalFlowTracker (from face_tracking.tracking.optical_flow): When dlib might miss a beat, optical flow steps in. It tracks points from one frame to the next, which helps keep the landmarks consistent and stable over time.

Processing

    SmoothingEngine (from face_tracking.processing.smoothing): This is a class designated for smoothing tasks, currently only smooths with EMA, but updates are planned.

    normalize_frame(frame, mask) (from face_tracking.processing.frame_processor): Prepares 32bit IR frames for processing. It handles things like converting to grayscale and normalizing pixel values.

    landmarks_to_points(landmarks) (from face_tracking.processing.landmark_processor): A helper that converts dlib's landmark objects into a simple NumPy array, which is easier to do math with.

    points_to_landmarks(points) (from face_tracking.processing.landmark_processor): Does the opposite of the above, turning a NumPy array of points back into a dlib-style object called SmoothedLandmarks (in utils).

Utilities

    MotionAnalyzer (from face_tracking.core.motion_analysis): This tool helps the tracker be smarter about smoothing by analyzing how much each landmark is moving and flagging any unnatural jumps using z-scores.

    MaskGenerator (from face_tracking.utils.mask_operations): If you need to create a cutout of the eyes, nose, or mouth, this utility makes it easy by building a mask from the landmark points.

    TrackingHistory (from face_tracking.utils.data_structs): This is like the tracker's short-term memory. It keeps track of where landmarks have been recently, which is crucial for the smoothing algorithms.

    plot_landmarks_on_frame(frame, landmarks) (from face_tracking.utils.visualizations): A simple function to draw the detected landmarks right onto an image, which is super helpful for checking your results.
    
    SmoothedLandmarks() storage for landmarks to provide support for legacy functions running dlib landmark code.

Confg

    PREDICTOR_PATH = './SF-TL54/dlib_landmark_predictor.dat'
    DETECTOR_PATH = './SF-TL54/dlib_face_detector.svm'
    NUM_LANDMARKS = 54
    HISTORY_WINDOW = 30
    SMOOTHING_WINDOW = 10
    DECAY_FACTOR = 0.8
    LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    Z_SCORE_THRESHOLD = 3.0

)


    
    
