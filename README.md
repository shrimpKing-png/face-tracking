# Advanced Face Tracking Library

[![Python](https://img.shields.io/badge/Python-Any-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21+-FF6B6B?logo=google&logoColor=white)](https://mediapipe.dev/)
[![dlib](https://img.shields.io/badge/dlib-19.0+-00D4AA?logo=cplusplus&logoColor=white)](http://dlib.net/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/ShrimpKing-png/face-tracking)](#)
[![GitHub last commit](https://img.shields.io/github/last-commit/ShrimpKing-png/face-tracking)](#)

⭐ Star us on GitHub — it motivates us a lot!

## Table of Contents
- [About](#-about)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
- [Feedback and Contributions](#-feedback-and-contributions)
- [License](#-license)

## 🚀 About

**Advanced Face Tracking Library** is a Python library dedicated to tracking faces in videos with high precision and in real-time. It's built to provide smooth, stable landmark tracking by intelligently combining robust face detection with optical flow and advanced smoothing techniques. The primary goal is to eliminate annoying jitter, ensuring reliable tracking even in challenging video conditions.

The library utilizes well-known computer vision techniques and employs a modular architecture that ensures:

- **Modularity**: Different tracking components can function independently, enhancing maintainability and allowing for easier updates.
- **Flexibility**: Support for multiple detection backends (dlib, MediaPipe) with seamless switching.
- **Performance**: Optimized hybrid approach combining model-based detection with optical flow for speed and accuracy.
- **Reliability**: Advanced smoothing algorithms that maintain tracking stability across challenging video conditions.

## ✨ Features

- **Hybrid Tracking**: Utilizes a powerful combination of model-based detectors (dlib or MediaPipe) for accuracy and optical flow for frame-to-frame consistency. This hybrid approach ensures both speed and stability.
- **Advanced Smoothing**: Implements a weighted moving average and z-score analysis to calm jittery landmarks, resulting in natural and stable motion for cleaner data extraction.
- **Motion Analysis**: Saves motion vectors for each landmark, allowing for in-depth analysis of facial part movements and the identification of problematic landmarks.
- **Modular Design**: Organized into distinct, easy-to-understand modules. This clean architecture makes it straightforward to extend, modify, or integrate custom components.
- **Mask Generation**: Includes tools to quickly generate masks for specific facial regions (e.g., eyes, mouth) using the tracked landmarks.
- **Multiple Detector Support**: Offers flexibility by allowing the user to choose between dlib and MediaPipe as the underlying landmark detector.

## 🛠️ Installation

You can install the library in one of two ways:

### Option 1: Install Directly from GitHub

Install the latest release directly using pip and the link to the repository's .whl file.

1. Navigate to the releases page and download the latest `face_tracking.whl` file.
2. Open your terminal and run the following pip command:

```bash
pip install face_tracking-x.x.x-py3-none-any.whl
```

### Option 2: Download and Install Manually

1. Go to the Releases Page and download the latest `.zip` or `.tar.gz` file.
2. Unzip the file and install it using pip:

```bash
# Navigate into the unzipped directory
cd face-tracking-vX.X.X/

# Install the package
pip install .
```

## 🚀 Quick Start

Getting started is straightforward. Here's a quick example of how to use the tracker on a list of video frames:

```python
import cv2
from face_tracking.core import FaceTracker
import face_tracking as ft

# Assume 'video_frames' is a list of frames (numpy arrays) from a video
# video_frames = load_your_video_frames()

# Initialize the tracker
# Detector options: 'dlib' or 'mediapipe'.
# Note: dlib is configured to 54 landmarks by default.
tracker = FaceTracker(
    use_optical_flow=True,
    use_moving_average=True,
    landmark_detector='dlib'
)

video_frames = ft.utils.video_to_list(ft.utils.filebrowser(select_directory=True))

# Process the frames
tracker.batch_process_frames(video_frames)

# Get the smoothed landmarks for a specific frame
frame_index = 10
smoothed_landmarks = tracker.get_smoothed_landmarks(frame_index)

if smoothed_landmarks:
    # You can now use the landmarks for further processing
    print(f"Found {smoothed_landmarks.num_parts} landmarks in frame {frame_index}.")
```

## 📚 API Reference

The library's functionality is broken down into a few key areas. The following API reference is based on the public modules exposed in the `__init__.py` files for each sub-package.

### Core (face_tracking.core)

This module contains the central components that orchestrate the face tracking process.

- **FaceTracker**: The main class that integrates detection, tracking, and smoothing. It manages the entire pipeline, from initial frame processing to generating the final, stabilized landmarks.
- **MotionAnalyzer**: A utility for analyzing landmark motion vectors. It calculates z-scores to detect and quantify jitter, helping to distinguish between natural facial movements and tracking noise.
- **MaskGenerator**: A helper class included from the utils package for convenience. It provides methods to create binary masks for specific facial regions (e.g., eyes, mouth) using landmark points.

### Tracking (face_tracking.tracking)

This module provides different strategies for detecting and tracking facial features.

- **DlibDetector**: A robust, model-based detector for faces and landmarks that leverages the dlib library. It is highly accurate and serves as the foundation for re-identifying faces when tracking is lost.
- **MediaPipeDetector**: An alternative detector that uses Google's MediaPipe framework. It offers a different set of performance characteristics and can be swapped in depending on the use case.
- **OpticalFlowTracker**: An efficient frame-to-frame tracker using the Lucas-Kanade optical flow method. It is responsible for maintaining smooth landmark paths between full detection runs, which is key to reducing jitter.

### Processing (face_tracking.processing)

This module provides tools for data transformation and enhancement.

- **SmoothingEngine**: Applies smoothing algorithms, such as a weighted moving average, to the landmark data. It takes a history of landmark positions and smooths out abrupt changes to produce more natural motion.
- **SmoothedLandmarks**: A custom data structure designed to hold smoothed landmark points while mimicking the dlib landmark object's interface. This ensures compatibility with functions that expect a dlib-style object.
- **landmarks_to_points(landmarks)**: A utility function that converts a dlib landmarks object into a NumPy array of points. This is necessary for numerical operations, especially for use with OpenCV functions like optical flow.
- **points_to_landmarks(points)**: The inverse of landmarks_to_points. It converts a NumPy array of points back into a SmoothedLandmarks object.
- **normalize_frame(frame, mask)**: A preprocessing function that prepares video frames for tracking. It converts frames to grayscale and normalizes pixel values to a 0-1 range to ensure consistent input for the tracking algorithms.

### Utilities (face_tracking.utils)

This module contains helper classes and functions used throughout the package.

- **TrackingHistory**: A custom data structure for managing the history of landmark positions and motion vectors over a sliding window of frames. This history is essential for both the smoothing engine and motion analysis.
- **MaskGenerator**: A utility to create binary masks for specific facial regions (e.g., eyes, mouth) from a set of landmark points. This is useful for isolating parts of the face for further analysis.
- **filebrowser()**: A helper function that opens a file dialog, allowing the user to select a video file interactively.
- **video_to_list(video_path)**: Reads a video file from the given path and converts it into a list of frames (as NumPy arrays).
- **list_to_video(frame_list, output_path, fps)**: Takes a list of frames and writes them to a video file at the specified path and frame rate.
- **visualizations**: This module (and its functions like plot_landmarks_on_frame) contains functions to help visualize tracking results by drawing landmarks directly onto video frames.

## ⚙️ Configuration

The library's behavior can be customized via the `config/settings.py` file. This allows you to fine-tune the tracking and smoothing parameters without modifying the core library code.

```python
# --- Model Paths ---
# Paths to the dlib face detector and landmark predictor models.
PREDICTOR_PATH = './SF-TL54/dlib_landmark_predictor.dat'
DETECTOR_PATH = './SF-TL54/dlib_face_detector.svm'

# --- Core Settings ---
# The number of landmarks your model is trained to detect.
NUM_LANDMARKS = 54
# The number of frames to store in history for motion analysis.
HISTORY_WINDOW = 30
# Enable or disable optical flow.
USE_OPTICAL_FLOW = True
# Enable or disable the moving average smoother.
USE_MOVING_AVERAGE = True

# --- Smoothing Parameters ---
# The number of frames to include in the weighted moving average.
SMOOTHING_WINDOW = 10
# The decay factor for the weighted moving average. A higher value gives more
# weight to older frames, resulting in smoother but less responsive tracking.
DECAY_FACTOR = 0.8

# --- Optical Flow Parameters ---
# Parameters for the Lucas-Kanade optical flow algorithm.
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)
# The z-score threshold for detecting jitter. A lower value makes the
# system more sensitive to small, rapid movements.
Z_SCORE_THRESHOLD = 3.0
# The weight given to dlib's prediction when motion is low (below z-score threshold).
LOW_MOTION_DLIB_WEIGHT = 0.9
# The weight given to optical flow's prediction when motion is low.
LOW_MOTION_FLOW_WEIGHT = 0.1
# The weight given to dlib's prediction during normal motion.
LANDMARK_WEIGHT = 0.7
# The weight given to optical flow's prediction during normal motion.
FLOW_WEIGHT = 0.3
```

## 📚 Documentation

For comprehensive documentation and advanced usage examples, please visit our documentation site. There you will find:

- **Integration Guides**: Detailed instructions for integrating the library into your projects
- **Advanced Configuration**: In-depth configuration options and performance tuning
- **API Documentation**: Complete API reference with examples
- **Troubleshooting**: Common issues and their solutions

## 🤝 Feedback and Contributions

I've made every effort to implement robust face tracking capabilities with the best possible performance and accuracy. However, the development journey doesn't end here, and your input is crucial for our continuous improvement.

> [!IMPORTANT]
> Whether you have feedback on features, have encountered any bugs, or have suggestions for enhancements, I'm eager to hear from you. Your insights help us make the Advanced Face Tracking Library more robust and user-friendly.

Please feel free to contribute by [submitting an issue](https://github.com/ShrimpKing-png/face-tracking/issues) or [joining the discussions](https://github.com/ShrimpKing-png/face-tracking/discussions).

I appreciate your support and look forward to making our product even better with your help!

## 📃 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.