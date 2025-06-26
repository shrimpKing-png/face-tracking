# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:10:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import numpy as np
import cv2 as cv
from typing import List

# Using relative imports for library structure
from config import settings as cfg
from tracking.dlib_detector import DlibDetector
from tracking.optical_flow import OpticalFlowTracker
from processing import landmark_processor, frame_processor
from processing.smoothing import SmoothingEngine
from core.motion_analysis import MotionAnalyzer
from utils.data_structs import TrackingHistory
from utils.mask_operations import MaskGenerator


class FaceTracker:
    """
    Orchestrates the face tracking process by coordinating dlib detection,
    optical flow, and smoothing algorithms.

    This class acts as the main entry point for the face tracking library. It
    initializes all necessary components and manages the state of the tracking
    process through a TrackingHistory object. Its main responsibility is to
    manage the flow of data between the detector, tracker, and smoother.
    """

    def __init__(self, use_optical_flow: bool = cfg.USE_OPTICAL_FLOW,
                 use_moving_average: bool = cfg.USE_MOVING_AVERAGE):
        """
        Initializes the FaceTracker and its components.

        Args:
            use_optical_flow (bool): Flag to enable/disable optical flow.
                                     Defaults to the value in config.settings.
            use_moving_average (bool): Flag to enable/disable weighted moving average.
                                       Defaults to the value in config.settings.
        """
        # --- Configuration ---
        self.use_optical_flow = use_optical_flow
        self.use_moving_average = use_moving_average
        self.num_landmarks = cfg.NUM_LANDMARKS

        # --- Component Initialization ---
        # Each component has a single, well-defined responsibility.
        self.detector = DlibDetector()
        self.optical_flow = OpticalFlowTracker(lk_params=cfg.LK_PARAMS)
        self.smoother = SmoothingEngine()
        self.motion_analyzer = MotionAnalyzer()

        # --- State Management ---
        # The history object encapsulates all frame-by-frame tracking data.
        self.history = TrackingHistory()
        # Stores the raw dlib detection result for each frame.
        self.raw_landmarks_per_frame = []
        # Stores the final, smoothed landmark result for each frame.
        self.smoothed_landmarks_per_frame = []

        self._print_initialization_status()

    def get_motion_stats(self):
        """
        :return: the motion analyzer for stats building
        """
        return self.motion_analyzer.build_motion_stat_df(self.history)

    def _print_initialization_status(self):
        """Prints the current configuration of the tracker for user feedback."""
        print("Initialized FaceTracker with:")
        print(f"  - Optical Flow: {'Enabled' if self.use_optical_flow else 'Disabled'}")
        print(f"  - Moving Average: {'Enabled' if self.use_moving_average else 'Disabled'}")
        if self.use_moving_average:
            print(f"  - Smoothing Window: {cfg.SMOOTHING_WINDOW} frames")
        if not self.use_optical_flow and not self.use_moving_average:
            print("Warning: Both smoothing methods are disabled. Using raw dlib detection only.")

    def process_frames(self, frames: List[np.ndarray]):
        """
        Main processing function to detect and track facial landmarks across a list of frames.
        This function orchestrates the entire pipeline from detection to smoothing.

        Args:
            frames (List[np.ndarray]): A list of video frames (in BGR or Grayscale format).
        """
        if not frames:
            print("Warning: No frames provided to process.")
            return

        print("Starting face tracking process...")

        # Step 1: Pre-process frames (e.g., convert to grayscale and normalize).
        normalized_frames = [self._normalize_frame(frame) for frame in frames]

        # Step 2: Run initial dlib detection on all frames to get a baseline.
        # Note: This is where you would integrate your multiprocessing utility.
        print(f"Detecting initial landmarks in {len(normalized_frames)} frames...")
        self.raw_landmarks_per_frame = [self.detector.extract_faces(frame)[0] for frame in normalized_frames]
        print("Initial landmark detection complete.")

        # Step 3: Iterate through the frames to apply optical flow and smoothing.
        print("Computing smoothed landmarks frame-by-frame...")
        self._compute_smoothing_pipeline(normalized_frames)

        print("Face tracking processing complete!")

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        """Helper to convert frame to grayscale and normalize it."""
        if len(frame.shape) == 3:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        # We assume a full mask for normalization, as dlib processes the whole image.
        return frame_processor.normalize_frame(gray_frame, np.ones_like(gray_frame)).astype(np.uint8)

    def _compute_smoothing_pipeline(self, frames: List[np.ndarray]):
        """
        Manages the frame-by-frame smoothing and tracking logic.
        This is the core loop that combines dlib, optical flow, and smoothing.
        """
        # --- First Frame Initialization ---
        if not self.raw_landmarks_per_frame or self.raw_landmarks_per_frame[0] is None:
            print("Error: Could not detect landmarks in the first frame. Cannot proceed.")
            self.smoothed_landmarks_per_frame = [None] * len(frames)
            return

        first_landmarks = self.raw_landmarks_per_frame[0]
        self.smoothed_landmarks_per_frame.append(first_landmarks)

        first_points = landmark_processor.landmarks_to_points(first_landmarks)

        # Initialize history and trackers with the first frame's data.
        if self.use_moving_average:
            self.history.add_position_to_history(first_points.reshape(-1, 2))
        if self.use_optical_flow:
            self.optical_flow.initialize(frames[0], first_points)

        self.history.log_motion_vector(np.zeros(self.num_landmarks))

        # --- Process Subsequent Frames ---
        for i in range(1, len(frames)):
            current_frame = frames[i]
            dlib_landmarks = self.raw_landmarks_per_frame[i]

            # We need the previous frame's smoothed points for motion calculation.
            prev_smoothed_points = landmark_processor.landmarks_to_points(self.smoothed_landmarks_per_frame[-1])

            # The core logic: process the frame using the best available data.
            final_points = self._process_single_frame(current_frame, dlib_landmarks, prev_smoothed_points)

            # Convert points back to a landmark object and store the result.
            if final_points is not None:
                if self.use_optical_flow:
                    self.optical_flow.initialize(frames[i], final_points)
                final_landmarks = landmark_processor.points_to_landmarks(final_points)
                self.smoothed_landmarks_per_frame.append(final_landmarks)
            else:
                self.smoothed_landmarks_per_frame.append(None)
                # If tracking is lost, log zero motion to avoid breaking motion analysis.
                self.history.log_motion_vector(np.zeros(self.num_landmarks))

            if (i + 1) % 50 == 0:
                print(f"  -> Smoothed frame {i + 1}/{len(frames)}")

    def _process_single_frame(self, frame: np.ndarray, dlib_landmarks, prev_points: np.ndarray) -> np.ndarray:
        """
        Determines the strategy for processing a single frame and returns the final smoothed points.
        This is a dispatcher function.
        """
        # Strategy 1 (Best): Hybrid optical flow and dlib detection.
        if self.use_optical_flow and dlib_landmarks and prev_points is not None:
            return self._apply_hybrid_smoothing(frame, dlib_landmarks, prev_points)

        # Strategy 2 (Fallback): Optical flow only (if dlib failed, but we were tracking).
        elif self.use_optical_flow and prev_points is not None:
            return self._apply_optical_flow_only(frame, prev_points)

        # Strategy 3 (Basic): Dlib detection only (if optical flow is off).
        elif dlib_landmarks and prev_points is not None:
            return self._apply_dlib_only(dlib_landmarks, prev_points)

        # No usable data for this frame.
        else:
            return None

    def _apply_hybrid_smoothing(self, frame, dlib_landmarks, prev_points):
        """Combines dlib, optical flow, and temporal smoothing."""
        flow_points, status, _ = self.optical_flow.track(frame)
        dlib_points = landmark_processor.landmarks_to_points(dlib_landmarks)

        # Calculate dlib motion vectors to check for jitter.
        prev_positions = prev_points.reshape(-1, 2)
        dlib_positions = dlib_points.reshape(-1, 2)
        dlib_motion_mags = np.linalg.norm(dlib_positions - prev_positions, axis=1)
        z_scores = self.motion_analyzer.calculate_z_scores(dlib_motion_mags, self.history.motion_vectors,
                                                           cfg.HISTORY_WINDOW)

        # Combine points based on z-score analysis.
        combined_positions = np.zeros_like(dlib_positions)
        flow_positions = flow_points.reshape(-1, 2)
        for i in range(self.num_landmarks):
            if status[i] == 0:  # Optical flow failed, trust dlib.
                combined_positions[i] = dlib_positions[i]
            else:  # Blend based on motion magnitude.
                is_jitter = z_scores[i] < cfg.Z_SCORE_THRESHOLD
                d_w = cfg.LOW_MOTION_DLIB_WEIGHT if is_jitter else cfg.DLIB_WEIGHT
                f_w = cfg.LOW_MOTION_FLOW_WEIGHT if is_jitter else cfg.FLOW_WEIGHT
                combined_positions[i] = (d_w * dlib_positions[i] + f_w * flow_positions[i])

        # Apply final temporal smoothing if enabled.
        final_positions = self._apply_temporal_smoothing(combined_positions)

        # Log motion for the next frame's analysis.
        motion_mags = np.linalg.norm(final_positions - prev_positions, axis=1)
        self.history.log_motion_vector(motion_mags)

        return final_positions.reshape(-1, 1, 2)

    def _apply_optical_flow_only(self, frame, prev_points):
        """Uses only optical flow when dlib fails."""
        flow_points, status, _ = self.optical_flow.track(frame)
        # Validate to prevent points from jumping on failed tracks.
        valid_points = self.optical_flow.validate_tracking(flow_points, prev_points, status)
        return self.compute_final_positions_ts(valid_points, prev_points)

    def compute_final_positions_ts(self, current_points: np.ndarray, prev_points) -> np.ndarray:
        """Compute the positions using temporal smoothing"""
        final_positions = self._apply_temporal_smoothing(current_points.reshape(-1, 2))

        motion_mags = np.linalg.norm(final_positions - prev_points.reshape(-1, 2), axis=1)
        self.history.log_motion_vector(motion_mags)

        return final_positions.reshape(-1, 1, 2)

    def _apply_dlib_only(self, dlib_landmarks, prev_points):
        """Uses only dlib, with optional temporal smoothing."""
        dlib_points = landmark_processor.landmarks_to_points(dlib_landmarks)
        return self.compute_final_positions_ts(dlib_points, prev_points)

    def _apply_temporal_smoothing(self, positions):
        """Applies the weighted moving average if enabled and updates history."""
        if self.use_moving_average:
            smoothed_positions = self.smoother.calculate_weighted_moving_average(
                positions, self.history
            )
            # Add the *newly smoothed* positions to history for the next frame.
            self.history.add_position_to_history(smoothed_positions)
            return smoothed_positions
        else:
            # If not smoothing, just return the original positions.
            return positions

    # --- Public Data Accessors ---

    def get_smoothed_landmarks(self, frame_index: int):
        """
        Public method to get smoothed landmarks for a specific frame index.
        """
        if 0 <= frame_index < len(self.smoothed_landmarks_per_frame):
            return self.smoothed_landmarks_per_frame[frame_index]
        return None

    def get_original_landmarks(self, frame_index: int):
        """
        Public method to get the original, raw dlib-detected landmarks.
        """
        if 0 <= frame_index < len(self.raw_landmarks_per_frame):
            return self.raw_landmarks_per_frame[frame_index]
        return None

    def ft_lndmrk_outline(self, frame_index: int,
                          frame: np.ndarray, masks: List[List[int]]):
        """
        Face tracking function that uses smoothed landmarks from FaceTracker

        Args:
            self: Initialized FaceTracker instance
            frame_index: Current frame index
            frame: Current frame
            masks: List of landmark indices for each mask

        Returns:
            Tuple of (processed_image, masked_images, new_masks)
        """
        img_normalized = frame_processor.normalize_frame(frame, np.ones_like(frame))

        # Get smoothed landmarks for this frame (prioritize smoothed, fallback to original)
        dst_landmarks = self.get_smoothed_landmarks(frame_index)

        if dst_landmarks is None:
            print("No landmarks found!")
            dst_landmarks = self.get_original_landmarks(frame_index)

        if dst_landmarks is None:
            return img_normalized, None, None

        # Apply masks using smoothed landmarks
        newmasks_list = []
        masked_images = []

        for landmark_list in masks:
            masked_image, newmask = MaskGenerator.define_mask_from_landmark(
                img_normalized, dst_landmarks, landmark_list
            )
            masked_images.append(masked_image)
            newmasks_list.append(newmask)

        # Visualize landmarks
        landmark_points = []
        for i in range(dst_landmarks.num_parts):
            point = dst_landmarks.part(i)
            landmark_points.append((point.x, point.y))

        # Draw landmarks on image
        img_normalized = cv.cvtColor(img_normalized, cv.COLOR_GRAY2BGR)
        for i, (x, y) in enumerate(landmark_points):
            cv.circle(img_normalized, (x, y), 3, (0, 255, 0), -1)
            cv.putText(img_normalized, str(i), (x + 5, y - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        return img_normalized, masked_images, newmasks_list
