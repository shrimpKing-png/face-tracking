# -*- coding: utf-8 -*-
"""
core/face_tracker.py
Created on Wed Jun 25 20:10:25 2025
Last Update: 27JUNE2025
@author: GPAULL
Optimized by Senior Engineering Standards
"""
import warnings
import numpy as np
import cv2 as cv
from typing import List, Optional, Tuple
from face_tracking.tracking.mediapipe_detector import MediaPipeDetector
from face_tracking.tracking.dlib_detector import DlibDetector
from face_tracking.tracking.optical_flow import OpticalFlowTracker
from face_tracking.processing import landmark_processor, frame_processor
from face_tracking.processing.smoothing import SmoothingEngine
from face_tracking.core.motion_analysis import MotionAnalyzer
from face_tracking.utils import TrackingHistory, MaskGenerator
from face_tracking.config import settings as cfg
from face_tracking.core.mask_ops import update_mask_positions, update_mask_positions_neighbors
from face_tracking.utils.visualizations import visualize_landmarks

class FrameResult:
    """Encapsulates the result of processing a single frame."""

    def __init__(self, raw_landmarks, smoothed_landmarks, final_points, motion_magnitudes):
        self.raw_landmarks = raw_landmarks
        self.smoothed_landmarks = smoothed_landmarks
        self.final_points = final_points
        self.motion_magnitudes = motion_magnitudes


class FaceTracker:
    """
    High-performance face tracking orchestrator optimized for mission-critical applications.

    Eliminates redundant processing and provides consistent, predictable behavior
    across single-frame and batch processing modes.
    """

    def __init__(self, use_optical_flow: bool = cfg.USE_OPTICAL_FLOW,
                 use_moving_average: bool = cfg.USE_MOVING_AVERAGE,
                 num_landmarks: int = cfg.NUM_LANDMARKS,
                 use_neighbors: bool = cfg.USE_NEIGHBORS_TRANSFORM,
                 num_neighbors: int = cfg.NUM_NEIGHBORS,
                 landmark_detector: str = 'dlib'):
        """Initialize FaceTracker with optimized component setup."""

        # Configuration
        self.use_optical_flow = use_optical_flow
        self.use_moving_average = use_moving_average
        self.num_landmarks = num_landmarks
        self.use_neighbors = use_neighbors
        self.num_neighbors = num_neighbors

        # Core components - single responsibility principle
        self.detector = self._create_detector(landmark_detector)
        self.optical_flow = OpticalFlowTracker(lk_params=cfg.LK_PARAMS) if use_optical_flow else None
        self.smoother = SmoothingEngine() if use_moving_average else None
        self.motion_analyzer = MotionAnalyzer()

        # Centralized state management
        self.history = TrackingHistory(num_landmarks=self.num_landmarks)
        self.raw_landmarks_per_frame = []
        self.smoothed_landmarks_per_frame = []
        self.mask_neighbors = None

        # Processing state
        self._is_initialized = False
        self._last_valid_points = None

        self._print_initialization_status()

    def _create_detector(self, detector_type: str):
        """Factory method for detector creation."""
        if detector_type.lower() == 'dlib':
            return DlibDetector()
        elif detector_type.lower() == 'mediapipe':
            return MediaPipeDetector()
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")

    def batch_process_frames(self, frames: List[np.ndarray]) -> None:
        """
        Optimized batch processing - single pass through frames.
        No redundant landmark detection.
        """
        if not frames:
            print("Warning: No frames provided to process.")
            return

        print(f"Processing {len(frames)} frames...")

        # Reset state for batch processing
        self._reset_state()

        # Single-pass processing
        for i, frame in enumerate(frames):
            self.process_frame(frame)

            if (i + 1) % 50 == 0:
                print(f"  -> Processed frame {i + 1}/{len(frames)}")

        print("Batch processing complete!")

    def process_frame(self, frame: np.ndarray):
        """
        Unified frame processing entry point.
        Used by both single-frame and batch processing.
        """
        # Preparation phase
        normalized_frame = self._prepare_frame(frame)
        raw_landmarks = self._detect_landmarks(normalized_frame)

        # Processing phase
        if not self._is_initialized:
            result = self._initialize_first_frame(raw_landmarks, normalized_frame)
        else:
            result = self._process_subsequent_frame(raw_landmarks, normalized_frame)

        # State update phase
        self._update_state(result)

        return result.smoothed_landmarks if result else None

    def _reset_state(self) -> None:
        """Reset tracker state for new processing session."""
        self.raw_landmarks_per_frame.clear()
        self.smoothed_landmarks_per_frame.clear()
        self.history = TrackingHistory(num_landmarks=self.num_landmarks)
        self._is_initialized = False
        self._last_valid_points = None

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optimized frame preparation."""
        if frame.dtype == np.float32:
            return self._normalize_frame(frame)
        return frame

    def _detect_landmarks(self, frame: np.ndarray):
        """Centralized landmark detection."""
        landmarks = self.detector.extract_faces(frame)[0]
        self.raw_landmarks_per_frame.append(landmarks)
        return landmarks

    def _initialize_first_frame(self, landmarks, frame: np.ndarray) -> Optional[FrameResult]:
        """Initialize tracking with first valid frame."""
        if landmarks is None:
            print("Error: Could not detect landmarks in the first frame. Cannot proceed.")
            return FrameResult(landmarks, landmarks, None, np.zeros(self.num_landmarks))

        points = landmark_processor.landmarks_to_points(landmarks)

        # Initialize tracking systems
        if self.use_moving_average:
            self.history.add_position_to_history(points.reshape(-1, 2))
        if self.use_optical_flow:
            self.optical_flow.initialize(frame, points)

        # Update state
        self._is_initialized = True
        self._last_valid_points = points

        motion_mags = np.zeros(self.num_landmarks)
        self.history.log_motion_vector(motion_mags)

        return FrameResult(landmarks, landmarks, points, motion_mags)

    def _process_subsequent_frame(self, raw_landmarks, frame: np.ndarray) -> FrameResult:
        """Process non-initial frames using established tracking."""
        prev_landmarks = self._get_last_valid_landmarks()

        if prev_landmarks is None:
            return self._handle_tracking_loss(raw_landmarks)

        prev_points = landmark_processor.landmarks_to_points(prev_landmarks)
        final_points = self._compute_tracking_result(frame, raw_landmarks, prev_points)

        if final_points is not None:
            return self._finalize_tracking_success(final_points, frame, prev_points, raw_landmarks)
        else:
            return self._handle_tracking_loss(raw_landmarks)

    def _compute_tracking_result(self, frame: np.ndarray, raw_landmarks, prev_points: np.ndarray) -> Optional[
        np.ndarray]:
        """
        Unified tracking computation - handles all tracking strategies.
        Eliminates duplicate logic across different processing paths.
        """
        # Strategy selection based on available data and configuration
        if self.use_optical_flow and raw_landmarks and prev_points is not None:
            return self._hybrid_tracking(frame, raw_landmarks, prev_points)
        elif self.use_optical_flow and prev_points is not None:
            return self._optical_flow_only_tracking(frame, prev_points)
        elif raw_landmarks is not None:
            return landmark_processor.landmarks_to_points(raw_landmarks)
        elif prev_points is not None:
            # Fallback: use last valid points when no new landmarks detected
            # This prevents accessing None landmarks when no smoothing is enabled
            return prev_points
        else:
            return None

    def _hybrid_tracking(self, frame: np.ndarray, raw_landmarks, prev_points: np.ndarray) -> np.ndarray:
        """Optimized hybrid tracking combining landmarks and optical flow."""
        # Get optical flow tracking
        flow_points, status, _ = self.optical_flow.track(frame)
        landmark_points = landmark_processor.landmarks_to_points(raw_landmarks)

        # Compute motion analysis once
        prev_positions = prev_points.reshape(-1, 2)
        landmark_positions = landmark_points.reshape(-1, 2)
        motion_mags = np.linalg.norm(landmark_positions - prev_positions, axis=1)
        z_scores = self.motion_analyzer.calculate_z_scores(
            motion_mags, self.history.motion_vectors, cfg.HISTORY_WINDOW
        )

        # Vectorized blending computation
        flow_positions = flow_points.reshape(-1, 2)
        combined_positions = self._blend_positions(
            landmark_positions, flow_positions, status, z_scores
        )

        return self._apply_temporal_smoothing(combined_positions).reshape(-1, 1, 2)

    def _blend_positions(self, landmark_pos: np.ndarray, flow_pos: np.ndarray,
                         status: np.ndarray, z_scores: np.ndarray) -> np.ndarray:
        """Vectorized position blending for performance."""
        combined = np.zeros_like(landmark_pos)

        # Failed optical flow - use landmarks
        failed_flow = (status == 0).flatten()
        combined[failed_flow] = landmark_pos[failed_flow]

        # Successful optical flow - blend based on motion
        success_flow = ~failed_flow
        is_low_motion = z_scores < cfg.Z_SCORE_THRESHOLD

        # Compute weights vectorized
        landmark_weights = np.where(is_low_motion, cfg.LOW_MOTION_LANDMARK_WEIGHT, cfg.LANDMARK_WEIGHT)
        flow_weights = np.where(is_low_motion, cfg.LOW_MOTION_FLOW_WEIGHT, cfg.FLOW_WEIGHT)

        combined[success_flow] = (
                landmark_weights[success_flow, np.newaxis] * landmark_pos[success_flow] +
                flow_weights[success_flow, np.newaxis] * flow_pos[success_flow]
        )

        return combined

    def _optical_flow_only_tracking(self, frame: np.ndarray, prev_points: np.ndarray) -> np.ndarray:
        """Optimized optical-flow-only tracking."""
        flow_points, status, _ = self.optical_flow.track(frame)
        valid_points = self.optical_flow.validate_tracking(flow_points, prev_points, status)

        final_positions = self._apply_temporal_smoothing(valid_points.reshape(-1, 2))
        return final_positions.reshape(-1, 1, 2)

    def _apply_temporal_smoothing(self, positions: np.ndarray) -> np.ndarray:
        """Centralized temporal smoothing application."""
        if self.use_moving_average:
            smoothed = self.smoother.calculate_weighted_moving_average(positions, self.history)
            self.history.add_position_to_history(smoothed)
            return smoothed
        return positions

    def _finalize_tracking_success(self, final_points: np.ndarray, frame: np.ndarray,
                                   prev_points: np.ndarray, raw_landmarks) -> FrameResult:
        """Finalize successful tracking with motion calculation."""
        # Motion calculation
        motion_mags = self._calculate_motion_magnitudes(final_points, prev_points)
        self.history.log_motion_vector(motion_mags)

        # Update optical flow state
        if self.use_optical_flow:
            self.optical_flow.initialize(frame, final_points)

        # Create result
        final_landmarks = landmark_processor.points_to_landmarks(final_points)
        self._last_valid_points = final_points

        return FrameResult(raw_landmarks, final_landmarks, final_points, motion_mags)

    def _handle_tracking_loss(self, raw_landmarks) -> FrameResult:
        """
        Unified tracking loss handling with intelligent fallback.
        When no landmarks are detected, attempts to use last valid landmarks.
        """
        motion_mags = np.zeros(self.num_landmarks)
        self.history.log_motion_vector(motion_mags)

        # If we have raw landmarks, use them
        if raw_landmarks is not None:
            return FrameResult(raw_landmarks, raw_landmarks, None, motion_mags)

        # Otherwise, try to fallback to last valid landmarks
        last_valid = self._get_last_valid_landmarks()
        if last_valid is not None:
            warnings.warn('Landmark detection failed - using last valid landmarks as fallback')
            return FrameResult(None, last_valid, None, motion_mags)

        # Complete failure
        warnings.warn('Complete tracking loss - no landmarks detected and no valid fallback!')
        return FrameResult(None, None, None, motion_mags)

    def _calculate_motion_magnitudes(self, current_points: np.ndarray, prev_points: np.ndarray) -> np.ndarray:
        """Optimized motion magnitude calculation."""
        current_pos = current_points.reshape(-1, 2)
        prev_pos = prev_points.reshape(-1, 2)
        return np.linalg.norm(current_pos - prev_pos, axis=1)

    def _get_last_valid_landmarks(self):
        """
        Get the most recent valid (non-None) landmarks.
        Critical for handling frames where landmark detection fails.
        """
        if not self.smoothed_landmarks_per_frame:
            return None

        # Search backwards for the last valid landmarks
        for landmarks in reversed(self.smoothed_landmarks_per_frame):
            if landmarks is not None:
                return landmarks

        # If all smoothed landmarks are None, fallback to raw landmarks
        for landmarks in reversed(self.raw_landmarks_per_frame):
            if landmarks is not None:
                return landmarks

        return None

    def _update_state(self, result: FrameResult) -> None:
        """Centralized state update."""
        self.smoothed_landmarks_per_frame.append(result.smoothed_landmarks)

    # --- Optimized utility methods ---

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        """Optimized frame normalization."""
        if len(frame.shape) == 3:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        return frame_processor.normalize_frame(gray_frame, np.ones_like(gray_frame)).astype(np.uint8)

    def _print_initialization_status(self) -> None:
        """Concise initialization status."""
        features = []
        if self.use_optical_flow:
            features.append("Optical Flow")
        if self.use_moving_average:
            features.append(f"Moving Average (window: {cfg.SMOOTHING_WINDOW})")

        status = f"FaceTracker initialized: {', '.join(features) if features else 'Raw detection only'}"
        print(status)

    # --- Public API (unchanged for compatibility) ---

    def get_motion_stats(self):
        """Return motion analyzer for stats building."""
        return self.motion_analyzer.build_motion_stat_df(self.history)

    def get_smoothed_landmarks(self, frame_index: int):
        """Get smoothed landmarks for specific frame."""
        if 0 <= frame_index < len(self.smoothed_landmarks_per_frame):
            return self.smoothed_landmarks_per_frame[frame_index]
        return None

    def get_original_landmarks(self, frame_index: int):
        """Get original raw landmarks."""
        if 0 <= frame_index < len(self.raw_landmarks_per_frame):
            return self.raw_landmarks_per_frame[frame_index]
        return None

    # face_tracking/core/face_tracker.py
    def ft_lndmrk_outline(self, frame_index: int, frame: np.ndarray,
                          masks: List[List[int]], org_landmarks=None) -> Tuple:
        """
        Optimized landmark outline processing.
        Public API maintained for compatibility.
        """
        # np.asarray is more efficient as it avoids copying data if not necessary
        frame = np.asarray(frame)

        # Normalize once if needed (no change here, was already efficient)
        img_normalized = frame_processor.normalize_frame(frame, np.ones_like(frame)) \
            if frame.dtype == np.float32 else frame

        # Get landmarks with fallback
        dst_landmarks = (self.get_smoothed_landmarks(frame_index) or
                         self.get_original_landmarks(frame_index))

        if dst_landmarks is None:
            warnings.warn(f"No landmarks found for frame_index: {frame_index}")
            return img_normalized, None, None

        # Apply masks
        if org_landmarks is None:
            # The optimizations in MaskGenerator.apply_masks provide the speedup here
            msk_gen = MaskGenerator()
            masked_images, newmasks_list = msk_gen.apply_masks(img_normalized, dst_landmarks, masks)
        else:
            # This part of the logic remains unchanged as it handles a different use case
            if self.use_neighbors:
                newmasks_list, self.mask_neighbors = update_mask_positions_neighbors(
                    org_landmarks, dst_landmarks, masks, self.num_neighbors, self.mask_neighbors
                )
            else:
                newmasks_list = update_mask_positions(org_landmarks, dst_landmarks, masks)

            # Using a generator expression can be slightly more memory-efficient
            masked_images = [img_normalized * mask for mask in newmasks_list]

        # Efficient landmark visualization
        vis_img = visualize_landmarks(img_normalized.copy(), dst_landmarks)

        return vis_img, masked_images, newmasks_list