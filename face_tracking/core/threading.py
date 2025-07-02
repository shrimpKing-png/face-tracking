# -*- coding: utf-8 -*-
"""
core/threading.py
Created on Mon Jun 30 2025
@author: GPAULL
"""
import threading
import queue
from typing import Optional
import time
from face_tracking.core.face_tracker import FaceTracker, FrameResult
from face_tracking.utils.visualizations import colored_mask_viseye
import numpy as np
import cv2 as cv
from face_tracking.utils.visualizations import render_visualization

class ThreadedTracker:
    """
    Manages running the FaceTracker in a separate thread for non-blocking,
    real-time applications.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the threaded tracker.

        Args:
            *args: Positional arguments to be passed to the FaceTracker constructor.
            **kwargs: Keyword arguments to be passed to the FaceTracker constructor.
        """
        # Queues for thread-safe communication
        # maxsize=1 ensures we only process the most recent frame, preventing lag
        self.frames_to_process_q = queue.Queue(maxsize=1)
        self.processed_results_q = queue.Queue(maxsize=1)

        # A flag to signal the thread to stop
        self._stop_event = threading.Event()

        # Instantiate the FaceTracker engine with provided arguments
        self.tracker_args = args
        self.tracker_kwargs = kwargs

        # The background thread that will run the tracker
        self.processing_thread = threading.Thread(target=self._run, daemon=True)
        self.tracker = FaceTracker(*self.tracker_args, **self.tracker_kwargs)

    def _run(self):
        """
        The main loop for the background processing thread.
        This function should not be called directly. Use start().
        """
        # The FaceTracker instance is created and used only within this thread
        print("Tracker thread started...")

        while not self._stop_event.is_set():
            try:
                # Wait for a frame to arrive from the main thread
                frame = self.frames_to_process_q.get(timeout=0.1)

                # Process the frame using the engine
                result = self.tracker.process_frame(frame)

                # If we have a result, put it in the output queue
                if result:
                    # To avoid blocking, we clear the queue before putting the new item
                    if self.processed_results_q.full():
                        self.processed_results_q.get_nowait()
                    self.processed_results_q.put((frame, result))

            except queue.Empty:
                # This is expected if the main thread is slower than the tracker
                continue
            except Exception as e:
                print(f"Error in tracker thread: {e}")

    def add_frame_to_process(self, frame):
        """
        Adds a new frame to the processing queue from the main thread.
        This is non-blocking. If the queue is full, the oldest frame is discarded.
        """
        if self.frames_to_process_q.full():
            try:
                # Discard the old frame to make room for the new one
                self.frames_to_process_q.get_nowait()
            except queue.Empty:
                pass  # Should not happen if full(), but good practice
        self.frames_to_process_q.put(frame)

    def get_latest_result(self) -> Optional[tuple]:
        """
        Retrieves the latest processed result from the queue.
        This is non-blocking and returns None if no new result is available.
        """
        try:
            return self.processed_results_q.get_nowait()
        except queue.Empty:
            return None

    def start(self):
        """Starts the background processing thread."""
        self.processing_thread.start()

    def stop(self):
        """Signals the background thread to stop and waits for it to exit."""
        print("Stopping tracker thread...")
        self._stop_event.set()
        self.processing_thread.join()
        print("Tracker thread stopped.")


class ThreadedFrameProcessor:
    """
    Manages mask operations while main thread handles display.
    Processes colored mask visualizations in a separate thread for non-blocking operation.
    """

    def __init__(self):
        # Queues for thread-safe communication
        # maxsize=1 ensures we only process the most recent frame, preventing lag
        self.frames_to_process_q = queue.Queue(maxsize=1)
        self.processed_results_q = queue.Queue(maxsize=1)

        # A flag to signal the thread to stop
        self._stop_event = threading.Event()

        # The background thread that will run the mask processing
        self.processing_thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        """
        The main loop for the background processing thread.
        This function should not be called directly. Use start().
        """
        print("Frame processor thread started...")

        while not self._stop_event.is_set():
            try:
                # Wait for a frame and masks to arrive from the main thread
                frame, newmasks_list = self.frames_to_process_q.get(timeout=0.1)

                # Process the colored mask visualization
                colored_masks_display = self._colored_mask_process(newmasks_list, frame)

                # If we have a result, put it in the output queue
                if colored_masks_display is not None:
                    # To avoid blocking, we clear the queue before putting the new item
                    if self.processed_results_q.full():
                        try:
                            self.processed_results_q.get_nowait()
                        except queue.Empty:
                            pass
                    self.processed_results_q.put(colored_masks_display)

            except queue.Empty:
                # This is expected if the main thread is slower than the processor
                continue
            except Exception as e:
                print(f"Error in frame processor thread: {e}")

    def _colored_mask_process(self, newmasks_list, frame):
        """
        Process the colored mask visualization.

        Args:
            newmasks_list: List of mask arrays
            frame: The frame to apply masks to

        Returns:
            The colored mask visualization result
        """
        try:
            # Ensure frame is grayscale for colored_mask_viseye
            if frame.ndim != 2:
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            else:
                gray_frame = frame

            result = colored_mask_viseye(newmasks_list, gray_frame)
            return result
        except Exception as e:
            print(f"Error processing colored masks: {e}")
            return None

    def add_frame_to_process(self, frame, newmasks_list):
        """
        Adds a new frame and masks to the processing queue from the main thread.
        This is non-blocking. If the queue is full, the oldest item is discarded.

        Args:
            frame: The video frame to process
            newmasks_list: List of mask arrays to apply
        """
        if self.frames_to_process_q.full():
            try:
                # Discard the old frame to make room for the new one
                self.frames_to_process_q.get_nowait()
            except queue.Empty:
                pass  # Should not happen if full(), but good practice
        self.frames_to_process_q.put((frame, newmasks_list))

    def get_latest_result(self) -> Optional[np.ndarray]:
        """
        Retrieves the latest processed colored mask result from the queue.
        This is non-blocking and returns None if no new result is available.

        Returns:
            The colored mask visualization or None if no result is available
        """
        try:
            return self.processed_results_q.get_nowait()
        except queue.Empty:
            return None

    def start(self):
        """Starts the background processing thread."""
        if not self.processing_thread.is_alive():
            self.processing_thread.start()

    def stop(self):
        """Signals the background thread to stop and waits for it to exit."""
        print("Stopping frame processor thread...")
        self._stop_event.set()
        if self.processing_thread.is_alive():
            self.processing_thread.join()
        print("Frame processor thread stopped.")


class ThreadedCamera:
    """
    Captures frames from camera in a separate thread with buffering
    to prevent blocking on frame reads.
    """

    def __init__(self, src=0, buffer_size=2):
        """
        Initialize threaded camera capture.

        Args:
            src: Camera source (0 for default webcam)
            buffer_size: Number of frames to buffer (2-3 is usually optimal)
        """
        self.src = src
        self.buffer_size = buffer_size

        # Initialize camera
        self.cap = cv.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {src}")

        # Optimize camera settings for performance
        self._optimize_camera_settings()

        # Threading components
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)

        # Statistics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_fps_time = time.time()

    def _optimize_camera_settings(self):
        """Optimize camera settings for better performance."""
        # Set buffer size to minimum to reduce latency

        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

        # Set higher FPS if supported
        self.cap.set(cv.CAP_PROP_FPS, 30)

        # Disable auto-exposure for more consistent timing (optional)
        # self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode

        print(f"Camera initialized:")
        print(f"  FPS: {self.cap.get(cv.CAP_PROP_FPS)}")
        print(f"  Buffer size: {int(self.cap.get(cv.CAP_PROP_BUFFERSIZE))}")

    def _capture_loop(self):
        """Main capture loop running in background thread."""
        print("Camera capture thread started...")

        while not self._stop_event.is_set():
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to read frame from camera")
                continue

            self.frames_captured += 1

            # Try to put frame in queue (non-blocking)
            try:
                # If queue is full, remove old frame and add new one
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # Discard old frame
                        self.frames_dropped += 1
                    except queue.Empty:
                        pass

                self.frame_queue.put(frame, block=False)

            except queue.Full:
                self.frames_dropped += 1

    def start(self):
        """Start the capture thread."""
        if not self._capture_thread.is_alive():
            self._capture_thread.start()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest frame from the buffer.

        Returns:
            (success, frame) tuple similar to cv.VideoCapture.read()
        """
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def get_fps_stats(self):
        """Get capture statistics."""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time

        if elapsed >= 1.0:  # Update every second
            capture_fps = self.frames_captured / elapsed if elapsed > 0 else 0
            drop_rate = (self.frames_dropped / max(1, self.frames_captured)) * 100

            self.frames_captured = 0
            self.frames_dropped = 0
            self.last_fps_time = current_time

            return capture_fps, drop_rate
        return None, None

    def stop(self):
        """Stop capture and cleanup."""
        print("Stopping camera capture...")
        self._stop_event.set()
        if self._capture_thread.is_alive():
            self._capture_thread.join()
        self.cap.release()
        print("Camera capture stopped.")