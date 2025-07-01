# -*- coding: utf-8 -*-
"""
core/threading.py
Created on Mon Jun 30 2025
@author: GPAULL
"""
import threading
import queue
from typing import Optional

from face_tracking.core.face_tracker import FaceTracker, FrameResult


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
