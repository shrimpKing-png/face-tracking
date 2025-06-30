# -*- coding: utf-8 -*-
"""
utils/mask_generator.py
Created on Wed Jun 25 20:10:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import numpy as np
import cv2 as cv
from typing import List
from face_tracking.processing.frame_processor import normalize_frame


class MaskGenerator:
    @staticmethod
    def define_mask_from_landmark(img: np.ndarray, landmarks, landmark_list: List[int]):
        """
        Creates a masked image from a single landmark group.
        This function is maintained for API compatibility but is no longer
        the primary method for performance-critical operations.

        Args:
            img: The input image to apply a mask to.
            landmarks: The landmarks detected by a model.
            landmark_list: The list of landmarks defining the mask region.

        Returns:
            A tuple of (masked_image, mask).
        """
        # Ensure landmark_list contains integers
        if landmark_list and not isinstance(landmark_list[0], int):
            landmark_list = [int(landmark) for landmark in landmark_list]

        mask = np.zeros_like(img, dtype=np.float32)

        # Pre-allocate numpy array for efficiency
        num_points = len(landmark_list)
        mask_pts = np.empty((num_points, 2), dtype=np.int32)

        for i, n in enumerate(landmark_list):
            part = landmarks.part(n)
            mask_pts[i, 0] = part.x
            mask_pts[i, 1] = part.y

        cv.fillConvexPoly(mask, mask_pts, 1.0)

        # Element-wise multiplication
        masked_image = img * mask
        return masked_image, mask

    def apply_masks(self, frame: np.ndarray, landmarks, masks: List[List[int]]):
        """
        Optimized creation of masked images based on landmark groups.

        Args:
            frame: The normalized source frame.
            landmarks: The landmarks object for the frame.
            masks: A list of lists, where each inner list contains landmark
                   indices for a specific facial feature.

        Returns:
            A tuple containing a list of generated masked images and a list
            of the corresponding mask arrays.
        """
        if not masks:
            return [], []

        # Prepare a list of polygons from the landmark indices
        polygons = []
        for landmark_list in masks:
            num_points = len(landmark_list)
            if num_points == 0:
                continue

            # Pre-allocate array for the polygon points
            poly_pts = np.empty((num_points, 2), dtype=np.int32)
            for i, n in enumerate(landmark_list):
                part = landmarks.part(n)
                poly_pts[i, 0] = part.x
                poly_pts[i, 1] = part.y
            polygons.append(poly_pts)

        if not polygons:
            return [], []

        # Create all masks in a single, efficient operation
        # We create a 3D array where each "slice" is a mask
        height, width = frame.shape[:2]
        num_masks = len(polygons)

        # Determine if the frame is color or grayscale for mask creation
        if frame.ndim == 3:
            all_masks_np = np.zeros((num_masks, height, width, frame.shape[2]), dtype=np.float32)
        else:
            all_masks_np = np.zeros((num_masks, height, width), dtype=np.float32)

        for i, poly in enumerate(polygons):
            # cv.fillPoly is efficient for multiple polygons, but here we use it
            # on each mask slice for clarity and to keep masks separate.
            cv.fillPoly(all_masks_np[i], [poly], 1.0)

        # Generate masked images using efficient element-wise multiplication
        # The frame is broadcasted across all masks
        masked_images_np = frame * all_masks_np

        # Convert back to lists to maintain API compatibility
        new_masks_lst = [all_masks_np[i] for i in range(num_masks)]
        masked_images = [masked_images_np[i] for i in range(num_masks)]

        return masked_images, new_masks_lst

    @staticmethod
    def plot_landmarks_and_select_roi(img, landmarks):
        """
        Plot landmark points on an image and allow interactive ROI selection.

        :param img: Input image
        :param landmarks: Landmarks object from dlib predictor
        :return: List of selected landmark indices for ROI
        """
        # Create a copy of the image to work with
        display_img = img.copy()

        # Convert landmarks to numpy array for easier handling
        landmark_points = []
        for i in range(landmarks.num_parts):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            landmark_points.append((x, y))

        # Plot all landmark points on the image
        for i, (x, y) in enumerate(landmark_points):
            cv.circle(display_img, (x, y), 3, (0, 255, 0), -1)  # Green circles
            cv.putText(display_img, str(i), (x + 5, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Variables for ROI selection
        selected_points = []
        selected_indices = []
        roi_complete = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal roi_complete

            if event == cv.EVENT_LBUTTONDOWN and not roi_complete:
                # Find the closest landmark point
                min_dist = float('inf')
                closest_idx = -1

                for i, (lx, ly) in enumerate(landmark_points):
                    dist = np.sqrt((x - lx) ** 2 + (y - ly) ** 2)
                    if dist < min_dist and dist < 15:  # Within 15 pixels
                        min_dist = dist
                        closest_idx = i

                if closest_idx != -1:
                    # Check if this is the first point clicked again (to close the ROI)
                    if len(selected_indices) > 2 and closest_idx == selected_indices[0]:
                        # Close the ROI by drawing line to first point
                        if len(selected_points) > 0:
                            cv.line(display_img, selected_points[-1], selected_points[0], (0, 0, 255), 2)
                        roi_complete = True
                        print(f"ROI completed! Selected landmark indices: {selected_indices}")
                    else:
                        # Add new point to selection
                        selected_points.append(landmark_points[closest_idx])
                        selected_indices.append(closest_idx)

                        # Highlight selected point
                        lx, ly = landmark_points[closest_idx]
                        cv.circle(display_img, (lx, ly), 5, (0, 0, 255), -1)  # Red circle

                        # Draw line to previous point if exists
                        if len(selected_points) > 1:
                            cv.line(display_img, selected_points[-2], selected_points[-1], (0, 0, 255), 2)

                        print(f"Selected landmark {closest_idx} at position {landmark_points[closest_idx]}")

            cv.imshow('Landmark ROI Selection', display_img)

        # Set up the window and mouse callback
        cv.namedWindow('Landmark ROI Selection', cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback('Landmark ROI Selection', mouse_callback)
        cv.imshow('Landmark ROI Selection', display_img)

        print("Instructions:")
        print("1. Click on landmark points to select them for ROI")
        print("2. Lines will be drawn between consecutive points")
        print("3. Click on the first point again to close the ROI")
        print("4. Press 'q' to quit or 'r' to reset selection")

        # Main interaction loop
        while True:
            key = cv.waitKey(1) & 0xFF

            # Reset selection
            if key == ord('r'):
                selected_points.clear()
                selected_indices.clear()
                roi_complete = False
                # Redraw original image with landmarks
                display_img = img.copy()
                for i, (x, y) in enumerate(landmark_points):
                    cv.circle(display_img, (x, y), 3, (0, 255, 0), -1)
                    cv.putText(display_img, str(i), (x + 5, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                cv.imshow('Landmark ROI Selection', display_img)
                print("Selection reset!")

            # Quit
            elif key == ord('q') or roi_complete:
                break

        cv.destroyAllWindows()
        return selected_indices

    def select_facial_roi(self, img, detector):
        """
        Complete workflow: extract faces and select ROI from landmarks

        :param img: Input image
        :param detector: dlib face detector
        :param predictor: dlib landmark predictor
        :return: Selected landmark indices for ROI
        """
        img = normalize_frame(img, np.ones_like(img)) if img.ndim == 2 else img
        # Extract faces and landmarks
        result = detector.extract_faces(img)

        if result[0] is None:
            print("No faces detected!")
            return []

        landmarks, faces, num_faces = result
        print(f"Found {num_faces} face(s)")

        # Select ROI from landmarks
        selected_roi = self.plot_landmarks_and_select_roi(img, landmarks)

        return selected_roi