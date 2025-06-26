from typing import List
import cv2 as cv
import numpy as np
from core.mask_operations import define_mask_from_landmark


def draw_landmarks_on_frame(frame: np.ndarray, landmarks) -> np.ndarray:
    """
    Draws landmark points and their indices on a frame.

    Args:
        frame: The image to draw on.
        landmarks: A landmarks object with a .num_parts and a .part(i) method.

    Returns:
        The frame with landmarks visualized.
    """
    vis_frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    for i in range(landmarks.num_parts):
        point = landmarks.part(i)
        cv.circle(vis_frame, (point.x, point.y), 3, (0, 255, 0), -1)
        cv.putText(vis_frame, str(i), (point.x + 5, point.y - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    return vis_frame


def create_masked_images(frame: np.ndarray, landmarks, masks: List[List[int]]):
    """
    Creates masked images based on landmark groups.

    Args:
        frame: The normalized source frame.
        landmarks: The landmarks object for the frame.
        masks: A list of lists, where each inner list contains the landmark
               indices for a specific facial feature.

    Returns:
        A tuple containing a list of the generated masked images and a list
        of the corresponding mask arrays.
    """
    masked_images = []
    new_masks_lst = []
    for landmark_list in masks:
        masked_image, new_mask = define_mask_from_landmark(
            frame, landmarks, landmark_list
        )
        masked_images.append(masked_image)
        new_masks_lst.append(new_mask)
    return masked_images, new_masks_lst
