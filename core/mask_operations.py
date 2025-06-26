# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:10:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import numpy as np
import cv2 as cv

def define_mask_from_landmark(img, landmarks, landmark_list):
    """

    Args:
        img: the input image you want to apply a mask to
        landmarks: the landmarks detected by a dlib/mediapipe model
        landmark_list: the list of landmarks that define the mask region's outline

    Returns:
        masked_image: the image with the mask applied to it.
        Esentially a cutout of the image in the shape of the mask
        mask: the mask, in the same data-format as the original image (maybe change to binary later)

    """
    if type(landmark_list[0]) is not int:
        landmark_list = [int(landmark) for landmark in landmark_list]
    mask = np.zeros(img.shape)
    mask_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in landmark_list], dtype=np.int32)
    mask = cv.fillConvexPoly(mask, mask_pts, color=1)
    masked_image = img * mask
    return masked_image, mask