# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:10:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import numpy as np


def normalize_frame(arr, imgmask):
    """
    normalize array to 0-1 range for all values

    Parameters
    ----------
    arr : np.array
    values you want to normalize across its min and max range
    booleanmask : mask you want to use np.array
        the mask you want to appy to the normalizeation process

    Returns
    -------
    new_arr : new image that is normalized to 0-1
        DESCRIPTION.

    """

    mask = np.array(imgmask)
    bool_mask = mask.astype(bool)
    new_arr = np.array(arr).astype(np.uint8)
    """
    get values of just the face without any zeros, etc
    """
    facevalues = new_arr[bool_mask != False]

    """
    get the min value in the range and then shift all values to left
    """
    minval = np.min(facevalues)
    new_arr[new_arr < minval] = minval
    new_arr = new_arr - minval

    "get the new max value and divide all values by the max value"
    maxval = np.max(new_arr)
    new_arr = new_arr / maxval if maxval != 0 else np.zeros_like(new_arr)
    return new_arr * 255

