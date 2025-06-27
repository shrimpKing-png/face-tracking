import numpy as np
from face_tracking.processing.landmark_processor import landmarks_to_points
import skimage as ski
import cv2 as cv


def update_mask_positions(org, dst, masks: list):
    """
    Args:
        org: origin landmarks - should be taken from reference image, not updated over time!
        dst: destination landmarks - either smoothed or raw landmarks
        masks: [list of all the masks you want to update]
    Returns:
        updated_masks: [updated masks in new position calculated with estimate_transform]
    """
    org_pts, dst_pts = landmarks_to_points(org), landmarks_to_points(dst)
    org_pts = np.array(org_pts).squeeze()  # (54, 1, 2) -> (54, 2)
    dst_pts = np.array(dst_pts).squeeze()  # (54, 1, 2) -> (54, 2)
    tform = ski.transform.estimate_transform('similarity', org_pts, dst_pts)

    # Apply transformation to each mask
    updated_masks = []
    for mask in masks:
        # Warp the mask using the estimated transformation
        # Use inverse_map to properly transform from destination back to source coordinates
        warped_mask = ski.transform.warp(mask, inverse_map=tform.inverse, preserve_range=True)
        updated_masks.append(warped_mask)

    return updated_masks


def get_mask_center(img):
    """
    Args:
        img: input image - should be a b&w image where white is the mask area and black is not.
    Returns:
        np.array([center_x, center_y]), the centerpoints of the masked regoin
    """
    if img.ndim != 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary = np.where(img > 0)
    # Find coordinates of white pixels
    white_pixels = np.where(binary)
    y_coords, x_coords = white_pixels
    # Calculate centroid
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    return np.array([center_x, center_y])


def update_mask_positions_neighbors(org, dst, masks: list, n_points=10, stored_mask_neighbors=None):
    """
    Args:
        org: origin landmarks - should be taken from reference image, not updated over time!
        dst: destination landmarks - either smoothed or raw landmarks
        masks: [list of all the masks you want to update]
        n_points: number of closest points to use for transformation (default: 10)
    Returns:
        updated_masks: [updated masks in new position calculated with estimate_transform using closest points]
    """
    org_pts, dst_pts = landmarks_to_points(org), landmarks_to_points(dst)
    org_pts = np.array(org_pts).squeeze()  # (54, 1, 2) -> (54, 2)
    dst_pts = np.array(dst_pts).squeeze()  # (54, 1, 2) -> (54, 2)

    updated_masks = []
    if stored_mask_neighbors is None:
        use_old_neighbors = False
    else:
        use_old_neighbors = True
        stored_mask_neighbors = []
    for i, mask in enumerate(masks):
        if not use_old_neighbors:
            # Get the center of the current mask
            mask_center = get_mask_center(mask)

            # Calculate distances from mask center to all landmark points
            distances = np.linalg.norm(org_pts - mask_center, axis=1)

            # Find indices of the n_points closest points
            closest_indices = np.argsort(distances)[:n_points]
            stored_mask_neighbors.append(closest_indices)
        closest_indices = stored_mask_neighbors[i]
        # Select the closest points from both origin and destination
        closest_org_pts = org_pts[closest_indices]
        closest_dst_pts = dst_pts[closest_indices]

        # Estimate transformation using only the closest points
        tform = ski.transform.estimate_transform('similarity', closest_org_pts, closest_dst_pts)

        # Warp the mask using the estimated transformation
        warped_mask = ski.transform.warp(mask, inverse_map=tform.inverse, preserve_range=True)
        updated_masks.append(warped_mask)

    return updated_masks, stored_mask_neighbors
