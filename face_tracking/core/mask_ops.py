from face_tracking.processing.landmark_processor import landmarks_to_points
import skimage as ski


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

    tform = ski.transform.estimate_transform('similarity', org_pts, dst_pts)

    # Apply transformation to each mask
    updated_masks = []
    for mask in masks:
        # Warp the mask using the estimated transformation
        # Use inverse_map to properly transform from destination back to source coordinates
        warped_mask = ski.transform.warp(mask, inverse_map=tform.inverse, preserve_range=True)
        updated_masks.append(warped_mask)

    return updated_masks
