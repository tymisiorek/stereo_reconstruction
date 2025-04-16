import os
import json
import numpy as np
import cv2
import util


def triangulate_and_merge(tri_data, chosen_folder):
    """
    For each image pair key in tri_data, constructs the projection matrices,
    filters inlier correspondences using the inlier mask, triangulates points using OpenCV,
    and then adds the triangulation results to the original data.
    Finally, the merged data is saved in JSON format.
    """
    for pair_key, pair_info in tri_data.items():
        print(f"Processing pair: {pair_key}")

        #load intrinsics, recovered pose, and matched points
        K = np.array(pair_info["camera_intrinsics"], dtype=np.float64)
        R2 = np.array(pair_info["rotation_recovered"], dtype=np.float64)
        t2 = np.array(pair_info["translation_recovered"], dtype=np.float64).reshape(3, 1)

        ptsA = np.array(pair_info["matched_points_imgA"], dtype=np.float64)
        ptsB = np.array(pair_info["matched_points_imgB"], dtype=np.float64)

        #use the inlier mask from recoverPose; filtering where mask == 255
        inlier_mask = np.array(pair_info["inlier_mask_pose_recovered"], dtype=np.int32)
        ptsA_inliers = ptsA[inlier_mask == 255]
        ptsB_inliers = ptsB[inlier_mask == 255]

        # 2) Construct projection matrices for the two cameras.
        # First camera is the reference: R1 = I, t1 = 0
        R1 = np.eye(3, dtype=np.float64)
        t1 = np.zeros((3, 1), dtype=np.float64)
        P1 = K @ np.hstack((R1, t1))
        P2 = K @ np.hstack((R2, t2))

        if ptsA_inliers.shape[0] < 2:
            print(f"Warning: Not enough inliers to triangulate for pair {pair_key}. Skipping triangulation for this pair.")
            pair_info["triangulated_points_3d"] = []
            pair_info["num_triangulated_points"] = 0
            continue

        #triangulate
        ptsA_for_triang = ptsA_inliers.T  # shape (2, N)
        ptsB_for_triang = ptsB_inliers.T  # shape (2, N)
        points_4d = cv2.triangulatePoints(P1, P2, ptsA_for_triang, ptsB_for_triang)

        # convert from homogeneous coordinates to 3D (divide by the last coordinate)
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T  # shape (N, 3)
        points_3d_list = points_3d.tolist()

        # apend triangulation results to the original data for this image pair.
        pair_info["triangulated_points_3d"] = points_3d_list
        pair_info["num_triangulated_points"] = len(points_3d_list)

    output_path = os.path.join(chosen_folder, "pose_and_triangulation_data.json")
    with open(output_path, 'w') as f:
        json.dump(tri_data, f, indent=2)

    print(f"Merged pose estimation and triangulated data saved to {output_path}")

def main():
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    chosen_folder = util.choose_image_set(parent_dir)
    pose_path = os.path.join(chosen_folder, "pose_estimation_data.json")

    tri_data = util.load_json_data(pose_path)
    triangulate_and_merge(tri_data, chosen_folder)

if __name__ == "__main__":
    main()
 