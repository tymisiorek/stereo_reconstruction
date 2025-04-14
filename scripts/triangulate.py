import os
import json
import numpy as np
import cv2

def choose_image_set(parent_dir):
    """
    Same as before; let user pick subfolder containing pose_estimation_data.json.
    """
    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    if not subdirs:
        print(f"No subdirectories found in {parent_dir}.")
        return None

    print("Choose one of the following image sets:")
    for idx, d in enumerate(subdirs, start=1):
        print(f"{idx}. {d}")

    while True:
        try:
            choice = int(input("Enter the number of the folder you want to use: "))
            if 1 <= choice <= len(subdirs):
                return os.path.join(parent_dir, subdirs[choice - 1])
            else:
                print(f"Invalid selection. Enter a number between 1 and {len(subdirs)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def load_pose_estimation(json_path):
    """
    Loads the dictionary with pose and correspondence data from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def triangulate_and_merge(tri_data, chosen_folder):
    """
    For each image pair key in tri_data, constructs the projection matrices,
    filters inlier correspondences using the inlier mask, triangulates points using OpenCV,
    and then adds the triangulation results to the original data.
    Finally, the merged data is saved in JSON format.
    """
    for pair_key, pair_info in tri_data.items():
        print(f"Processing pair: {pair_key}")

        # 1) Load intrinsics, recovered pose, and matched points
        K = np.array(pair_info["camera_intrinsics"], dtype=np.float64)
        R2 = np.array(pair_info["rotation_recovered"], dtype=np.float64)
        t2 = np.array(pair_info["translation_recovered"], dtype=np.float64).reshape(3, 1)

        ptsA = np.array(pair_info["matched_points_imgA"], dtype=np.float64)
        ptsB = np.array(pair_info["matched_points_imgB"], dtype=np.float64)

        # Use the inlier mask from recoverPose; filtering where mask == 255
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

        # 3) Triangulate points using OpenCV.
        # OpenCV expects points in shape (2, N)
        ptsA_for_triang = ptsA_inliers.T  # shape (2, N)
        ptsB_for_triang = ptsB_inliers.T  # shape (2, N)
        points_4d = cv2.triangulatePoints(P1, P2, ptsA_for_triang, ptsB_for_triang)

        # 4) Convert from homogeneous coordinates to 3D (divide by the last coordinate)
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T  # shape (N, 3)
        points_3d_list = points_3d.tolist()

        # 5) Append triangulation results to the original data for this image pair.
        pair_info["triangulated_points_3d"] = points_3d_list
        pair_info["num_triangulated_points"] = len(points_3d_list)

    # Save the merged data to a new JSON file
    output_path = os.path.join(chosen_folder, "pose_and_triangulation_data.json")
    with open(output_path, 'w') as f:
        json.dump(tri_data, f, indent=2)

    print(f"Merged pose estimation and triangulated data saved to {output_path}")

def main():
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    chosen_folder = choose_image_set(parent_dir)
    if not chosen_folder:
        print("No valid folder selected. Exiting.")
        return

    pose_path = os.path.join(chosen_folder, "pose_estimation_data.json")
    if not os.path.isfile(pose_path):
        print(f"No pose_estimation_data.json found in {chosen_folder}.")
        return

    # Load pose estimation data (which also contains matches, intrinsics, and recovered poses)
    tri_data = load_pose_estimation(pose_path)

    # Triangulate and merge the data, then save the combined file
    triangulate_and_merge(tri_data, chosen_folder)

if __name__ == "__main__":
    main()
