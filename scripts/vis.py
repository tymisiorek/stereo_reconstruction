import os
import json
import numpy as np
import cv2
import util

def visualize_correspondences(img_path, pts2d, pts3d, R, t, K, window_name="Correspondences"):
    """
    Visualize 2D keypoints (pts2d) and the reprojected 3D points (pts3d) on the image.
    
    Parameters:
      img_path: Path to the image.
      pts2d: 2D keypoints (Nx2 array) from the image.
      pts3d: 3D points (Nx3 array) that correspond to the 2D keypoints.
      R: Rotation matrix (3x3) from the pose estimation.
      t: Translation vector (3x1) from the pose estimation.
      K: Camera intrinsic matrix.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image: {img_path}")
        return

    # Convert rotation matrix to rotation vector.
    rvec, _ = cv2.Rodrigues(R)
    # Project the 3D points into the image.
    projected_points, _ = cv2.projectPoints(pts3d, rvec, t, K, None)
    projected_points = projected_points.reshape(-1, 2)

    # Create a copy for visualization.
    vis_img = img.copy()

    # Draw the original 2D keypoints (in green).
    for pt in pts2d:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

    # Draw the reprojected 3D points (in red).
    for pt in projected_points:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

    # Draw lines connecting corresponding points (in blue).
    for pt2d, proj_pt in zip(pts2d, projected_points):
        cv2.line(vis_img, (int(pt2d[0]), int(pt2d[1])), (int(proj_pt[0]), int(proj_pt[1])), (255, 0, 0), 1)

    # Create a named window and resize it to be larger.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)
    cv2.imshow(window_name, vis_img)
    print("Press any key to close the visualization window...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def triangulate_and_merge(tri_data, chosen_folder):
    """
    For each image pair key in tri_data, constructs the projection matrices,
    filters inlier correspondences using the inlier mask, triangulates points using OpenCV,
    and then adds the triangulation results to the original data.
    Finally, the merged data is saved in JSON format.
    Also, for each pair, visualizes the correspondences on the first image.
    """
    for pair_key, pair_info in tri_data.items():
        print(f"Processing pair: {pair_key}")

        # Load camera intrinsics, recovered pose, and matched points.
        K = np.array(pair_info["camera_intrinsics"], dtype=np.float64)
        R2 = np.array(pair_info["rotation_recovered"], dtype=np.float64)
        t2 = np.array(pair_info["translation_recovered"], dtype=np.float64).reshape(3, 1)

        ptsA = np.array(pair_info["matched_points_imgA"], dtype=np.float64)
        ptsB = np.array(pair_info["matched_points_imgB"], dtype=np.float64)

        # Use the inlier mask from recoverPose (mask value 255 means inlier).
        inlier_mask = np.array(pair_info["inlier_mask_pose_recovered"], dtype=np.int32)
        ptsA_inliers = ptsA[inlier_mask == 255]
        ptsB_inliers = ptsB[inlier_mask == 255]

        # Construct projection matrices.
        R1 = np.eye(3, dtype=np.float64)
        t1 = np.zeros((3, 1), dtype=np.float64)
        P1 = K @ np.hstack((R1, t1))
        P2 = K @ np.hstack((R2, t2))

        if ptsA_inliers.shape[0] < 2:
            print(f"Warning: Not enough inliers to triangulate for pair {pair_key}. Skipping triangulation for this pair.")
            pair_info["triangulated_points_3d"] = []
            pair_info["num_triangulated_points"] = 0
            continue

        # Triangulate.
        ptsA_for_triang = ptsA_inliers.T  # shape (2, N)
        ptsB_for_triang = ptsB_inliers.T  # shape (2, N)
        points_4d = cv2.triangulatePoints(P1, P2, ptsA_for_triang, ptsB_for_triang)

        # Convert from homogeneous to 3D.
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T  # shape (N, 3)
        points_3d_list = points_3d.tolist()

        # Append triangulation results to the original data.
        pair_info["triangulated_points_3d"] = points_3d_list
        pair_info["num_triangulated_points"] = len(points_3d_list)

        # Visualization: extract the two image paths from the pair_key.
        # Assuming pair_key is in the format "imgA_path::imgB_path"
        if "::" in pair_key:
            imgA_path, imgB_path = pair_key.split("::")
        else:
            print(f"Warning: pair_key {pair_key} is not in the expected format. Skipping visualization.")
            continue

        # Visualize correspondences on the first image.
        # Using all 2D keypoints from image A.
        visualize_correspondences(imgA_path, ptsA, np.array(points_3d_list, dtype=np.float32), 
                                  R2, t2, K, window_name=f"Correspondences: {pair_key}")



def main():
    # Define the parent directory containing your images.
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    chosen_folder = util.choose_image_set(parent_dir)
    
    # Path to the pose estimation data (output of your previous step).
    pose_path = os.path.join(chosen_folder, "pose_estimation_data.json")
    tri_data = util.load_json_data(pose_path)
    
    # Run triangulation and visualization.
    triangulate_and_merge(tri_data, chosen_folder)

if __name__ == "__main__":
    main()
