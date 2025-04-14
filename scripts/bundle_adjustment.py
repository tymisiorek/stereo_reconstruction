import os
import json
import numpy as np
import cv2
from scipy.optimize import least_squares

def choose_image_set(parent_dir):
    """
    Let user pick a subfolder containing triangulated_data.json.
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
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def rodrigues_to_mat(rvec):
    """Converts a Rodrigues rotation vector to a 3x3 rotation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    return R

def mat_to_rodrigues(R):
    """Converts a 3x3 rotation matrix to a Rodrigues rotation vector."""
    rvec, _ = cv2.Rodrigues(R)
    return rvec

def project_points(points_3d, rvec, tvec, K):
    """
    Projects 3D points into 2D given camera intrinsics K and extrinsics (rvec, tvec).
    """
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 1, 3)
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)
    projected, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
    return projected.reshape(-1, 2)

def bundle_adjustment_residual(params, n_points, K, pts2d_cam1, pts2d_cam2):
    """
    Computes reprojection error residuals for:
      - Camera1 (fixed: R=I, t=0)
      - Camera2 (refined using Rodrigues + translation)
      - 3D points (refined)
    
    Parameter vector layout:
      [rvec_cam2 (3), tvec_cam2 (3), X1 (3), X2 (3), ..., Xn (3)]
    """
    # Extract camera2 parameters
    rvec2 = params[0:3]
    tvec2 = params[3:6]
    # Extract 3D points (n_points x 3)
    pts_3d = params[6:].reshape((n_points, 3))
    
    # Projection for Camera1 (identity pose)
    rvec1 = np.zeros((3,), dtype=np.float64)
    tvec1 = np.zeros((3,), dtype=np.float64)
    projected_cam1 = project_points(pts_3d, rvec1, tvec1, K)
    
    # Projection for Camera2 (using current estimates)
    projected_cam2 = project_points(pts_3d, rvec2, tvec2, K)
    
    # Compute residuals (flatten differences)
    residuals_cam1 = (projected_cam1 - pts2d_cam1).ravel()
    residuals_cam2 = (projected_cam2 - pts2d_cam2).ravel()
    residuals = np.concatenate((residuals_cam1, residuals_cam2), axis=0)
    return residuals

def refine_pose_and_points_pair(pair_info):
    """
    Refines the second camera’s pose and the 3D points for a pair using two-view bundle adjustment.
    
    Also computes the RMS reprojection error before and after refinement.
    
    Returns:
      R2_refined: refined rotation matrix for camera2.
      tvec2_refined: refined translation vector for camera2.
      points_3d_refined: refined 3D points (n x 3).
      error_before: RMS reprojection error before optimization.
      error_after: RMS reprojection error after optimization.
    """
    # --- Gather Data ---
    K = np.array(pair_info["camera_intrinsics"], dtype=np.float64)
    R2_init = np.array(pair_info["rotation_recovered"], dtype=np.float64)
    t2_init = np.array(pair_info["translation_recovered"], dtype=np.float64).reshape(3)
    points_3d = np.array(pair_info["triangulated_points_3d"], dtype=np.float64)
    
    ptsA = np.array(pair_info["matched_points_imgA"], dtype=np.float64)
    ptsB = np.array(pair_info["matched_points_imgB"], dtype=np.float64)
    inlier_mask = np.array(pair_info["inlier_mask_pose_recovered"], dtype=np.int32)
    
    # Filter to only inliers that were triangulated
    valid_idx = (inlier_mask == 255)
    ptsA_inliers = ptsA[valid_idx]
    ptsB_inliers = ptsB[valid_idx]
    
    # Use the minimum number of points available
    n_2d = len(ptsA_inliers)
    n_3d = len(points_3d)
    n = min(n_2d, n_3d)
    
    if n < 6:
        print("Not enough points to run bundle adjustment for this pair.")
        # Compute reprojection error using the available points
        rvec_zero = np.zeros((3,), dtype=np.float64)
        proj1 = project_points(points_3d, rvec_zero, np.zeros((3,), dtype=np.float64), K)
        rvec2_init = mat_to_rodrigues(R2_init).flatten()
        proj2 = project_points(points_3d, rvec2_init, t2_init, K)
        resid_before = np.concatenate(((proj1 - ptsA_inliers).ravel(), (proj2 - ptsB_inliers).ravel()))
        error = np.sqrt(np.mean(resid_before ** 2))
        return R2_init, t2_init, points_3d, error, error

    ptsA_inliers = ptsA_inliers[:n]
    ptsB_inliers = ptsB_inliers[:n]
    points_3d = points_3d[:n]
    
    # --- Compute Reprojection Error Before Optimization ---
    rvec_zero = np.zeros((3,), dtype=np.float64)
    projected_cam1_before = project_points(points_3d, rvec_zero, np.zeros((3,), dtype=np.float64), K)
    rvec2_init = mat_to_rodrigues(R2_init).flatten()
    projected_cam2_before = project_points(points_3d, rvec2_init, t2_init, K)
    residuals_before = np.concatenate(((projected_cam1_before - ptsA_inliers).ravel(),
                                       (projected_cam2_before - ptsB_inliers).ravel()))
    error_before = np.sqrt(np.mean(residuals_before ** 2))
    
    # --- Bundle Adjustment ---
    x0 = np.hstack((rvec2_init, t2_init, points_3d.reshape(-1)))
    fun = lambda params: bundle_adjustment_residual(
        params,
        n,
        K,
        ptsA_inliers,
        ptsB_inliers
    )
    result = least_squares(
        fun,
        x0,
        method='lm',       # Levenberg–Marquardt
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=1000
    )
    refined = result.x
    rvec2_refined = refined[0:3]
    tvec2_refined = refined[3:6]
    points_3d_refined = refined[6:].reshape((n, 3))
    R2_refined = rodrigues_to_mat(rvec2_refined)
    
    # --- Compute Reprojection Error After Optimization ---
    projected_cam1_after = project_points(points_3d_refined, rvec_zero, np.zeros((3,), dtype=np.float64), K)
    projected_cam2_after = project_points(points_3d_refined, rvec2_refined, tvec2_refined, K)
    residuals_after = np.concatenate(((projected_cam1_after - ptsA_inliers).ravel(),
                                      (projected_cam2_after - ptsB_inliers).ravel()))
    error_after = np.sqrt(np.mean(residuals_after ** 2))
    
    return R2_refined, tvec2_refined, points_3d_refined, error_before, error_after

def write_ply(filename, points):
    """
    Write a simple ASCII PLY file for a point cloud.
    Points should be an (N,3) array or list of [x, y, z].
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def main():
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    chosen_folder = choose_image_set(parent_dir)
    if not chosen_folder:
        print("No valid folder selected. Exiting.")
        return

    pose_path = os.path.join(chosen_folder, "pose_and_triangulation_data.json")
    if not os.path.isfile(pose_path):
        print(f"No pose_and_triangulation_data.json found in {chosen_folder}. Run triangulation first.")
        return

    tri_data = load_pose_estimation(pose_path)
    
    # These lists will hold errors from all pairs for global statistics.
    errors_before_list = []
    errors_after_list = []
    
    # This list will aggregate all refined 3D points for the final point cloud.
    all_refined_points = []
    
    # Process each pair in the dataset.
    for pair_key, pair_info in tri_data.items():
        print(f"Running bundle adjustment for pair: {pair_key}")
        if "triangulated_points_3d" not in pair_info or len(pair_info["triangulated_points_3d"]) < 1:
            print(f"No triangulated points found for pair {pair_key}, skipping.")
            continue
        
        R2_refined, t2_refined, points_3d_refined, error_before, error_after = refine_pose_and_points_pair(pair_info)
        
        pair_info["rotation_refined"] = R2_refined.tolist()
        pair_info["translation_refined"] = t2_refined.tolist()
        pair_info["triangulated_points_3d_refined"] = points_3d_refined.tolist()
        pair_info["num_points_refined"] = len(points_3d_refined)
        pair_info["reprojection_error_before"] = error_before
        pair_info["reprojection_error_after"] = error_after
        
        errors_before_list.append(error_before)
        errors_after_list.append(error_after)
        all_refined_points.extend(points_3d_refined.tolist())
        
        print(f"Pair {pair_key}: Reprojection error before = {error_before:.4f}, after = {error_after:.4f}")

    # Save the refined bundle-adjusted data.
    output_path = os.path.join(chosen_folder, "bundle_adjusted_data.json")
    with open(output_path, 'w') as f:
        json.dump(tri_data, f, indent=2)
    print(f"Bundle adjustment complete. Refined data saved to {output_path}")
    
    # (Optional) Print global average reprojection errors.
    if errors_before_list and errors_after_list:
        global_error_before = np.mean(errors_before_list)
        global_error_after = np.mean(errors_after_list)
        print(f"Global average reprojection error before: {global_error_before:.4f}")
        print(f"Global average reprojection error after: {global_error_after:.4f}")
    
    # Write the aggregated refined point cloud to a PLY file for MeshLab.
    if all_refined_points:
        ply_filename = os.path.join(chosen_folder, "refined_point_cloud.ply")
        write_ply(ply_filename, all_refined_points)
        print(f"Refined point cloud saved to {ply_filename}")
    else:
        print("No refined points available to write a point cloud file.")

if __name__ == "__main__":
    main()
