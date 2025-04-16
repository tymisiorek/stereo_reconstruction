import os
import json
import numpy as np
import cv2
from scipy.optimize import least_squares
import util

ZERO_VEC3 = np.zeros(3, dtype=np.float64)

def rodrigues_to_mat(rvec):
    """Converts a Rodrigues rotation vector to a 3x3 rotation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    return R

def mat_to_rodrigues(R):
    """Converts a 3x3 rotation matrix to a Rodrigues rotation vector."""
    rvec, _ = cv2.Rodrigues(R)
    return rvec

def project_points_no_dist(pts_3d, K, R, t):
    """
    Vectorized version of projecting 3D points into 2D without distortion.
    Args:
      pts_3d: (N, 3) array of 3D points.
      K:      (3, 3) camera intrinsics.
      R:      (3, 3) rotation matrix.
      t:      (3,)   translation vector.

    Returns:
      proj_2d: (N, 2) array of projected 2D points.
    """
    # 1) Rotate + translate: X_cam = R * X + t
    #    Here we do it in a vectorized way:
    #    R @ pts_3d^T => shape (3, N)
    #    then add t.reshape(3,1)
    X_cam = R @ pts_3d.T + t.reshape(3, 1)   # shape => (3, N)

    # 2) Apply camera intrinsics: X_img = K * X_cam
    X_img = K @ X_cam  # shape => (3, N)

    # 3) Perspective divide
    #    x' = X_img[0]/X_img[2], y' = X_img[1]/X_img[2]
    #    We'll stack them to get shape (N, 2)
    uv = X_img[:2] / X_img[2]  # shape => (2, N)

    # Transpose to (N,2)
    return uv.T

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
    rvec2 = params[:3]
    tvec2 = params[3:6]
    # Extract 3D points
    pts_3d = params[6:].reshape(n_points, 3)

    # --- Camera1: identity pose ---
    # R=I, t=0
    R1 = np.eye(3, dtype=np.float64)
    t1 = ZERO_VEC3

    # --- Camera2: from rvec2, tvec2 ---
    R2 = rodrigues_to_mat(rvec2)

    # Project points in a vectorized way
    proj_cam1 = project_points_no_dist(pts_3d, K, R1, t1)
    proj_cam2 = project_points_no_dist(pts_3d, K, R2, tvec2)

    # Allocate one residual array
    # Each camera has n_points * 2  => total 2 * n_points * 2
    residuals = np.empty(2 * n_points * 2, dtype=np.float64)

    # Fill camera1 residual
    residuals[: n_points * 2] = (proj_cam1 - pts2d_cam1).ravel()
    # Fill camera2 residual
    residuals[n_points * 2:] = (proj_cam2 - pts2d_cam2).ravel()

    return residuals

def refine_pose_and_points_pair(pair_info):
    """
    Refines the second camera’s pose and the 3D points for a pair using two-view bundle adjustment.
    Computes RMS reprojection error before and after refinement.
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

    n_2d = len(ptsA_inliers)
    n_3d = len(points_3d)
    n = min(n_2d, n_3d)

    # Not enough points => skip
    if n < 6:
        print("Not enough points to run bundle adjustment for this pair.")
        rvec2_init = mat_to_rodrigues(R2_init).flatten()
        if len(points_3d) == 0:
            return R2_init, t2_init, points_3d, 0.0, 0.0
        
        # Evaluate reprojection error with whatever we have
        # We'll do a quick vectorized projection or fall back to projectPoints
        # (Just to show you could do the same approach.)
        R2_eval = rodrigues_to_mat(rvec2_init)
        # clamp to the min of 2D or 3D points
        n_eval = min(len(ptsA_inliers), len(points_3d))
        if n_eval == 0:
            return R2_init, t2_init, points_3d, 0.0, 0.0

        X_eval = points_3d[:n_eval]
        A_eval = ptsA_inliers[:n_eval]
        B_eval = ptsB_inliers[:n_eval]

        R1 = np.eye(3, dtype=np.float64)
        t1 = ZERO_VEC3
        proj1 = project_points_no_dist(X_eval, K, R1, t1)
        proj2 = project_points_no_dist(X_eval, K, R2_eval, t2_init)

        residuals = np.concatenate([proj1.ravel() - A_eval.ravel(), 
                                    proj2.ravel() - B_eval.ravel()])
        error = np.sqrt(np.mean(residuals**2))
        return R2_init, t2_init, points_3d, error, error

    # Use only the first n valid inliers & points
    ptsA_inliers = ptsA_inliers[:n]
    ptsB_inliers = ptsB_inliers[:n]
    points_3d = points_3d[:n]

    # --- Compute Reprojection Error Before Optimization ---
    rvec2_init = mat_to_rodrigues(R2_init).flatten()
    R2_eval = rodrigues_to_mat(rvec2_init)
    R1 = np.eye(3, dtype=np.float64)
    t1 = ZERO_VEC3

    proj_cam1_before = project_points_no_dist(points_3d, K, R1, t1)
    proj_cam2_before = project_points_no_dist(points_3d, K, R2_eval, t2_init)
    residuals_before = np.concatenate([(proj_cam1_before - ptsA_inliers).ravel(),
                                       (proj_cam2_before - ptsB_inliers).ravel()])
    error_before = np.sqrt(np.mean(residuals_before**2))

    # --- Bundle Adjustment ---
    x0 = np.hstack((rvec2_init, t2_init, points_3d.ravel()))
    fun = lambda p: bundle_adjustment_residual(p, n, K, ptsA_inliers, ptsB_inliers)
    result = least_squares(fun, x0, method='lm',
                           xtol=1e-12, ftol=1e-12, gtol=1e-12,
                           max_nfev=1000)
    refined = result.x
    rvec2_refined = refined[:3]
    tvec2_refined = refined[3:6]
    points_3d_refined = refined[6:].reshape((n, 3))
    R2_refined = rodrigues_to_mat(rvec2_refined)

    # --- Compute Reprojection Error After Optimization ---
    proj_cam1_after = project_points_no_dist(points_3d_refined, K, R1, t1)
    proj_cam2_after = project_points_no_dist(points_3d_refined, K, R2_refined, tvec2_refined)
    residuals_after = np.concatenate([(proj_cam1_after - ptsA_inliers).ravel(),
                                      (proj_cam2_after - ptsB_inliers).ravel()])
    error_after = np.sqrt(np.mean(residuals_after**2))

    return R2_refined, tvec2_refined, points_3d_refined, error_before, error_after

def write_ply(filename, points):
    """
    Write a simple ASCII PLY file for a point cloud.
    Points should be an (N,3) array or list of [x, y, z].
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def main():
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    chosen_folder = util.choose_image_set(parent_dir)
    pose_path = os.path.join(chosen_folder, "pose_and_triangulation_data.json")

    tri_data = util.load_json_data(pose_path)
    errors_before_list = []
    errors_after_list = []
    all_refined_points = []

    for pair_key, pair_info in tri_data.items():
        print(f"Running bundle adjustment for pair: {pair_key}")

        if "triangulated_points_3d" not in pair_info or len(pair_info["triangulated_points_3d"]) < 1:
            print(f"No triangulated points found for pair {pair_key}, skipping.")
            continue

        R2_refined, t2_refined, points_3d_refined, error_before, error_after = refine_pose_and_points_pair(pair_info)

        # Store results
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

    # Write out updated JSON
    output_path = os.path.join(chosen_folder, "bundle_adjusted_data.json")
    with open(output_path, 'w') as f:
        json.dump(tri_data, f, indent=2)
    print(f"Bundle adjustment complete. Refined data saved to {output_path}")

    # Print global mean errors
    if errors_before_list and errors_after_list:
        global_error_before = np.mean(errors_before_list)
        global_error_after = np.mean(errors_after_list)
        print(f"Global average reprojection error before: {global_error_before:.4f}")
        print(f"Global average reprojection error after:  {global_error_after:.4f}")

    # Write combined PLY
    if all_refined_points:
        ply_filename = os.path.join(chosen_folder, "refined_point_cloud.ply")
        write_ply(ply_filename, all_refined_points)
        print(f"Refined point cloud saved to {ply_filename}")
    else:
        print("No refined points available to write a point cloud file.")

if __name__ == "__main__":
    main()
