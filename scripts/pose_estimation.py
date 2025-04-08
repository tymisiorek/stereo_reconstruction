import os
import json
import cv2
import numpy as np
from scipy.optimize import least_squares  # For sparse bundle adjustment

########################################
# Utility functions
########################################

def load_feature_data(feature_file):
    """Load the feature data JSON produced by your SIFT + RANSAC script."""
    with open(feature_file, 'r') as f:
        data = json.load(f)
    return data

def rebuild_keypoints(kp_list):
    """Convert JSON keypoints to cv2.KeyPoint objects."""
    keypoints = []
    for kp_dict in kp_list:
        x = kp_dict['pt'][0]
        y = kp_dict['pt'][1]
        size = kp_dict.get('size', 1.0)
        angle = kp_dict.get('angle', -1.0)
        response = kp_dict.get('response', 0.0)
        octave = kp_dict.get('octave', 0)
        class_id = kp_dict.get('class_id', -1)
        kp = cv2.KeyPoint(x, y, size, angle, response, octave, class_id)
        keypoints.append(kp)
    return keypoints

def rebuild_matches(match_list):
    """Convert JSON matches to cv2.DMatch objects."""
    matches = []
    for m_dict in match_list:
        q = m_dict.get('queryIdx', 0)
        t = m_dict.get('trainIdx', 0)
        d = m_dict.get('distance', 0.0)
        dm = cv2.DMatch(q, t, d)
        matches.append(dm)
    return matches

def extract_matched_points(imgA, imgB, data):
    """
    Return the 2D correspondences (ptsA, ptsB) from the inlier matches,
    along with the raw DMatch objects (dmatches) and the keypoints arrays (kpsA, kpsB).
    """
    keyA = f"{imgA}::{imgB}"
    keyB = f"{imgB}::{imgA}"
    matches_data = None
    if keyA in data['refined_matches_dict']:
        matches_data = data['refined_matches_dict'][keyA]
    elif keyB in data['refined_matches_dict']:
        matches_data = data['refined_matches_dict'][keyB]
    else:
        return None, None, None, None, None

    dmatches = rebuild_matches(matches_data)
    kpA_data = data['sift_results'][imgA]['keypoints']
    kpB_data = data['sift_results'][imgB]['keypoints']
    kpsA = rebuild_keypoints(kpA_data)
    kpsB = rebuild_keypoints(kpB_data)

    ptsA = np.float32([kpsA[m.queryIdx].pt for m in dmatches]).reshape(-1, 2)
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in dmatches]).reshape(-1, 2)
    return ptsA, ptsB, dmatches, kpsA, kpsB

def visualize_reprojection(image, projected_pts, original_pts, window_title="Reprojection"):
    """
    Draw the reprojected 3D points (green) and the original 2D feature locations (red) on the same image.
    """
    vis_img = image.copy()

    # Draw the reprojected points in green
    for pt in projected_pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)

    # Draw the original points in red
    for pt in original_pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis_img, (x, y), 3, (0, 0, 255), 1)

    cv2.imshow(window_title, vis_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

########################################
# Sparse Bundle Adjustment (2-view version)
########################################

def axis_angle_to_R(axis_angle):
    """Convert a 3D axis-angle rotation vector to a 3x3 rotation matrix via cv2.Rodrigues."""
    R, _ = cv2.Rodrigues(axis_angle.reshape(3,1))
    return R

def reproject_point(pt_3d, R, t, K):
    """
    Reproject a single 3D point into a camera with rotation R, translation t, and intrinsics K.
    Returns (x, y) in image coordinates.
    """
    # 3D -> camera coords
    pt_cam = R.dot(pt_3d) + t
    # project onto image plane
    x = pt_cam[0] / (pt_cam[2] + 1e-12)
    y = pt_cam[1] / (pt_cam[2] + 1e-12)
    # apply intrinsics
    px = K[0, 0]*x + K[0, 2]
    py = K[1, 1]*y + K[1, 2]
    return np.array([px, py], dtype=np.float64)

def sba_residuals(params, n_points, K, ptsA, ptsB):
    """
    Residual function for 2-view SBA:
      - The first camera is fixed at identity pose (R=I, t=0).
      - The second camera has 6 parameters for rotation (axis-angle) + translation.
      - Each 3D point has 3 parameters.
    
    params layout:
      [ rx, ry, rz, tx, ty, tz, X1, Y1, Z1, X2, Y2, Z2, ..., Xn, Yn, Zn ]
    """
    # 1) Extract the second camera pose parameters
    rvec = params[0:3]  # axis-angle
    tvec = params[3:6]
    # 2) Convert axis-angle to rotation matrix
    R = axis_angle_to_R(rvec)
    t = tvec.reshape(3,)

    # 3) Extract 3D points
    points_3d = params[6:].reshape(n_points, 3)

    residuals = []

    for i in range(n_points):
        X3d = points_3d[i]
        # Reproject to first camera (assume R=I, t=0)
        projA = reproject_point(X3d, np.eye(3), np.zeros(3), K)
        errA = projA - ptsA[i]  # shape (2,)

        # Reproject to second camera
        projB = reproject_point(X3d, R, t, K)
        errB = projB - ptsB[i]  # shape (2,)

        # Combine
        residuals.extend(errA)
        residuals.extend(errB)

    return np.array(residuals)

def bundle_adjustment_two_view(K, ptsA, ptsB, R_init, t_init, points3D_init):
    """
    Perform a naive 2-view SBA that refines:
      - The second camera pose (R, t)
      - The 3D points
    """
    n_points = len(points3D_init)

    # Flatten initial guess into a 1D parameter array
    #   param[0:3] -> axis-angle
    #   param[3:6] -> translation
    #   param[6:...] -> all 3D points (X, Y, Z for each)
    r_init, _ = cv2.Rodrigues(R_init)
    r_init = r_init.ravel()  # axis-angle vector
    t_init = t_init.ravel()

    param_init = []
    param_init.extend(r_init.tolist())    # 3
    param_init.extend(t_init.tolist())    # 3
    param_init.extend(points3D_init.reshape(-1).tolist())  # 3*n_points

    param_init = np.array(param_init, dtype=np.float64)

    # Use Levenberg-Marquardt via least_squares
    result = least_squares(
        fun=sba_residuals,
        x0=param_init,
        args=(n_points, K, ptsA, ptsB),
        verbose=1,
        method='lm'  # Levenberg–Marquardt
    )

    # Extract optimized parameters
    opt_params = result.x
    r_opt = opt_params[0:3]
    t_opt = opt_params[3:6]
    points_3d_opt = opt_params[6:].reshape(n_points, 3)

    # Convert axis-angle back to rotation matrix
    R_opt = axis_angle_to_R(r_opt)

    return R_opt, t_opt.reshape(3,1), points_3d_opt

########################################
# XYZ Export
########################################

def write_xyz(filename, points):
    """
    Write a set of 3D points to an XYZ file, each line: "X Y Z"
    No header lines.
    """
    with open(filename, 'w') as f:
        for p in points:
            x, y, z = p
            # Format to e.g. 6 decimal places
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

########################################
# Main pipeline function
########################################

def main():
    # Replace with your folder and sequence
    folder_path = r"C:\Projects\Semester6\CS4501\stereo_reconstruction\data\tripod-seq"
    sequence_id = "01"
    feature_json = os.path.join(folder_path, sequence_id, "feature_data.json")

    # 1) Load the JSON feature data
    data = load_feature_data(feature_json)

    # 2) Get the image paths
    img_paths = sorted(data['sift_results'].keys())
    if len(img_paths) < 2:
        print("Need at least two images for reconstruction.")
        return

    # 3) Camera intrinsics (example only; replace with real calibration)
    fx = 1000
    fy = 1000
    cx = 960
    cy = 540
    K = np.array([[fx,  0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    # We'll just do a pairwise reconstruction for consecutive images
    for i in range(len(img_paths) - 1):
        imgA_path = img_paths[i]
        imgB_path = img_paths[i+1]

        imageA = cv2.imread(imgA_path, cv2.IMREAD_COLOR)
        imageB = cv2.imread(imgB_path, cv2.IMREAD_COLOR)
        if imageA is None or imageB is None:
            print(f"Could not read {imgA_path} or {imgB_path}. Skipping.")
            continue

        # Extract correspondences
        ptsA, ptsB, dmatches, kpsA, kpsB = extract_matched_points(imgA_path, imgB_path, data)
        if ptsA is None or ptsB is None or len(ptsA) < 8:
            print(f"Not enough matches between {imgA_path} and {imgB_path}. Skipping.")
            continue

        # 4) Initial pose with Essential matrix
        E, mask_pose = cv2.findEssentialMat(
            ptsA, ptsB, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            print("No Essential Matrix found. Skipping.")
            continue
        _, R_init, t_init, mask_pose = cv2.recoverPose(E, ptsA, ptsB, K, mask=mask_pose)
        mask_pose = mask_pose.ravel()

        inlier_ptsA = ptsA[mask_pose == 1]
        inlier_ptsB = ptsB[mask_pose == 1]
        if len(inlier_ptsA) < 5:
            print("Too few inlier points after recoverPose. Skipping.")
            continue

        # 5) Triangulate initial 3D points
        P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = K @ np.hstack((R_init, t_init))
        points4D = cv2.triangulatePoints(P1, P2, inlier_ptsA.T, inlier_ptsB.T)
        points3D_init = (points4D[:3, :] / (points4D[3, :] + 1e-12)).T  # shape (N, 3)

        # 6) Run a simple 2-view SBA to refine (R, t) and the 3D points
        R_opt, t_opt, points3D_opt = bundle_adjustment_two_view(
            K, inlier_ptsA, inlier_ptsB, R_init, t_init, points3D_init
        )

        # 7) Reproject the refined 3D points onto image B for visualization
        projected_pts, _ = cv2.projectPoints(
            points3D_opt, cv2.Rodrigues(R_opt)[0], t_opt, K, None
        )
        projected_pts = projected_pts.reshape(-1, 2)

        # visualize_reprojection(imageB, projected_pts, inlier_ptsB,
        #                        window_title=f"Refined Reprojection: {os.path.basename(imgB_path)}")

        # 8) Write the final 3D points to an XYZ file
        #    Note: Each pair produces its own set of points. Adjust naming as needed.
        output_xyz = os.path.join(folder_path, "reconstructed_points.xyz")
        write_xyz(output_xyz, points3D_opt)
        print(f"Exported final 3D points to {output_xyz}")

    print("\nDone. Press any key in an OpenCV window (if open) to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
