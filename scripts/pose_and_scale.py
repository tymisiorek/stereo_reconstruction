import os
import json
import numpy as np
import cv2
import util


def draw_inlier_keypoints(img_path, keypoints, inlier_mask, window_name="Inliers"):
    """
    Draw circles on the inlier keypoints on a resized (half-size) image.
    
    The keypoints represent the feature matches between images that were
    identified as inliers during the pose estimation. They indicate where 
    corresponding features were detected in the original image.
    
    :param img_path: Path to the image file.
    :param keypoints: Array of keypoints (Nx2) from the matched points.
    :param inlier_mask: List or array of 0/1 indicating inlier status.
    :param window_name: Window name for display.
    """
    img = cv2.imread(img_path)

    #resize image to fraction of its original size so it fits
    half_width = img.shape[1] // 5
    half_height = img.shape[0] // 5
    img_resized = cv2.resize(img, (half_width, half_height))

    #draw each inlier point in green, scaling the coordinates accordingly
    for pt, valid in zip(keypoints, inlier_mask):
        if valid:
            x, y = int(pt[0] * 0.2), int(pt[1] * 0.2)
            cv2.circle(img_resized, (x, y), 4, (0, 255, 0), -1)

    # cv2.imshow(window_name, img_resized)
    # cv2.waitKey(0)
    # cv2.destroyWindow(window_name)

def recover_pose_and_resolve_scale(tri_data, known_baseline=None):
    """
    For each pair in tri_data, recover the pose (R, t) from the essential matrix
    using cv2.recoverPose, then optionally resolve the translation scale if a
    known baseline is provided.
    """
    updated_data = {}

    for pair_key, pair_info in tri_data.items():
        print(f"Processing pair: {pair_key}")

        K = np.array(pair_info["camera_intrinsics"], dtype=np.float64)
        E = np.array(pair_info["essential_matrix_direct"], dtype=np.float64)  # or use essential_matrix_from_F if desired
        ptsA = np.array(pair_info["matched_points_imgA"], dtype=np.float64)
        ptsB = np.array(pair_info["matched_points_imgB"], dtype=np.float64)

        #Reshape points to (N,1,2) for recoverPose
        ptsA_reshaped = ptsA.reshape(-1, 1, 2)
        ptsB_reshaped = ptsB.reshape(-1, 1, 2)

        #Recover pose (R, t) with cv2.recoverPose
        retval, R, t, pose_mask = cv2.recoverPose(E, ptsA_reshaped, ptsB_reshaped, K)

        #Resolve scale. If a known baseline is given, scale so that norm(t) == known_baseline.
        scale_factor = 1
        if known_baseline is not None:
            t_norm = np.linalg.norm(t)
            if t_norm > 1e-12:
                scale_factor = known_baseline / t_norm
            else:
                print(f"Warning: Norm of translation is nearly zero for pair {pair_key}, skipping scale resolution.")
                scale_factor = 1.0

        t_scaled = t * scale_factor

        pair_info["recovered_pose_inliers"] = int(retval) 
        pair_info["rotation_recovered"] = R.tolist()
        pair_info["translation_recovered"] = t_scaled.tolist()
        pair_info["scale_factor"] = scale_factor
        pair_info["inlier_mask_pose_recovered"] = pose_mask.flatten().astype(int).tolist()

        updated_data[pair_key] = pair_info

        #Visualize inlier keypoints on image a
        if "imgA" in pair_info:
            print(f"Visualizing inlier keypoints for {pair_info['imgA']}")
            draw_inlier_keypoints(pair_info["imgA"], ptsA, pair_info["inlier_mask_pose_recovered"],
                                    window_name=f"Inlier Keypoints - {pair_key} - Image A")
    return updated_data

def main():
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    chosen_folder = util.choose_image_set(parent_dir)
    tri_path = os.path.join(chosen_folder, "triangulation_data.json")

    tri_data = util.load_json_data(tri_path)

    # If known baseline, set it here (meters) (doesnt seem to do anyhting)
    known_baseline = 15
    updated_data = recover_pose_and_resolve_scale(tri_data, known_baseline=known_baseline)

    pose_estimation_path = os.path.join(chosen_folder, "pose_estimation_data.json")
    with open(pose_estimation_path, 'w') as f:
        json.dump(updated_data, f, indent=2)

    print(f"Pose recovery and scale resolution complete. Updated pose estimation data saved to:\n{pose_estimation_path}")

if __name__ == "__main__":
    main()
