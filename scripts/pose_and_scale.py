import os
import json
import numpy as np
import cv2

def choose_image_set(parent_dir):
    """
    Same as before; let user pick subfolder containing triangulation_data.json.
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

def load_triangulation_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

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
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image: {img_path}")
        return

    # Resize image to half of its original size
    half_width = img.shape[1] // 2
    half_height = img.shape[0] // 2
    img_resized = cv2.resize(img, (half_width, half_height))

    # Draw each inlier point in green, scaling the coordinates accordingly
    for pt, valid in zip(keypoints, inlier_mask):
        if valid:
            x, y = int(pt[0] * 0.5), int(pt[1] * 0.5)
            cv2.circle(img_resized, (x, y), 4, (0, 255, 0), -1)

    cv2.imshow(window_name, img_resized)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def recover_pose_and_resolve_scale(tri_data, known_baseline=None):
    """
    For each pair in tri_data, recover the pose (R, t) from the essential matrix
    using cv2.recoverPose, then optionally resolve the translation scale if a
    known baseline is provided.
    """
    updated_data = {}

    for pair_key, pair_info in tri_data.items():
        print(f"Processing pair: {pair_key}")

        # 1) Load intrinsics, essential matrix, and matched points
        K = np.array(pair_info["camera_intrinsics"], dtype=np.float64)
        E = np.array(pair_info["essential_matrix_direct"], dtype=np.float64)  # or use essential_matrix_from_F if desired
        ptsA = np.array(pair_info["matched_points_imgA"], dtype=np.float64)
        ptsB = np.array(pair_info["matched_points_imgB"], dtype=np.float64)

        # Reshape points to (N,1,2) for recoverPose
        ptsA_reshaped = ptsA.reshape(-1, 1, 2)
        ptsB_reshaped = ptsB.reshape(-1, 1, 2)

        # 2) Recover pose (R, t) with cv2.recoverPose
        retval, R, t, pose_mask = cv2.recoverPose(E, ptsA_reshaped, ptsB_reshaped, K)

        # 3) Resolve scale. If a known baseline is given, scale so that norm(t) == known_baseline.
        scale_factor = 1
        if known_baseline is not None:
            t_norm = np.linalg.norm(t)
            # Avoid division by zero if recoverPose fails or t is degenerate
            if t_norm > 1e-12:
                scale_factor = known_baseline / t_norm
            else:
                print(f"Warning: Norm of translation is nearly zero for pair {pair_key}, skipping scale resolution.")
                scale_factor = 1.0

        t_scaled = t * scale_factor

        # 4) Store the updated pose and scale in the data for this pair
        pair_info["recovered_pose_inliers"] = int(retval)  # number of inliers from recoverPose
        pair_info["rotation_recovered"] = R.tolist()
        pair_info["translation_recovered"] = t_scaled.tolist()
        pair_info["scale_factor"] = scale_factor
        pair_info["inlier_mask_pose_recovered"] = pose_mask.flatten().astype(int).tolist()

        updated_data[pair_key] = pair_info

        # Optional: Visualize inlier keypoints on image A (similar visualization can be done for image B)
        if "imgA" in pair_info:
            print(f"Visualizing inlier keypoints for {pair_info['imgA']}")
            draw_inlier_keypoints(pair_info["imgA"], ptsA, pair_info["inlier_mask_pose_recovered"],
                                    window_name=f"Inlier Keypoints - {pair_key} - Image A")
    return updated_data

def main():
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    chosen_folder = choose_image_set(parent_dir)
    if not chosen_folder:
        print("No valid folder selected. Exiting.")
        return

    tri_path = os.path.join(chosen_folder, "triangulation_data.json")
    if not os.path.isfile(tri_path):
        print(f"No triangulation_data.json found in {chosen_folder}. Run pose_and_triangulation first.")
        return

    # Load existing data
    tri_data = load_triangulation_data(tri_path)

    # If you have a known baseline, set it here, e.g. known_baseline = 0.12 meters; otherwise, set to None.
    known_baseline = .12  # Example value; adjust as necessary

    # Perform pose recovery and scale resolution
    updated_data = recover_pose_and_resolve_scale(tri_data, known_baseline=known_baseline)

    # 5) Save the updated pose estimation data to a new JSON file in the same folder
    pose_estimation_path = os.path.join(chosen_folder, "pose_estimation_data.json")
    with open(pose_estimation_path, 'w') as f:
        json.dump(updated_data, f, indent=2)

    print(f"Pose recovery and scale resolution complete. Updated pose estimation data saved to:\n{pose_estimation_path}")

if __name__ == "__main__":
    main()
