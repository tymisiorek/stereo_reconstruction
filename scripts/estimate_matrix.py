import os
import json
import cv2
import numpy as np

def choose_image_set(parent_dir):
    """
    List all subdirectories under 'parent_dir' and prompt the user to pick one.
    Returns the subdirectory name chosen by the user (full path), or None if none found.
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

def load_feature_data(json_path):
    """
    Load the feature data (sift_results, refined_matches_dict) from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_matched_points(sift_results, refined_matches_dict, pair_key):
    """
    From the JSON data structures, extract matched 2D points for a specific 'imgA::imgB' key.
    
    :param sift_results: dict from JSON containing keypoints and descriptors
    :param refined_matches_dict: dict from JSON containing matches for each image pair
    :param pair_key: string key in the format 'imgA::imgB'
    :return: two NumPy arrays (pointsA, pointsB) of shape (N, 2)
    """
    # Split the pair_key to get actual image names
    imgA, imgB = pair_key.split("::")
    
    # Get the stored data for each image
    imgA_data = sift_results[imgA]
    imgB_data = sift_results[imgB]
    
    # Deserialize keypoints for each image using positional arguments
    keypointsA = [
        cv2.KeyPoint(
            float(kp_dict["pt"][0]),
            float(kp_dict["pt"][1]),
            float(kp_dict["size"]),
            float(kp_dict["angle"]),
            float(kp_dict["response"]),
            int(kp_dict["octave"]),
            int(kp_dict["class_id"])
        ) for kp_dict in imgA_data["keypoints"]
    ]
    
    keypointsB = [
        cv2.KeyPoint(
            float(kp_dict["pt"][0]),
            float(kp_dict["pt"][1]),
            float(kp_dict["size"]),
            float(kp_dict["angle"]),
            float(kp_dict["response"]),
            int(kp_dict["octave"]),
            int(kp_dict["class_id"])
        ) for kp_dict in imgB_data["keypoints"]
    ]
    
    # Pull matches for this pair
    matches = refined_matches_dict[pair_key]
    
    pointsA = []
    pointsB = []
    for m in matches:
        # queryIdx corresponds to keypoints in imgA, trainIdx to keypoints in imgB.
        queryIdx = m["queryIdx"]
        trainIdx = m["trainIdx"]
        
        ptA = keypointsA[queryIdx].pt
        ptB = keypointsB[trainIdx].pt
        pointsA.append(ptA)
        pointsB.append(ptB)
    
    pointsA = np.array(pointsA, dtype=np.float32)
    pointsB = np.array(pointsB, dtype=np.float32)
    
    return pointsA, pointsB

def process_image_pairs(sift_results, refined_matches, K):
    """
    Process all image pairs: for each pair, extract matched points,
    estimate the Fundamental and Essential matrices, recover the pose,
    and collect data for point triangulation.
    
    :param sift_results: dict containing keypoints/descriptors for each image.
    :param refined_matches: dict with keys as "imgA::imgB" and list of match dicts.
    :param K: Camera intrinsic matrix.
    :return: A dictionary containing triangulation data for each image pair.
    """
    triangulation_data = {}
    
    for pair_key in refined_matches:
        print(f"\nProcessing image pair: {pair_key}")
        pointsA, pointsB = extract_matched_points(sift_results, refined_matches, pair_key)
        print(f"  Extracted {pointsA.shape[0]} matched points for each image.")
        
        # Skip pair if too few points for a reliable estimation.
        if pointsA.shape[0] < 8:
            print("  Not enough matches to compute the matrices (minimum 8 required). Skipping.")
            continue
        
        # Estimate the Fundamental Matrix using RANSAC
        F, mask_F = cv2.findFundamentalMat(pointsA, pointsB, cv2.FM_RANSAC, 1.0, 0.99)
        if F is None or F.shape[0] == 0:
            print("  Fundamental matrix estimation failed.")
            continue
        
        print("  Estimated Fundamental Matrix (F):")
        print(F)
        
        # Derive the Essential Matrix from F
        E_from_F = K.T @ F @ K
        print("  Essential Matrix derived from F:")
        print(E_from_F)
        
        # Directly compute the Essential Matrix using RANSAC
        E_direct, mask_E = cv2.findEssentialMat(pointsA, pointsB, cameraMatrix=K,
                                                  method=cv2.FM_RANSAC, prob=0.999, threshold=1.0)
        if E_direct is None or E_direct.shape[0] == 0:
            print("  Essential matrix estimation failed.")
            continue
        print("  Essential Matrix computed directly from matched points:")
        print(E_direct)
        
        # Recover the relative camera pose (rotation and translation) from the essential matrix.
        retval, R, t, mask_pose = cv2.recoverPose(E_direct, pointsA, pointsB, K)
        print("  Recovered rotation (R):")
        print(R)
        print("  Recovered translation (t):")
        print(t)
        
        # Collect data for triangulation.
        # Convert numpy arrays/matrices to lists for JSON serialization.
        pair_data = {
            "imgA": pair_key.split("::")[0],
            "imgB": pair_key.split("::")[1],
            "camera_intrinsics": K.tolist(),
            "fundamental_matrix": F.tolist(),
            "essential_matrix_from_F": E_from_F.tolist(),
            "essential_matrix_direct": E_direct.tolist(),
            "matched_points_imgA": pointsA.tolist(),
            "matched_points_imgB": pointsB.tolist(),
            "recovered_rotation": R.tolist(),
            "recovered_translation": t.tolist(),
            "inlier_mask_fundamental": mask_F.tolist(),
            "inlier_mask_pose": mask_pose.tolist()
        }
        
        triangulation_data[pair_key] = pair_data

    return triangulation_data

def main():
    # -------------------------------------------------------------------------
    # 1. Prompt for the folder that contains "feature_data.json"
    # -------------------------------------------------------------------------
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    chosen_folder = choose_image_set(parent_dir)
    if not chosen_folder:
        print("No valid folder was selected. Exiting.")
        return
    
    # Construct the path to "feature_data.json"
    json_path = os.path.join(chosen_folder, "feature_data.json")
    if not os.path.isfile(json_path):
        print(f"No feature_data.json found in {chosen_folder}. Exiting.")
        return
    
    # -------------------------------------------------------------------------
    # 2. Load the JSON data (keypoints, descriptors, and inlier matches)
    # -------------------------------------------------------------------------
    data = load_feature_data(json_path)
    sift_results = data["sift_results"]               # dict: image_path -> { "keypoints": [...], "descriptors": [...] }
    refined_matches = data["refined_matches_dict"]    # dict: "imgA::imgB" -> list of match dicts
    
    # -------------------------------------------------------------------------
    # 3. Define the Camera Intrinsics (Adjust these parameters as necessary)
    # -------------------------------------------------------------------------
    fx = 1844.65813
    fy = 1848.40594
    cx = 560.584384
    cy = 972.188133

    K = np.array([
        [fx,   0, cx],
        [0,   fy, cy],
        [0,    0,  1 ]
    ], dtype=np.float64)

    
    # -------------------------------------------------------------------------
    # 4. Process each image pair found in the JSON file and collect triangulation data
    # -------------------------------------------------------------------------
    if not refined_matches:
        print("No refined matches found. Exiting.")
        return
    
    triangulation_data = process_image_pairs(sift_results, refined_matches, K)
    
    # -------------------------------------------------------------------------
    # 5. Save the triangulation data to a file in the same directory as feature_data.json
    # -------------------------------------------------------------------------
    output_path = os.path.join(chosen_folder, "triangulation_data.json")
    with open(output_path, 'w') as outfile:
        json.dump(triangulation_data, outfile, indent=2)
    print(f"Triangulation data has been saved to {output_path}")

if __name__ == "__main__":
    main()
