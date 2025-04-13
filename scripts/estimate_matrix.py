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

def main():
    # -------------------------------------------------------------------------
    # 1. Prompt for the folder that contains feature_data.json
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
    
    if not refined_matches:
        print("No matches found in the JSON. Exiting.")
        return
    
    # -------------------------------------------------------------------------
    # 3. Pick the first matched pair of images in the dictionary
    #    (Alternatively, you could prompt the user or pick a specific pair)
    # -------------------------------------------------------------------------
    first_pair_key = next(iter(refined_matches.keys()))
    matches_list = refined_matches[first_pair_key]
    if not matches_list:
        print(f"No inlier matches for pair: {first_pair_key}. Exiting.")
        return
    
    # The pair key is "imgA::imgB"
    imgA_path, imgB_path = first_pair_key.split("::")
    
    # -------------------------------------------------------------------------
    # 4. Gather the (x,y) points from keypoints based on the match indices
    # -------------------------------------------------------------------------
    # Reconstruct the keypoint list for each image from the JSON
    kpsA_json = sift_results[imgA_path]["keypoints"]  # list of dicts
    kpsB_json = sift_results[imgB_path]["keypoints"]  # list of dicts
    
    # We'll build arrays of points for the matched keypoints
    ptsA = []
    ptsB = []
    
    for m in matches_list:
        # m is a dict with 'queryIdx', 'trainIdx', etc.
        query_idx = m["queryIdx"]
        train_idx = m["trainIdx"]
        
        # Keypoint A
        kpA = kpsA_json[query_idx]  # the dictionary for that keypoint
        xA, yA = kpA["pt"][0], kpA["pt"][1]
        
        # Keypoint B
        kpB = kpsB_json[train_idx]
        xB, yB = kpB["pt"][0], kpB["pt"][1]
        
        ptsA.append([xA, yA])
        ptsB.append([xB, yB])
    
    ptsA = np.float32(ptsA)
    ptsB = np.float32(ptsB)
    
    if len(ptsA) < 8:
        print(f"Not enough matches ({len(ptsA)}) to reliably compute F. Exiting.")
        return
    
    # -------------------------------------------------------------------------
    # 5. Estimate Fundamental Matrix (F) using RANSAC
    # -------------------------------------------------------------------------
    F, mask = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None or F.shape != (3, 3):
        print("Fundamental matrix estimation failed. Exiting.")
        return
    
    print("Estimated Fundamental Matrix (F):")
    print(F)
    
    # -------------------------------------------------------------------------
    # 6. Estimate Essential Matrix (E) using placeholder intrinsics
    #    E = K^T * F * K
    #    Replace fx, fy, cx, cy with your real camera calibration params
    # -------------------------------------------------------------------------
    fx = 1000.0  # placeholder
    fy = 1000.0  # placeholder
    cx = 640.0   # placeholder
    cy = 360.0   # placeholder
    
    K = np.array([
        [fx,   0, cx],
        [0,   fy, cy],
        [0,    0,  1 ]
    ], dtype=np.float64)
    
    E = K.T @ F @ K
    
    print("\nEstimated Essential Matrix (E):")
    print(E)

if __name__ == "__main__":
    main()
