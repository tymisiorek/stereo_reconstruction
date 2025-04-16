import os
import cv2
import numpy as np
import json
import util

def compute_sift_features(image_paths):
    """
    Runs SIFT on each image (grayscale) and returns a dictionary mapping image_path -> (keypoints, descriptors).
    Also prints out the number of features detected per image.
    """
    cv2.setUseOptimized(True)
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.013, edgeThreshold=10, sigma=1.6)
    
    features = {}
    gray_cache = {}
    for img_path in image_paths:
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
        gray_cache[img_path] = gray  # cache grayscale image
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        features[img_path] = (keypoints, descriptors)
        print(f"{img_path}: {len(keypoints)} features detected.")
    return features, gray_cache

def match_features(features, ratio_threshold=0.8):
    """
    Performs feature matching between consecutive images using a FLANN-based matcher and Lowe's ratio test.
    Returns a dict: (imgA, imgB) -> list of 'good' matches.
    Also prints out the number of good matches found between each pair.
    """
    # FLANN parameters for SIFT descriptors
    index_params = dict(algorithm=1, trees=5)  # algorithm=1 for KDTree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    image_paths = sorted(features.keys())
    matches_dict = {}
    
    for i in range(len(image_paths) - 1):
        imgA = image_paths[i]
        imgB = image_paths[i + 1]
        
        kpsA, descsA = features[imgA]
        kpsB, descsB = features[imgB]
        
        if descsA is None or descsB is None:
            matches_dict[(imgA, imgB)] = []
            print(f"Matching {imgA} and {imgB}: 0 good matches (missing descriptors).")
            continue
        
        knn_matches = flann.knnMatch(descsA, descsB, k=2)
        good_matches = [m for m, n in knn_matches if m.distance < ratio_threshold * n.distance]
        matches_dict[(imgA, imgB)] = good_matches
        print(f"Matching {imgA} and {imgB}: {len(good_matches)} good matches found.")
        
    return matches_dict

def apply_ransac(matches_dict, features, ransac_thresh=5.0):
    """
    For each consecutive pair, fit a homography using RANSAC and return only inlier matches.
    Also prints out the number of inlier matches after RANSAC.
    """
    refined_matches_dict = {}
    
    for (imgA, imgB), good_matches in matches_dict.items():
        kpsA, _ = features[imgA]
        kpsB, _ = features[imgB]
        
        if len(good_matches) < 4:
            refined_matches_dict[(imgA, imgB)] = []
            print(f"RANSAC {imgA} and {imgB}: insufficient matches for homography.")
            continue
        
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransac_thresh)
        if mask is None:
            refined_matches_dict[(imgA, imgB)] = []
            print(f"RANSAC {imgA} and {imgB}: homography could not be computed.")
            continue
        
        mask = mask.ravel().tolist()
        inlier_matches = [gm for gm, mk in zip(good_matches, mask) if mk == 1]
        refined_matches_dict[(imgA, imgB)] = inlier_matches
        print(f"RANSAC {imgA} and {imgB}: {len(inlier_matches)} inlier matches found.")
        
    return refined_matches_dict

def serialize_keypoints(keypoints):
    """
    Serialize keypoints by storing only the 'pt' and 'size' fields.
    """
    serialized = []
    for kp in keypoints:
        kp_dict = {
            'pt': kp.pt,
            'size': kp.size
        }
        serialized.append(kp_dict)
    return serialized

def serialize_matches(matches):
    """
    Serialize matches by storing only queryIdx and trainIdx.
    """
    serialized = []
    for m in matches:
        m_dict = {
            'queryIdx': m.queryIdx,
            'trainIdx': m.trainIdx
        }
        serialized.append(m_dict)
    return serialized

def save_feature_data(features, refined_matches_dict, output_folder):
    """
    Saves SIFT keypoints (only pt and size) and inlier matches to 'feature_data.json'
    inside the chosen image folder.
    """
    output_file = os.path.join(output_folder, "feature_data.json")
    
    data = {}
    sift_data = {}
    for img_path, (keypoints, _) in features.items():
        sift_data[img_path] = {
            'keypoints': serialize_keypoints(keypoints)
        }
    data['sift_results'] = sift_data

    matches_data = {}
    for (imgA, imgB), matches in refined_matches_dict.items():
        key = f"{imgA}::{imgB}"
        matches_data[key] = serialize_matches(matches)
    data['refined_matches_dict'] = matches_data

    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"Feature data saved to {output_file}")

def main():
    cv2.setUseOptimized(True)
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    
    chosen_folder = util.choose_image_set(parent_dir)
    if not chosen_folder:
        print("No valid folder was selected. Exiting.")
        return
    
    all_files = os.listdir(chosen_folder)
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_paths = [
        os.path.join(chosen_folder, f)
        for f in sorted(all_files)
        if f.lower().endswith(valid_exts)
    ]
    
    if not image_paths:
        print(f"No image files found in {chosen_folder}. Exiting.")
        return
    
    print(f"Processing {len(image_paths)} images in {chosen_folder}...")
    
    # Compute features (keypoints and descriptors) and cache grayscale images
    features, _ = compute_sift_features(image_paths)
    matches_dict = match_features(features, ratio_threshold=0.8)
    refined_matches_dict = apply_ransac(matches_dict, features, ransac_thresh=5.0)
    
    save_feature_data(features, refined_matches_dict, chosen_folder)
    
    # (Optional) Visualization of matches
    for (imgA, imgB), inlier_matches in refined_matches_dict.items():
        if not inlier_matches:
            continue
        
        imageA = cv2.imread(imgA, cv2.IMREAD_COLOR)
        imageB = cv2.imread(imgB, cv2.IMREAD_COLOR)
        if imageA is None or imageB is None:
            continue
        
        kpsA, _ = features[imgA]
        kpsB, _ = features[imgB]
        
        match_img = cv2.drawMatches(
            imageA,
            kpsA,
            imageB,
            kpsB,
            inlier_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
    #     window_name = "Matches"
    #     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow(window_name, 600, 400)
    #     cv2.imshow(window_name, match_img)
    #     cv2.waitKey(0)
    
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
