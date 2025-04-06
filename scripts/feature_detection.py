import os
import cv2
import numpy as np
from util import get_images_for_sequence

# SIFT algorithm
def sift_keypoint_detection(image_paths):
    sift = cv2.SIFT_create()
    results = {}
    for img_path in image_paths:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        results[img_path] = (keypoints, descriptors)
    return results

# Match features using Lowe's ratio
def match_features(sift_results, ratio_threshold=0.8):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    image_paths = sorted(sift_results.keys())
    matches_dict = {}
    
    # Match only consecutive images: 0 with 1, 1 with 2, etc.
    for i in range(len(image_paths) - 1):
        imgA = image_paths[i]
        imgB = image_paths[i + 1]
        
        kpsA, descsA = sift_results[imgA]
        kpsB, descsB = sift_results[imgB]
        
        if descsA is None or descsB is None:
            matches_dict[(imgA, imgB)] = []
            continue
        
        knn_matches = bf.knnMatch(descsA, descsB, k=2)
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        
        matches_dict[(imgA, imgB)] = good_matches
        
    return matches_dict

# RANSAC for removing outliers
def apply_ransac(matches_dict, sift_results, ransac_thresh=5.0):
    refined_matches_dict = {}
    
    for (imgA, imgB), good_matches in matches_dict.items():
        kpsA, _ = sift_results[imgA]
        kpsB, _ = sift_results[imgB]
        
        # Ensure there are enough matches to compute a homography.
        if len(good_matches) < 4:
            refined_matches_dict[(imgA, imgB)] = []
            continue
        
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransac_thresh)
        if mask is None:
            refined_matches_dict[(imgA, imgB)] = []
            continue
        
        mask = mask.ravel().tolist()
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i] == 1]
        refined_matches_dict[(imgA, imgB)] = inlier_matches
        
    return refined_matches_dict

if __name__ == "__main__":
    folder_path = r"stereo_reconstruction\data\tripod-seq" 
    sequence_id = "01" 

    paths = get_images_for_sequence(folder_path, sequence_id)
    
    sift_results = sift_keypoint_detection(paths)
    
    for path, (kps, descs) in sift_results.items():
        print(f"Image: {path}")
        print(f"  Number of keypoints detected: {len(kps)}")
        print(f"  Descriptor shape: {descs.shape if descs is not None else None}")
    
    matches_dict = match_features(sift_results, ratio_threshold=0.8)
    refined_matches_dict = apply_ransac(matches_dict, sift_results, ransac_thresh=5.0)
    
    # display correspondence
    for (imgA, imgB), inlier_matches in refined_matches_dict.items():
        imageA = cv2.imread(imgA, cv2.IMREAD_COLOR)
        imageB = cv2.imread(imgB, cv2.IMREAD_COLOR)
        if imageA is None or imageB is None:
            continue
        
        kpsA, _ = sift_results[imgA]
        kpsB, _ = sift_results[imgB]
        
        match_img = cv2.drawMatches(imageA, kpsA, imageB, kpsB, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        window_name = f"RANSAC Matches: {os.path.basename(imgA)} - {os.path.basename(imgB)}"
        cv2.imshow(window_name, match_img)
        
        cv2.waitKey(500)
        cv2.destroyWindow(window_name)
    
    cv2.destroyAllWindows()
