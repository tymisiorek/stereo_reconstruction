import os
import cv2
import numpy as np
import json
import io

from PIL import Image
import matplotlib.pyplot as plt

from rembg import remove
from rembg.session_factory import new_session

from util import get_images_for_sequence

# Lightweight model of u2net (will install on first run, then subsequently run from cache)
u2net_lite_session = new_session('u2net_lite')

def remove_background_rembg(image):
    is_success, buffer = cv2.imencode(".png", image)
    if not is_success:
        return image
    
    pil_image = Image.open(io.BytesIO(buffer))
    pil_no_bg = remove(pil_image, session=u2net_lite_session)
    np_no_bg = np.array(pil_no_bg)
    if np_no_bg.shape[-1] == 4:
        alpha = np_no_bg[:, :, 3]
        np_no_bg = np_no_bg[:, :, :3]
        np_no_bg[alpha == 0] = [0, 0, 0]

    no_bg_bgr = cv2.cvtColor(np_no_bg, cv2.COLOR_RGB2BGR)
    return no_bg_bgr

def sift_keypoint_detection(image_paths):
    sift = cv2.SIFT_create()
    results = {}
    for img_path in image_paths:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
        
        image_no_bg = remove_background_rembg(image)
        # Convert background-removed image to grayscale for SIFT
        gray = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        results[img_path] = (keypoints, descriptors)
    return results

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

# --- Helper functions to serialize keypoints and matches ---

def serialize_keypoints(keypoints):
    serialized = []
    for kp in keypoints:
        kp_dict = {
            'pt': kp.pt,          # (x, y)
            'size': kp.size,
            'angle': kp.angle,
            'response': kp.response,
            'octave': kp.octave,
            'class_id': kp.class_id
        }
        serialized.append(kp_dict)
    return serialized

def serialize_matches(matches):
    serialized = []
    for m in matches:
        m_dict = {
            'queryIdx': m.queryIdx,
            'trainIdx': m.trainIdx,
            'imgIdx': m.imgIdx,
            'distance': m.distance
        }
        serialized.append(m_dict)
    return serialized

def save_feature_data(sift_results, refined_matches_dict, folder_path, sequence_id):
    output_dir = os.path.join(folder_path, sequence_id)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "feature_data.json")
    
    data = {}
    # Serialize sift_results for each image.
    sift_data = {}
    for img_path, (keypoints, descriptors) in sift_results.items():
        sift_data[img_path] = {
            'keypoints': serialize_keypoints(keypoints),
            'descriptors': descriptors.tolist() if descriptors is not None else None
        }
    data['sift_results'] = sift_data

    # Serialize refined_matches_dict. Convert the tuple key (imgA, imgB) to a string "imgA::imgB".
    matches_data = {}
    for (imgA, imgB), matches in refined_matches_dict.items():
        key = f"{imgA}::{imgB}"
        matches_data[key] = serialize_matches(matches)
    data['refined_matches_dict'] = matches_data

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Feature data saved to {output_file}")

def load_feature_data(feature_file):
    """
    Load feature data from a JSON file.
    """
    with open(feature_file, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    folder_path = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\tripod-seq'
    sequence_id = "01" 
    paths = get_images_for_sequence(folder_path, sequence_id)

    # 1. Detect keypoints & descriptors (with background removed using rembg)
    sift_results = sift_keypoint_detection(paths)
    
    # 2. Match features between consecutive images
    matches_dict = match_features(sift_results, ratio_threshold=0.8)

    # 3. Apply RANSAC to filter outliers
    refined_matches_dict = apply_ransac(matches_dict, sift_results, ransac_thresh=5.0)
    
    # 4. Save the feature data to a JSON file in the same folder structure.
    save_feature_data(sift_results, refined_matches_dict, folder_path, sequence_id)
    
    # 5. (Optional) Display the inlier matches for each image pair.
    for (imgA, imgB), inlier_matches in refined_matches_dict.items():
        imageA = cv2.imread(imgA, cv2.IMREAD_COLOR)
        imageB = cv2.imread(imgB, cv2.IMREAD_COLOR)
        if imageA is None or imageB is None:
            continue
        
        kpsA, _ = sift_results[imgA]
        kpsB, _ = sift_results[imgB]
        
        match_img = cv2.drawMatches(
            imageA, 
            kpsA, 
            imageB, 
            kpsB, 
            inlier_matches,
            None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        window_name = f"RANSAC Matches: {os.path.basename(imgA)} - {os.path.basename(imgB)}"
        cv2.imshow(window_name, match_img)
        cv2.waitKey(500)
        cv2.destroyWindow(window_name)
    
    cv2.destroyAllWindows()
