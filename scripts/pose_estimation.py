import os
import json
import cv2
import numpy as np

from feature_detection import load_feature_data
from util import get_images_for_sequence

def convert_keypoint_dicts_to_cv2(keypoint_dicts):
    """
    Convert a list of serialized keypoint dictionaries back into cv2.KeyPoint objects.
    """
    keypoints = []
    for kp_dict in keypoint_dicts:
        kp = cv2.KeyPoint(
            kp_dict['pt'][0],
            kp_dict['pt'][1],
            kp_dict['size'],
            kp_dict['angle'],
            kp_dict['response'],
            int(kp_dict['octave']),
            int(kp_dict['class_id'])
        )
        keypoints.append(kp)
    return keypoints

def get_keypoints_and_descriptors(sift_results_dict, img_path):
    """
    Convenience function to get keypoints and descriptors for a given image.
    """
    kp_desc_dict = sift_results_dict[img_path]
    kp_cv2 = convert_keypoint_dicts_to_cv2(kp_desc_dict['keypoints'])
    desc = np.array(kp_desc_dict['descriptors'], dtype=np.float32) if kp_desc_dict['descriptors'] is not None else None
    return kp_cv2, desc

def find_essential_and_pose(kpsA, kpsB, matches, K):
    """
    Given matched keypoints between two images, compute the Essential matrix and recover the relative pose.
    
    Returns:
      R, t, mask
    """
    ptsA = []
    ptsB = []
    for m in matches:
        ptsA.append(kpsA[m['queryIdx']].pt)
        ptsB.append(kpsB[m['trainIdx']].pt)
    ptsA = np.array(ptsA, dtype=np.float32)
    ptsB = np.array(ptsB, dtype=np.float32)
    
    E, mask = cv2.findEssentialMat(ptsA, ptsB, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        print("Could not compute Essential matrix.")
        return None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, ptsA, ptsB, K)
    return R, t, mask_pose

def triangulate_points_from_matches(kpsA, kpsB, matches, R, t, K):
    """
    Triangulate 3D points from matched keypoints between two images.
    
    Parameters:
      kpsA (list): cv2.KeyPoint objects from image A.
      kpsB (list): cv2.KeyPoint objects from image B.
      matches (list): List of match dictionaries (with 'queryIdx' and 'trainIdx').
      R (np.ndarray): Relative rotation matrix from image A to image B.
      t (np.ndarray): Relative translation vector from image A to image B.
      K (np.ndarray): Camera intrinsic matrix.
    
    Returns:
      points3D_dict (dict): Dictionary mapping keypoint indices from image A to triangulated 3D point [X, Y, Z].
    """
    # Build projection matrices.
    projMat1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    projMat2 = K @ np.hstack((R, t))
    
    ptsA = []
    ptsB = []
    match_indices = []  # Track which keypoint in image A each point comes from.
    
    for m in matches:
        ptsA.append(kpsA[m['queryIdx']].pt)
        ptsB.append(kpsB[m['trainIdx']].pt)
        match_indices.append(m['queryIdx'])
    
    if len(ptsA) == 0:
        print("No valid points for triangulation.")
        return {}
    
    # Convert to NumPy arrays with shape (2, N)
    ptsA = np.array(ptsA, dtype=np.float32).T
    ptsB = np.array(ptsB, dtype=np.float32).T
    
    # Triangulate using OpenCV's function.
    pts4D = cv2.triangulatePoints(projMat1, projMat2, ptsA, ptsB)
    # Convert from homogeneous to Euclidean coordinates.
    pts3D = pts4D[:3] / (pts4D[3] + 1e-8)
    pts3D = pts3D.T  # shape: (N, 3)
    
    # Build a dictionary mapping image A keypoint index to its 3D point.
    points3D_dict = {}
    for idx, pt3d in zip(match_indices, pts3D):
        points3D_dict[idx] = pt3d.tolist()
    
    return points3D_dict

def main():
    # Set up folder path and sequence ID.
    folder_path = r"C:\Projects\Semester6\CS4501\stereo_reconstruction\data\tripod-seq"
    sequence_id = "01"
    
    # Build path to the feature data JSON file.
    feature_file = os.path.join(folder_path, sequence_id, "feature_data.json")
    
    # Load the feature data.
    data = load_feature_data(feature_file)
    sift_results_dict = data['sift_results']
    refined_matches_dict = data['refined_matches_dict']
    
    # Get sorted image paths.
    all_img_paths = sorted(sift_results_dict.keys())
    if len(all_img_paths) < 2:
        print("Not enough images to perform triangulation.")
        return
    
    # Use the first two images.
    img0 = all_img_paths[0]
    img1 = all_img_paths[1]
    
    # Build key for refined matches. Keys are stored as "imgA::imgB" or vice versa.
    key1 = f"{img0}::{img1}"
    key2 = f"{img1}::{img0}"
    if key1 in refined_matches_dict:
        matches_01 = refined_matches_dict[key1]
    elif key2 in refined_matches_dict:
        matches_01 = refined_matches_dict[key2]
    else:
        print("No matches found between the first two images.")
        return
    
    if len(matches_01) < 8:
        print("Not enough matches between the first two images for reliable triangulation.")
        return
    
    # Retrieve keypoints and descriptors.
    kps0, _ = get_keypoints_and_descriptors(sift_results_dict, img0)
    kps1, _ = get_keypoints_and_descriptors(sift_results_dict, img1)
    
    # Define camera intrinsics.
    camera_matrix = np.array([
        [800,   0, 320],
        [  0, 800, 240],
        [  0,   0,   1]
    ], dtype=np.float32)
    
    # Compute relative pose between img0 and img1.
    R, t, mask_pose = find_essential_and_pose(kps0, kps1, matches_01, camera_matrix)
    if R is None or t is None:
        print("Failed to recover relative pose between the first two images.")
        return
    
    print("Recovered relative pose:")
    print("Rotation (R):\n", R)
    print("Translation (t):\n", t)
    
    # Triangulate 3D points.
    points3D_dict = triangulate_points_from_matches(kps0, kps1, matches_01, R, t, camera_matrix)
    print(f"\nTriangulated {len(points3D_dict)} 3D points.")
    
    # --- Visualization ---
    # Reproject the triangulated 3D points onto the first image.
    # For image 0, the pose is identity (rvec = [0,0,0], tvec = [0,0,0]).
    rvec0 = np.zeros((3, 1), dtype=np.float32)
    tvec0 = np.zeros((3, 1), dtype=np.float32)
    
    # Collect all 3D points into a NumPy array.
    pts3D = []
    for idx, pt in points3D_dict.items():
        pts3D.append(pt)
    pts3D = np.array(pts3D, dtype=np.float32)  # shape: (N, 3)
    
    # Project these 3D points onto image 0.
    projected_points, _ = cv2.projectPoints(pts3D, rvec0, tvec0, camera_matrix, None)
    projected_points = projected_points.reshape(-1, 2)
    
    # Load the first image.
    image0 = cv2.imread(img0, cv2.IMREAD_COLOR)
    if image0 is None:
        print(f"Failed to load image: {img0}")
        return
    
    # Draw each projected point on the image.
    for pt in projected_points:
        cv2.circle(image0, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
    
    # Display the image with reprojected points.
    cv2.imshow("Triangulated Points Reprojected on Image 0", image0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
