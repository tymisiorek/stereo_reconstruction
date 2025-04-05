import cv2
import numpy as np
import glob
import os
import re

def natural_sort_key(s):
    """
    Sort key for natural/human sorting of filenames
    (e.g. 'tripod_seq_01_001', 'tripod_seq_01_002', ...).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split('([0-9]+)', s)
    ]

def load_images_in_range(folder_pattern="../data/tripod-seq/tripod_seq_02_*.jpg",
                         max_index=128):
    """
    Loads images named tripod_seq_01_###.jpg where ### <= max_index.
    Returns a list of (filename, image) in natural order.
    """
    print(f"[INFO] Loading images from pattern: {folder_pattern}")
    print(f"[INFO] Limiting to images with index ≤ {max_index}")
    
    image_files = sorted(glob.glob(folder_pattern), key=natural_sort_key)
    filtered_files = []
    
    # Filter only those with the numeric portion up to 'max_index'
    for fpath in image_files:
        filename = os.path.basename(fpath)
        # Expect a pattern: tripod_seq_01_###.jpg
        match = re.search(r"tripod_seq_02_(\d+)\.jpg", filename)
        if not match:
            continue  # Not matching the expected pattern
        
        idx_str = match.group(1)
        idx = int(idx_str)
        if idx <= max_index:
            filtered_files.append(fpath)
    
    images = []
    for f in filtered_files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is not None:
            images.append((os.path.basename(f), img))
        else:
            print(f"[WARNING] Unable to load image {f}")
    
    print(f"[INFO] Found {len(images)} images matching your limit.")
    return images

def detect_and_describe(image, sift):
    """
    Detect keypoints and extract SIFT descriptors over the entire image.
    Returns (keypoints, descriptors).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if keypoints is None or descriptors is None:
        return [], None
    return keypoints, descriptors

def match_features(desc1, desc2, ratio_threshold=0.75):
    """
    Match features between two sets of SIFT descriptors using BFMatcher.
    Applies Lowe's ratio test to remove ambiguous matches.
    Returns a list of good matches (cv2.DMatch).
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # knnMatch to get two nearest neighbors
    knn_matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    return good_matches

def extract_matched_keypoints(kp1, kp2, matches):
    """
    Given keypoints in two images and a list of matches,
    return the matched keypoint coordinates as two arrays of shape (N, 2).
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def estimate_pose(pts1, pts2, K):
    """
    Estimate the Essential Matrix using matched points, then recover
    the relative pose (R, t) from that. Assumes known camera intrinsics K.
    Returns (R, t, inlier_mask).
    """
    E, inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                      prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask

def triangulate_points(kp1, kp2, matches, R1, t1, R2, t2, K):
    """
    Triangulate matched keypoints from two camera poses.
    R1, t1: rotation, translation for camera 1
    R2, t2: rotation, translation for camera 2
    Returns a set of 3D points in Euclidean coordinates: shape (N, 3).
    """
    #Projection matrix for camera 1: P1 = K [R1 | t1]
    P1 = K @ np.hstack((R1, t1))
    #Projection matrix for camera 2: P2 = K [R2 | t2]
    P2 = K @ np.hstack((R2, t2))
    
    pts1, pts2 = extract_matched_keypoints(kp1, kp2, matches)
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)  # shape: (4, N)
    pts3D = pts4D[:3, :] / pts4D[3, :]
    return pts3D.T  # shape: (N, 3)

def main():
    print("[INFO] Starting SfM pipeline without bounding boxes ...")
    
    # 1. Load images up to index 128 in the sequence
    images = load_images_in_range("../data/tripod-seq/tripod_seq_02_*.jpg",
                                  max_index=111)
    num_images = len(images)
    if num_images < 2:
        print("[ERROR] Need at least two images to run the pipeline.")
        return
    
    # 2. Camera Intrinsics (dummy; replace with real if known)
    fx, fy = 800.0, 800.0
    cx, cy = 640.0, 360.0
    K = np.array([[fx,   0, cx],
                  [ 0,  fy, cy],
                  [ 0,   0,  1]], dtype=np.float32)
    
    # 3. Create SIFT detector
    print("[INFO] Creating SIFT detector ...")
    sift = cv2.SIFT_create()
    
    # 4. Detect/describe features for each image
    keypoints_list = []
    descriptors_list = []
    for i, (fname, img) in enumerate(images):
        print(f"[INFO] Detecting features for image {i+1}/{num_images}: {fname}")
        kp, desc = detect_and_describe(img, sift)
        if desc is None or len(kp) == 0:
            print(f"[WARNING] No keypoints detected in {fname}")
        keypoints_list.append(kp)
        descriptors_list.append(desc)
    
    # 5. Initialize first camera pose at world origin
    print("[INFO] Initializing first camera pose at origin ...")
    R_prev = np.eye(3, dtype=np.float32)
    t_prev = np.zeros((3, 1), dtype=np.float32)
    camera_poses = [(R_prev, t_prev)]
    
    # 6. Pairwise incremental reconstruction
    print("[INFO] Starting pairwise reconstruction ...")
    global_points_3d = []
    
    for i in range(num_images - 1):
        kp1, desc1 = keypoints_list[i], descriptors_list[i]
        kp2, desc2 = keypoints_list[i+1], descriptors_list[i+1]
        
        if desc1 is None or desc2 is None:
            print(f"[WARNING] Missing descriptors for images {i} or {i+1}. Skipping.")
            camera_poses.append(camera_poses[-1])
            continue
        
        print(f"[INFO] Matching features between images {i+1} and {i+2} ...")
        matches = match_features(desc1, desc2, ratio_threshold=0.75)
        if len(matches) < 8:
            print(f"[WARNING] Not enough matches between images {i} and {i+1}. Skipping pose.")
            camera_poses.append(camera_poses[-1])
            continue
        
        pts1, pts2 = extract_matched_keypoints(kp1, kp2, matches)
        
        print("[INFO] Estimating relative pose ...")
        R_rel, t_rel, inlier_mask = estimate_pose(pts1, pts2, K)
        
        # changed absolute pose
        # R_abs = R_prev @ R_rel
        # t_abs = t_prev + R_prev @ t_rel
        R_abs = R_prev @ R_rel
        t_abs = t_prev + (R_prev @ t_rel)
        
        camera_poses.append((R_abs, t_abs))
        
        print("[INFO] Triangulating 3D points ...")
        pts_3d = triangulate_points(kp1, kp2, matches, R_prev, t_prev, R_abs, t_abs, K)
        
        # Filter out negative or zero depth
        valid_mask = (pts_3d[:, 2] > 0)
        pts_3d = pts_3d[valid_mask]
        global_points_3d.append(pts_3d)
        
        # Update for next iteration
        R_prev, t_prev = R_abs, t_abs
    
    # 7. Combine all 3D points
    if global_points_3d:
        global_points_3d = np.vstack(global_points_3d)
    else:
        global_points_3d = np.zeros((0, 3), dtype=np.float32)
    
    print("[INFO] Reconstruction completed.")
    print(f"[INFO] Total 3D points: {global_points_3d.shape[0]}")
    
    if global_points_3d.shape[0] > 0:
        out_file = "reconstructed_points2.xyz"
        np.savetxt(out_file, global_points_3d, delimiter=" ")
        print(f"[INFO] Saved reconstructed 3D points to {out_file}")

if __name__ == "__main__":
    main()
