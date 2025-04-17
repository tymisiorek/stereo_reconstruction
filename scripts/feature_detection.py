import os
import cv2
import numpy as np
import json
import util


# ────────────────────────────────────────────────────────────────────────────
# 1.  SIFT feature extraction
# ────────────────────────────────────────────────────────────────────────────
def compute_sift_features(image_paths):
    """
    Detect SIFT key‑points & descriptors for each image.
    Returns:
        features : dict  image_path → (keypoints, descriptors)
        gray_cache : dict image_path → grayscale image
    """
    cv2.setUseOptimized(True)
    sift = cv2.SIFT_create(
        nfeatures=0,           # keep all
        contrastThreshold=0.013,
        edgeThreshold=10,
        sigma=1.6
    )

    features, gray_cache = {}, {}
    for img_path in image_paths:
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[WARN] Could not read {img_path} – skipped.")
            continue

        gray_cache[img_path] = gray
        kps, descs = sift.detectAndCompute(gray, None)
        features[img_path] = (kps, descs)
        print(f"{img_path}: {len(kps)} features.")

    return features, gray_cache


# ────────────────────────────────────────────────────────────────────────────
# 2.  Feature matching (all image pairs)
# ────────────────────────────────────────────────────────────────────────────
def match_features(features, ratio_threshold=0.8):
    """
    Match descriptors for *every* pair of images using FLANN + Lowe ratio test.
    Returns:
        matches_dict : {(imgA, imgB)} → list[cv2.DMatch]
    """
    # FLANN params for SIFT (KD‑Tree)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    image_paths = sorted(features.keys())
    matches_dict = {}

    for i in range(len(image_paths) - 1):
        for j in range(i + 1, len(image_paths)):
            imgA, imgB = image_paths[i], image_paths[j]
            kpsA, descA = features[imgA]
            kpsB, descB = features[imgB]

            if descA is None or descB is None:
                matches_dict[(imgA, imgB)] = []
                print(f"Match {imgA} ↔ {imgB}: 0 (missing desc).")
                continue

            knn = flann.knnMatch(descA, descB, k=2)
            good = [m for m, n in knn if m.distance < ratio_threshold * n.distance]
            matches_dict[(imgA, imgB)] = good
            print(f"Match {imgA} ↔ {imgB}: {len(good)} good.")

    return matches_dict


# ────────────────────────────────────────────────────────────────────────────
# 3.  RANSAC with Fundamental matrix (epipolar geometry)
# ────────────────────────────────────────────────────────────────────────────
def apply_ransac(matches_dict, features, ransac_thresh=1.0):
    """
    Robustly estimate a Fundamental matrix for each pair, keep inlier matches.
    Works for arbitrary 3‑D scenes (no planar assumption).
    Returns:
        refined_matches_dict : {(imgA, imgB)} → inlier list[cv2.DMatch]
    """
    refined = {}

    for (imgA, imgB), good in matches_dict.items():
        kpsA, _ = features[imgA]
        kpsB, _ = features[imgB]

        if len(good) < 8:            # 8‑point alg. minimum
            refined[(imgA, imgB)] = []
            print(f"RANSAC {imgA} ↔ {imgB}: <8 matches.")
            continue

        ptsA = np.float32([kpsA[m.queryIdx].pt for m in good])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in good])

        F, mask = cv2.findFundamentalMat(
            ptsA, ptsB, cv2.FM_RANSAC, ransac_thresh, 0.99
        )
        if mask is None:
            refined[(imgA, imgB)] = []
            print(f"RANSAC {imgA} ↔ {imgB}: F‑matrix failed.")
            continue

        inliers = [gm for gm, mk in zip(good, mask.ravel()) if mk]
        refined[(imgA, imgB)] = inliers
        print(f"RANSAC {imgA} ↔ {imgB}: {len(inliers)} inliers.")

    return refined


# ────────────────────────────────────────────────────────────────────────────
# 4.  JSON helpers
# ────────────────────────────────────────────────────────────────────────────
def serialize_keypoints(kps):
    return [{'pt': kp.pt, 'size': kp.size} for kp in kps]


def serialize_matches(matches):
    return [{'queryIdx': m.queryIdx, 'trainIdx': m.trainIdx} for m in matches]


def save_feature_data(features, refined, out_folder):
    """
    Write SIFT key‑points (pt & size) and inlier matches to feature_data.json.
    """
    out_file = os.path.join(out_folder, "feature_data.json")
    data = {
        'sift_results': {
            img: {'keypoints': serialize_keypoints(kps)}
            for img, (kps, _) in features.items()
        },
        'refined_matches_dict': {
            f"{imgA}::{imgB}": serialize_matches(m)
            for (imgA, imgB), m in refined.items()
        }
    }
    with open(out_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Feature data → {out_file}")


# ────────────────────────────────────────────────────────────────────────────
# 5.  Main driver
# ────────────────────────────────────────────────────────────────────────────
def main():
    cv2.setUseOptimized(True)

    parent_dir = r'C:\Users\Owen-McKenney\OneDrive\Desktop\CS4501 Computer Vision\stereo_reconstruction\data\images'
    chosen_folder = util.choose_image_set(parent_dir)
    if not chosen_folder:
        print("No folder selected – exiting.")
        return

    valid_exts = ('.png', '.jpg', '.jpeg')
    image_paths = sorted(
        os.path.join(chosen_folder, f)
        for f in os.listdir(chosen_folder)
        if f.lower().endswith(valid_exts)
    )
    if not image_paths:
        print("No images found – exiting.")
        return

    print(f"Processing {len(image_paths)} images …")

    # 1. SIFT
    features, _ = compute_sift_features(image_paths)

    # 2. Matching
    matches_dict = match_features(features, ratio_threshold=0.8)

    # 3. Epipolar‑RANSAC
    refined_matches = apply_ransac(matches_dict, features, ransac_thresh=1.0)

    # 4. Save
    save_feature_data(features, refined_matches, chosen_folder)

    # (Optional) visualise inliers
    for (imgA, imgB), inliers in refined_matches.items():
        if not inliers:
            continue

        imgA_color = cv2.imread(imgA)
        imgB_color = cv2.imread(imgB)
        if imgA_color is None or imgB_color is None:
            continue

        kpsA, _ = features[imgA]
        kpsB, _ = features[imgB]

        vis = cv2.drawMatches(
            imgA_color, kpsA,
            imgB_color, kpsB,
            inliers, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.namedWindow("Inlier matches", cv2.WINDOW_NORMAL)
        cv2.imshow("Inlier matches", vis)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
