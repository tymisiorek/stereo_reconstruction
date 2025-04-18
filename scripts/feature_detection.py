# feature_detection.py
import os, json
import cv2
import numpy as np
import util

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
RATIO_TEST       = 0.8      # Lowe’s ratio
RANSAC_CONF      = 0.99
RANSAC_REPROJ    = 3.0      # in pixels
SAMPSON_THRESH   = 1.0      # prune large‐error inliers
MIN_F_INLIERS    = 30

# -----------------------------------------------------------------------------
# 1.  Detect & describe
# -----------------------------------------------------------------------------
def compute_sift_features(image_paths):
    sift = cv2.SIFT_create(contrastThreshold=0.013, edgeThreshold=10, sigma=1.6)
    features = {}
    for p in image_paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[WARN] Could not read {p}")
            continue
        kps, desc = sift.detectAndCompute(gray, None)
        features[p] = (kps, desc)
        print(f"[FEAT] {os.path.basename(p)} → {len(kps)} keypoints")
    return features

# -----------------------------------------------------------------------------
# 2.  Pairwise match + ratio test
# -----------------------------------------------------------------------------
def match_features(features):
    index_params = dict(algorithm=1, trees=5)
    flann = cv2.FlannBasedMatcher(index_params, dict(checks=50))
    paths = sorted(features.keys())
    matches = {}
    for i in range(len(paths)-1):
        for j in range(i+1, len(paths)):
            a,b = paths[i], paths[j]
            kpsA, dA = features[a]
            kpsB, dB = features[b]
            if dA is None or dB is None:
                matches[(a,b)] = []
                continue
            knn = flann.knnMatch(dA, dB, k=2)
            good = [m for m,n in knn if m.distance < RATIO_TEST*n.distance]
            print(f"[MATCH] {os.path.basename(a)}↔{os.path.basename(b)}: {len(good)}")
            matches[(a,b)] = good
    return matches

# -----------------------------------------------------------------------------
# 3.  Fundamental RANSAC + Sampson prune
# -----------------------------------------------------------------------------
def prune_matches(matches, features):
    refined = {}
    for (a,b), mlist in matches.items():
        kpsA,_ = features[a]
        kpsB,_ = features[b]
        if len(mlist) < MIN_F_INLIERS:
            print(f"[SKIP] {os.path.basename(a)}↔{os.path.basename(b)} (<{MIN_F_INLIERS} matches)")
            continue

        ptsA = np.float32([kpsA[m.queryIdx].pt for m in mlist])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in mlist])

        F, mask = cv2.findFundamentalMat(
            ptsA, ptsB,
            cv2.FM_RANSAC,
            RANSAC_REPROJ, RANSAC_CONF
        )
        if F is None or mask.shape[0] != len(ptsA):
            print(f"[F FAIL] {os.path.basename(a)}↔{os.path.basename(b)}")
            continue

        # Sampson error prune
        idx = mask.ravel()==1
        ptsA_h = np.hstack([ptsA[idx], np.ones((idx.sum(),1))])
        ptsB_h = np.hstack([ptsB[idx], np.ones((idx.sum(),1))])
        Fx1 = (F @ ptsA_h.T)
        Ftx2= (F.T @ ptsB_h.T)
        num = np.sum(ptsB_h*(F @ ptsA_h.T).T, axis=1)**2
        den = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2 + 1e-12
        sampson = num/den
        keep = sampson<=SAMPSON_THRESH

        new_mask = np.zeros_like(mask.ravel(),dtype=bool)
        idxs = np.flatnonzero(idx)
        new_mask[idxs[keep]] = True

        inliers = [m for m,k in zip(mlist,new_mask) if k]
        print(f"[F INL] {os.path.basename(a)}↔{os.path.basename(b)}: {len(inliers)}/{len(mlist)}")
        refined[(a,b)] = inliers
    return refined

# -----------------------------------------------------------------------------
# 4.  Serialize
# -----------------------------------------------------------------------------
def save_feature_data(features, refined, folder):
    out = {}
    # keypoints
    out['sift_results'] = {
        p: [{'pt':kp.pt,'size':kp.size} for kp in kps]
        for p,(kps,_) in features.items()
    }
    # matches
    out['refined_matches_dict'] = {}
    for (a,b),mlist in refined.items():
        kpsA,_ = features[a]
        kpsB,_ = features[b]
        lst = []
        for m in mlist:
            lst.append({
                'queryIdx': m.queryIdx,
                'trainIdx': m.trainIdx,
                'ptA':      kpsA[m.queryIdx].pt,
                'ptB':      kpsB[m.trainIdx].pt
            })
        out['refined_matches_dict'][f"{a}::{b}"] = lst

    path = os.path.join(folder,'feature_data.json')
    with open(path,'w') as f:
        json.dump(out,f,indent=2)
    print(f"[OK] saved → {path}")

# -----------------------------------------------------------------------------
# 5.  Main
# -----------------------------------------------------------------------------
def main():
    parent = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    folder = util.choose_image_set(parent)
    if not folder: return

    imgs = sorted([
        os.path.join(folder,f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.png','.jpg','.jpeg'))
    ])
    print(f"Found {len(imgs)} images")

    feats   = compute_sift_features(imgs)
    matches = match_features(feats)
    refined = prune_matches(matches, feats)
    save_feature_data(feats, refined, folder)

if __name__=='__main__':
    main()
