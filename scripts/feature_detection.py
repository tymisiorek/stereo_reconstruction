# feature_detection.py
import os, json
import cv2
import numpy as np
import util

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
RATIO_TEST     = .99
RANSAC_CONF    = 0.5
RANSAC_REPROJ  = 10.0
SAMPSON_THRESH = 1.0
MIN_F_INLIERS  = 10

# NEW FLAG: set True to bypass Sampson pruning
SKIP_SAMPSON   = True

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

def match_features(features):
    index_params = dict(algorithm=1, trees=5)
    flann = cv2.FlannBasedMatcher(index_params, dict(checks=50))
    paths = sorted(features.keys())
    matches = {}
    for i in range(len(paths)-1):
        for j in range(i+1, len(paths)):
            a, b = paths[i], paths[j]
            kpsA, dA = features[a]
            kpsB, dB = features[b]
            if dA is None or dB is None:
                matches[(a,b)] = []
                continue
            knn = flann.knnMatch(dA, dB, k=2)
            good = [m for m,n in knn if m.distance < RATIO_TEST * n.distance]
            print(f"[MATCH] {os.path.basename(a)}↔{os.path.basename(b)}: {len(good)}")
            matches[(a,b)] = good
    return matches

def prune_matches(matches, features):
    refined = {}
    for (a,b), mlist in matches.items():
        kpsA, _ = features[a]
        kpsB, _ = features[b]
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

        mask = mask.ravel().astype(bool)

        if SKIP_SAMPSON:
            inliers = [m for m,k in zip(mlist, mask) if k]
            print(f"[F INL] {os.path.basename(a)}↔{os.path.basename(b)}: {len(inliers)}/{len(mlist)} (skipped Sampson)")
        else:
            # Sampson error prune
            idx = mask
            ptsA_h = np.hstack([ptsA[idx], np.ones((idx.sum(),1))])
            ptsB_h = np.hstack([ptsB[idx], np.ones((idx.sum(),1))])
            Fx1   = (F @ ptsA_h.T)
            Ftx2  = (F.T @ ptsB_h.T)
            num   = np.sum(ptsB_h * (F @ ptsA_h.T).T, axis=1)**2
            den   = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2 + 1e-12
            sampson = num/den
            keep    = sampson <= SAMPSON_THRESH

            new_mask = np.zeros_like(mask, dtype=bool)
            idxs = np.flatnonzero(idx)
            new_mask[idxs[keep]] = True

            inliers = [m for m,k in zip(mlist,new_mask) if k]
            print(f"[F INL] {os.path.basename(a)}↔{os.path.basename(b)}: {len(inliers)}/{len(mlist)}")

        # visualize inliers full-screen
        imgA = cv2.imread(a)
        imgB = cv2.imread(b)
        vis = cv2.drawMatches(
            imgA, kpsA, imgB, kpsB, inliers, None,
            matchColor=(0,255,0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        win = f"Inliers {os.path.basename(a)}↔{os.path.basename(b)}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(win, vis)
        cv2.waitKey(0)
        cv2.destroyWindow(win)

        refined[(a,b)] = inliers

    return refined

def save_feature_data(features, refined, folder):
    out = {}
    out['sift_results'] = {
        p: [{'pt': kp.pt, 'size': kp.size} for kp in kps]
        for p,(kps,_) in features.items()
    }
    out['refined_matches_dict'] = {}
    for (a,b), mlist in refined.items():
        kpsA, _ = features[a]
        kpsB, _ = features[b]
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

def main():
    parent = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    folder = util.choose_image_set(parent)
    if not folder: return

    imgs = sorted([
        os.path.join(folder,f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.png','.jpg','.jpeg'))
    ])
    print(f"[INFO] Found {len(imgs)} images")

    feats   = compute_sift_features(imgs)
    matches = match_features(feats)
    refined = prune_matches(matches, feats)
    save_feature_data(feats, refined, folder)

if __name__=='__main__':
    main()
