# estimate_matrix.py
import os, json
import cv2, numpy as np
import util

# thresholds (reuse feature_detection settings)
SAMPSON_THRESH = 1.0
MIN_INLIERS    = 30
RANSAC_CONF    = 0.99
RANSAC_REPROJ  = 3.0

def load_features(folder):
    with open(os.path.join(folder,'feature_data.json')) as f:
        data = json.load(f)
    return data['sift_results'], data['refined_matches_dict']

def process_pairs(sift_results, refined, K):
    out = {}
    for key, matches in refined.items():
        a,b = key.split("::")
        kpA = sift_results[a]
        kpB = sift_results[b]

        ptsA = np.float32([m['ptA'] for m in matches])
        ptsB = np.float32([m['ptB'] for m in matches])
        n    = len(ptsA)
        if n < MIN_INLIERS:
            continue

        # F
        F, mF = cv2.findFundamentalMat(ptsA, ptsB,
                         cv2.FM_RANSAC, RANSAC_REPROJ, RANSAC_CONF)
        if F is None or mF.shape[0] != n:
            continue
        # E
        E, mE = cv2.findEssentialMat(ptsA, ptsB, K,
                         cv2.FM_RANSAC, RANSAC_CONF, RANSAC_REPROJ)
        if E is None or mE.shape[0] != n:
            continue

        # recoverPose
        _,R,t,mP = cv2.recoverPose(E, ptsA, ptsB, K, mask=mE)
        out[key] = {
            'imgA': a, 'imgB': b,
            'camera_intrinsics': K.tolist(),
            'matched_points_imgA': ptsA.tolist(),
            'matched_points_imgB': ptsB.tolist(),
            'fundamental_matrix': F.tolist(),
            'inlier_mask_fundamental': mF.ravel().astype(int).tolist(),
            'essential_matrix': E.tolist(),
            'inlier_mask_essential': mE.ravel().astype(int).tolist(),
            'recovered_rotation': R.tolist(),
            'recovered_translation': t.ravel().tolist(),
            'inlier_mask_pose': mP.ravel().astype(int).tolist()
        }
        print(f"[EST] {os.path.basename(a)}→{os.path.basename(b)}: pose inliers {int(mP.sum())}/{n}")
    return out

def main():
    parent = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    folder = util.choose_image_set(parent)
    if not folder: return

    # load intrinsics
    fx,fy,cx,cy = 1826.35890,1826.55090,520.668647,955.447831
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float64)

    sift_results, refined = load_features(folder)
    tri_data = process_pairs(sift_results, refined, K)

    out = os.path.join(folder,'triangulation_data.json')
    with open(out,'w') as f:
        json.dump(tri_data,f,indent=2)
    print(f"[OK] wrote → {out}")

if __name__=='__main__':
    main()
