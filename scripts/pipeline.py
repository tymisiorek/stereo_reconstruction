#!/usr/bin/env python3
# pipeline.py

import os, re, json
import cv2
import numpy as np
import util

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
RANSAC_CONF       = 0.99     # confidence for Essential‐matrix RANSAC
RANSAC_REPROJ     = 0.005    # 0.5% of image diagonal → pixels
MIN_INLIERS       = 30       # drop any pair with fewer pose inliers
MIN_RATIO         = 0.20     # or with inlier ratio below this
VERBOSE           = True     # print debug info

# -----------------------------------------------------------------------------
# HELPERS TO SORT PAIRS IN SEQUENTIAL ORDER
# -----------------------------------------------------------------------------
def extract_index(path):
    """chapel_12.jpeg → 12 (for sorting)."""
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1

def sort_pairs(pair_keys):
    """Sort 'imgA::imgB' by (index(imgA), index(imgB))."""
    def keyfn(k):
        a,b = k.split("::")
        return (extract_index(a), extract_index(b))
    return sorted(pair_keys, key=keyfn)

# -----------------------------------------------------------------------------
# STEP 1: ESTIMATE ESSENTIAL MATRIX + RELATIVE POSE
# -----------------------------------------------------------------------------
def estimate_relposes(folder):
    # — load intrinsics —
    fx, fy = 1826.35890, 1826.55090
    cx, cy =  520.668647, 955.447831
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    if VERBOSE: print(f"[1] Using intrinsics:\n{K}")

    # — load SIFT results + refined matches —
    feat_f = os.path.join(folder, "feature_data.json")
    if not os.path.exists(feat_f):
        raise FileNotFoundError(f"Missing {feat_f}")
    with open(feat_f,'r') as f:
        feat = json.load(f)
    refined_matches = feat["refined_matches_dict"]

    # — dynamic RANSAC threshold (~0.5% of diagonal) —
    sample_img = cv2.imread(next(iter(feat["sift_results"].keys())))
    h, w = sample_img.shape[:2]
    pix_thresh = RANSAC_REPROJ * np.hypot(w, h)
    if VERBOSE:
        print(f"[1] RANSAC reproj‐threshold = {pix_thresh:.2f}px")

    tri_data = {}
    for pair_key, matches in refined_matches.items():
        ptsA = np.asarray([m["ptA"] for m in matches], np.float64)
        ptsB = np.asarray([m["ptB"] for m in matches], np.float64)
        n    = ptsA.shape[0]
        if VERBOSE:
            print(f"\n[1] Pair {pair_key}: {n} raw matches")
        if n < 8:
            if VERBOSE: print("   ↳ too few matches, skipping")
            continue

        # — findEssentialMat over all matches →
        E, maskE = cv2.findEssentialMat(
            ptsA, ptsB, K,
            method=cv2.FM_RANSAC,
            prob=RANSAC_CONF,
            threshold=pix_thresh
        )
        if E is None or maskE is None or maskE.shape[0] != n:
            if VERBOSE: print("   ↳ Essential failed, skipping")
            continue
        maskE = maskE.ravel().astype(bool)
        inE   = maskE.sum()
        if VERBOSE:
            print(f"   ↳ Essential inliers: {inE}/{n}")
        if inE < MIN_INLIERS or (inE / n) < MIN_RATIO:
            if VERBOSE: print("   ↳ too few essential inliers, skipping")
            continue

        # — recoverPose on only the essential inliers →
        ptsA_e = ptsA[maskE]
        ptsB_e = ptsB[maskE]
        _, R_rel, t_rel, maskP = cv2.recoverPose(E, ptsA_e, ptsB_e, K)
        maskP = maskP.ravel().astype(bool)
        inP   = maskP.sum()
        if VERBOSE:
            print(f"   ↳ Pose inliers:     {inP}/{inE}")
        if inP < MIN_INLIERS or (inP / inE) < MIN_RATIO:
            if VERBOSE: print("   ↳ too few pose inliers, skipping")
            continue

        # — build full‐length pose‐mask over original matches →
        full_mask = np.zeros(n, bool)
        idxs = np.nonzero(maskE)[0]
        full_mask[idxs[maskP]] = True

        tri_data[pair_key] = {
            "imgA":                   pair_key.split("::")[0],
            "imgB":                   pair_key.split("::")[1],
            "camera_intrinsics":      K.tolist(),
            "matched_points_imgA":    ptsA.tolist(),
            "matched_points_imgB":    ptsB.tolist(),
            "essential_matrix":       E.tolist(),
            "inlier_mask_essential":  maskE.astype(int).tolist(),
            "recovered_rotation":     R_rel.tolist(),
            "recovered_translation":  t_rel.ravel().tolist(),
            "inlier_mask_pose":       full_mask.astype(int).tolist()
        }

    out1 = os.path.join(folder, "triangulation_data.json")
    with open(out1, "w") as f:
        json.dump(tri_data, f, indent=2)
    print(f"[1] Wrote relative‐pose data → {out1}")
    return tri_data

# -----------------------------------------------------------------------------
# STEP 2: CHAIN INTO METRIC GLOBAL FRAME
# -----------------------------------------------------------------------------
def chain_global_poses(tri_data):
    keys = sort_pairs(tri_data.keys())

    # start first image at identity
    first_img = tri_data[keys[0]]["imgA"]
    extrinsics = {
        first_img: (
            np.eye(3, dtype=np.float64),
            np.zeros((3,1), dtype=np.float64)
        )
    }

    pose_data = {}
    for key in keys:
        info = tri_data[key]
        A, B = info["imgA"], info["imgB"]
        maskP = np.asarray(info["inlier_mask_pose"], np.int32).astype(bool)
        if maskP.sum() < MIN_INLIERS or (maskP.sum()/maskP.size) < MIN_RATIO:
            if VERBOSE: print(f"[2] skip chain {key} ({maskP.sum()}/{maskP.size})")
            continue

        R_rel = np.asarray(info["recovered_rotation"],    np.float64)
        t_rel = np.asarray(info["recovered_translation"], np.float64).reshape(3,1)

        # forward: if we know A, build B
        if A in extrinsics and B not in extrinsics:
            R_A, t_A = extrinsics[A]
            R_B = R_rel @ R_A
            t_B = R_rel @ t_A + t_rel
            extrinsics[B] = (R_B, t_B)

        # backward: if we know B, build A
        elif B in extrinsics and A not in extrinsics:
            R_B, t_B = extrinsics[B]
            R_A = R_rel.T @ R_B
            t_A = R_rel.T @ (t_B - t_rel)
            extrinsics[A] = (R_A, t_A)

        else:
            if VERBOSE:
                print(f"[2] cannot chain {key} (both known or both unknown)")
            continue

        # compute camera centers C = -R^T t
        Rg_A, tg_A = extrinsics[A]
        Rg_B, tg_B = extrinsics[B]
        Cg_A = (-Rg_A.T @ tg_A).ravel().tolist()
        Cg_B = (-Rg_B.T @ tg_B).ravel().tolist()

        info.update({
            "rotation_global_A":      Rg_A.tolist(),
            "camera_center_global_A": Cg_A,
            "rotation_global_B":      Rg_B.tolist(),
            "camera_center_global_B": Cg_B
        })
        pose_data[key] = info

        if VERBOSE:
            print(f"[2] chained {A}→{B}:  C_A={Cg_A}  C_B={Cg_B}")

    out2 = os.path.join(
        os.path.dirname(next(iter(pose_data.values()))["imgA"]),
        "pose_estimation_data.json"
    )
    with open(out2, "w") as f:
        json.dump(pose_data, f, indent=2)
    print(f"[2] Wrote global‑pose data → {out2}")
    return pose_data



def triangulate_and_write(pose_data):
    """
    STEP 3: exactly your original P = K [R | -R C]  +  cheirality test.
    """
    # single K
    K = np.asarray(next(iter(pose_data.values()))["camera_intrinsics"], np.float64)

    all_pts = []
    for key, info in pose_data.items():
        A, B = info["imgA"], info["imgB"]
        R1 = np.asarray(info["rotation_global_A"],      np.float64)
        C1 = np.asarray(info["camera_center_global_A"], np.float64).reshape(3,1)
        R2 = np.asarray(info["rotation_global_B"],      np.float64)
        C2 = np.asarray(info["camera_center_global_B"], np.float64).reshape(3,1)

        # build your exact P1,P2
        P1 = K @ np.hstack((R1, -R1 @ C1))
        P2 = K @ np.hstack((R2, -R2 @ C2))

        ptsA = np.asarray(info["matched_points_imgA"], np.float64)
        ptsB = np.asarray(info["matched_points_imgB"], np.float64)
        mask = np.asarray(info["inlier_mask_pose"], np.int32).ravel().astype(bool)
        ptsA_in, ptsB_in = ptsA[mask], ptsB[mask]
        if len(ptsA_in) < 2:
            if VERBOSE: print(f"[3] skip {key}: <2 inliers")
            continue

        X4 = cv2.triangulatePoints(P1, P2, ptsA_in.T, ptsB_in.T)
        X3 = (X4[:3] / X4[3]).T

        # cheirality in each camera
        t1 = -R1 @ C1
        t2 = -R2 @ C2
        Z1 = ((R1 @ X3.T) + t1).T[:,2]
        Z2 = ((R2 @ X3.T) + t2).T[:,2]
        keep = (Z1 > 0) & (Z2 > 0)

        pts3 = X3[keep]
        info["triangulated_points_3d"]  = pts3.tolist()
        info["num_triangulated_points"] = int(len(pts3))
        all_pts.append(pts3)

        if VERBOSE:
            print(f"[3] {key}: kept {len(pts3)}/{len(X3)} pts")

    out3 = os.path.join(
        os.path.dirname(next(iter(pose_data.values()))["imgA"]),
        "pose_and_triangulation_data.json"
    )
    with open(out3, "w") as f:
        json.dump(pose_data, f, indent=2)
    print(f"[3] Wrote triangulation data → {out3}")

    # unified PLY (exactly as you had it)
    if all_pts:
        cloud = np.vstack(all_pts)
        ply   = os.path.join(os.path.dirname(out3), "cloud.ply")
        with open(ply, "w") as fp:
            fp.write("ply\nformat ascii 1.0\n")
            fp.write(f"element vertex {len(cloud)}\n")
            fp.write("property float x\nproperty float y\nproperty float z\n")
            fp.write("end_header\n")
            for x,y,z in cloud:
                fp.write(f"{x} {y} {z}\n")
        print(f"[3] Wrote PLY → {ply}")
    else:
        print("[3] No points to write to PLY")
# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parent = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    folder = util.choose_image_set(parent)
    if not folder:
        print("No folder selected – exiting.")
        return

    tri_data  = estimate_relposes(folder)
    pose_data = chain_global_poses(tri_data)
    triangulate_and_write(pose_data)

if __name__=="__main__":
    main()
