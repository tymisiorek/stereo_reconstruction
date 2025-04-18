import os, re
import sys
import json
import cv2
import numpy as np
import util


RANSAC_CONF    = 0.99
RANSAC_REPROJ  = 0.005
MIN_INLIERS    = 30
MIN_RATIO      = 0.20
MAX_MATCHES    = 100000

# If True, only process one pair and then stop.
BOOTSTRAP_ONLY = True

# You can hard‑code a pair key here (exactly as "pathA::pathB").
# If None, the script will list all pairs and ask you to pick one.
BOOTSTRAP_PAIR = None

def extract_index(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1

def sort_pairs(pair_keys):
    return sorted(pair_keys,
                  key=lambda k: (extract_index(k.split("::")[0]),
                                 extract_index(k.split("::")[1])))


def estimate_relposes(folder):
    fx, fy = 1826.35890, 1826.55090
    cx, cy =  520.668647, 955.447831
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)
    print(f"[1] Using intrinsics:\n{K}")

    # load feature_data.json
    feat = json.load(open(os.path.join(folder, "feature_data.json")))
    matches_dict = feat["refined_matches_dict"]

    # dynamic reproj threshold
    sample = cv2.imread(next(iter(feat["sift_results"])))
    thresh = RANSAC_REPROJ * np.hypot(*sample.shape[:2])
    print(f"[1] RANSAC reproj‑threshold = {thresh:.1f}px")

    # get sorted list of all candidate pairs
    all_pairs = sort_pairs(matches_dict.keys())

    # if bootstrap‑only, choose exactly one
    if BOOTSTRAP_ONLY:
        global BOOTSTRAP_PAIR
        if BOOTSTRAP_PAIR is None:
            print("\nAvailable image pairs:")
            for i, p in enumerate(all_pairs, 1):
                A, B = p.split("::")
                print(f"  {i}. {os.path.basename(A)} ↔ {os.path.basename(B)}")
            sel = input("Enter the pair number to bootstrap: ")
            try:
                idx = int(sel) - 1
                BOOTSTRAP_PAIR = all_pairs[idx]
            except Exception:
                print("Invalid selection. Exiting.")
                sys.exit(1)
        if BOOTSTRAP_PAIR not in all_pairs:
            print(f"BOOTSTRAP_PAIR '{BOOTSTRAP_PAIR}' not found. Exiting.")
            sys.exit(1)
        pair_list = [BOOTSTRAP_PAIR]
        print(f"\n[1] Bootstrapping only on: {BOOTSTRAP_PAIR}")
    else:
        pair_list = all_pairs

    tri_data = {}
    for pair in pair_list:
        matches = matches_dict[pair]
        ptsA = np.float64([m["ptA"] for m in matches])
        ptsB = np.float64([m["ptB"] for m in matches])

        if len(ptsA) > MAX_MATCHES:
            sel = np.random.choice(len(ptsA), MAX_MATCHES, replace=False)
            ptsA, ptsB = ptsA[sel], ptsB[sel]

        if len(ptsA) < 8:
            print(f"[1] Pair {pair}: too few matches ({len(ptsA)}), skipping")
            continue

        E, maskE = cv2.findEssentialMat(
            ptsA, ptsB, K,
            method=cv2.FM_RANSAC,
            prob=RANSAC_CONF,
            threshold=thresh
        )
        maskE = maskE.ravel().astype(bool) if maskE is not None else np.zeros(len(ptsA),bool)
        if maskE.sum() < MIN_INLIERS or maskE.mean() < MIN_RATIO:
            print(f"[1] Pair {pair}: too few essential inliers ({maskE.sum()}/{len(ptsA)})")
            continue

        _, R_rel, t_rel, maskP = cv2.recoverPose(E, ptsA[maskE], ptsB[maskE], K)
        maskP = maskP.ravel().astype(bool)
        if maskP.sum() < MIN_INLIERS or maskP.mean() < MIN_RATIO:
            print(f"[1] Pair {pair}: too few pose inliers ({maskP.sum()}/{maskE.sum()})")
            continue

        # build full‑length pose mask
        full_mask = np.zeros(len(ptsA), bool)
        idxE = np.nonzero(maskE)[0]
        full_mask[idxE[maskP]] = True

        tri_data[pair] = {
            "imgA": pair.split("::")[0],
            "imgB": pair.split("::")[1],
            "camera_intrinsics": K.tolist(),
            "matched_points_imgA": ptsA.tolist(),
            "matched_points_imgB": ptsB.tolist(),
            "inlier_mask_pose": full_mask.astype(int).tolist(),
            "recovered_rotation": R_rel.tolist(),
            "recovered_translation": t_rel.ravel().tolist()
        }

        print(f"[1] kept {pair} ({maskP.sum()}/{len(ptsA)})")
        # no need to break here—pair_list already has just one if bootstrap

    out1 = os.path.join(folder, "triangulation_data.json")
    with open(out1, "w") as f:
        json.dump(tri_data, f, indent=2)
    print(f"[1] Wrote → {out1}\n")
    return tri_data


def chain_global_poses(tri_data):
    keys = sort_pairs(tri_data.keys())
    first = tri_data[keys[0]]["imgA"]
    extr  = { first: (np.eye(3), np.zeros((3,1))) }
    pose  = {}

    for k in keys:
        d = tri_data[k]
        A,B = d["imgA"], d["imgB"]
        mask = np.array(d["inlier_mask_pose"], bool)
        if mask.sum() < MIN_INLIERS or mask.mean() < MIN_RATIO:
            continue

        Rr = np.array(d["recovered_rotation"])
        tr = np.array(d["recovered_translation"]).reshape(3,1)

        if A in extr and B not in extr:
            RA, tA = extr[A]
            RB = Rr @ RA; tB = Rr @ tA + tr
            extr[B] = (RB, tB)
        elif B in extr and A not in extr:
            RB, tB = extr[B]
            RA = Rr.T @ RB; tA = Rr.T @ (tB - tr)
            extr[A] = (RA, tA)
        else:
            continue

        RgA, tgA = extr[A]; CgA = (-RgA.T @ tgA).ravel().tolist()
        RgB, tgB = extr[B]; CgB = (-RgB.T @ tgB).ravel().tolist()

        d.update({
            "rotation_global_A":      RgA.tolist(),
            "camera_center_global_A": CgA,
            "rotation_global_B":      RgB.tolist(),
            "camera_center_global_B": CgB
        })
        pose[k] = d
        print(f"[2] chained {A} → {B}")

    out2 = os.path.join(os.path.dirname(pose[keys[0]]["imgA"]),
                        "pose_estimation_data.json")
    with open(out2, "w") as f:
        json.dump(pose, f, indent=2)
    print(f"[2] Wrote → {out2}\n")
    return pose


def triangulate_and_write(pose_data):
    K = np.array(next(iter(pose_data.values()))["camera_intrinsics"])
    all_pts, all_cols = [], []

    for k, d in pose_data.items():
        ptsA = np.array(d["matched_points_imgA"], float)
        ptsB = np.array(d["matched_points_imgB"], float)
        mask = np.array(d["inlier_mask_pose"], bool)

        ptsA_i, ptsB_i = ptsA[mask], ptsB[mask]
        if len(ptsA_i) < 2:
            continue

        R1, C1 = np.array(d["rotation_global_A"]), np.array(d["camera_center_global_A"]).reshape(3,1)
        R2, C2 = np.array(d["rotation_global_B"]), np.array(d["camera_center_global_B"]).reshape(3,1)

        P1 = K @ np.hstack((R1, -R1@C1))
        P2 = K @ np.hstack((R2, -R2@C2))

        X4 = cv2.triangulatePoints(P1, P2, ptsA_i.T, ptsB_i.T)
        X3 = (X4[:3]/X4[3]).T

        Z1 = ((R1 @ X3.T) + (-R1@C1)).T[:,2]
        Z2 = ((R2 @ X3.T) + (-R2@C2)).T[:,2]
        keep = (Z1>0) & (Z2>0)
        cloud_pts = X3[keep]
        all_pts.append(cloud_pts)

        imgA = cv2.imread(d["imgA"])
        vis2d = ptsA_i[keep].astype(int)
        cols  = [ imgA[v,u].tolist() for u,v in vis2d ]
        all_cols.append(cols)

        print(f"[3] {k}: kept {len(cloud_pts)} pts")

    if not all_pts:
        print("[3] No points to write – check your bootstrap pair or thresholds.")
        return

    cloud  = np.vstack(all_pts)
    colors = np.vstack(all_cols)

    ply = os.path.join(os.path.dirname(next(iter(pose_data.values()))["imgA"]),
                       "cloud_colored.ply")
    with open(ply, "w") as fp:
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {len(cloud)}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fp.write("end_header\n")
        for (x,y,z),(b,g,r) in zip(cloud, colors):
            fp.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"[3] Wrote colored PLY → {ply}")

# -----------------------------------------------------------------------------
def main():
    parent = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    folder = util.choose_image_set(parent)
    if not folder:
        print("No folder selected – exiting.")
        return

    tri_data  = estimate_relposes(folder)
    if not tri_data:
        print("[!] No valid relative‑pose pair found.")
        return

    pose_data = chain_global_poses(tri_data)
    if not pose_data:
        print("[!] Failed to chain global poses.")
        return

    triangulate_and_write(pose_data)

if __name__ == "__main__":
    main()
