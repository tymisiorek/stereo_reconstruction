# triangulate.py
import os
import json
import cv2
import numpy as np
import util

# -----------------------------------------------------------------------------
# 1) Build a pose‐dictionary using (R, t) directly from your JSON
#    (translation_global_* is already -R * C)
# -----------------------------------------------------------------------------
def build_pose_dict(pairs_json):
    pose_dict = {}
    for info in pairs_json.values():
        for side in ("A","B"):
            img_key = f"img{side}"
            R_key   = f"rotation_global_{side}"
            t_key   = f"translation_global_{side}"

            if img_key not in info or R_key not in info or t_key not in info:
                raise KeyError(f"Missing {img_key}/{R_key}/{t_key} in JSON entry")

            img = info[img_key]
            Rg  = np.asarray(info[R_key], dtype=np.float64)
            tg  = np.asarray(info[t_key], dtype=np.float64).reshape(3,1)
            pose_dict[img] = (Rg, tg)
    return pose_dict

# -----------------------------------------------------------------------------
# 2) Triangulate one pair, with debug prints
# -----------------------------------------------------------------------------
def triangulate_pair(info, K, pose_dict):
    a, b = os.path.basename(info["imgA"]), os.path.basename(info["imgB"])
    ptsA = np.asarray(info["matched_points_imgA"], np.float64)
    ptsB = np.asarray(info["matched_points_imgB"], np.float64)
    mask = np.asarray(info["inlier_mask_pose"], np.int32).ravel()

    A_in = ptsA[mask==1]
    B_in = ptsB[mask==1]
    print(f"[TRI] {a} ↔ {b}: {len(A_in)} inlier correspondences")

    if len(A_in) < 2:
        return np.zeros((0,3), np.float64)

    R1, t1 = pose_dict[info["imgA"]]
    R2, t2 = pose_dict[info["imgB"]]

    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    print("  P1:\n", P1)
    print("  P2:\n", P2)

    X4 = cv2.triangulatePoints(P1, P2, A_in.T, B_in.T)
    print("  Triangulated homogeneous shape:", X4.shape)

    X3 = (X4[:3]/X4[3]).T
    print("  Converted to Euclidean shape:", X3.shape)

    # Cheirality check
    # Depth in cam1: z = row‑3 of (R1 @ X + t1)
    z1 = (R1[2] @ (X3.T - (-R1.T @ t1))).ravel()
    z2 = (R2[2] @ (X3.T - (-R2.T @ t2))).ravel()
    good = (z1>0) & (z2>0)
    print(f"  In front of cam1: {good.sum()}/{len(good)}, cam2: {good.sum()}/{len(good)}")

    return X3[good]

# -----------------------------------------------------------------------------
# 3) Main: load pose JSON, triangulate all pairs, write JSON + PLY
# -----------------------------------------------------------------------------
def main():
    parent = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    folder = util.choose_image_set(parent)
    if not folder:
        print("No folder selected—exiting.")
        return

    # 3.1 load your pose‐estimation output
    pose_path = os.path.join(folder, "pose_estimation_data.json")
    with open(pose_path,'r') as f:
        pairs_json = json.load(f)

    # 3.2 camera intrinsics
    fx, fy = 1826.35890, 1826.55090
    cx, cy = 520.668647, 955.447831
    K = np.array([[fx,0,cx],
                  [0,fy,cy],
                  [0, 0,  1]], dtype=np.float64)

    # 3.3 build pose dictionary
    pose_dict = build_pose_dict(pairs_json)
    print(f"Loaded global poses for {len(pose_dict)} cameras")

    # 3.4 triangulate each pair
    all_pts = []
    for key, info in pairs_json.items():
        pts3 = triangulate_pair(info, K, pose_dict)
        info["triangulated_points_3d"] = pts3.tolist()
        info["num_triangulated_points"] = int(len(pts3))
        all_pts.append(pts3) 
        print(f"  → kept {len(pts3)} points\n")

    # 3.5 write merged JSON
    out_json = os.path.join(folder, "pose_and_triangulation_data.json")
    with open(out_json,'w') as f:
        json.dump(pairs_json, f, indent=2)
    print(f"[✓] Wrote merged data → {out_json}")

    # 3.6 write PLY
    all_pts = np.vstack(all_pts) if all_pts else np.zeros((0,3))
    ply_path = os.path.join(folder, "cloud.ply")
    with open(ply_path,'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(all_pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x,y,z in all_pts:
            f.write(f"{x} {y} {z}\n")
    print(f"[✓] Wrote point‐cloud → {ply_path}")

if __name__=="__main__":
    main()
