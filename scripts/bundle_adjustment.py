# ────────────────────── triangulate_and_ba.py ────────────────────────────
"""
1. Load pose_estimation_data.json (global poses + matches)
2. For every image pair:
      • build global projection matrices
      • filter pose‑inliers, triangulate, cheirality‑filter
      • subsample & run a *local* 2‑view bundle adjustment (BA)
3. Append refined pose / points back into JSON
4. Dump pose_and_triangulation_data.json and an aggregate PLY
"""
import os, json, cv2, numpy as np, util
from scipy.optimize import least_squares

SUBSET_SIZE = 2000        # max points per pair in BA
MIN_INLIERS_BA = 6        # need at least this many after cheirality
SKIP_RMS_THRESH = 0.5     # skip BA if pre-BA RMS < 0.5px
ZERO3       = np.zeros(3, dtype=np.float64)

# ──────────────────────────────────────────────────────────────────────────
# Rodrigues helpers
# ──────────────────────────────────────────────────────────────────────────
def m2rvec(R):  return cv2.Rodrigues(R)[0].ravel()
def rvec2m(r):  return cv2.Rodrigues(r)[0]

# ──────────────────────────────────────────────────────────────────────────
# project 3D→2D (no distortion)
# ──────────────────────────────────────────────────────────────────────────
def project(X, K, R, t):
    Xc = R @ X.T + t[:,None]        # 3×N
    uv = (K @ Xc)[:2] / Xc[2]       # 2×N
    return uv.T                     # N×2

# ──────────────────────────────────────────────────────────────────────────
# build residual + sparsity for BA (camera1 fixed)
# ──────────────────────────────────────────────────────────────────────────
def make_ba_fun(K, pts1, pts2, n):
    def residual(p):
        r2, t2, X = p[:3], p[3:6], p[6:].reshape(n,3)
        R2 = rvec2m(r2)
        proj1 = project(X, K, np.eye(3), ZERO3)
        proj2 = project(X, K, R2, t2)
        return np.hstack([(proj1-pts1).ravel(),
                          (proj2-pts2).ravel()])
    # sparsity mask: 4n residuals × (6 + 3n) parameters
    J = np.zeros((4*n, 6 + 3*n), dtype=int)
    J[2*n:, :6] = 1
    for i in range(n):
        J[2*i:2*i+2,     6+3*i:6+3*i+3] = 1
        J[2*n+2*i:2*n+2*i+2, 6+3*i:6+3*i+3] = 1
    return residual, J

# ──────────────────────────────────────────────────────────────────────────
# process one pair: triangulate + BA
# ──────────────────────────────────────────────────────────────────────────
def process_pair(info, pose_dict):
    K = np.asarray(info["camera_intrinsics"], np.float64)
    R1, t1 = pose_dict[info["imgA"]]
    R2, t2 = pose_dict[info["imgB"]]

    ptsA = np.asarray(info["matched_points_imgA"], np.float64)
    ptsB = np.asarray(info["matched_points_imgB"], np.float64)
    mask = np.asarray(info["inlier_mask_pose"], np.int32)

    # filter to pose inliers
    ptsA, ptsB = ptsA[mask==1], ptsB[mask==1]
    if len(ptsA) < MIN_INLIERS_BA:
        print("  <6 inliers – skipping")
        return None

    # triangulate in global frame
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    X4 = cv2.triangulatePoints(P1, P2, ptsA.T, ptsB.T)
    X3 = (X4[:3]/X4[3]).T

    # cheirality filter
    z1 = (R1[2] @ (X3.T - t1))
    z2 = (R2[2] @ (X3.T - t2))
    keep = (z1>0) & (z2>0)
    X3, ptsA, ptsB = X3[keep], ptsA[keep], ptsB[keep]
    n0 = len(X3)
    if n0 < MIN_INLIERS_BA:
        print("  <6 valid after cheirality – skipping")
        return None

    # initial reprojection RMS (on all valid)
    fun0, _ = make_ba_fun(K, ptsA, ptsB, n0)
    p0 = np.hstack([m2rvec(R2), t2.ravel(), X3.ravel()])
    rms0 = np.sqrt(np.mean(fun0(p0)**2))
    print(f"  pre-BA RMS = {rms0:.3f}px  ({n0} points)")

    # skip BA if already low error
    if rms0 < SKIP_RMS_THRESH:
        return {
            **info,
            "triangulated_points_3d": X3.tolist(),
            "num_triangulated_points": n0,
            "rms_before": rms0,
            "rms_after": rms0,
            "ba_performed": False
        }

    # subsample to at most SUBSET_SIZE
    if n0 > SUBSET_SIZE:
        idx = np.random.choice(n0, SUBSET_SIZE, replace=False)
        X3, ptsA, ptsB = X3[idx], ptsA[idx], ptsB[idx]
    n = len(X3)

    # now build BA fun & sparsity with the **subsampled** n
    fun, spars = make_ba_fun(K, ptsA, ptsB, n)
    p0 = np.hstack([m2rvec(R2), t2.ravel(), X3.ravel()])  # length 6 + 3n

    # run sparse‑Jacobian BA (trf supports jac_sparsity)
    res = least_squares(fun, p0,
                        jac_sparsity=spars,
                        method="trf",
                        xtol=1e-8,
                        ftol=1e-8,
                        max_nfev=300)

    p_opt = res.x
    R2_ref = rvec2m(p_opt[:3])
    t2_ref = p_opt[3:6]
    X3_ref = p_opt[6:].reshape(-1,3)
    rms1 = np.sqrt(np.mean(fun(p_opt)**2))

    return {
        **info,
        "rotation_refined_global":       R2_ref.tolist(),
        "translation_refined_global":    t2_ref.tolist(),
        "triangulated_points_3d":        X3_ref.tolist(),
        "num_triangulated_points":       len(X3_ref),
        "rms_before":                    rms0,
        "rms_after":                     rms1,
        "ba_performed":                  True
    }

# ──────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────
def main():
    parent = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    folder = util.choose_image_set(parent)
    if not folder:
        return

    with open(os.path.join(folder, "pose_estimation_data.json")) as f:
        pairs = json.load(f)

    # build global pose dictionary
    pose_dict = {pairs[next(iter(pairs))]["imgA"]: (np.eye(3), np.zeros((3,1)))}
    for v in pairs.values():
        pose_dict.setdefault(
            v["imgB"],
            (np.asarray(v["rotation_recovered_global"], np.float64),
             np.asarray(v["translation_recovered_global"], np.float64).reshape(3,1))
        )

    # process each pair in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed
    args = [(info, pose_dict) for info in pairs.values()]
    updated = {}
    with ProcessPoolExecutor(max_workers=4) as exe:
        futures = [exe.submit(process_pair, info, pose_dict) for info in pairs.values()]
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                key = f"{result['imgA']}::{result['imgB']}"
                updated[key] = result
                print(f"[OK] {key}: {result['num_triangulated_points']} pts, RMS {result['rms_after']:.3f}, BA={result['ba_performed']}")

    # merge back and save
    for k,v in updated.items():
        pairs[k] = v

    out = os.path.join(folder, "pose_and_triangulation_data.json")
    with open(out, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"\n[✓] Wrote refined data → {out}")

    # write aggregate PLY
    all_pts = [pt for info in pairs.values() if "triangulated_points_3d" in info
               for pt in info["triangulated_points_3d"]]
    if all_pts:
        ply = os.path.join(folder, "refined_cloud.ply")
        with open(ply, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex "+str(len(all_pts))+"\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for x,y,z in all_pts:
                f.write(f"{x} {y} {z}\n")
        print(f"[✓] Saved cloud → {ply}")

if __name__ == "__main__":
    main()
