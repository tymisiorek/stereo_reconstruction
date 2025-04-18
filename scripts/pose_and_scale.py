# ───────────────────────── pose_and_scale.py (fixed v2) ───────────────────────
"""
Chain every pair’s relative pose into a *metric‑consistent* global frame.
translation is kept unit‑length for the first pair; subsequent poses inherit the
same global scale automatically.
"""
import os, json, cv2, numpy as np, util

def show_inliers(img_path, pts, mask):
    img = cv2.imread(img_path);  h,w = img.shape[:2]
    s   = min(800/w, 600/h, 1.0)
    disp= cv2.resize(img, None, fx=s, fy=s)
    for (x,y),m in zip(pts,mask):
        if m: cv2.circle(disp,(int(x*s),int(y*s)),3,(0,255,0),-1)
    cv2.imshow("Pose inliers",disp); cv2.waitKey(0); cv2.destroyAllWindows()

def run_pose_estimation(tri_data, min_inl=30, min_ratio=0.2):
    keys     = list(tri_data.keys())
    updated  = {}

    # first camera: R = I ,  C = (0,0,0)
    Rg_last  = np.eye(3);          C_last  = np.zeros(3)
    shown    = False

    for idx,key in enumerate(keys,1):
        info  = tri_data[key]
        R_rel = np.asarray(info['recovered_rotation']     ,np.float64)
        t_rel = np.asarray(info['recovered_translation']  ,np.float64).ravel()

        mask  = np.asarray(info['inlier_mask_pose'],np.int32).ravel()
        nin, ntot = int(mask.sum()), len(mask)
        print(f"\nPAIR {idx}/{len(keys)}: {key}  ({nin}/{ntot} pose inliers)")
        if nin < min_inl or nin/ntot < min_ratio:
            print("  – skipped")
            continue

        # optional visual check for *every* kept pair
        show_inliers(info['imgA'], np.asarray(info['matched_points_imgA']), mask)

        # normalise t once (unit baseline)
        s = np.linalg.norm(t_rel)
        if s < 1e-9:
            print("  – skipped (zero baseline)")
            continue
        t_unit = t_rel / s

        # ------- chain into global frame ----------
        Rg_new = R_rel @ Rg_last                     # world→camB
        C_new  = C_last + Rg_last.T @ t_unit         # camera centre in world
        tg_new = -Rg_new @ C_new                     # convert to t = −R C

        jump = np.linalg.norm(C_new - C_last)
        if jump > 5: print(f"  • warning: camera jump {jump:.2f}")

        # write back
        info.update({
            "rotation_global_A"    : Rg_last.tolist(),
            "translation_global_A" : (-Rg_last @ C_last).tolist(),
            "rotation_global_B"    : Rg_new.tolist(),
            "translation_global_B" : tg_new.tolist(),
            "camera_center_global_B": C_new.tolist()
        })
        updated[key] = info

        Rg_last, C_last = Rg_new, C_new      # next iteration

    return updated

def main():
    parent = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    folder = util.choose_image_set(parent);  1/0 if not folder else None

    tri_path = os.path.join(folder,'triangulation_data.json')
    tri_data = util.load_json_data(tri_path)

    updated  = run_pose_estimation(tri_data)
    out_path = os.path.join(folder,'pose_estimation_data.json')
    with open(out_path,'w') as f: json.dump(updated,f,indent=2)
    print(f"[✓] global poses → {out_path}")

if __name__=='__main__': main()
