# pose_and_scale.py
import os, json
import cv2, numpy as np
import util

def show_inliers(img, pts, mask):
    img = cv2.imread(img)
    if img is None: return
    h,w = img.shape[:2]
    s = min(800/w,600/h,1.0)
    disp = cv2.resize(img,None,fx=s,fy=s)
    for (x,y),m in zip(pts,mask):
        if m: cv2.circle(disp,(int(x*s),int(y*s)),3,(0,255,0),-1)
    cv2.imshow("Inliers",disp); cv2.waitKey(0); cv2.destroyAllWindows()

def run_pose_estimation(tri_data):
    keys = list(tri_data.keys())
    updated = {}
    # first camera at origin
    Rg, tg = np.eye(3), np.zeros(3)

    for idx,key in enumerate(keys,1):
        info = tri_data[key]
        ptsA = np.asarray(info['matched_points_imgA'],np.float64)
        mask = np.asarray(info['inlier_mask_pose'],np.int32).ravel()
        Rr   = np.asarray(info['recovered_rotation'],np.float64)
        tr   = np.asarray(info['recovered_translation'],np.float64).ravel()
        nin  = mask.sum(); ntot=len(mask)
        if nin<30 or nin/ntot<0.2: 
            print(f"[SKIP] {key} ({nin}/{ntot})")
            continue

        show_inliers(info['imgA'], ptsA, mask)

        norm_t = np.linalg.norm(tr)
        if norm_t<1e-6: continue
        t_unit = tr/norm_t

        # chain
        Rg_new = Rr @ Rg
        tg_new = Rr @ tg + t_unit

        info.update({
            'rotation_global_A':     Rg.tolist(),
            'translation_global_A':  tg.tolist(),
            'rotation_global_B':     Rg_new.tolist(),
            'translation_global_B':  tg_new.tolist()
        })
        updated[key] = info
        Rg, tg = Rg_new, tg_new

    return updated

def main():
    parent = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    folder = util.choose_image_set(parent)
    if not folder: return

    path = os.path.join(folder,'triangulation_data.json')
    tri_data = util.load_json_data(path)
    updated = run_pose_estimation(tri_data)

    out = os.path.join(folder,'pose_estimation_data.json')
    with open(out,'w') as f:
        json.dump(updated,f,indent=2)
    print(f"[OK] wrote → {out}")

if __name__=='__main__':
    main()
