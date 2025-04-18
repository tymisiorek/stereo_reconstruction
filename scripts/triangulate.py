# triangulate_global.py
import os, json
import cv2, numpy as np
import util

def triangulate_pair(info):
    K  = np.asarray(info['camera_intrinsics'],np.float64)
    R1 = np.asarray(info['rotation_global_A'],   np.float64)
    t1 = np.asarray(info['translation_global_A'],np.float64).reshape(3,1)
    R2 = np.asarray(info['rotation_global_B'],   np.float64)
    t2 = np.asarray(info['translation_global_B'],np.float64).reshape(3,1)

    P1 = K @ np.hstack((R1,t1))
    P2 = K @ np.hstack((R2,t2))

    ptsA = np.asarray(info['matched_points_imgA'],np.float64)
    ptsB = np.asarray(info['matched_points_imgB'],np.float64)
    mask = np.asarray(info['inlier_mask_pose'],np.int32)
    ptsA, ptsB = ptsA[mask==1], ptsB[mask==1]
    if len(ptsA)<2:
        return np.zeros((0,3),np.float64)

    X4 = cv2.triangulatePoints(P1,P2,ptsA.T,ptsB.T)
    X3 = (X4[:3]/X4[3]).T
    # cheirality
    Z1 = (R1@X3.T + t1).T[:,2]
    Z2 = (R2@X3.T + t2).T[:,2]
    good = (Z1>0)&(Z2>0)
    return X3[good]

def main():
    parent = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    folder = util.choose_image_set(parent)
    if not folder: return

    with open(os.path.join(folder,'pose_estimation_data.json')) as f:
        pairs = json.load(f)

    all_pts=[]
    for key,info in pairs.items():
        pts3 = triangulate_pair(info)
        info['triangulated_points_3d']  = pts3.tolist()
        info['num_triangulated_points'] = int(len(pts3))
        all_pts.extend(pts3)
        print(f"[TRI] {key}: {len(pts3)} points")

    out = os.path.join(folder,'pose_and_triangulation_data.json')
    with open(out,'w') as f:
        json.dump(pairs,f,indent=2)
    print(f"[OK] wrote → {out}")

    # optional PLY
    if all_pts:
        ply = os.path.join(folder,'cloud.ply')
        with open(ply,'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(all_pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for x,y,z in all_pts:
                f.write(f"{x} {y} {z}\n")
        print(f"[OK] PLY → {ply}")

if __name__=='__main__':
    main()
