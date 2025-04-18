# final_triangulate.py

import os
import sys
import json
import cv2
import numpy as np

def load_global_poses(path):
    with open(path, 'r') as f:
        data = json.load(f)
    poses = {}
    for img, v in data.items():
        R = np.array(v['R'], dtype=np.float64)
        t = np.array(v['t'], dtype=np.float64).reshape(3,1)
        poses[img] = (R, t)
    return poses

def load_feature_data(folder):
    """Loads feature_data.json from the given image-folder."""
    feat_file = os.path.join(folder, 'feature_data.json')
    if not os.path.isfile(feat_file):
        raise FileNotFoundError(f"{feat_file} not found")
    with open(feat_file, 'r') as f:
        fd = json.load(f)

    # rebuild keypoints 
    kp_map = {}
    for img_path, lst in fd['sift_results'].items():
        kps = []
        for kp in lst:
            x, y = kp['pt']
            size = kp['size']
            # positional args: x, y, size
            kps.append(cv2.KeyPoint(float(x), float(y), float(size)))
        kp_map[img_path] = kps

    # load tracks.json, if present
    track_map = {}
    tracks_file = os.path.join(folder, 'reconstruction', 'tracks.json')
    if os.path.isfile(tracks_file):
        with open(tracks_file, 'r') as f:
            raw = json.load(f)
        for key, pid in raw.items():
            cam, kp_idx = key.split(':')
            kp_idx = int(kp_idx)
            # find full img_path that ends with cam
            img = cam if os.path.isabs(cam) else next(p for p in kp_map if p.endswith(cam))
            track_map.setdefault(int(pid), []).append((img, kp_idx))
    else:
        raise RuntimeError("tracks.json not found; please run the BA‑pipeline first")

    return kp_map, track_map

def triangulate_tracks(poses, kp_map, track_map, K, output_ply):
    all_pts = []
    for pid, obs in track_map.items():
        if len(obs) < 2:
            continue
        (img1, k1), (img2, k2) = obs[0], obs[1]
        R1, t1 = poses[img1]; R2, t2 = poses[img2]
        P1 = K @ np.hstack([R1, t1])
        P2 = K @ np.hstack([R2, t2])

        u1 = np.array(kp_map[img1][k1].pt, dtype=np.float64)
        u2 = np.array(kp_map[img2][k2].pt, dtype=np.float64)

        X4 = cv2.triangulatePoints(
            P1, P2,
            u1[:2].reshape(2,1),
            u2[:2].reshape(2,1)
        )
        X3 = (X4[:3] / X4[3]).reshape(3)
        all_pts.append(X3)

    with open(output_ply, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(all_pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for X in all_pts:
            f.write(f"{X[0]} {X[1]} {X[2]}\n")

def main():
    # 1) Determine reconstruction folder
    if len(sys.argv) > 1:
        recon_dir = sys.argv[1]
    else:
        from feature_detection import util
        parent = r"C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images"
        folder = util.choose_image_set(parent)
        if not folder:
            print("No folder selected."); return
        recon_dir = os.path.join(folder, "reconstruction")

    if not os.path.isdir(recon_dir):
        print(f"Error: reconstruction dir not found:\n  {recon_dir}")
        return

    poses_file = os.path.join(recon_dir, "poses_global.json")
    if not os.path.isfile(poses_file):
        print(f"Error: {poses_file} not found")
        return

    # 2) Load global poses
    poses = load_global_poses(poses_file)

    # 3) Intrinsics (hard‑coded)
    fx, fy = 1826.35890, 1826.55090
    cx, cy =  520.668647, 955.447831
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

    # 4) Load features & tracks from the image‑set folder
    img_folder = os.path.dirname(recon_dir)   # now points to .../images/chapel
    kp_map, track_map = load_feature_data(img_folder)

    # 5) Triangulate and write final cloud
    out_ply = os.path.join(recon_dir, "cloud_global.ply")
    triangulate_tracks(poses, kp_map, track_map, K, out_ply)
    print("Wrote final global cloud to", out_ply)

if __name__ == "__main__":
    main()
