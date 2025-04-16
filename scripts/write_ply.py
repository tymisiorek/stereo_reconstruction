import os
import json
import numpy as np
import util

def write_ply(filename, points):
    """
    Write a simple ASCII PLY file for a point cloud.
    Points should be an (N,3) array or list of [x, y, z].
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def main():
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    chosen_folder = util.choose_image_set(parent_dir)
    pose_path = os.path.join(chosen_folder, "pose_and_triangulation_data.json")
    tri_data = util.load_json_data(pose_path)

    all_points = []

    # Accumulate all triangulated 3D points from the JSON data.
    for pair_key, pair_info in tri_data.items():
        if "triangulated_points_3d" in pair_info and pair_info["triangulated_points_3d"]:
            all_points.extend(pair_info["triangulated_points_3d"])
    
    if all_points:
        ply_filename = os.path.join(chosen_folder, "point_cloud.ply")
        write_ply(ply_filename, all_points)
        print(f"Point cloud saved to {ply_filename}")
    else:
        print("No 3D points found to write PLY file.")

if __name__ == "__main__":
    main()
