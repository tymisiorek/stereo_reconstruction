import os
import json
import cv2
import numpy as np

def choose_image_set(parent_dir):
    """
    List all subdirectories under 'parent_dir' and prompt the user to pick one.
    Returns the subdirectory name chosen by the user (full path), or None if none found.
    """
    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    if not subdirs:
        print(f"No subdirectories found in {parent_dir}.")
        return None

    print("Choose one of the following image sets:")
    for idx, d in enumerate(subdirs, start=1):
        print(f"{idx}. {d}")

    while True:
        try:
            choice = int(input("Enter the number of the folder you want to use: "))
            if 1 <= choice <= len(subdirs):
                return os.path.join(parent_dir, subdirs[choice - 1])
            else:
                print(f"Invalid selection. Enter a number between 1 and {len(subdirs)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def load_feature_data(json_path):
    """
    Load the feature data (sift_results, refined_matches_dict) from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_matched_points(sift_results, refined_matches_dict, pair_key):
    """
    From the JSON data structures, extract matched 2D points for a specific 'imgA::imgB' key.
    :return: two NumPy arrays (pointsA, pointsB) of shape (N, 2)
    """
    imgA, imgB = pair_key.split("::")
    imgA_data = sift_results[imgA]
    imgB_data = sift_results[imgB]

    # Deserialize keypoints for each image.
    keypointsA = [
        cv2.KeyPoint(
            float(kp_dict["pt"][0]),
            float(kp_dict["pt"][1]),
            float(kp_dict["size"]),
            float(kp_dict["angle"]),
            float(kp_dict["response"]),
            int(kp_dict["octave"]),
            int(kp_dict["class_id"])
        ) for kp_dict in imgA_data["keypoints"]
    ]
    keypointsB = [
        cv2.KeyPoint(
            float(kp_dict["pt"][0]),
            float(kp_dict["pt"][1]),
            float(kp_dict["size"]),
            float(kp_dict["angle"]),
            float(kp_dict["response"]),
            int(kp_dict["octave"]),
            int(kp_dict["class_id"])
        ) for kp_dict in imgB_data["keypoints"]
    ]

    matches = refined_matches_dict[pair_key]

    pointsA = []
    pointsB = []
    for m in matches:
        queryIdx = m["queryIdx"]
        trainIdx = m["trainIdx"]
        ptA = keypointsA[queryIdx].pt
        ptB = keypointsB[trainIdx].pt
        pointsA.append(ptA)
        pointsB.append(ptB)

    return np.array(pointsA, dtype=np.float32), np.array(pointsB, dtype=np.float32)

def draw_epilines(img, lines, pts):
    """
    Draw epipolar lines on an image.
    :param img: The image on which to draw the lines.
    :param lines: Epipolar lines (each line is [a, b, c]).
    :param pts: The corresponding points (for visualization).
    :return: The image with epipolar lines drawn.
    """
    r, c, _ = img.shape
    img_with_lines = img.copy()
    for r_line, pt in zip(lines, pts):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # Compute line endpoints: ax + by + c = 0 => y = (-c - ax)/b.
        if r_line[1] != 0:
            x0, y0 = 0, int(-r_line[2] / r_line[1])
            x1, y1 = c, int(-(r_line[2] + r_line[0] * c) / r_line[1])
        else:
            # In case b is zero, choose default endpoints.
            x0, y0 = int(-r_line[2] / r_line[0]), 0
            x1, y1 = int(-r_line[2] / r_line[0]), r
        img_with_lines = cv2.line(img_with_lines, (x0, y0), (x1, y1), color, 1)
        img_with_lines = cv2.circle(img_with_lines, (int(pt[0]), int(pt[1])), 5, color, -1)
    return img_with_lines

def process_image_pairs(sift_results, refined_matches, K):
    """
    Process each image pair: for each pair, extract matched points,
    estimate the Fundamental and Essential matrices, recover the pose,
    and display epipolar lines for visual validation.
    """
    triangulation_data = {}

    for pair_key in refined_matches:
        print(f"\nProcessing image pair: {pair_key}")
        pointsA, pointsB = extract_matched_points(sift_results, refined_matches, pair_key)
        print(f"  Extracted {pointsA.shape[0]} matched points for each image.")

        if pointsA.shape[0] < 8:
            print("  Not enough matches to compute the matrices (minimum 8 required). Skipping.")
            continue

        # Estimate the Fundamental Matrix using RANSAC.
        F, mask_F = cv2.findFundamentalMat(pointsA, pointsB, cv2.FM_RANSAC, 1.0, 0.99)
        if F is None or F.shape[0] == 0:
            print("  Fundamental matrix estimation failed.")
            continue

        print("  Estimated Fundamental Matrix (F):")
        print(F)

        # Derive the Essential Matrix from F.
        E_from_F = K.T @ F @ K
        print("  Essential Matrix derived from F:")
        print(E_from_F)

        # Directly compute the Essential Matrix using RANSAC.
        E_direct, mask_E = cv2.findEssentialMat(pointsA, pointsB, cameraMatrix=K,
                                                  method=cv2.FM_RANSAC, prob=0.999, threshold=1.0)
        if E_direct is None or E_direct.shape[0] == 0:
            print("  Essential matrix estimation failed.")
            continue
        print("  Essential Matrix computed directly from matched points:")
        print(E_direct)

        # Recover the relative camera pose.
        retval, R, t, mask_pose = cv2.recoverPose(E_direct, pointsA, pointsB, K)
        print("  Recovered rotation (R):")
        print(R)
        print("  Recovered translation (t):")
        print(t)

        imgA_path, imgB_path = pair_key.split("::")
        imgA = cv2.imread(imgA_path)
        imgB = cv2.imread(imgB_path)
        if imgA is None or imgB is None:
            print("  Could not load one or both images for visualization. Skipping visualization.")
        else:
            #Compute and draw epipolar lines on image B using points from image A.
            linesB = cv2.computeCorrespondEpilines(pointsA.reshape(-1, 1, 2), 1, F)
            linesB = linesB.reshape(-1, 3)
            imgB_with_lines = draw_epilines(imgB, linesB, pointsB)

            #Compute and draw epipolar lines on image A using points from image B.
            linesA = cv2.computeCorrespondEpilines(pointsB.reshape(-1, 1, 2), 2, F)
            linesA = linesA.reshape(-1, 3)
            imgA_with_lines = draw_epilines(imgA, linesA, pointsA)

            cv2.imshow("Image A with Epipolar Lines", imgA_with_lines)
            cv2.imshow("Image B with Epipolar Lines", imgB_with_lines)
            print("  Press any key in the image window to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        pair_data = {
            "imgA": imgA_path,
            "imgB": imgB_path,
            "camera_intrinsics": K.tolist(),
            "fundamental_matrix": F.tolist(),
            "essential_matrix_from_F": E_from_F.tolist(),
            "essential_matrix_direct": E_direct.tolist(),
            "matched_points_imgA": pointsA.tolist(),
            "matched_points_imgB": pointsB.tolist(),
            "recovered_rotation": R.tolist(),
            "recovered_translation": t.tolist(),
            "inlier_mask_fundamental": mask_F.tolist(),
            "inlier_mask_pose": mask_pose.tolist()
        }
        triangulation_data[pair_key] = pair_data

    return triangulation_data

def main():
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    chosen_folder = choose_image_set(parent_dir)
    if not chosen_folder:
        print("No valid folder was selected. Exiting.")
        return

    json_path = os.path.join(chosen_folder, "feature_data.json")
    if not os.path.isfile(json_path):
        print(f"No feature_data.json found in {chosen_folder}. Exiting.")
        return

    data = load_feature_data(json_path)
    sift_results = data["sift_results"]
    refined_matches = data["refined_matches_dict"]

    # fx = 1844.65813
    # fy = 1848.40594
    # cx = 560.584384
    # cy = 972.188133

    # K = np.array([
    #     [fx,   0, cx],
    #     [0,   fy, cy],
    #     [0,    0,  1 ]
    # ], dtype=np.float64)
    fx = 1826.35890
    fy = 1826.55090
    cx = 520.668647
    cy = 955.447831

    K = np.array([
        [fx,   0, cx],
        [0,   fy, cy],
        [0,    0,  1]
    ], dtype=np.float64)

    if not refined_matches:
        print("No refined matches found. Exiting.")
        return

    triangulation_data = process_image_pairs(sift_results, refined_matches, K)

    output_path = os.path.join(chosen_folder, "triangulation_data.json")
    with open(output_path, 'w') as outfile:
        json.dump(triangulation_data, outfile, indent=2)
    print(f"Triangulation data has been saved to {output_path}")

if __name__ == "__main__":
    main()
