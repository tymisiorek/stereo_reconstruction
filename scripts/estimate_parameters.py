import cv2
import numpy as np
import glob
import os

# Define chessboard size (number of inner corners per row and column)
chessboard_size = (8, 5)  # Change to match your printed pattern (e.g., 9x6)
square_size = 0.029  # Size of a square in meters (or any consistent unit)

# Prepare object points (3D points in the world)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in the world
imgpoints = []  # 2D points in the image

# Load all images for calibration
directory = r"C:\Projects\Semester6\CS4501\stereo_reconstruction\data\chessboard"
pattern = os.path.join(directory, "*.*")
image_files = glob.glob(pattern)

# Process each file.
for image_file in image_files:
    img = cv2.imread(image_file)
    if img is None:
        print(f"Unable to load {image_file}")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # Refine the corners for better accuracy
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)
        objpoints.append(objp)
        
        # Optional: Draw and display the corners for visual confirmation
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow("Chessboard", img)
        cv2.waitKey(100)
    else:
        print(f"Chessboard not found in {image_file}")

cv2.destroyAllWindows()

# Check that we have at least one valid calibration image
if not objpoints:
    raise ValueError("No chessboard corners were found in any of the images. Check your images and chessboard pattern.")

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the results
if ret:
    print("Camera matrix (Intrinsic parameters):")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)
else:
    print("Calibration failed. Ensure the chessboard pattern is clear in the images.")

# Save the calibration results
np.savez("calibration_results.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# Optional: Compute the reprojection error to check calibration quality
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

print(f"\nReprojection error: {total_error / len(objpoints):.4f}")
