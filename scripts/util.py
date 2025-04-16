import os
import glob
import json

def get_images_for_sequence(folder, sequence):
    pattern = os.path.join(folder, f"tripod_seq_{sequence}_*.jpg")
    file_list = glob.glob(pattern)
    file_list.sort()
    
    return file_list


def choose_image_set():
    """
    List all subdirectories under 'parent_dir' and prompt the user to pick one.
    Returns the subdirectory name chosen by the user (full path), or None if none found.
    """
    parent_dir = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data\images'
    subdirs = [
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
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

def load_json_data(json_path):
    """
    Load the feature data (sift_results, refined_matches_dict) from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    folder_path = r"stereo_reconstruction\data\tripod-seq" 
    sequence_id = "01" 
    
    images = get_images_for_sequence(folder_path, sequence_id)
    print("Found images:")
    for image in images:
        print(image)

