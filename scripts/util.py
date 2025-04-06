import os
import glob

def get_images_for_sequence(folder, sequence):
    pattern = os.path.join(folder, f"tripod_seq_{sequence}_*.jpg")
    file_list = glob.glob(pattern)
    file_list.sort()
    
    return file_list

if __name__ == "__main__":
    folder_path = r"stereo_reconstruction\data\tripod-seq" 
    sequence_id = "01" 
    
    images = get_images_for_sequence(folder_path, sequence_id)
    print("Found images:")
    for image in images:
        print(image)
