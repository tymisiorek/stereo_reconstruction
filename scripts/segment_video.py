import os
import math
import cv2

def extract_frames(video_path, n):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate interval between frames
    interval = total_frames / float(n)
    
    extracted_frames = []
    for i in range(n):
        frame_number = int(round(i * interval))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rotate the frame by 270 degrees (i.e. 90 degrees counterclockwise)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        
        extracted_frames.append(frame)
    
    cap.release()
    return extracted_frames

def list_video_files(video_path):
    # List all files in the directory
    files = os.listdir(video_path)
    # Filter out files that are likely videos based on common extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in files if f.lower().endswith(video_extensions)]
    return video_files

def choose_video_file(video_path):
    video_files = list_video_files(video_path)
    if not video_files:
        print("No video files found in the directory.")
        return None
    
    print("Select a video file to process:")
    for idx, file_name in enumerate(video_files, start=1):
        print(f"{idx}. {file_name}")
    
    while True:
        try:
            choice = int(input("Enter the number corresponding to the video: "))
            if 1 <= choice <= len(video_files):
                selected_file = video_files[choice - 1]
                full_path = os.path.join(video_path, selected_file)
                return full_path
            else:
                print(f"Please enter a number between 1 and {len(video_files)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def save_frames(frames, video_path, output_base):
    # Get the video file name without extension
    video_filename = os.path.basename(video_path)
    video_name, _ = os.path.splitext(video_filename)
    
    # Create output directory: output_base/video_name
    output_dir = os.path.join(output_base, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Write each frame to the output directory with sequential numbering
    for idx, frame in enumerate(frames, start=1):
        image_filename = f"{video_name}_{idx}.jpg"
        image_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")

def main():
    data_path = r'C:\Projects\Semester6\CS4501\stereo_reconstruction\data'
    video_path = os.path.join(data_path, 'videos')
    selected_video = choose_video_file(video_path)
    if not selected_video:
        print("No video was selected.")
        return

    print(f"Selected video: {selected_video}")
    
    images = extract_frames(selected_video, 20)
    
    # Save the frames into the 'data/images/<video_name>' directory
    output_base = os.path.join(data_path, 'images')
    save_frames(images, selected_video, output_base)

if __name__ == "__main__":
    main()
