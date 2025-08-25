import cv2
import os
import sys

def extract_frames(video_path, output_folder):
    """
    Extracts all frames from a video file and saves them to an output folder.
    """
    # --- 1. Basic Setup and Checks ---
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: '{output_folder}'")

    # --- 2. Open the Video File ---
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return
        
    print("Starting frame extraction...")
    
    # --- 3. Loop Through Video and Extract Frames ---
    frame_count = 0
    while True:
        success, frame = video_capture.read()

        if not success:
            break
        
        # ------------------------------------------------------------------
        # NEW: Fix for upside-down frames. Flip 180 degrees.
        # frame = cv2.flip(frame, -1)
        # ------------------------------------------------------------------

        filename = f"frame_{str(frame_count).zfill(5)}.png"
        save_path = os.path.join(output_folder, filename)
        
        cv2.imwrite(save_path, frame)
        
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  ... extracted {frame_count} frames")

    # --- 4. Clean Up ---
    video_capture.release()
    
    print("-" * 30)
    print(f"Successfully extracted {frame_count} frames.")
    print(f"Frames are saved in: '{output_folder}'")
    print("-" * 30)


# --- Main execution block ---
if __name__ == "__main__":
    vid_path = input("Enter the full path to your video file: ")
    out_folder = input("Enter the name of the folder to save frames: ")
    extract_frames(vid_path, out_folder)