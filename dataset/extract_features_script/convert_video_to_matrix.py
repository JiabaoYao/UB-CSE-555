import os
import numpy as np
import pandas as pd
from extract_upperbody_hands_3dkeypoints import extract_3d_keypoints
import mediapipe as mp
 
input_folder = r"D:\Courses\Pattern Recognition\WLASL\WLASL\data\videos"
output_folder = r"D:\Courses\Pattern Recognition\WLASL\WLASL\data"

for video in os.listdir(input_folder):
    if not video.endswith(".mp4"):
        continue

    video_name = os.path.splitext(video)[0]  # This is "00338"
    full_video_filename = video  # This is "00338.mp4"


    # Skip already-processed
    if os.path.exists(os.path.join(output_folder, video_name + ".npy")):
        print(f"✔ Already done: {video_name}")
        continue

    print(f"▶ Processing {video_name}...")
    extract_3d_keypoints(input_folder, output_folder, full_video_filename)
    print(f"▶ {video_name} done.")