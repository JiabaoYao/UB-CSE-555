import os
import json
import numpy as np

def video_to_stgcn_data(folder_path, max_frame=64):
    frames_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])
    keypoints_per_frame = []

    for fname in frames_files[:max_frame]:
        with open(os.path.join(folder_path, fname)) as f:
            data = json.load(f)
            if not data['people']:
                continue

            signer = data['people'][0]
            left_hand = np.array(signer.get('hand_left_keypoints_2d', [])).reshape(-1, 3)
            right_hand = np.array(signer.get('hand_right_keypoints_2d', [])).reshape(-1, 3)

            if left_hand.shape[0] != 21 or right_hand.shape[0] != 21:
                continue

            full_frame = np.concatenate([left_hand, right_hand], axis=0)
            keypoints_per_frame.append(full_frame[:, :2])  # keep only x, y

    if not keypoints_per_frame:
        return None

    T = len(keypoints_per_frame)
    if T < max_frame:
        pad = [np.zeros((42, 2))] * (max_frame - T)
        keypoints_per_frame.extend(pad)
    else:
        keypoints_per_frame = keypoints_per_frame[:max_frame]

    arr = np.stack(keypoints_per_frame)       # (T, 42, 2)
    arr = arr.transpose(2, 0, 1)              # (2, T, 42)
    arr = arr[:, :, :, None]                 # (2, T, 42, 1)
    return arr

            

data= video_to_stgcn_data(r'C:\Users\19692\Downloads\UB-CSE-555\data_processing_wlasl\pose_per_videoes\04444');
print(data[0][7][12][0])