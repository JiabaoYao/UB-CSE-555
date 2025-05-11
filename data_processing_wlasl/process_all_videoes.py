import os
import csv
import numpy as np
from video_to_stgcn_data import video_to_stgcn_data
from video_to_label import video_to_label

def process_all_videos(data_root, output_path, label_dict, splits_dict, max_frame=64):
    video_tensors_test = []
    video_labels_test = []
    video_tensors_train = []
    video_labels_train = []
    video_tensors_val = []
    video_labels_val = []



    for folder_name in os.listdir(data_root):
        print(folder_name)
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
    
        tensor = video_to_stgcn_data(folder_path, max_frame=max_frame)
        label = label_dict.get(str(folder_name))
        splits = splits_dict.get(str(folder_name))
        

        
        # means the video is not in the current subset
        if tensor is None or label is None:
            continue
        
        if splits == 'test':
            video_tensors_test.append(tensor)
            video_labels_test.append(label)
        elif splits == 'train':
            video_tensors_train.append(tensor)
            video_labels_train.append(label)
        else:
            video_tensors_val.append(tensor)
            video_labels_val.append(label)

    # stack all videos: (N, 2, 64, 67, 1)
    all_data_test = np.stack(video_tensors_test)
    np.save(os.path.join(output_path, "videos_300_test.npy"), all_data_test)

    all_data_train = np.stack(video_tensors_train)
    np.save(os.path.join(output_path, "videos_300_train.npy"), all_data_train)

    all_data_val = np.stack(video_tensors_val)
    np.save(os.path.join(output_path, "videos_300_val.npy"), all_data_val)

    # Save labels
    with open(os.path.join(output_path, "labels_300_test.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for idx, label in enumerate(video_labels_test):
            writer.writerow([f"{idx}.npy", label])
    
    # Save labels
    with open(os.path.join(output_path, "labels_300_train.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for idx, label in enumerate(video_labels_train):
            writer.writerow([f"{idx}.npy", label])
    
    # Save labels
    with open(os.path.join(output_path, "labels_300_val.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for idx, label in enumerate(video_labels_val):
            writer.writerow([f"{idx}.npy", label])


labels, splits = video_to_label(r'C:\Users\19692\Downloads\UB-CSE-555\data_processing_wlasl\splits\splits\asl300.json')
process_all_videos(
    data_root=r'C:\Users\19692\Downloads\UB-CSE-555\data_processing_wlasl\pose_per_videoes',
    output_path=r'C:\Users\19692\Downloads\UB-CSE-555\data_processing_wlasl\output',
    label_dict=labels,
    splits_dict = splits,
    max_frame=64
)
