import json
from collections import Counter


# map each video number (e.g. 25481) to its word (e.g. "book")
# json_path should be openpose's split json
def video_to_label(json_path):
    with open(json_path) as f:
        metadata = json.load(f)
    video_label = {}
    video_split = {}
    for data in metadata:
        label = data['gloss']
        for instance in data['instances']:
            video_id = instance['video_id']
            split = instance['split']
            video_label[video_id] = label
            video_split[video_id] = split

    return video_label, video_split

video_label, video_split = video_to_label(r"C:\Users\19692\Downloads\UB-CSE-555\data_processing_wlasl\splits\splits\asl1000.json")

video_label_values = video_label.values()
label_counts = Counter(video_label_values)
top_50_labels = [label for label, _ in label_counts.most_common(50)]

print(top_50_labels)