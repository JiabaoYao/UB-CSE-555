import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


class SignDataset:
        def __init__(self, x, y):
            self.X = x
            self.Y = y
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            key_points = self.X[idx]
            label = self.Y[idx]
            return (key_points,label)

## Data Structure 📖
# data[i] -> ith data  
# data = label (data[i][0]) + 3d key points (data[i][1])   
# 3d key point's shape is (195,200) => 195 frames, 200 joint data / frame  
# 200 joint data => 50 joint x 4 values (x,y,z,confidence) 

# this function process raw open pose data to data required by st-gcn
# however, the graph is not built in this step
def process_data(data, frame_num):
    # convert openpose raw data to st-gcn preferred data
    labels = []
    all_key_points = []
    for label, key_points in data:
        labels.append(label)
        key_points = key_points.reshape(frame_num, 50, 4) # frame-joint-channels(x,y,z,c)
        key_points = key_points.transpose(2, 0, 1) # st-gcn requires (channels, time_stamp, joints)
        all_key_points.append(key_points)
    all_key_points = np.stack(all_key_points)

    # label encoding
    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(labels)

    # convert to pytorch-like data
    x = torch.tensor(all_key_points, dtype=torch.float32)
    y = torch.tensor(encoded_label,dtype=torch.long)

   
    
    return SignDataset(x,y), label_encoder

