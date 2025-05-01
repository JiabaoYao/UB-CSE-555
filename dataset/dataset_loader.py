from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np
import os
import math

class GlossKeypointDataset(Dataset):
    def __init__(self):
        base_path = os.path.dirname(__file__)
        self.video_matrix = np.load(os.path.join(base_path, "dataset_all_padding_new.npz"), allow_pickle=True)['data']  # shape: [N, 2]
        self.n_samples = len(self.video_matrix)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx): # return single video information
        # gloss = torch.tensor(self.gloss[idx][0], dtype=torch.long)
        # keypoints = torch.tensor(self.keypoints[idx][1], dtype=torch.float32)
        gloss = self.video_matrix[idx][0]
        keypoints = torch.tensor(self.video_matrix[idx][1], dtype=torch.float32)
        return gloss, keypoints

def dataset_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert(math.isclose(train_ratio + val_ratio + test_ratio, 1.0)) # summation of ratios must be 1
    total_size = len(dataset)
    train_size, val_size = int(train_ratio * total_size), int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    return random_split(dataset, [train_size, val_size, test_size])


def data_loader(batch_size, num_workers=2):
    dataset = GlossKeypointDataset()
    print(f"Total dataset size: {len(dataset)}")

    train_data, val_data, test_data = dataset_split(dataset)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# Test case
# def run():
#     train_loader, test_loader, val_loader = data_loader(batch_size=3)

#     print("Train test")
#     for epoch in range(2):
#         for i, (gloss, keypoints) in enumerate(train_loader):
#             print(f"Epoch {epoch}, Step {i}, Gloss: {gloss}, Keypoints: {keypoints}")
#             break

#     print("Validation test")
#     for epoch in range(2):
#         for i, (gloss, keypoints) in enumerate(train_loader):
#             print(f"Epoch {epoch}, Step {i}, Gloss: {gloss}, Keypoints: {keypoints[1]}")
#             break
    
#     print("Test test")
#     for epoch in range(2):
#         for i, (gloss, keypoints) in enumerate(test_loader):
#             print(f"Epoch {epoch}, Step {i}, Gloss: {gloss}, Keypoints: {keypoints[1]}")
#             break

# if __name__ == "__main__":
#     run()

