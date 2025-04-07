from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import math

class GlossKeypointDataset(Dataset):
    def __init__(self):
        base_path = os.path.dirname(__file__)
        self.video_matrix = np.load(os.path.join(base_path, "dataset_all_padding.npz"), allow_pickle=True)['data']  # shape: [N, 2]
        self.n_samples = len(self.video_matrix)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx): # return single video information
        # gloss = torch.tensor(self.gloss[idx][0], dtype=torch.long)
        # keypoints = torch.tensor(self.keypoints[idx][1], dtype=torch.float32)
        gloss = self.video_matrix[idx][0]
        keypoints = self.video_matrix[idx][1]
        return gloss, keypoints

def data_loader(batch_size, num_workers=2):
    dataset = GlossKeypointDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

# Test case
# def run():
#     loader = data_loader(batch_size=2)

#     for epoch in range(2):
#         for i, (gloss, keypoints) in enumerate(loader):
#             print(f"Epoch {epoch}, Step {i}, Gloss: {gloss}, Keypoints: {keypoints}")

# if __name__ == "__main__":
#     run()
