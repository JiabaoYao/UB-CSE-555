# **Word-Level Sign Language Recognition**

## 1 Data download

> ../data_download  
> `../raw_videos`: original videos (not covert to mp4)  
> `../raw_videos_mp4`: convert all videos to mp4 format
> `../videos`: covert youtube video name to simple video name  
> `../script`: data download and process scripts

### 1.1 Data process

1. Run `video_downloader.py` to download videos from youtube based on `WLASL_v0.3.json`, download log will be generated to check

2. Run `preprocess.py` to covert raw videos to a standand video format: videoId.mp4

3. Run `find_missing.py` to figure out the missing data, a missing.txt will be generated.

## 2 Dataset preprocess for training

### 2.1 data source and scripts

> ../videos/  
> ../extract_features_scripts/

**Due to git repo size limit, source files are saved at google drive:`https://drive.google.com/drive/folders/18hM71LUlha7Km0lqN5W33RX98gAsLr0I?usp=sharing`.**

### 2.2 Preprocess steps

1. Run `data_match_process.ipynb`to generate `matched_samples.txt`

2. Run `extract_features_script/convert_video_to_matrix.py` to handle series of videos and generate their json and npy files, seperately save into `/dataset/json` and `/dataset/matrix` folders.

3. Check and compare json and npy via `compare_json_and_matrix.py`.

4. Copy and paste `blender_upperbody_hands.py` to blender and render it via blender app. (remember to modify the path). Points which visibility < 0.5 are not replayed.

5. In order to train our model, dataset architecture is like [x,y,z,v]. Joints are not valid when `v <= 0` but we still remain them as paddings to maintain a consistent structure.

6. Due to different videos have different number of frames, we add paddings to transfer all video matrices to the same structure for trainging.

### 2.3 Dataset and loader
<<<<<<< HEAD
1. Generate dataset by word num.   
Run this script and enter word number. Before run this script, you need to ensure you folder have all video npy files. All training datasets are finally from folder `./data_preprocess/dataset`
> python create_dataset_by_wordnum.py  

=======
1. Dataset source:  
All dataset with paddings:
> dataset_all_padding_new.npz

All dataset without paddings:  
> dataset_all_new.npz

>>>>>>> cea5df6cda301e79667a80b69bfce859f06792f3
2. Dataset loader
`dataset_loader.py` contains the integrated Dataset ([N, 2]) definition and a loader function. Use the loader as model input.

How to use it?

```
    train_loader, test_loader, val_loader = data_loader(batch_size=3)

    for epoch in range(2):
        for i, (gloss, keypoints) in enumerate(train_loader):
            ...
```

3. Dataset structure

- Each video has series of frames (images).
- One frame contains upper body (exclude face) and hands, 8 + 21 \* 2 = 50 joints. For one 3d key points, four features ([x, y, z, visibility]) are saved.
- Dataset combines 3d key points from all videos. A series of 3d key points from the same frame are flatten into one array.
- Among all videos, the maximum number of frames is 233. All matrices are padded with 0 to keep size = 233.

4. Skeleton connection
```
# pose skeletons and mapping
pose_edges = [(11, 12), (12, 14), (14, 16), (11, 13), (13, 15),
              (12, 24), (11, 23), (24, 23)]
pose_joints_map = {11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 23: 6, 24: 7}

# hand skeletons
hand_edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9,10), (10,11), (11,12),     # Middle
    (0,13), (13,14), (14,15), (15,16),    # Ring
    (0,17), (17,18), (18,19), (19,20)     # Pinky
]
```

### 2.4 Text-Embedding

`text-embeddings` folder contains embeddings files and the script to obtain it.

`bert_text_embedding.ipynb` saves the logic of converting text gloss to embeddings. Bert
large model is used instead of CLIP since Bert is more suitbale for our project which needs word embedding. CLIP in the other hand is more suitbale for phrase and sentence embeddings.
`gloss_embeddings.

`gloss_embeddings.npz`saves both gloss and their embeddings. This could be useful for debugging and visualization.

`text_embeddings.npy` saves embeddings ONLY. It should be the one used for model building and training though the exact usage may vary depending on the design process.
