# **Multimodal 3D Sign Language Translation**

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

5. After finishing generating all matrix and json files, combine them via `data_match_process.ipynb` to generate `dataset_all_padding.npz`. This is the input include all gloss and 3d keypoints matrix of all vaild videos.

### 2.3 Dataset and loader

`dataset_loader.py` contains the integrated Dataset ([N, 2]) definition and a loader function. Use the loader as model input.

1. How to use it?

```
    loader = data_loader(batch_size=10)

    for epoch in range(2):
        for i, (gloss, keypoints) in enumerate(loader):
            ...
```

2. Dadaset structure

- Each video has series of frames (images).
- One frame contains upper body and hands, 15 + 21 \* 2 = 57 joints. For one 3d key points, four features ([x, y, z, visibility]) are saved.
- Dataset combines 3d key points from all videos. A series of 3d key points from the same frame are flatten into one array.
- Among all videos, the maximum number of frames is 233. All matrix are padding with 0 to keep size = 233.

### 2.4 Text-Embedding

`text-embeddings` folder contains embeddings files and the script to obtain it.  
`bert_text_embedding.ipynb` saves the logic of converting text gloss to embeddings. Bert
large model is used instead of CLIP since Bert is more suitbale for our project which needs word embedding. CLIP in the other hand is more suitbale for phrase and sentence embeddings.
`gloss_embeddings.  
`gloss_embeddings.npz`saves both gloss and their embeddings. This could be useful for debugging and visualization.`text_embeddings.npy` saves embeddings ONLY. It should be the one used for model building and training though the exact usage may vary depending on the design process.
