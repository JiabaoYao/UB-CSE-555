import pandas as pd
import numpy as np
import os

def save_sort_content_map(output_dir):
    all_data_df = pd.read_csv(r"D:\Courses\Pattern Recognition\UB-CSE-555\data_preprocess\matched_samples.txt", sep='\t', dtype={'video_file': str})
    gloss_counts = all_data_df['gloss'].value_counts()
    all_data_df['count'] = all_data_df['gloss'].map(gloss_counts)
    sorted_df = all_data_df.sort_values(['count', 'gloss'], ascending=[False, True])

    file_path = os.path.join(output_dir, "sort_matched_samples.txt")
    with open(file_path, 'w') as f:
        f.write("video_file\tgloss\n")
        for row in sorted_df.itertuples(index=False):
            f.write(f"{row.video_file}\t{row.gloss}\n")

    return sorted_df

def count_datasize_by_wordnum(word_num, path):
    file_path = os.path.join(path, "sort_matched_samples.txt")
    data_map = pd.read_csv(file_path, sep='\t')
    top_glosses = (
        data_map['gloss']
        .value_counts()
        .nlargest(word_num)
        .index
    )
    print(f"top_glosses:{top_glosses}")
    count = data_map[data_map['gloss'].isin(top_glosses)].shape[0]
    return count, top_glosses

def creat_dataset_by_size(data_df, top_glosses, count, save_path):
    base_path = r"D:\Courses\Pattern Recognition\WLASL\WLASL\dataset\matrix"
    T_max = 0
    data_padding = []

    # filter only top glosses
    filtered_df = data_df[data_df['gloss'].isin(top_glosses)]

    for row in filtered_df.itertuples(index=False):
        file_path = os.path.join(base_path, f"{row.video_file}.npy")
        if os.path.exists(file_path):
            matrix = np.load(file_path)
            T_max = max(T_max, matrix.shape[0])
    print(f"maximum number of frames: {T_max}")

    for i, row in enumerate(filtered_df.itertuples(index=False)):
        file_path = os.path.join(base_path, f"{row.video_file}.npy")
        if os.path.exists(file_path):
            matrix = np.load(file_path)
            T, D = matrix.shape
            if T < T_max:
                pad_matrix = np.zeros((T_max - T, D), dtype=np.float32)
                matrix = np.vstack([matrix, pad_matrix])
            data_padding.append([row.gloss, matrix])

            if i >= count:
                break
        else:
            print(f"{row.video_file}-{row.gloss} miss!!")

    np.save(save_path, np.array(data_padding, dtype=object))

def data_compress(npy_path, npz_path):
    data_samples = np.load(npy_path, allow_pickle=True)
    np.savez_compressed(npz_path, data=data_samples)

if __name__ == '__main__':
    print("Enter word number:")
    word_num = int(input())

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(output_dir, "dataset")
    os.makedirs(output_dir, exist_ok=True)

    data_samples = save_sort_content_map(output_dir)
    count, top_glosses = count_datasize_by_wordnum(word_num, output_dir)
    print(count, top_glosses)

    npy_path = os.path.join(output_dir, "dataset_padding_new.npy")
    npz_path = os.path.join(output_dir, "dataset_padding_new.npz")

    creat_dataset_by_size(data_samples, top_glosses, count, npy_path)
    data_compress(npy_path, npz_path)
