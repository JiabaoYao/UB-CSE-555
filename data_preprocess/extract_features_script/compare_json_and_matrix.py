import json
import numpy as np

# Load JSON
with open(r"D:\Courses\Pattern Recognition\WLASL\WLASL\data\json\00414.json", "r") as f:
    json_data = json.load(f)

# Load Matrix
matrix = np.load(r"D:\Courses\Pattern Recognition\WLASL\WLASL\data\matrix\00414.npy")

frame_idx = 0
frame = json_data[frame_idx]
matrix_frame = matrix[frame_idx]

upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 24]

print("Pose comparison:")
for i, idx in enumerate(upper_body_indices):
    # Position in matrix
    start = i * 4
    mat_x, mat_y, mat_z, mat_vis = matrix_frame[start:start+4]

    # Look for corresponding index in JSON
    json_joint = next((j for j in frame["pose"] if j["index"] == idx), None)

    if json_joint:
        print(f"Pose[{idx}] OK ✔️")
        print(f"  Matrix: {mat_x:.4f}, {mat_y:.4f}, {mat_z:.4f}, {mat_vis:.4f}")
        print(f"  JSON:   {json_joint['x']:.4f}, {json_joint['y']:.4f}, {json_joint['z']:.4f}, {json_joint['visibility']:.4f}")
    else:
        print(f"Pose[{idx}] missing in JSON ❌ → Matrix value was: {mat_x:.4f}, {mat_y:.4f}, {mat_z:.4f}, {mat_vis:.4f}")

print("\nLeft hand comparison:")
offset = len(upper_body_indices) * 4
for i in range(21):
    start = offset + i * 4
    mat = matrix_frame[start:start+4]
    print(f"Left hand joint {i}: {mat}")

print("\nRight hand comparison:")
offset = len(upper_body_indices) * 4 + 21 * 4  # pose + left hand
for i in range(21):  # right hand joints
    start = offset + i * 4
    mat = matrix_frame[start:start + 4]
    print(f"Right hand joint {i}: {mat}")
