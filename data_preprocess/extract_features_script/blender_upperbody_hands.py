import bpy
import json

# === Clear existing objects ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# === Load JSON ===
with open(r"D:\Courses\Pattern Recognition\WLASL\WLASL\data\json\00338.json", "r") as f:
    frames = json.load(f)

# === Create joint objects only for expected indices ===
joint_objs = {}

# Unique pose indices (those that ever appear in the JSON)
pose_joint_indices = sorted({lm["index"] for frame in frames for lm in frame["pose"]})

# Create pose joints
for idx in pose_joint_indices:
    name = f"pose_{idx}"
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = name
    joint_objs[name] = obj

# Create 21 joints for each hand
for hand_label in ["left", "right"]:
    for idx in range(21):
        name = f"{hand_label}_hand_{idx}"
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=(0, 0, 0))
        obj = bpy.context.object
        obj.name = name
        joint_objs[name] = obj

# === Animate each frame ===
scale = 2.0  # Optional: change scale for visibility

for frame_idx, frame in enumerate(frames):
    frame_number = frame_idx + 1

    # Pose joints (only if visibility > 0.5)
    for lm in frame["pose"]:
        if lm.get("visibility", 0.0) <= 0.5:
            continue
        idx = lm["index"]
        name = f"pose_{idx}"
        if name in joint_objs:
            x = (lm["x"] - 0.5) * scale
            y = lm["z"] * scale
            z = -(lm["y"] - 0.5) * scale
            obj = joint_objs[name]
            obj.location = (x, y, z)
            obj.keyframe_insert(data_path="location", frame=frame_number)

    # Hand joints (only if visibility > 0.5)
    for hand in frame["hands"]:
        label = hand["label"].lower()
        for lm in hand["landmarks"]:
            if lm.get("visibility", 0.0) <= 0.5:
                continue
            idx = lm["index"]
            name = f"{label}_hand_{idx}"
            if name in joint_objs:
                x = (lm["x"] - 0.5) * scale
                y = lm["z"] * scale
                z = -(lm["y"] - 0.5) * scale
                obj = joint_objs[name]
                obj.location = (x, y, z)
                obj.keyframe_insert(data_path="location", frame=frame_number)
