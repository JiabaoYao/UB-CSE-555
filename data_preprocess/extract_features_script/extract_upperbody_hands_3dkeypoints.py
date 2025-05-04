import os
import cv2
import json
import numpy as np
import mediapipe as mp

# === Config ===
upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 24]
hand_joint_count = 21
test_flag = False  # Visualization toggle

# === Keypoint processor ===
def build_frame_data(pose_dict, hand_dict):
    frame_data_json = {"pose": [], "hands": []}
    frame_vector = []

    # --- Pose ---
    for idx in upper_body_indices:
        if idx in pose_dict:
            x, y, z, v = pose_dict[idx]
        else:
            x = y = z = v = 0.0
        frame_vector.extend([x, y, z, v])
        frame_data_json["pose"].append({"index": idx, "x": x, "y": y, "z": z, "visibility": v})

    # --- Hands ---
    for hand_label in ["left", "right"]:
        hand_landmarks = []
        for i in range(hand_joint_count):
            if i in hand_dict[hand_label]:
                x, y, z, v = hand_dict[hand_label][i]
            else:
                x = y = z = v = 0.0
            frame_vector.extend([x, y, z, v])
            hand_landmarks.append({"index": i, "x": x, "y": y, "z": z, "visibility": v})
        frame_data_json["hands"].append({"label": hand_label, "landmarks": hand_landmarks})

    return frame_data_json, frame_vector

# === Main extraction function ===
def extract_3d_keypoints(path, output_folder, video_name):
    video_path = os.path.join(path, video_name)
    cap = cv2.VideoCapture(video_path)
    all_data = []
    single_video_matrix = []

    print("Trying to open:", video_path)
    print("Video opened:", cap.isOpened())

    # Init MediaPipe
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)

        pose_dict = {}
        hands_dict = {"left": {}, "right": {}}

        # --- Pose landmarks ---
        if pose_results.pose_landmarks:
            for idx in upper_body_indices:
                lm = pose_results.pose_landmarks.landmark[idx]
                pose_dict[idx] = [lm.x, lm.y, max(min(lm.z, 1.0), -1.0), lm.visibility]
            if test_flag:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- Hands landmarks ---
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_lms, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                label = handedness.classification[0].label.lower()
                for idx, lm in enumerate(hand_lms.landmark):
                    hands_dict[label][idx] = [lm.x, lm.y, max(min(lm.z, 1.0), -1.0), 1.0]
                if test_flag:
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        frame_data, frame_vector = build_frame_data(pose_dict, hands_dict)
        all_data.append(frame_data)
        single_video_matrix.append(frame_vector)

        if test_flag:
            cv2.imshow("Pose + Hands", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # === Save outputs ===
    os.makedirs(os.path.join(output_folder, "json"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "matrix"), exist_ok=True)

    json_out = os.path.join(output_folder, "json", f"{os.path.splitext(video_name)[0]}.json")
    matrix_out = os.path.join(output_folder, "matrix", f"{os.path.splitext(video_name)[0]}.npy")

    with open(json_out, "w") as f:
        json.dump(all_data, f)

    np.save(matrix_out, np.array(single_video_matrix, dtype=np.float32))

    cap.release()
    pose.close()
    hands.close()
    cv2.destroyAllWindows()