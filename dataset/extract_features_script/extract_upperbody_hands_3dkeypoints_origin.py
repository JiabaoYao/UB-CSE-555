import cv2
import mediapipe as mp
import json
import os
import numpy as np

# Upper-body pose indices
upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 24]
test_flag = False  # Set to False to disable GUI

def matrix_process(pose_data, hand_data):
    frame_matrix = []

    for idx in upper_body_indices:
        frame_matrix.extend(pose_data.get(idx, [0.0, 0.0, 0.0, 0.0]))

    for idx in range(21):  # Left hand
        frame_matrix.extend(hand_data["left"].get(idx, [0.0, 0.0, 0.0, 0.0]))

    for idx in range(21):  # Right hand
        frame_matrix.extend(hand_data["right"].get(idx, [0.0, 0.0, 0.0, 0.0]))

    return frame_matrix

def extract_3d_keypoints(path, output_folder, video_name):
    video_path = os.path.join(path, video_name)
    cap = cv2.VideoCapture(video_path)
    all_data = []
    single_video_matrix = []

    print("Trying to open:", video_path)

    # Init MediaPipe
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    print("Video opened:", cap.isOpened())

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)

        frame_data = {"pose": [], "hands": []}
        pose_dict = {}
        hands_dict = {"left": {}, "right": {}}

        if pose_results.pose_landmarks:
            for idx in upper_body_indices:
                lm = pose_results.pose_landmarks.landmark[idx]
                if lm.visibility > 0.5:
                    frame_data["pose"].append({
                        "index": idx,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    })
                pose_dict[idx] = [lm.x, lm.y, lm.z, lm.visibility]
            if test_flag:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_lms, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                label = handedness.classification[0].label.lower()  # "left" or "right"
                for idx, lm in enumerate(hand_lms.landmark):
                    hands_dict[label][idx] = [lm.x, lm.y, lm.z, 1.0]

                landmarks = [{
                    "index": idx,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": 1.0
                } for idx, lm in enumerate(hand_lms.landmark)]

                frame_data["hands"].append({
                    "label": label,
                    "landmarks": landmarks
                })

                if test_flag:
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        single_video_matrix.append(matrix_process(pose_dict, hands_dict))
        all_data.append(frame_data)

        if test_flag:
            cv2.imshow("Pose + Hands", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # Ensure output folders exist
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
