import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

needed_landmarks = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]


def extract_keypoints(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        keypoints = []
        for landmark in needed_landmarks:
            point = results.pose_landmarks.landmark[landmark]
            keypoints.extend([point.x, point.y, point.z, point.visibility])
        keypoints = np.array(keypoints)
    else:
        keypoints = np.zeros(len(needed_landmarks) * 4)

    return keypoints


def process_video(video_path, output_file):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame'] + [f'kp_{i}' for i in range(len(needed_landmarks) * 4)])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = extract_keypoints(frame)

            writer.writerow([frame_count] + list(keypoints))

            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

    cap.release()
    print(f"Finished processing {video_path}")


if __name__ == "__main__":
    video_path = "train2.mp4"
    output_dir = "./features"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"{os.path.splitext(video_path)[0]}_features.csv")
    process_video(video_path, output_file)
