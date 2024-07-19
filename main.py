import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk

import pandas as pd
from scipy.spatial.distance import euclidean


class ActionRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("800x700")

        self.cv2offset = 50
        self.counts = []

        self.width = 800
        self.height = 600
        self.standard_push_up = pd.read_csv('standard_push_up.csv', header=None).values.flatten()
        self.standard_pull_up = pd.read_csv('standard_pull_up.csv', header=None).values.flatten()
        self.init_mediapipe()
        self.load_model()
        self.setup_gui()
        self.reset_variables()

        self.window.mainloop()

    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array(a)  # 第一个点
        b = np.array(b)  # 中间点（关节）
        c = np.array(c)  # 最后一个点

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def init_mediapipe(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_hands = mp.solutions.hands
        # self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.35, min_tracking_confidence=0.5)
        # self.mp_draw = mp.solutions.drawing_utils

    def load_model(self):
        self.model = joblib.load('action_recognition_model.joblib')
        self.label_encoder = joblib.load('label_encoder.joblib')

    def setup_gui(self):
        self.start_button = tk.Button(self.window, text="开始", width=25, command=self.toggle_capture)
        self.start_button.pack(pady=20)
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.pack()

    def reset_variables(self):
        self.is_capturing = False
        self.delay = 15
        self.prev_landmarks = None
        self.movement_threshold = 0.01
        self.state = "neutral"
        self.count = 0
        self.queue_len = 15
        self.predictions = deque(maxlen=self.queue_len)
        self.is_ready = False
        self.current_action = None
        self.action_count = 0

    def toggle_capture(self):
        if not self.is_capturing:
            self.start_capture()
        else:
            self.stop_capture()

    def start_capture(self):
        self.is_capturing = True
        self.cap = cv2.VideoCapture("train.mp4")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.start_button.config(text="停止")
        self.update_frame()

    def stop_capture(self):
        self.is_capturing = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.start_button.config(text="开始")
        self.canvas.delete("all")

        self.counts.append(f"{self.current_action}: {self.action_count}")
        self.show_results()

    def update_frame(self):
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))
                self.process_frame(frame)

            self.window.after(self.delay, self.update_frame)

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not self.is_ready:
            self.is_ready = self.check_ready(results.pose_landmarks)
            if results.pose_landmarks:
                self.draw_pose_landmarks(frame, results.pose_landmarks)
            mask = self.create_human_mask(frame)
            frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
            cv2.putText(frame, "Not Ready", (10, self.cv2offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if self.is_ready:
            self.handle_ready_state(frame, results)

        self.display_frame(frame)

    def draw_angles(self, frame, results):
        landmarks = results.pose_landmarks.landmark

        # 获取需要的关键点坐标
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # 计算肘部角度
        angle = self.calculate_angle(shoulder, elbow, wrist)

        # 可视化角度
        cv2.putText(frame, str(int(angle)),
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

    def euclidean_similarity(self, v1, v2):
        distance = euclidean(v1, v2)
        return 1 / (1 + distance)

    def handle_ready_state(self, frame, results):
        cv2.rectangle(frame, (10, 10), (60, 60), (0, 0, 255), 2)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

            h, w, c = frame.shape

            # 转换左手腕坐标
            left_cx, left_cy = int(left_wrist.x * w), int(left_wrist.y * h)

            # 转换右手腕坐标
            right_cx, right_cy = int(right_wrist.x * w), int(right_wrist.y * h)

            # 检查左手或右手是否在红色方框内
            if (10 < left_cx < 60 and 10 < left_cy < 60) or (10 < right_cx < 60 and 10 < right_cy < 60):
                self.stop_capture()

            self.draw_pose_landmarks(frame, results.pose_landmarks)
            keypoints = self.extract_keypoints(results.pose_landmarks)
            movement = self.get_vertical_movement(results.pose_landmarks)
            self.prev_landmarks = results.pose_landmarks

            pred = self.model.predict(keypoints.reshape(1, -1))[0]

            action = self.label_encoder.inverse_transform([pred])[0]
            self.predictions.append(action)

            self.update_action_count(movement)

            self.draw_angles(frame, results)  # TODO

            similarity = None
            if action == "push_up":
                similarity = self.euclidean_similarity(self.standard_push_up, list(keypoints))

            if action == "pull_up":
                similarity = self.euclidean_similarity(self.standard_pull_up, list(keypoints))

            if similarity:
                cv2.putText(frame, f"score:{similarity}", (10, self.cv2offset + 40 * 3), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                if similarity > 0.5:
                    cv2.putText(frame, f"score:{similarity}", (10, self.cv2offset + 40 * 3), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

            self.display_action_info(frame)

        else:
            cv2.putText(frame, "No person detected", (10, self.cv2offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

    def display_frame(self, frame):
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def draw_pose_landmarks(self, frame, pose_landmarks):
        self.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    def extract_keypoints(self, pose_landmarks):
        keypoints = []
        for landmark in self.get_needed_landmarks():
            point = pose_landmarks.landmark[landmark]
            keypoints.extend([point.x, point.y, point.z, point.visibility])
        return np.array(keypoints)

    def get_needed_landmarks(self):
        return [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

    def get_vertical_movement(self, current_landmarks):
        if self.prev_landmarks is None or current_landmarks is None:
            return 0

        current_y = self.get_average_y(current_landmarks)
        prev_y = self.get_average_y(self.prev_landmarks)
        return prev_y - current_y

    def get_average_y(self, landmarks):
        return np.mean([
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y,
            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y,
            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y
        ])

    def check_ready(self, landmarks):
        if landmarks is None:
            return False
        for landmark in self.get_needed_landmarks():
            if landmarks.landmark[landmark].visibility < 0.5:
                return False
        return True

    @staticmethod
    def create_human_mask(frame):
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        head_radius = max(center_x, center_y) // 10
        body_width = int(head_radius * 2.5)
        body_height = head_radius * 4
        limb_width = body_width * 3 // 8

        mask = np.zeros_like(frame)

        # Head
        cv2.circle(mask, (center_x, center_y - body_height + head_radius // 2), head_radius, (255, 255, 255), -1)

        # Body
        cv2.rectangle(mask, (center_x - body_width // 2, center_y - body_height // 2),
                      (center_x + body_width // 2, center_y + body_height // 2), (255, 255, 255), -1)

        # Legs
        cv2.line(mask, (center_x - body_width // 2 + limb_width // 2, center_y + body_height // 2),
                 (center_x - body_width // 2 + limb_width // 2, center_y + body_height * 3 // 2), (255, 255, 255),
                 limb_width)
        cv2.line(mask, (center_x + body_width // 2 - limb_width // 2, center_y + body_height // 2),
                 (center_x + body_width // 2 - limb_width // 2, center_y + body_height * 3 // 2), (255, 255, 255),
                 limb_width)

        # Arms
        cv2.line(mask, (center_x - body_width // 2, center_y - body_height // 2 + limb_width // 2),
                 (center_x - body_width // 2 - head_radius, center_y + head_radius), (255, 255, 255), limb_width)
        cv2.line(mask, (center_x + body_width // 2, center_y - body_height // 2 + limb_width // 2),
                 (center_x + body_width // 2 + head_radius, center_y + head_radius), (255, 255, 255), limb_width)

        return mask

    def update_action_count(self, movement):
        if len(self.predictions) == self.queue_len:
            most_common = max(set(self.predictions), key=self.predictions.count)
            if self.predictions.count(most_common) >= self.queue_len // 3 * 2:

                if most_common != self.current_action:
                    self.counts.append(f"{self.current_action}: {self.action_count}")
                    self.current_action = most_common
                    self.action_count = 0
                if most_common != 'ignore':
                    if movement > self.movement_threshold and self.state != "up":
                        self.state = "up"
                    elif movement < -self.movement_threshold and self.state == "up":
                        self.state = "down"
                        self.action_count += 1

    def display_action_info(self, frame):
        cv2.putText(frame, f"Action: {self.current_action}", (10, self.cv2offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {self.action_count}", (10, self.cv2offset + 40 * 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    def show_results(self):
        # smoothed_predictions = self.smooth_predictions(self.predictions)
        result_window = tk.Toplevel(self.window)
        result_window.title("结果")

        text_widget = tk.Text(result_window, height=20, width=40)
        text_widget.pack()

        for i, count in enumerate(self.counts):
            text_widget.insert(tk.END, f"{i} {count}\n")

        # for i, action in enumerate(smoothed_predictions):
        #     text_widget.insert(tk.END, f"帧 {i}: {action}\n")

    # def smooth_predictions(self, preds, window_size=5):
    #     smoothed = []
    #     history = deque(maxlen=window_size)
    #     for pred in preds:
    #         history.append(pred)
    #         smoothed.append(max(set(history), key=history.count))
    #     return smoothed


if __name__ == "__main__":
    print(f"Model expects {joblib.load('action_recognition_model.joblib').n_features_in_} features")
    ActionRecognitionApp(tk.Tk(), "动作识别应用")

# TODO: 自更新（增强稳定性）
# TODO: android

# TODO: review code
# TODO: fine tune

# TODO: Jump rope from side
