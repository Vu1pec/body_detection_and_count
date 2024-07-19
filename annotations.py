import cv2
import os
import json


def annotate_video(video_path, output_file):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    annotations = []
    current_action = None
    start_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Current Action: {current_action}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            current_action = 'idle'
        elif key == ord('j'):
            current_action = 'jump_rope'
        elif key == ord('p'):
            current_action = 'pull_up'
        elif key == ord('d'):
            current_action = 'ignore'
        elif key == ord('e'):
            if current_action:
                annotations.append({
                    'action': current_action,
                    'start_frame': start_frame,
                    'end_frame': frame_count
                })
                current_action = None

        frame_count += 1
        # 241 723
        if frame_count > 239:
            cv2.waitKey(0)

        if current_action is None:
            start_frame = frame_count

    cap.release()
    cv2.destroyAllWindows()

    with open(output_file, 'w') as f:
        json.dump(annotations, f)


if __name__ == "__main__":
    video_path = "train2.mp4"
    annotation_dir = "./annotations"

    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)

    annotation_file = os.path.join(annotation_dir, f"{os.path.splitext(video_path)[0]}_annotation.json")
    annotate_video(video_path, annotation_file)