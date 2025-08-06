from ultralytics import YOLO
import cv2
import os
from collections import defaultdict

model = YOLO("yolov8n.pt")  # Load once globally

def detect_objects_in_frames(frames_folder, output_folder):
    frame_paths = sorted(os.listdir(frames_folder))
    results = []

    for i, frame_name in enumerate(frame_paths):
        frame_path = os.path.join(frames_folder, frame_name)
        output_path = os.path.join(output_folder, frame_name)

        frame = cv2.imread(frame_path)
        result = model.predict(frame, save=False)[0]

        label_counts = defaultdict(int)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            label_counts[label] += 1

        # Save the annotated frame
        annotated = result.plot()
        cv2.imwrite(output_path, annotated)

        # ✅ Matching what event_summarizer expects
        results.append({
            "timestamp": i,
            "objects": dict(label_counts)
        })

    return results
