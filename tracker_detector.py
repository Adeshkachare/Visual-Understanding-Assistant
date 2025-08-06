# tracker_detector.py

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os

def track_objects(frames_folder, output_folder, model_path='yolov8n.pt'):
    model = YOLO(model_path)
    tracker = DeepSort(max_age=30)

    results = []
    frame_files = sorted(os.listdir(frames_folder))

    for frame_index, file in enumerate(frame_files):
        frame_path = os.path.join(frames_folder, file)
        frame = cv2.imread(frame_path)

        detections = model(frame)[0].boxes
        det_list = []
        for box in detections:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            det_list.append(([xyxy[0], xyxy[1], xyxy[2], xyxy[3]], conf, label))

        tracks = tracker.update_tracks(det_list, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            label = track.get_det_class()
            results.append({
                "timestamp": frame_index,
                "track_id": track_id,
                "label": label,
                "bbox": [l, t, r, b]
            })

            # Optional: draw + save frame
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {track_id}", (int(l), int(t) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out_path = os.path.join(output_folder, file)
        cv2.imwrite(out_path, frame)

    return results
