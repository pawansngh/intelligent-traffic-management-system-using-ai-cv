from ultralytics import YOLO
import cv2
import supervision as sv

model = YOLO("yolov8n.pt")

tracker = sv.ByteTrack()

cap = cv2.VideoCapture("videos/traffic.mp4")


vehicle_classes = ["car", "bus", "truck", "motorcycle"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    unique_ids = set()

    for i, class_id in enumerate(detections.class_id):
        label = model.names[class_id]
        if label in vehicle_classes:
            unique_ids.add(detections.tracker_id[i])

    annotated = results.plot()

    cv2.putText(annotated,
                f"Unique Vehicles: {len(unique_ids)}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2)

    cv2.imshow("Vehicle Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
