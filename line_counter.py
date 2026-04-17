from ultralytics import YOLO
import cv2
import supervision as sv

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
cap = cv2.VideoCapture("videos/traffic.mp4")


vehicle_classes = ["car", "bus", "truck", "motorcycle"]

line = sv.LineZone(
    start=sv.Point(100, 300),
    end=sv.Point(500, 300)
)

line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    mask = [model.names[c] in vehicle_classes for c in detections.class_id]
    detections = detections[mask]

    line.trigger(detections)

    annotated = results.plot()
    annotated = line_annotator.annotate(annotated, line)

    cv2.putText(annotated,
                f"Vehicles Passed: {line.in_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)

    cv2.imshow("Line Counter", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
