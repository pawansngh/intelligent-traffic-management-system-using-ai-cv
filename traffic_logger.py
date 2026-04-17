from ultralytics import YOLO
import cv2
import supervision as sv
import time
import csv

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()

cap = cv2.VideoCapture("videos/traffic.mp4")


vehicle_classes = ["car", "bus", "truck", "motorcycle"]

line = sv.LineZone(
    start=sv.Point(100, 300),
    end=sv.Point(500, 300)
)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2)

start_time = time.time()

file = open("traffic_data.csv", "w", newline="")
writer = csv.writer(file)
writer.writerow(["Time(sec)", "Vehicles Passed", "Traffic Status"])

last_log_time = 0

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

    elapsed = time.time() - start_time
    vehicles_per_min = int((line.in_count / elapsed) * 60) if elapsed > 0 else 0

    if vehicles_per_min < 10:
        status = "LOW"
        color = (0,255,0)
    elif vehicles_per_min < 20:
        status = "MEDIUM"
        color = (0,255,255)
    else:
        status = "HIGH"
        color = (0,0,255)

    # log every 5 seconds
    if elapsed - last_log_time >= 5:
        writer.writerow([int(elapsed), line.in_count, status])
        last_log_time = elapsed

    annotated = results.plot()
    annotated = line_annotator.annotate(annotated, line)

    cv2.putText(annotated, f"Vehicles Passed: {line.in_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    cv2.putText(annotated, f"Traffic Status: {status}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Traffic Logger", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()
