from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/traffic.mp4")


vehicle_classes = ["car", "bus", "truck", "motorcycle"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in vehicle_classes:
            count += 1

    annotated = results.plot()

    cv2.putText(annotated, f"Vehicle Count: {count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Vehicle Counting", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
