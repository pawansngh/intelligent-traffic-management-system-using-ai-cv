import cv2

cap = cv2.VideoCapture("videos/traffic.mp4")

if not cap.isOpened():
    raise RuntimeError("Unable to open videos/traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Traffic Video", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
