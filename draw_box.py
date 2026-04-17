import cv2

cap = cv2.VideoCapture("videos/traffic.mp4")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # draw a rectangle in center
    x1 = int(w * 0.3)
    y1 = int(h * 0.3)
    x2 = int(w * 0.7)
    y2 = int(h * 0.7)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Test Box", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Drawing Test", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
