from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

img = cv2.imread("test.jpg")

results = model(img)

annotated = results[0].plot()

cv2.imshow("YOLO Test", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
