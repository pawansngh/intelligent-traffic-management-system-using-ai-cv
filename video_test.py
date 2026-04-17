import cv2

import os

video_folder = "videos"
video_files = os.listdir(video_folder)

for video in video_files:
    path = os.path.join(video_folder, video)
    cap = cv2.VideoCapture(path)
    print(f"Processing: {video} | opened={cap.isOpened()}")
    cap.release()
