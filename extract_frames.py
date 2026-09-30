import cv2
import os

video_path = "input/Profit and Loss 2 (23rd June & 9 PM).mp4"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

frame_interval = int(fps * 10)   # every 10 seconds

count = 0
saved = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if count % frame_interval == 0:
        timestamp = int(count / fps)

        filename = f"{output_folder}/frame_{timestamp}.jpg"

        cv2.imwrite(filename, frame)

        saved += 1

    count += 1

cap.release()

print(f"Saved {saved} frames.")