import cv2
import numpy as np
import os

VIDEO = r"input\Profit and Loss 2 (23rd June & 9 PM).mp4"
OUT = "selected"

os.makedirs(OUT, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
sample_fps = 2               # analyze 2 frames/sec
threshold = 35               # adjust 25-60

prev_frame = None
prev_gray = None
prev_time = 0

saved = 0

while True:

    cap.set(cv2.CAP_PROP_POS_MSEC, prev_time * 1000)

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:

        diff = cv2.absdiff(prev_gray, gray)

        score = np.mean(diff)

        if score > threshold:

            filename = os.path.join(
                OUT,
                f"{prev_time:.1f}.jpg"
            )

            cv2.imwrite(filename, prev_frame)

            print(
                f"Saved {filename}  Score={score:.2f}"
            )

            saved += 1

    prev_gray = gray
    prev_frame = frame.copy()

    prev_time += 1 / sample_fps

cap.release()

print(f"\nDone. Saved {saved} screenshots.")