from PIL import Image
import imagehash
import os
import shutil

INPUT = "selected"
OUTPUT = "final"

THRESHOLD = 5      # smaller = stricter

os.makedirs(OUTPUT, exist_ok=True)

hashes = []

for file in sorted(os.listdir(INPUT)):

    if not file.lower().endswith(".jpg"):
        continue

    path = os.path.join(INPUT, file)

    img = Image.open(path)

    h = imagehash.phash(img)

    duplicate = False

    for old_hash in hashes:

        if abs(h - old_hash) <= THRESHOLD:
            duplicate = True
            break

    if not duplicate:
        hashes.append(h)
        shutil.copy(path, os.path.join(OUTPUT, file))
        print(f"Saved {file}")

print("\nDone")