"""
Take one clear photo of each ISL letter. These pictures are shown to the deaf person
in the "Reply" tab, so they can SEE the hearing person's words in sign language.

Usage (from the project root):
    python src/capture_signs.py                 # all letters A-Z
    python src/capture_signs.py --labels M N    # retake only some letters

Controls (click the camera window first):
    SPACE -> take the photo for the current letter (hands must be visible)
    N     -> skip this letter
    Q     -> quit (photos taken so far are already saved)

Tips: plain wall behind you, good light, sign slowly and clearly, hands in the middle.
Photos are saved to web/signs/A.jpg, B.jpg, ... (cropped around the hands).
"""
import argparse
import os
import string
import time

import cv2
import numpy as np

from utils import ROOT, create_hand_detector, detect_hands, draw_hands

OUT_DIR = os.path.join(ROOT, "web", "signs")
SIZE = 320  # saved image size (pixels, square)


def hands_box(result, w, h, pad=0.35):
    """Square box around all detected hands, with some padding."""
    xs, ys = [], []
    for hand in result.hand_landmarks:
        xs += [p.x * w for p in hand]
        ys += [p.y * h for p in hand]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    side = max(x1 - x0, y1 - y0) * (1 + pad * 2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(side, w, h)
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return x0, y0, int(side)


def text(img, s, y, color=(255, 255, 255), scale=0.9):
    cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=list(string.ascii_uppercase))
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    labels = [l.upper() for l in args.labels]

    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1")
    detector = create_hand_detector(video_mode=True)
    start = time.time()

    for i, label in enumerate(labels):
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Camera stopped.")
            h, w = frame.shape[:2]
            # Detect on the real (un-mirrored) frame: the saved photo shows the sign
            # the way a person sitting in front of you sees it.
            result = detect_hands(detector, frame, (time.time() - start) * 1000)
            n = len(result.hand_landmarks or [])

            view = frame.copy()
            draw_hands(view, result)
            if n:
                x0, y0, s = hands_box(result, w, h)
                cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), (0, 200, 255), 2)
            view = cv2.flip(view, 1)  # mirror only for the preview, feels natural
            text(view, f"Letter {label}  ({i + 1}/{len(labels)})", 36, (0, 220, 255), 1.1)
            text(view, "SPACE = take photo   N = skip   Q = quit", 72, scale=0.7)
            if not n:
                text(view, "Show your hand(s)", 108, (0, 0, 255), 0.8)
            cv2.imshow("Capture ISL letters", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release(); cv2.destroyAllWindows()
                print("Stopped. Photos are in", OUT_DIR)
                return
            if key == ord("n"):
                print(f"{label}: skipped")
                break
            if key == ord(" ") and n:
                crop = cv2.resize(frame[y0:y0 + s, x0:x0 + s], (SIZE, SIZE))
                cv2.imwrite(os.path.join(OUT_DIR, f"{label}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                print(f"{label}: saved")
                text(view, "Saved!", 150, (0, 255, 0), 1.4)
                cv2.imshow("Capture ISL letters", view)
                cv2.waitKey(500)
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done! Photos are in", OUT_DIR)


if __name__ == "__main__":
    main()
