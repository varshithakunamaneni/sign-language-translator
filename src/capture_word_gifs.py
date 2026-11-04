"""
Record a short looping GIF of each WORD sign, so people who don't know ISL
(e.g. judges) can watch it and copy it. Shown in the app and the README.

Usage (from the project root) - ideally the same person who recorded the word data:
    python src/capture_word_gifs.py                    # all 14 words
    python src/capture_word_gifs.py --words PAIN       # retake one word

Controls (click the camera window first):
    SPACE -> record this word (1 s get ready, then 2 s of signing)
    N     -> skip      Q -> quit (GIFs made so far are kept)

Tips: start with hands down, do the whole sign once, slowly, then lower your hands.
Saved to web/word_signs/HELLO.gif, THANK_YOU.gif, ...
"""
import argparse
import os
import time

import cv2
import numpy as np
from PIL import Image

from utils import ROOT, create_hand_detector, detect_hands

OUT_DIR = os.path.join(ROOT, "web", "word_signs")
WORDS = ["HELLO", "THANK YOU", "PLEASE", "YES", "NO", "HELP", "DOCTOR", "PAIN", "WATER",
         "TOILET", "WHERE", "FAMILY", "NAME", "DEAF"]
SIZE = 260            # GIF size in pixels (square)
RECORD_SECONDS = 2.0


def file_name(word):
    return word.replace(" ", "_") + ".gif"


def crop_box(boxes, w, h, pad=0.35):
    """One square around everywhere the hands went during the clip."""
    if not boxes:                                   # no hands seen: centre square
        side = min(w, h)
        return (w - side) // 2, (h - side) // 2, side
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    side = int(min(max(x1 - x0, y1 - y0) * (1 + 2 * pad), w, h))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x = int(np.clip(cx - side / 2, 0, w - side)); y = int(np.clip(cy - side / 2, 0, h - side))
    return x, y, side


def save_gif(frames, boxes, path, fps=12):
    """frames: list of BGR images (already mirrored). Keeps ~12 frames per second."""
    h, w = frames[0].shape[:2]
    x, y, side = crop_box(boxes, w, h)
    step = max(1, round(len(frames) / (RECORD_SECONDS * fps)))
    imgs = []
    for f in frames[::step]:
        crop = cv2.resize(f[y:y + side, x:x + side], (SIZE, SIZE))
        imgs.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).quantize(colors=128))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return len(imgs)


def text(img, s, y, color=(255, 255, 255), scale=0.9):
    cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="+", default=WORDS)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    words = [" ".join(w.upper().split()) for w in args.words]

    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1")
    detector = create_hand_detector(video_mode=True)
    start = time.time()

    def grab():
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Camera stopped.")
        frame = cv2.flip(frame, 1)       # mirror view: easiest for viewers to copy
        res = detect_hands(detector, frame, (time.time() - start) * 1000)
        h, w = frame.shape[:2]
        boxes = [(min(p.x for p in hand) * w, min(p.y for p in hand) * h,
                  max(p.x for p in hand) * w, max(p.y for p in hand) * h) for hand in (res.hand_landmarks or [])]
        return frame, boxes

    for i, word in enumerate(words):
        while True:
            frame, _ = grab()
            view = frame.copy()
            text(view, f"Word: {word}  ({i + 1}/{len(words)})", 36, (0, 220, 255), 1.1)
            text(view, "SPACE = record   N = skip   Q = quit", 72, scale=0.7)
            cv2.imshow("Record word-sign GIFs", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                cap.release(); cv2.destroyAllWindows(); print("Stopped. GIFs are in", OUT_DIR); return
            if k == ord("n"):
                print(f"{word}: skipped"); break
            if k == ord(" "):
                t_end = time.time() + 1.0
                while time.time() < t_end:                       # get ready
                    frame, _ = grab(); view = frame.copy()
                    text(view, f"{word}: get ready... hands down", 60, (0, 165, 255), 1.0)
                    cv2.imshow("Record word-sign GIFs", view); cv2.waitKey(1)
                frames, boxes = [], []
                t0 = time.time()
                while time.time() - t0 < RECORD_SECONDS:         # record
                    frame, bx = grab(); frames.append(frame.copy()); boxes += bx
                    view = frame.copy()
                    text(view, f"{word}: SIGN NOW", 60, (0, 0, 255), 1.3)
                    cv2.rectangle(view, (12, 80), (12 + int(600 * (time.time() - t0) / RECORD_SECONDS), 92), (0, 0, 255), -1)
                    cv2.imshow("Record word-sign GIFs", view); cv2.waitKey(1)
                n = save_gif(frames, boxes, os.path.join(OUT_DIR, file_name(word)))
                print(f"{word}: saved ({n} frames)")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done! GIFs are in", OUT_DIR)


if __name__ == "__main__":
    main()
