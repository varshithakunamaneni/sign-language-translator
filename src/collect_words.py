"""
Record WORD signs (moving gestures) such as HELLO, THANK YOU, HELP.

Usage (from the project root):
    python src/collect_words.py --person harshita
    python src/collect_words.py --person harshita --words HELP WATER     # only some words
    python src/collect_words.py --person harshita --clips 20             # more clips per word

How it works:
    * Each word is recorded as short clips of 1.5 seconds.
    * Press S once per word: it then records all clips automatically:
        "Get ready" (1 s)  ->  "SIGN NOW" (1.5 s)  ->  next clip ...
    * Do the full sign inside the 1.5 s, starting from hands down / relaxed.
    * "NOTHING" = hands resting or moving casually WITHOUT signing. This teaches
      the app when NOT to type a word, so record it like the others.

Controls (click the camera window first):
    S -> start recording the current word      N -> skip word      Q -> quit
"""
import argparse
import csv
import os
import time

import cv2

from utils import (ROOT, NUM_MOTION_FEATURES, WORD_FRAMES, WORD_SECONDS, create_hand_detector,
                   detect_hands, draw_hands, extract_motion_features, resample_clip)

DATA_FILE = os.path.join(ROOT, "data", "words.csv")
DEFAULT_WORDS = ["HELLO", "THANK YOU", "PLEASE", "SORRY", "YES", "NO", "HELP", "DOCTOR", "PAIN",
                 "WATER", "FOOD", "TOILET", "HOME", "GOOD", "WHERE", "DEAF", "UNDERSTAND", "NAME",
                 "LEARN", "FAMILY", "NOTHING"]
READY_SECONDS = 1.0


def ensure_csv():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["label", "person"] + [f"f{i}" for i in range(WORD_FRAMES * NUM_MOTION_FEATURES)])


def put(img, s, y, color=(255, 255, 255), scale=0.9, thick=2):
    cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--words", nargs="+", default=DEFAULT_WORDS)
    ap.add_argument("--clips", type=int, default=15, help="clips per word")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    words = [" ".join(w.upper().split()) for w in args.words]

    ensure_csv()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1")
    detector = create_hand_detector(video_mode=True)
    start = time.time()

    def grab():
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Camera stopped.")
        frame = cv2.flip(frame, 1)                       # same mirroring as the letters
        result = detect_hands(detector, frame, (time.time() - start) * 1000)
        feats, n = extract_motion_features(result)
        draw_hands(frame, result)
        return frame, feats, n

    for wi, word in enumerate(words):
        # ---- wait for S ----
        while True:
            frame, _, n = grab()
            put(frame, f"Word: {word}   ({wi + 1}/{len(words)})", 40, (0, 220, 255), 1.1, 3)
            put(frame, f"Press S to record {args.clips} clips,  N = skip,  Q = quit", 78, scale=0.7)
            if word == "NOTHING":
                put(frame, "Do NOT sign: rest your hands, scratch, move casually", 114, (0, 200, 255), 0.7)
            cv2.imshow("Record word signs", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                cap.release(); cv2.destroyAllWindows(); print("Stopped. Saved to", DATA_FILE); return
            if k == ord("n"):
                break
            if k == ord("s"):
                # ---- record all clips for this word ----
                saved = 0
                while saved < args.clips:
                    t_end = time.time() + READY_SECONDS
                    while time.time() < t_end:                       # get ready
                        frame, _, _ = grab()
                        put(frame, f"{word}: clip {saved + 1}/{args.clips}", 40, (0, 220, 255), 1.0, 3)
                        put(frame, "Get ready... hands down", 90, (0, 165, 255), 1.1, 3)
                        cv2.imshow("Record word signs", frame)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            cap.release(); cv2.destroyAllWindows(); return
                    times, frames = [], []
                    t0 = time.time()
                    while time.time() - t0 < WORD_SECONDS:            # record
                        frame, feats, _ = grab()
                        times.append(time.time()); frames.append(feats)
                        put(frame, f"{word}: clip {saved + 1}/{args.clips}", 40, (0, 220, 255), 1.0, 3)
                        put(frame, "SIGN NOW", 100, (0, 0, 255), 1.6, 4)
                        bar = int(600 * (time.time() - t0) / WORD_SECONDS)
                        cv2.rectangle(frame, (12, 120), (12 + bar, 132), (0, 0, 255), -1)
                        cv2.imshow("Record word signs", frame)
                        cv2.waitKey(1)
                    hands_seen = sum(1 for f in frames if f[:126].any())
                    if word != "NOTHING" and hands_seen < len(frames) * 0.3:
                        print(f"  {word}: clip skipped (hands not visible), recording it again")
                        continue
                    row = resample_clip(times, frames)
                    with open(DATA_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([word, args.person] + [f"{v:.4f}" for v in row])
                    saved += 1
                print(f"{word}: {saved} clips saved")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done! Saved to", DATA_FILE)


if __name__ == "__main__":
    main()
