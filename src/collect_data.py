"""
STEP 2: Collect training data from your webcam.

Usage (run from the project root):
    python src/collect_data.py --person harshita --labels A B C
    python src/collect_data.py --person harshita --labels A-Z
    python src/collect_data.py --person harshita --labels HELLO THANKYOU --samples 200

Controls in the camera window:
    S  -> start recording the current sign (wait for the 3-second countdown)
    N  -> skip to the next sign
    Q  -> quit (everything recorded so far is already saved)

Tips for GOOD data (this decides your accuracy!):
    * Move your hand slightly while recording: closer/further, a little left/right, small tilts.
    * Record each sign from EVERY teammate (different hand sizes = model generalizes).
    * Use different lighting/backgrounds across sessions.
"""
import argparse
import csv
import os
import string
import time

import cv2

from utils import ROOT, NUM_FEATURES, create_hand_detector, detect_hands, draw_hands, extract_features

DATA_FILE = os.path.join(ROOT, "data", "landmarks.csv")


def parse_labels(raw):
    labels = []
    for item in raw:
        if item.upper() == "A-Z":
            labels += list(string.ascii_uppercase)
        elif item.upper() == "0-9":
            labels += [str(i) for i in range(10)]
        else:
            labels.append(item.upper())
    return labels


def ensure_csv():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["label", "person"] + [f"f{i}" for i in range(NUM_FEATURES)])


def put_text(frame, text, y, color=(255, 255, 255), scale=0.8):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="your name (used to test on unseen people)")
    ap.add_argument("--labels", nargs="+", required=True, help="e.g. A B C, or A-Z, or HELLO")
    ap.add_argument("--samples", type=int, default=300, help="samples per sign")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    labels = parse_labels(args.labels)
    ensure_csv()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1")

    detector = create_hand_detector(video_mode=True)
    start_time = time.time()

    for label in labels:
        recording, countdown_end, count = False, None, 0
        while count < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view, feels natural
            ts = (time.time() - start_time) * 1000
            result = detect_hands(detector, frame, ts)
            feats, n_hands = extract_features(result)
            draw_hands(frame, result)

            put_text(frame, f"Sign: {label}   ({labels.index(label) + 1}/{len(labels)})", 35, (0, 255, 255), 1.0)
            put_text(frame, f"Saved: {count}/{args.samples}   Hands: {n_hands}", 70)

            if countdown_end and time.time() < countdown_end:
                put_text(frame, f"Get ready... {int(countdown_end - time.time()) + 1}", 110, (0, 165, 255), 1.2)
            elif countdown_end:
                recording, countdown_end = True, None

            if recording:
                put_text(frame, "RECORDING - move your hand a little", 110, (0, 0, 255))
                if n_hands > 0:  # only save frames where a hand is visible
                    with open(DATA_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([label, args.person] + [f"{v:.5f}" for v in feats])
                    count += 1
            elif not countdown_end:
                put_text(frame, "Press S to record, N to skip, Q to quit", 110)

            cv2.imshow("Collect Data", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and not recording:
                countdown_end = time.time() + 3
            elif key == ord("n"):
                break
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Stopped. Data saved to", DATA_FILE)
                return
        print(f"{label}: {count} samples saved")

    cap.release()
    cv2.destroyAllWindows()
    print("Done! Data saved to", DATA_FILE)


if __name__ == "__main__":
    main()
