"""
STEP 4: Real-time Sign Language -> Text -> Speech translator (local webcam).

Usage (from the project root):
    python src/realtime.py

How it works:
    * Hold a sign steady for ~1 second -> the letter is added to the text.
    * Remove your hands for ~1.5 seconds -> a SPACE is added (word finished).
    * The app only accepts a letter when the model is confident AND the
      prediction stays the same for many frames in a row (smoothing).
      This makes the live demo far more reliable than single-frame accuracy.

Keys (click the camera window first):
    ENTER / S  -> speak the sentence out loud
    SPACE      -> add a space
    BACKSPACE  -> delete last character
    C          -> clear everything
    Q / ESC    -> quit
"""
import os
import subprocess
import sys
import threading
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np

from utils import ROOT, create_hand_detector, detect_hands, draw_hands, extract_features

MODEL_FILE = os.path.join(ROOT, "models", "sign_model.joblib")

# ---- Tuning knobs ----
CONFIDENCE_THRESHOLD = 0.60   # ignore predictions below this probability
HOLD_FRAMES = 20              # frames a sign must be stable before it's accepted (~0.7-1 s)
WINDOW = 10                   # frames used for majority-vote smoothing
NO_HAND_SPACE_FRAMES = 40     # frames without hands before a space is added (~1.5 s)


# ---------------- Text to speech ----------------
def speak(text):
    """Speak in the background so the video never freezes."""
    text = text.strip()
    if not text:
        return

    def _run():
        try:
            if sys.platform == "darwin":            # macOS built-in voice
                subprocess.run(["say", text])
            else:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
        except Exception as e:
            print("Speech error:", e)

    threading.Thread(target=_run, daemon=True).start()


# ---------------- Drawing helpers ----------------
def text(img, s, pos, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(img, s, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, s, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_panel(frame, letter, conf, progress, sentence, n_hands):
    h, w = frame.shape[:2]
    panel_h = 150
    canvas = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    canvas[:h] = frame
    canvas[h:] = (35, 35, 35)

    # Current prediction (top-left of the video)
    if letter:
        color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        text(canvas, f"{letter}", (20, 80), 2.5, color, 5)
        text(canvas, f"{conf:.0%}", (20, 115), 0.8, color)
        # Hold progress bar
        cv2.rectangle(canvas, (110, 50), (110 + 200, 75), (80, 80, 80), -1)
        cv2.rectangle(canvas, (110, 50), (110 + int(200 * progress), 75), color, -1)
        text(canvas, "hold steady", (110, 100), 0.6)
    else:
        text(canvas, "Show your hand" if n_hands == 0 else "...", (20, 50), 0.9, (200, 200, 200))

    # Sentence panel
    shown = sentence[-40:] if len(sentence) > 40 else sentence
    text(canvas, "Text:", (15, h + 40), 0.8, (180, 180, 180))
    text(canvas, shown + "_", (15, h + 85), 1.2, (0, 255, 255), 3)
    text(canvas, "ENTER/S: speak   SPACE: space   BACKSPACE: delete   C: clear   Q: quit",
         (15, h + 130), 0.55, (180, 180, 180), 1)
    return canvas


def main():
    if not os.path.exists(MODEL_FILE):
        raise SystemExit("No trained model. Run: python src/train.py")
    bundle = joblib.load(MODEL_FILE)
    model, classes = bundle["model"], list(bundle["model"].classes_)
    print(f"Loaded {bundle.get('model_name', 'model')} with {len(classes)} signs.")

    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try: python src/realtime.py 1")

    detector = create_hand_detector(video_mode=True)
    start = time.time()

    recent = deque(maxlen=WINDOW)    # recent confident predictions
    stable_label, stable_count = None, 0
    last_added = None                # prevents "AAAA" when holding one sign
    no_hand_count = 0
    sentence = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)   # same mirroring as during data collection
        result = detect_hands(detector, frame, (time.time() - start) * 1000)
        feats, n_hands = extract_features(result)
        draw_hands(frame, result)

        letter, conf, progress = None, 0.0, 0.0
        if n_hands > 0:
            no_hand_count = 0
            probs = model.predict_proba(feats.reshape(1, -1))[0]
            idx = int(np.argmax(probs))
            letter, conf = classes[idx], float(probs[idx])

            if conf >= CONFIDENCE_THRESHOLD:
                recent.append(letter)
            else:
                recent.append(None)

            # Majority vote over the last WINDOW frames
            vote, votes = Counter(recent).most_common(1)[0]
            if vote is not None and votes >= WINDOW * 0.7:
                if vote == stable_label:
                    stable_count += 1
                else:
                    stable_label, stable_count = vote, 1
            else:
                stable_label, stable_count = None, 0

            if stable_label:
                progress = min(1.0, stable_count / HOLD_FRAMES)
                if stable_count == HOLD_FRAMES and stable_label != last_added:
                    sentence += stable_label
                    last_added = stable_label
        else:
            recent.clear()
            stable_label, stable_count = None, 0
            last_added = None        # hand removed -> same letter allowed again
            no_hand_count += 1
            if no_hand_count == NO_HAND_SPACE_FRAMES and sentence and not sentence.endswith(" "):
                sentence += " "

        canvas = draw_panel(frame, letter, conf, progress, sentence, n_hands)
        cv2.imshow("Sign Language Translator", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (13, ord("s")):
            speak(sentence)
        elif key == ord(" "):
            sentence += " "
        elif key in (8, 127):        # backspace (Windows / Mac)
            sentence = sentence[:-1]
            last_added = None
        elif key == ord("c"):
            sentence, last_added = "", None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
