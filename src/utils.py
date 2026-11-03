"""
Shared helpers: hand detection (MediaPipe Tasks API) + feature extraction.

Why landmarks instead of raw images?
- MediaPipe gives 21 (x, y, z) points per hand -> tiny, fast, background-independent.
- A small classifier on these points trains in seconds on a laptop (no GPU needed).

ISL uses BOTH hands for many letters, so we always build a 126-value vector:
    [left hand 63 values] + [right hand 63 values]   (zeros if a hand is missing)
"""
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

NUM_LANDMARKS = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3          # 63
NUM_FEATURES = FEATURES_PER_HAND * 2           # 126

# Bone connections, used to draw the hand skeleton on screen
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),
]


def ensure_hand_model():
    """Download MediaPipe's hand model (~7.5 MB) once, on first run."""
    if not os.path.exists(HAND_MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading MediaPipe hand model (one time)...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    return HAND_MODEL_PATH


def create_hand_detector(video_mode=True):
    """video_mode=True for webcam streams, False for single images (web app)."""
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_hand_model()),
        running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(detector, frame_bgr, timestamp_ms=None):
    """Run detection on an OpenCV BGR frame. Pass timestamp_ms in video mode."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if timestamp_ms is None:
        return detector.detect(mp_image)
    return detector.detect_for_video(mp_image, int(timestamp_ms))


def _normalize_hand(landmarks):
    """Make features independent of where the hand is and how big it looks.
    1) translate so the wrist is the origin, 2) scale to fit in [-1, 1]."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts.flatten()


def extract_features(result):
    """Returns (126-dim feature vector, number of hands found)."""
    left = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    right = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    if not result.hand_landmarks:
        return np.concatenate([left, right]), 0

    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name   # "Left" or "Right"
        if label == "Left":
            left = _normalize_hand(landmarks)
        else:
            right = _normalize_hand(landmarks)
    return np.concatenate([left, right]), len(result.hand_landmarks)


# ---------------- Word signs (movement over time) ----------------
# For words, WHERE the hand is and HOW it moves matters too (e.g. a wave, a hand moving
# to the chin). So each frame gets 6 extra values: for each hand its position in the
# picture (centred: -0.5 .. 0.5) and its size. 126 + 6 = 132 values per frame.
NUM_MOTION_FEATURES = NUM_FEATURES + 6           # 132
WORD_FRAMES = 10                                 # every clip is resampled to 10 frames
WORD_SECONDS = 1.5                               # length of one word clip


def _hand_position(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    size = float(np.max(np.abs(pts - pts[0])))
    return [pts[0][0] - 0.5, pts[0][1] - 0.5, size]


def extract_motion_features(result):
    """132 values for one frame: [left shape 63, right shape 63,
    left wrist x, y, size, right wrist x, y, size] (zeros for a missing hand)."""
    shape, n = extract_features(result)
    left_pos, right_pos = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    for landmarks, handedness in zip(result.hand_landmarks or [], result.handedness or []):
        if handedness[0].category_name == "Left":
            left_pos = _hand_position(landmarks)
        else:
            right_pos = _hand_position(landmarks)
    return np.concatenate([shape, np.array(left_pos + right_pos, dtype=np.float32)]), n


def resample_clip(times, frames, n=WORD_FRAMES):
    """Pick n frames evenly spread over the clip's time (works at any camera speed)."""
    times = np.asarray(times, dtype=np.float64)
    targets = np.linspace(times[0], times[-1], n)
    idx = [int(np.argmin(np.abs(times - t))) for t in targets]
    return np.concatenate([frames[i] for i in idx]).astype(np.float32)


def draw_hands(frame_bgr, result):
    """Draw the hand skeleton on the frame (in place)."""
    h, w = frame_bgr.shape[:2]
    for landmarks in result.hand_landmarks or []:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame_bgr, p, 4, (0, 0, 255), -1)
    return frame_bgr
