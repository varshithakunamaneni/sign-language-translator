"""
Measure speed (latency) of each part of SignSpeak, for the README.

Usage (from the project root):
    python src/benchmark.py

Prints a Markdown table you can paste into the README.
"""
import json
import os
import time

import cv2
import joblib
import numpy as np
import pandas as pd

from utils import ROOT, create_hand_detector, detect_hands

RUNS = 300


def timed(fn, runs=RUNS):
    fn()                                    # warm-up
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) / runs * 1000     # ms per call


def mlp_forward(model, x):
    """Same maths the browser runs (StandardScaler + MLP + softmax)."""
    a = (x - np.array(model["mean"])) / np.array(model["scale"])
    for i, L in enumerate(model["layers"]):
        a = a @ np.array(L["W"]).reshape(L["n_in"], L["n_out"]) + np.array(L["b"])
        if i < len(model["layers"]) - 1:
            a = np.maximum(a, 0)
    e = np.exp(a - a.max())
    return e / e.sum()


def main():
    rows = []

    # 1) Hand detection (MediaPipe) on a real photo if we have one
    sample = next((os.path.join(ROOT, "web", "signs", f) for f in sorted(os.listdir(os.path.join(ROOT, "web", "signs")))
                   if f.endswith(".jpg")), None) if os.path.isdir(os.path.join(ROOT, "web", "signs")) else None
    img = cv2.imread(sample) if sample else np.full((480, 640, 3), 128, np.uint8)
    img = cv2.resize(img, (640, 480))
    try:
        det = create_hand_detector(video_mode=False)
        rows.append(("Hand tracking (MediaPipe, 640x480 frame)", timed(lambda: detect_hands(det, img), 50)))
    except Exception as e:                  # e.g. no internet to download the hand model
        print("(Skipping hand-tracking timing:", e, ")")
        rows.append(("Hand tracking (MediaPipe, 640x480 frame)", float("nan")))

    # 2) Letter models
    df = pd.read_csv(os.path.join(ROOT, "data", "landmarks.csv"), nrows=50)
    x = df[[c for c in df.columns if c.startswith("f")]].to_numpy(np.float32)[:1]
    bundle = joblib.load(os.path.join(ROOT, "models", "sign_model.joblib"))
    rows.append((f"Letter model, Python ({bundle.get('model_name', 'model')})",
                 timed(lambda: bundle["model"].predict_proba(x), 100)))
    with open(os.path.join(ROOT, "models", "web_model.json")) as f:
        web = json.load(f)
    rows.append(("Letter model, browser network (same maths)", timed(lambda: mlp_forward(web, x[0]))))

    # 3) Word model
    wpath = os.path.join(ROOT, "models", "words_web_model.json")
    if os.path.exists(wpath):
        with open(wpath) as f:
            words = json.load(f)
        xw = np.zeros(words["frames"] * words["features"], np.float32)
        rows.append(("Word model, browser network (10-frame clip)", timed(lambda: mlp_forward(words, xw))))

    total = rows[0][1] + rows[2][1]
    print("\n| Step | Time per call |\n|---|---|")
    for name, ms in rows:
        print(f"| {name} | {ms:.2f} ms |")
    if total == total:                      # not NaN
        print(f"| **Per frame, letters (tracking + browser model)** | **{total:.1f} ms ≈ {1000 / total:.0f} frames/s max** |")
    print("\n(Measured on this computer. The live page also shows real frames per second under 'Show details'.)")


if __name__ == "__main__":
    main()
