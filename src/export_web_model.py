"""
Export a browser-ready version of our sign model (for the LIVE web mode).

Usage (from the project root):
    python src/export_web_model.py

Trains the Neural Network (MLP) in Python on our dataset, then saves its learned
weights to models/web_model.json. The website loads these weights and runs the
same maths in the browser, so live translation is instant (no server round-trip)
and also works on phones.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from train import augment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, "data", "landmarks.csv")
OUT_FILE = os.path.join(ROOT, "models", "web_model.json")


def fit(X, y):
    scaler = StandardScaler().fit(X)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                        early_stopping=True, random_state=42)
    mlp.fit(scaler.transform(X), y)
    return scaler, mlp


def main():
    df = pd.read_csv(DATA_FILE)
    X = df[[c for c in df.columns if c.startswith("f")]].to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy(dtype=object)

    # Quick check on held-out data
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler, mlp = fit(*augment(X_tr, y_tr))
    acc = accuracy_score(y_te, mlp.predict(scaler.transform(X_te)))
    print(f"Web model (MLP) test accuracy: {acc:.2%}")

    # Final model on all data
    scaler, mlp = fit(*augment(X, y))
    r = lambda a: np.round(np.asarray(a, dtype=np.float64), 5).tolist()
    model = {
        "classes": [str(c) for c in mlp.classes_],
        "mean": r(scaler.mean_),
        "scale": r(scaler.scale_),
        # each layer: W is (n_in x n_out) stored row by row, b is (n_out)
        "layers": [{"W": r(W.flatten()), "b": r(b), "n_in": W.shape[0], "n_out": W.shape[1]}
                   for W, b in zip(mlp.coefs_, mlp.intercepts_)],
        "test_accuracy": round(float(acc), 4),
    }
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    print(f"Saved {OUT_FILE} ({os.path.getsize(OUT_FILE) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
