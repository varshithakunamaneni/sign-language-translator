"""
Train the WORD sign model (moving gestures) and export it for the website.

Usage (from the project root):
    python src/train_words.py

Each recorded clip = 10 frames x 132 values (hand shapes + where the hands are).
We compare a Random Forest and a Neural Network, test on held-out clips AND on a
person the model never saw, then export the neural network for the browser.

Saves:
    models/words_web_model.json         -> used by the website (Words mode)
    assets/words_confusion_matrix.png   -> for README / slides
    assets/words_metrics.json           -> numbers for README
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import NUM_MOTION_FEATURES, WORD_FRAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, "data", "words.csv")
WEB_FILE = os.path.join(ROOT, "models", "words_web_model.json")
ASSETS = os.path.join(ROOT, "assets")
F = NUM_MOTION_FEATURES


def mirror(X):
    """Same word signed by a LEFT-handed person: swap hands and flip left/right."""
    S = X.reshape(len(X), WORD_FRAMES, F).copy()
    out = np.empty_like(S)
    out[:, :, :63], out[:, :, 63:126] = S[:, :, 63:126], S[:, :, :63]       # swap hand shapes
    out[:, :, 0:126:3] *= -1                                                # flip x of shapes
    out[:, :, 126:129], out[:, :, 129:132] = S[:, :, 129:132], S[:, :, 126:129]  # swap positions
    out[:, :, 126] *= -1; out[:, :, 129] *= -1                              # flip wrist x
    return out.reshape(len(X), -1)


def augment(X, y, copies=3, seed=0):
    """Extra examples: mirrored (left-handed), small noise, slightly faster/slower signing."""
    rng = np.random.default_rng(seed)
    bx, by = np.vstack([X, mirror(X)]), np.concatenate([y, y])
    xs, ys = [bx], [by]
    for _ in range(copies):
        S = bx.reshape(len(bx), WORD_FRAMES, F)
        shift = rng.integers(-1, 2, len(bx))                     # timing shift by one frame
        S = np.stack([np.roll(s, k, axis=0) for s, k in zip(S, shift)])
        noise = rng.normal(0, 0.02, S.shape).astype(np.float32)
        Xa = (S + noise).reshape(len(bx), -1)
        Xa[bx == 0] = 0
        xs.append(Xa); ys.append(by)
    return np.vstack(xs).astype(np.float32), np.concatenate(ys)


def models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        "Neural Network (MLP)": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True, random_state=42)),
    }


def main():
    if not os.path.exists(DATA_FILE):
        raise SystemExit("No word data yet. Run: python src/collect_words.py --person yourname")
    df = pd.read_csv(DATA_FILE)
    X = df[[c for c in df.columns if c.startswith("f")]].to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy(dtype=object)
    person = df["person"].astype(str).to_numpy(dtype=object)
    labels, people = sorted(set(y)), sorted(set(person))
    counts = pd.Series(y).value_counts()
    print("=" * 60)
    print(f"Clips: {len(df)} | Words: {len(labels)} | People: {len(people)} {people}")
    print("Clips per word:", counts.min(), "to", counts.max())
    print("=" * 60)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xa, ya = augment(X_tr, y_tr)
    results, best = {}, None
    for name, m in models().items():
        m.fit(Xa, ya)
        pred = m.predict(X_te)
        acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")
        results[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
        print(f"{name:22s} accuracy={acc:.2%}  F1={f1:.2%}")
        if best is None or acc > best[1]:
            best = (name, acc, pred)
    print(f"\nBest: {best[0]} ({best[1]:.2%})")

    unseen = {}
    if len(people) >= 2:
        print("\nUnseen-person test (train on others, test on this person):")
        for p in people:
            tr, te = person != p, person == p
            m = clone(models()[best[0]]).fit(*augment(X[tr], y[tr]))
            unseen[p] = round(accuracy_score(y[te], m.predict(X[te])), 4)
            print(f"  {p:15s} {unseen[p]:.2%}")
        print(f"  Average: {np.mean(list(unseen.values())):.2%}")

    os.makedirs(ASSETS, exist_ok=True)
    cm = confusion_matrix(y_te, best[2], labels=labels)
    plt.figure(figsize=(max(8, len(labels) * 0.55), max(7, len(labels) * 0.48)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Predicted word"); plt.ylabel("Actual word")
    plt.title(f"Word signs - {best[0]} (accuracy {best[1]:.1%})")
    plt.tight_layout(); plt.savefig(os.path.join(ASSETS, "words_confusion_matrix.png"), dpi=150); plt.close()
    report = classification_report(y_te, best[2], digits=3)

    # ---- export the neural network for the browser (trained on ALL data) ----
    web = models()["Neural Network (MLP)"].fit(*augment(X, y))
    scaler, mlp = web.named_steps["standardscaler"], web.named_steps["mlpclassifier"]
    r = lambda a: np.round(np.asarray(a, dtype=np.float64), 5).tolist()
    os.makedirs(os.path.dirname(WEB_FILE), exist_ok=True)
    with open(WEB_FILE, "w") as f:
        json.dump({"classes": [str(c) for c in mlp.classes_], "mean": r(scaler.mean_),
                   "scale": r(np.where(scaler.scale_ == 0, 1, scaler.scale_)),
                   "layers": [{"W": r(W.flatten()), "b": r(b), "n_in": W.shape[0], "n_out": W.shape[1]}
                              for W, b in zip(mlp.coefs_, mlp.intercepts_)],
                   "frames": WORD_FRAMES, "features": F,
                   "test_accuracy": results["Neural Network (MLP)"]["accuracy"]}, f, separators=(",", ":"))
    with open(os.path.join(ASSETS, "words_metrics.json"), "w") as f:
        json.dump({"num_clips": int(len(df)), "words": labels, "people": people, "best_model": best[0],
                   "test_split": results, "unseen_person_accuracy": unseen,
                   "unseen_person_average": round(float(np.mean(list(unseen.values()))), 4) if unseen else None},
                  f, indent=2)
    print(f"\nSaved {WEB_FILE} ({os.path.getsize(WEB_FILE) / 1024:.0f} KB)")
    print("Saved assets/words_confusion_matrix.png and assets/words_metrics.json")
    print("\nPer-word results:\n" + report)


if __name__ == "__main__":
    main()
