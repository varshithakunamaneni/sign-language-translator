"""
STEP 3: Train the sign classifier and generate metrics for the README.

Usage (from the project root):
    python src/train.py

What it does:
    1. Loads data/landmarks.csv
    2. Trains 2 models (Random Forest + Neural Network/MLP) and keeps the best
    3. Reports accuracy on a held-out 20% test split
    4. If 2+ people recorded data: "unseen person" test -> trains on everyone
       except one person, tests on that person (shows real-world generalization)
    5. Saves:
         models/sign_model.joblib         -> the trained model (used by the app)
         assets/confusion_matrix.png      -> picture for README / slides
         assets/metrics.json              -> all numbers for README
         assets/classification_report.txt -> per-letter precision / recall
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, "data", "landmarks.csv")
MODEL_FILE = os.path.join(ROOT, "models", "sign_model.joblib")
ASSETS = os.path.join(ROOT, "assets")


def augment(X, y, copies=2, seed=42):
    """Create extra training samples with small random noise + scaling.
    Helps the model handle slightly different hands/positions."""
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    for _ in range(copies):
        noise = rng.normal(0, 0.02, X.shape).astype(np.float32)
        scale = rng.uniform(0.9, 1.1, (X.shape[0], 1)).astype(np.float32)
        Xa = (X + noise) * scale
        Xa[X == 0] = 0  # keep "missing hand" zeros as zeros
        Xs.append(Xa)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def build_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        "Neural Network (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                          early_stopping=True, random_state=42),
        ),
    }


def main():
    if not os.path.exists(DATA_FILE):
        raise SystemExit("No data found. Run src/collect_data.py first.")

    df = pd.read_csv(DATA_FILE)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    # plain numpy string arrays (pandas 3 uses Arrow strings, which break sklearn indexing)
    y = df["label"].astype(str).to_numpy(dtype=object)
    person_arr = df["person"].astype(str).to_numpy(dtype=object)
    people = sorted(set(person_arr))
    labels = sorted(set(y))

    print("=" * 60)
    print(f"Samples: {len(df)} | Signs: {len(labels)} | People: {len(people)} {people}")
    counts = pd.Series(y).value_counts()
    print("Samples per sign:", counts.min(), "to", counts.max())
    print("=" * 60)

    # ---------- 1) Standard 80/20 split, compare models ----------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr_aug, y_tr_aug = augment(X_tr, y_tr)

    results, best_name, best_acc, best_model = {}, None, -1, None
    for name, model in build_models().items():
        t0 = time.time()
        model.fit(X_tr_aug, y_tr_aug)
        train_time = time.time() - t0

        t0 = time.time()
        pred = model.predict(X_te)
        ms_per_sample = (time.time() - t0) / len(X_te) * 1000

        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        results[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4),
                         "train_time_s": round(train_time, 2),
                         "inference_ms_per_sample": round(ms_per_sample, 3)}
        print(f"{name:22s} accuracy={acc:.2%}  F1={f1:.2%}  train={train_time:.1f}s")
        if acc > best_acc:
            best_name, best_acc, best_model, best_pred = name, acc, model, pred

    print(f"\nBest model: {best_name} ({best_acc:.2%})")

    # Confusion matrix + per-class report for the best model
    os.makedirs(ASSETS, exist_ok=True)
    cm = confusion_matrix(y_te, best_pred, labels=labels)
    size = max(8, len(labels) * 0.45)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=len(labels) <= 40, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Predicted sign")
    plt.ylabel("Actual sign")
    plt.title(f"Confusion Matrix - {best_name} (accuracy {best_acc:.1%})")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, "confusion_matrix.png"), dpi=150)
    plt.close()

    report = classification_report(y_te, best_pred, digits=3)
    with open(os.path.join(ASSETS, "classification_report.txt"), "w") as f:
        f.write(report)

    # ---------- 2) Unseen-person test (leave-one-person-out) ----------
    unseen = {}
    if len(people) >= 2:
        print("\nUnseen-person test (train on others, test on this person):")
        model_proto = build_models()[best_name]
        for p in people:
            tr, te = person_arr != p, person_arr == p
            if te.sum() == 0 or len(set(y[tr])) < 2:
                continue
            Xa, ya = augment(X[tr], y[tr])
            from sklearn.base import clone
            m = clone(model_proto).fit(Xa, ya)
            acc = accuracy_score(y[te], m.predict(X[te]))
            unseen[p] = round(acc, 4)
            print(f"  {p:15s} {acc:.2%}")
        if unseen:
            print(f"  Average: {np.mean(list(unseen.values())):.2%}")
    else:
        print("\n(Only 1 person recorded -> skipping unseen-person test.)")

    # ---------- 3) Retrain best model on ALL data and save ----------
    X_all, y_all = augment(X, y)
    final_model = build_models()[best_name].fit(X_all, y_all)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump({"model": final_model, "labels": labels, "model_name": best_name}, MODEL_FILE, compress=3)

    metrics = {
        "num_samples": int(len(df)), "num_signs": len(labels), "signs": labels,
        "people": people, "best_model": best_name,
        "test_split": results,
        "unseen_person_accuracy": unseen,
        "unseen_person_average": round(float(np.mean(list(unseen.values()))), 4) if unseen else None,
    }
    with open(os.path.join(ASSETS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print("  models/sign_model.joblib")
    print("  assets/confusion_matrix.png")
    print("  assets/metrics.json")
    print("  assets/classification_report.txt")
    print("\nPer-sign results:\n" + report)


if __name__ == "__main__":
    main()
