"""
Replace signers' real names in our public datasets with "signer_1", "signer_2", ...
(privacy: the hackathon asks us not to expose personal information).

Usage (from the project root):
    python src/anonymize.py
Then retrain so the metrics use the new labels:
    python src/train.py && python src/export_web_model.py && python src/train_words.py

The same person gets the same number in both files. The name -> number mapping is
printed once on screen and NOT saved anywhere.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [os.path.join(ROOT, "data", "landmarks.csv"), os.path.join(ROOT, "data", "words.csv")]


def main():
    tables = {}
    names = []
    for path in FILES:
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={"person": str, "label": str})
            tables[path] = df
            for n in df["person"].astype(str):
                if n not in names:
                    names.append(n)
    if not tables:
        raise SystemExit("No data files found.")

    already = all(n.startswith("signer_") for n in names)
    if already:
        print("Already anonymised:", names)
        return

    mapping = {n: f"signer_{i + 1}" for i, n in enumerate(names)}
    for path, df in tables.items():
        df["person"] = df["person"].astype(str).map(mapping)
        df.to_csv(path, index=False)
        print(f"Updated {os.path.relpath(path, ROOT)}: {df['person'].nunique()} signers")
    print("\nMapping (shown only here, not saved):")
    for n, s in mapping.items():
        print(f"  {n} -> {s}")

    backup = os.path.join(ROOT, "data", "landmarks_backup.csv")
    if os.path.exists(backup):
        os.remove(backup)
        print("\nRemoved data/landmarks_backup.csv (it contained real names).")


if __name__ == "__main__":
    main()
