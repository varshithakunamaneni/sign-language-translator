"""
Fix person names when several people recorded under the SAME --person name.

Rows are saved in recording order (A, B, ... Z, then the next person's A, B, ... Z),
so each time the letter jumps BACK (Z -> A), a new recording session started.

Usage (from the project root):
    1) See the sessions:
         python src/fix_names.py
    2) Give one name per session, in recording order:
         python src/fix_names.py --names varshitha harshita

A backup of the original file is saved as data/landmarks_backup.csv first.
"""
import argparse
import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, "data", "landmarks.csv")
BACKUP_FILE = os.path.join(ROOT, "data", "landmarks_backup.csv")


def find_sessions(labels):
    """Return a list of (start_row, end_row) for each recording session."""
    sessions, start = [], 0
    for i in range(1, len(labels)):
        if labels[i] < labels[i - 1]:      # letter went backwards -> new session
            sessions.append((start, i))
            start = i
    sessions.append((start, len(labels)))
    return sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", help="one name per session, in recording order")
    args = ap.parse_args()

    df = pd.read_csv(DATA_FILE, dtype={"label": str, "person": str})
    labels = df["label"].tolist()
    sessions = find_sessions(labels)

    print(f"Found {len(sessions)} recording session(s):\n")
    for n, (a, b) in enumerate(sessions, 1):
        part = labels[a:b]
        letters = sorted(set(part))
        print(f"  Session {n}: rows {a + 1}-{b}  ({b - a} samples)  "
              f"signs {part[0]} -> {part[-1]}  ({len(letters)} different signs)  "
              f"current name: {df['person'].iloc[a]}")

    if not args.names:
        print("\nNow run again with one name per session, in the order you recorded, e.g.:")
        print("  python src/fix_names.py --names " + " ".join(f"name{i}" for i in range(1, len(sessions) + 1)))
        return

    if len(args.names) != len(sessions):
        raise SystemExit(f"\nYou gave {len(args.names)} names but there are {len(sessions)} sessions. "
                         "Give exactly one name per session (repeat a name if the same person "
                         "recorded twice).")

    shutil.copy(DATA_FILE, BACKUP_FILE)
    for (a, b), name in zip(sessions, args.names):
        df.loc[a:b - 1, "person"] = name
    df.to_csv(DATA_FILE, index=False)

    print("\nUpdated! Samples per person now:")
    print(df.groupby("person").size().to_string())
    print(f"\n(Backup of the old file: {BACKUP_FILE})")


if __name__ == "__main__":
    main()
