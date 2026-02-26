"""
data/scripts/preprocess.py
--------------------------
Processes raw landmark recordings into training-ready numpy arrays.

Input:  data/datasets/raw/*.npy or CSV files
Output: data/datasets/X_train.npy, y_train.npy, X_val.npy, y_val.npy

Usage:
    python data/scripts/preprocess.py
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json


RAW_DIR = Path("data/datasets/raw")
OUT_DIR = Path("data/datasets")


def load_raw_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load raw recordings.
    Expects subfolders named by sign label, each containing .npy files.

    data/datasets/raw/
        A/
            session1.npy   ← shape (N, 178) float32
            session2.npy
        B/
            session1.npy
        ...
    """
    X, y = [], []

    if not RAW_DIR.exists():
        print(f"[Preprocess] Raw data directory not found: {RAW_DIR}")
        return np.array([]), np.array([])

    for sign_dir in sorted(RAW_DIR.iterdir()):
        if not sign_dir.is_dir():
            continue
        sign = sign_dir.name.upper()

        for npy_file in sign_dir.glob("*.npy"):
            data = np.load(npy_file)
            if data.ndim == 1:
                data = data[np.newaxis, :]  # single sample
            X.append(data)
            y.extend([sign] * len(data))
            print(f"[Preprocess] Loaded {len(data)} samples for '{sign}' from {npy_file.name}")

    if not X:
        return np.array([]), np.array([])

    return np.vstack(X).astype(np.float32), np.array(y)


def preprocess(val_size: float = 0.2, random_seed: int = 42):
    """
    Load, encode, split, and save training data.
    """
    print("[Preprocess] Loading raw data...")
    X, y_labels = load_raw_data()

    if len(X) == 0:
        print("[Preprocess] No data found. Run data/scripts/collect.py first.")
        return

    print(f"[Preprocess] Total samples: {len(X)}")
    print(f"[Preprocess] Signs found: {sorted(set(y_labels))}")

    # Encode labels to integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    # Save label mapping
    label_map = {int(i): label for i, label in enumerate(encoder.classes_)}
    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[Preprocess] Label map saved: {label_map}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_seed, stratify=y
    )

    # Save
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "X_val.npy", X_val)
    np.save(OUT_DIR / "y_val.npy", y_val)

    print(f"\n[Preprocess] ✓ Done!")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Saved to: {OUT_DIR}/")


if __name__ == "__main__":
    preprocess()
