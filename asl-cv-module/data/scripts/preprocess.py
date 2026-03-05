"""
data/scripts/preprocess.py
--------------------------
Processes raw landmark recordings into training-ready numpy arrays.
Includes class balancing, outlier removal, and feature normalization.

Usage:
    python data/scripts/preprocess.py
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RAW_DIR = DATA_DIR / "datasets" / "raw"
OUT_DIR = DATA_DIR / "datasets"

# Minimum samples required to include a class
MIN_SAMPLES = 500

# Classes to explicitly exclude
EXCLUDE_CLASSES = {"NOTHING", "SPACE", "DEL"}


def load_raw_data():
    X, y = [], []

    if not RAW_DIR.exists():
        print(f"[Preprocess] Raw data directory not found: {RAW_DIR}")
        return np.array([]), np.array([])

    for sign_dir in sorted(RAW_DIR.iterdir()):
        if not sign_dir.is_dir():
            continue
        sign = sign_dir.name.upper()

        # Skip excluded classes
        if sign in EXCLUDE_CLASSES:
            print(f"[Preprocess] Skipping excluded class: {sign}")
            continue

        samples = []
        for npy_file in sign_dir.glob("*.npy"):
            data = np.load(npy_file)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            samples.append(data)

        if not samples:
            continue

        sign_data = np.vstack(samples)

        # Skip classes with too few samples
        if len(sign_data) < MIN_SAMPLES:
            print(f"[Preprocess] Skipping '{sign}' — only {len(sign_data)} samples (min={MIN_SAMPLES})")
            continue

        X.append(sign_data)
        y.extend([sign] * len(sign_data))
        print(f"[Preprocess] Loaded {len(sign_data):5d} samples for '{sign}'")

    if not X:
        return np.array([]), np.array([])

    return np.vstack(X).astype(np.float32), np.array(y)


def remove_nan_inf(X, y):
    """Remove samples with NaN or Inf values."""
    mask = np.isfinite(X).all(axis=1)
    removed = np.sum(~mask)
    if removed > 0:
        print(f"[Preprocess] Removed {removed} samples with NaN/Inf values")
    return X[mask], y[mask]


def balance_classes(X, y, max_samples: int = 2500):
    """Cap each class at max_samples to prevent imbalance."""
    X_bal, y_bal = [], []
    counts = Counter(y)

    for label in sorted(set(y)):
        idx = np.where(y == label)[0]
        if len(idx) > max_samples:
            idx = np.random.choice(idx, max_samples, replace=False)
        X_bal.append(X[idx])
        y_bal.extend([label] * len(idx))

    return np.vstack(X_bal), np.array(y_bal)


def preprocess(val_size: float = 0.2, random_seed: int = 42):
    np.random.seed(random_seed)

    print("[Preprocess] Loading raw data...")
    X, y_labels = load_raw_data()

    if len(X) == 0:
        print("[Preprocess] No data found.")
        return

    print(f"\n[Preprocess] Raw total: {len(X)} samples")

    # ── Remove bad samples ──
    X, y_labels = remove_nan_inf(X, y_labels)

    # ── Balance classes ──
    print("[Preprocess] Balancing classes...")
    X, y_labels = balance_classes(X, y_labels, max_samples=2500)

    # ── Print class distribution ──
    counts = Counter(y_labels)
    print(f"\n[Preprocess] Class distribution after balancing:")
    for label, count in sorted(counts.items()):
        print(f"  {label:10s} → {count} samples")

    print(f"\n[Preprocess] Total samples: {len(X)}")
    print(f"[Preprocess] Classes: {sorted(set(y_labels))}")

    # ── Encode labels ──
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)
    label_map = {int(i): str(label) for i, label in enumerate(encoder.classes_)}

    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n[Preprocess] Label map: {label_map}")

    # ── Train/val split ──
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=random_seed,
        stratify=y
    )

    # ── Normalize features ──
    print("[Preprocess] Fitting StandardScaler on train set...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)

    # Save scaler — needed at inference time
    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("[Preprocess] Scaler saved to data/datasets/scaler.pkl")

    # ── Save ──
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "X_val.npy",   X_val)
    np.save(OUT_DIR / "y_val.npy",   y_val)

    print(f"\n[Preprocess] ✓ Done!")
    print(f"  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)} samples")
    print(f"  Saved to: {OUT_DIR}/")


if __name__ == "__main__":
    preprocess()