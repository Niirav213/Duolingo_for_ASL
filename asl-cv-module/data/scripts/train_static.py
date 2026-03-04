"""
data/scripts/train_static.py
-----------------------------
Trains the static sign classifier (A-Z letters) using
preprocessed numpy arrays from preprocess.py.

Run this AFTER preprocess.py has been run.

Usage:
    python data/scripts/train_static.py
"""

import sys
import json
import numpy as np
from pathlib import Path

# So Python can find your modules regardless of where you run from
SCRIPT_DIR   = Path(__file__).resolve().parent      # data/scripts/
DATA_DIR     = SCRIPT_DIR.parent                    # data/
PROJECT_ROOT = SCRIPT_DIR.parent.parent             # asl-cv-module/

sys.path.append(str(PROJECT_ROOT))

from models.static import StaticSignClassifier

# ─────────────────────────────────────────────
# Load preprocessed data
# ─────────────────────────────────────────────

DATASETS_DIR = DATA_DIR / "datasets"

print("[Train] Loading preprocessed data...")

X_train = np.load(DATASETS_DIR / "X_train.npy")
y_train = np.load(DATASETS_DIR / "y_train.npy")
X_val   = np.load(DATASETS_DIR / "X_val.npy")
y_val   = np.load(DATASETS_DIR / "y_val.npy")

# Load label map
with open(DATASETS_DIR / "label_map.json") as f:
    label_map = json.load(f)

# Convert {"0": "A", "1": "B", ...} → ["A", "B", ...]
labels = [label_map[str(i)] for i in range(len(label_map))]
num_classes = len(labels)

print(f"[Train] Train samples  : {len(X_train)}")
print(f"[Train] Val samples    : {len(X_val)}")
print(f"[Train] Num classes    : {num_classes}")
print(f"[Train] Labels         : {labels}")
print()

# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = str(CHECKPOINT_DIR / "static_sign_best.pt")

clf = StaticSignClassifier(num_classes=num_classes, labels=labels)

clf.train_model(
    X_train   = X_train,
    y_train   = y_train,
    X_val     = X_val,
    y_val     = y_val,
    epochs    = 50,
    batch_size= 64,
    lr        = 1e-3,
    save_path = SAVE_PATH,
)

print(f"\n[Train] ✓ Done! Model saved to {SAVE_PATH}")
print(f"[Train] Next step: uvicorn api.router:app --reload --port 8000")
