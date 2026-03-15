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
import torch
import torch.nn as nn
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
print(f"[Train] Batches/epoch  : {len(X_train) // 64 + 1}")
print()

# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = str(CHECKPOINT_DIR / "static_sign_best.pt")

clf = StaticSignClassifier(num_classes=num_classes, labels=labels)

device = clf.device
X_t = torch.FloatTensor(X_train).to(device)
y_t = torch.LongTensor(y_train).to(device)

optimizer = torch.optim.Adam(clf.model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# Class weights for imbalance (M and N have fewer samples)
class_counts = torch.bincount(y_t)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

epochs     = 50
batch_size = 64
best_val_acc = 0.0

print(f"{'Epoch':>6}  {'Avg Loss':>10}  {'Val Acc':>9}  {'Best':>9}")
print("-" * 42)

for epoch in range(epochs):
    clf.model.train()
    perm        = torch.randperm(len(X_t))
    epoch_loss  = 0.0
    num_batches = 0

    for i in range(0, len(X_t), batch_size):
        idx    = perm[i:i + batch_size]
        bx, by = X_t[idx], y_t[idx]

        optimizer.zero_grad()
        loss = criterion(clf.model(bx), by)
        loss.backward()
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

    scheduler.step()

    # Average loss — divide by number of batches, not total samples
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

    # Validate every epoch, print every 5
    val_acc = clf._evaluate(X_val, y_val)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        clf.save(SAVE_PATH)
        saved_marker = " <-- saved"
    else:
        saved_marker = ""

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"{epoch+1:>6}  {avg_loss:>10.4f}  {val_acc:>8.3f}%  {best_val_acc:>8.3f}%{saved_marker}")

print("-" * 42)
print(f"\n[Train] Done! Best val acc : {best_val_acc:.3f}%")
print(f"[Train] Model saved to     : {SAVE_PATH}")
print(f"[Train] Next step          : uvicorn api.router:app --reload --port 8000")