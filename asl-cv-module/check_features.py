import numpy as np
import json

X_train = np.load("data/datasets/X_train.npy")
y_train = np.load("data/datasets/y_train.npy")

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y unique values: {sorted(set(y_train.tolist()))}")
print(f"X_train mean: {X_train.mean():.3f}")
print(f"X_train std: {X_train.std():.3f}")
print(f"Any NaN: {np.isnan(X_train).any()}")
print(f"Any Inf: {np.isinf(X_train).any()}")

# Check class distribution
from collections import Counter
counts = Counter(y_train.tolist())
for k, v in sorted(counts.items()):
    print(f"  Class {k}: {v} samples")