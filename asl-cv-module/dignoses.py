import torch
import json
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))

from models.static import StaticSignClassifier

# Load label map
with open("data/datasets/label_map.json") as f:
    label_map = json.load(f)
labels = [label_map[str(i)] for i in range(len(label_map))]

clf = StaticSignClassifier(num_classes=len(labels), labels=labels)
clf.load("models/checkpoints/static_sign_best.pt")

# Test with random input
for i in range(5):
    random_vector = np.random.rand(178).astype(np.float32)
    sign, confidence = clf.predict(random_vector)
    print(f"Random input {i+1}: {sign} ({confidence*100:.1f}%)")

# Test with all zeros
sign, conf = clf.predict(np.zeros(178, dtype=np.float32))
print(f"Zero input: {sign} ({conf*100:.1f}%)")