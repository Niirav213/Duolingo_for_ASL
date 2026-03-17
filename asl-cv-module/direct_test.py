import cv2
import numpy as np
import sys
import json
from pathlib import Path

sys.path.append(".")

import mediapipe as mp
from core.extractor import ASLFeatureExtractor
from core.detector import ASLDetector, DetectionResult
from models.static import StaticSignClassifier

# Load model
with open("data/datasets/label_map.json") as f:
    label_map = json.load(f)
labels = [label_map[str(i)] for i in range(len(label_map))]
clf = StaticSignClassifier(num_classes=len(labels), labels=labels)
clf.load("models/checkpoints/static_sign_best.pt")
extractor = ASLFeatureExtractor()

# Test on raw Kaggle image using same settings as extract_kaggle.py
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,
    model_complexity=1,
)

import os
folder = r"data\datasets\raw_images2\Test_Alphabet\A"
files = os.listdir(folder)

correct = 0
total = 10
for f in files[:total]:
    img = cv2.imread(os.path.join(folder, f))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)
    
    if not result.multi_hand_landmarks:
        print(f"{f}: No hand detected")
        continue

    # Build DetectionResult manually
    from core.detector import Landmark
    landmarks = [
        Landmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in result.multi_hand_landmarks[0].landmark
    ]
    
    detection = DetectionResult()
    detection.right_hand = landmarks
    detection.right_hand_detected = True

    features = extractor.extract(detection)
    sign, conf = clf.predict(features.vector)
    status = "✅" if sign == "A" else "❌"
    print(f"{status} {f}: Got {sign} ({conf*100:.1f}%)")

mp_hands.close()