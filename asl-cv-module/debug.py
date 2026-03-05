import cv2
import base64
import requests
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.detector import ASLDetector
from core.extractor import ASLFeatureExtractor

# ── Load your test image ──
image_path = r"C:\Users\ASUS\Downloads\asl_alphabet_train\asl_alphabet_train\A\A8.jpg"   # your test image
frame = cv2.imread(image_path)

if frame is None:
    print("[ERROR] Could not load image")
    exit()

print(f"Image shape: {frame.shape}")

# ── Run detector directly ──
detector = ASLDetector(draw_landmarks=True, model_complexity=1)
extractor = ASLFeatureExtractor()

detection = detector.process_frame(frame)
print(f"Hand detected: {detection.is_valid()}")
print(f"Right hand: {detection.right_hand_detected}")
print(f"Left hand: {detection.left_hand_detected}")

if detection.is_valid():
    features = extractor.extract(detection)
    print(f"Feature vector shape: {features.vector.shape}")
    print(f"First 10 features: {features.vector[:10]}")
    print(f"Any NaN: {np.isnan(features.vector).any()}")
    print(f"Any Inf: {np.isinf(features.vector).any()}")

    # Show annotated frame
    cv2.imshow("Detection", detection.annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("[WARN] No hand detected in image!")
    print("This is why the model fails — MediaPipe can't find the hand")
    cv2.imshow("Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detector.release()