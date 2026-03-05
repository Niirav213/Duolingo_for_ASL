import cv2
import base64
import requests
import os

# Test with actual Kaggle training images
test_cases = [
    ("data/datasets/raw_images/A", "A"),
    ("data/datasets/raw_images/B", "B"),
    ("data/datasets/raw_images/L", "L"),
]

for folder, expected in test_cases:
    files = os.listdir(folder)
    img = cv2.imread(os.path.join(folder, files[0]))
    _, buf = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buf).decode('utf-8')

    r = requests.post("http://localhost:8000/analyze", json={
        "frame_base64": b64,
        "target_sign": expected,
        "mode": "static"
    }).json()

    print(f"Expected: {expected} | Got: {r['detected_sign']} | Confidence: {r['confidence']*100:.1f}%")