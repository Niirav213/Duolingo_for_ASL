import sys
sys.path.append(r"c:\Users\kisho\Duolingo_for_ASL\asl-cv-module")
import json
import base64
import cv2
import numpy as np
import requests

# Create a blank white image to keep the payload valid
img = np.ones((480, 640, 3), dtype=np.uint8) * 255
_, buffer = cv2.imencode('.jpg', img)
b64 = base64.b64encode(buffer).decode('utf-8')

# Send to API
payload = {
    "frame_base64": f"data:image/jpeg;base64,{b64}",
    "target_sign": "B",
    "mode": "static",
    "include_landmarks": True
}
try:
    res = requests.post("http://localhost:8002/analyze", json=payload)
    print("STATUS:", res.status_code)
    print("JSON:", json.dumps(res.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
