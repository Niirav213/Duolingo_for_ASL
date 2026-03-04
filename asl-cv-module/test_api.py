"""
test_api.py
-----------
Quick test script that captures a single webcam frame
and sends it to the running API server.

Usage:
    python test_api.py
    python test_api.py --sign B
"""

import cv2
import base64
import json
import argparse
import requests
import numpy as np

API_URL = "http://localhost:8000/analyze"


def capture_frame():
    """Capture a single frame from webcam."""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return None

    print("[Test] Show your hand sign to the camera...")
    print("[Test] Press SPACE to capture, Q to quit\n")

    frame = None
    while True:
        ret, f = cap.read()
        if not ret:
            break

        f = cv2.flip(f, 1)
        cv2.putText(f, "Press SPACE to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capture - Press SPACE", f)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            frame = f.copy()
            print("[Test] Frame captured!")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert OpenCV frame to base64 string."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def test_api(sign: str = "A"):
    """Capture a frame and send to /analyze endpoint."""

    # Step 1 — capture frame
    frame = capture_frame()
    if frame is None:
        print("[ERROR] No frame captured.")
        return

    # Step 2 — convert to base64
    frame_b64 = frame_to_base64(frame)
    print(f"[Test] Frame encoded ({len(frame_b64)} chars)")

    # Step 3 — send to API
    print(f"[Test] Sending to API... (target sign: {sign})")
    payload = {
        "frame_base64": frame_b64,
        "target_sign": sign,
        "mode": "static",
        "include_landmarks": False
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        # Step 4 — display results
        print("\n" + "="*50)
        print("API RESPONSE")
        print("="*50)
        print(f"Hand detected   : {result['hand_detected']}")
        print(f"Detected sign   : {result['detected_sign']}")
        print(f"Confidence      : {result['confidence'] * 100:.1f}%")
        print(f"Overall score   : {result['overall_score']}/100")
        print(f"Is correct      : {result['is_correct']}")
        print(f"Emoji           : {result['emoji']}")
        print(f"Praise          : {result['praise']}")
        print(f"\nFeedback messages:")
        for msg in result['messages']:
            print(f"  → {msg}")
        print(f"\nJoint colors (red = wrong, green = correct):")
        for joint, color in result['joint_colors'].items():
            print(f"  {joint:20s} : {color}")
        print("="*50)

    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to API.")
        print("        Make sure the server is running:")
        print("        uvicorn api.router:app --reload --port 8000")
    except Exception as e:
        print(f"[ERROR] {e}")
        if response:
            print(f"Response: {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign", default="A", help="Target sign to test against (default: A)")
    args = parser.parse_args()

    test_api(sign=args.sign.upper())
