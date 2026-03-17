"""
test_api.py
-----------
Captures a webcam frame, sends it to the running API server,
then shows AND saves the photo with:
  - MediaPipe finger tracking overlay (landmarks + connections)
  - API result overlay (sign, score, feedback, joint colors)

Usage:
    python test_api.py
    python test_api.py --sign B
    python test_api.py --sign A --save result.png
"""

import cv2
import base64
import json
import argparse
import requests
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.detector import ASLDetector

API_URL = "http://localhost:8000/analyze"

# Joint color mapping from API response → BGR for OpenCV
COLOR_MAP = {
    "green":  (0, 200, 80),
    "orange": (0, 165, 255),
    "red":    (0, 60, 220),
}

# MediaPipe hand connections (21 landmarks, standard connections)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17),           # palm base
]

# Map joint names from API → landmark indices
JOINT_TO_INDICES = {
    "thumb_mcp":  [2],
    "thumb_ip":   [3, 4],
    "index_mcp":  [5],
    "index_pip":  [6, 7, 8],
    "middle_mcp": [9],
    "middle_pip": [10, 11, 12],
    "ring_mcp":   [13],
    "ring_pip":   [14, 15, 16],
    "pinky_mcp":  [17],
    "pinky_pip":  [18, 19, 20],
}


def capture_frame(camera_index: int = 2) -> np.ndarray | None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}.")
        return None

    print("[Test] Show your hand sign to the camera.")
    print("[Test] Press SPACE to capture, Q to quit.\n")

    frame = None
    while True:
        ret, f = cap.read()
        if not ret:
            break

        f = cv2.flip(f, 1)

        # Draw overlay on a COPY — so the saved frame stays clean
        display = f.copy()
        cv2.putText(display, "Press SPACE to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "Q = quit", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.imshow("Capture - Press SPACE", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            frame = f.copy()   # clean frame, no text overlay
            print("[Test] Frame captured.")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


def draw_tracking(
    frame: np.ndarray,
    landmarks_raw: dict | None,
    joint_colors: dict,
    detection_result,
) -> np.ndarray:
    """
    Draw finger tracking on the frame.
    Uses API landmark data if include_landmarks=True was set,
    otherwise falls back to the local detection result.
    """
    h, w = frame.shape[:2]

    # Prefer API landmarks if available, else use local detection
    if landmarks_raw and landmarks_raw.get("right_hand"):
        pts = [(int(lm["x"] * w), int(lm["y"] * h))
               for lm in landmarks_raw["right_hand"]]
    elif detection_result and detection_result.right_hand:
        pts = [(int(lm.x * w), int(lm.y * h))
               for lm in detection_result.right_hand]
    else:
        return frame

    # Draw connections first (underneath dots)
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)

    # Draw each landmark dot colored by joint score
    for joint_name, indices in JOINT_TO_INDICES.items():
        color_name = joint_colors.get(joint_name, "green")
        color = COLOR_MAP.get(color_name, (200, 200, 200))
        for idx in indices:
            cv2.circle(frame, pts[idx], 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pts[idx], 6, (255, 255, 255), 1, cv2.LINE_AA)

    # Wrist always white
    cv2.circle(frame, pts[0], 6, (255, 255, 255), -1, cv2.LINE_AA)

    return frame


def draw_results(frame: np.ndarray, result: dict, sign: str) -> np.ndarray:
    """Draw API result overlay: sign, score, confidence, feedback messages."""
    h, w = frame.shape[:2]

    # Semi-transparent dark panel on the left
    overlay = frame.copy()
    panel_w = 320
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Score color
    score = result.get("overall_score", 0)
    if score >= 75:
        score_color = (0, 200, 80)
    elif score >= 50:
        score_color = (0, 165, 255)
    else:
        score_color = (0, 60, 220)

    # Target sign label
    cv2.putText(frame, f"Target: {sign}", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # Detected sign large
    detected = result.get("detected_sign", "?")
    conf = result.get("confidence", 0)
    cv2.putText(frame, f"Detected: {detected}", (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {conf*100:.1f}%", (12, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # Score bar
    bar_y = 120
    cv2.putText(frame, f"Score: {score:.1f}/100", (12, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    bar_bg_w = panel_w - 24
    cv2.rectangle(frame, (12, bar_y + 8), (12 + bar_bg_w, bar_y + 22), (60, 60, 60), -1)
    fill_w = int(bar_bg_w * score / 100)
    cv2.rectangle(frame, (12, bar_y + 8), (12 + fill_w, bar_y + 22), score_color, -1)

    # Correct / not correct
    is_correct = result.get("is_correct", False)
    status_txt = "CORRECT" if is_correct else "KEEP TRYING"
    status_col = (0, 200, 80) if is_correct else (0, 100, 220)
    cv2.putText(frame, status_txt, (12, bar_y + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2, cv2.LINE_AA)

    # Praise — ASCII only (OpenCV putText does not support unicode emoji on Windows)
    praise = result.get("praise", "")
    emoji  = result.get("emoji", "")
    emoji_ascii = {"\U0001f31f":"[*]","\u2705":"[OK]","\U0001f44d":"[+]","\U0001f914":"[?]","\u274c":"[X]"}.get(emoji, "")
    cv2.putText(frame, emoji_ascii + " " + praise, (12, bar_y + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # Feedback messages
    messages = result.get("messages", [])
    cv2.putText(frame, "Feedback:", (12, bar_y + 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 200, 255), 1, cv2.LINE_AA)
    for i, msg in enumerate(messages[:3]):
        # Word-wrap at ~38 chars
        words = msg.split()
        lines, line = [], ""
        for word in words:
            if len(line) + len(word) + 1 <= 36:
                line = (line + " " + word).strip()
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        for j, l in enumerate(lines[:2]):
            cv2.putText(frame, f"  {l}", (12, bar_y + 124 + i*42 + j*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    # Joint color legend bottom
    legend_y = h - 60
    cv2.putText(frame, "Joint colors:", (12, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    for i, (label, color) in enumerate([("Good", (0,200,80)),
                                         ("Close", (0,165,255)),
                                         ("Off", (0,60,220))]):
        x = 12 + i * 96
        cv2.circle(frame, (x, legend_y + 18), 6, color, -1)
        cv2.putText(frame, label, (x + 10, legend_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def frame_to_base64(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode('utf-8')


def test_api(sign: str = "A", save_path: str = None, camera: int = 1):

    # ── 1. Capture frame ──
    frame = capture_frame(camera_index=camera)
    if frame is None:
        print("[ERROR] No frame captured.")
        return

    # ── 2. Run local detection for landmark drawing ──
    detector = ASLDetector(draw_landmarks=False, model_complexity=1)
    local_detection = detector.process_frame(frame.copy())
    detector.release()

    # ── 3. Send to API ──
    print(f"[Test] Sending to API (target sign: {sign})...")
    payload = {
        "frame_base64": frame_to_base64(frame),
        "target_sign": sign,
        "mode": "static",
        "include_landmarks": True,   # ask API for landmark coords too
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to API.")
        print("        Start the server first:")
        print("        uvicorn api.router:app --reload --port 8000")
        return
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # ── 4. Build annotated image ──
    annotated = frame.copy()

    # Draw finger tracking with joint colors from API
    annotated = draw_tracking(
        annotated,
        landmarks_raw=result.get("landmarks"),
        joint_colors=result.get("joint_colors", {}),
        detection_result=local_detection,
    )

    # Draw result overlay panel
    annotated = draw_results(annotated, result, sign)

    # ── 5. Save ──
    out_path = save_path or f"result_{sign}.png"
    cv2.imwrite(out_path, annotated)
    print(f"[Test] Saved annotated image → {out_path}")

    # ── 6. Show ──
    cv2.imshow(f"ASL Test — Sign {sign}", annotated)
    print("[Test] Press any key to close the image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── 7. Print summary ──
    print("\n" + "="*50)
    print("API RESPONSE SUMMARY")
    print("="*50)
    print(f"Hand detected   : {result['hand_detected']}")
    print(f"Detected sign   : {result['detected_sign']}")
    print(f"Confidence      : {result['confidence']*100:.1f}%")
    print(f"Overall score   : {result['overall_score']}/100")
    print(f"Is correct      : {result['is_correct']}")
    print(f"Emoji           : {result['emoji']}")
    print(f"Praise          : {result['praise']}")
    print(f"\nFeedback:")
    for msg in result['messages']:
        print(f"  → {msg}")
    print(f"\nJoint colors:")
    for joint, color in result.get('joint_colors', {}).items():
        dot = {"green": "●", "orange": "◐", "red": "○"}.get(color, "·")
        print(f"  {dot} {joint:<20} : {color}")
    print("="*50)
    print(f"\nAnnotated image saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test ASL API and save annotated photo with tracking."
    )
    parser.add_argument("--sign",   default="A",         help="Target sign (default: A)")
    parser.add_argument("--save",   default=None,        help="Output image path (default: result_<sign>.png)")
    parser.add_argument("--camera", default=1, type=int, help="Camera index (default: 1)")
    args = parser.parse_args()

    test_api(
        sign=args.sign.upper(),
        save_path=args.save,
        camera=args.camera,
    )