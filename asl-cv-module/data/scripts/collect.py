"""
data/scripts/collect.py
-----------------------
Record expert ASL pose references from webcam.
Run this to create the reference JSON files for each sign.

Usage:
    python data/scripts/collect.py --sign A
    python data/scripts/collect.py --sign B
"""

import cv2
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.detector import ASLDetector
from core.extractor import ASLFeatureExtractor
from core.scorer import ASLScorer


def collect_reference(sign: str, num_samples: int = 30):
    """
    Collect N frames of a sign and save the median as reference.

    Args:
        sign: ASL sign label (e.g. "A")
        num_samples: How many frames to average over
    """
    print(f"\n[Collect] Recording reference for sign '{sign}'")
    print(f"[Collect] Will capture {num_samples} frames.")
    print("[Collect] Press SPACE to start capturing, Q to quit.\n")

    cap = cv2.VideoCapture(0)
    detector = ASLDetector(draw_landmarks=True)
    extractor = ASLFeatureExtractor()
    scorer = ASLScorer()

    samples = []
    capturing = False
    captured = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        detection = detector.process_frame(frame)
        features = extractor.extract(detection)

        # UI overlay
        status = f"Sign: {sign} | Capturing: {captured}/{num_samples}" if capturing \
                 else f"Sign: {sign} | Press SPACE to start"
        color = (0, 255, 0) if capturing else (0, 200, 255)

        cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if detection.is_valid():
            cv2.putText(frame, "Hand detected ✓", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if capturing and features.is_valid:
            samples.append(features.vector.copy())
            captured += 1
            if captured >= num_samples:
                break

        cv2.imshow(f"Recording: {sign}", detection.annotated_frame or frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            capturing = True
            print("[Collect] Capturing started...")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()

    if len(samples) < 5:
        print("[Collect] Not enough samples collected. Aborting.")
        return

    # Use median to be robust to outlier frames
    median_vector = np.median(np.array(samples), axis=0).astype(np.float32)

    # Reconstruct FeatureVector from median
    from core.extractor import FeatureVector
    fv = FeatureVector(
        right_hand_angles=median_vector[0:15],
        right_hand_position=median_vector[162:165],
        right_hand_orientation=median_vector[156:159],
        right_finger_extensions=median_vector[168:173],
        vector=median_vector,
        is_valid=True,
    )

    scorer.save_reference(fv, sign)
    print(f"[Collect] ✓ Reference for '{sign}' saved successfully from {len(samples)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect ASL sign reference poses.")
    parser.add_argument("--sign", required=True, help="ASL sign label, e.g. A, B, HELLO")
    parser.add_argument("--samples", type=int, default=30, help="Number of frames to capture")
    args = parser.parse_args()

    collect_reference(sign=args.sign.upper(), num_samples=args.samples)
