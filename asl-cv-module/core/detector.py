"""
core/detector.py
----------------
MediaPipe Hands wrapper for ASL landmark detection.
Updated to use the mediapipe Tasks API (mediapipe 0.10+).

Usage:
    detector = ASLDetector()
    result = detector.process_frame(frame)  # frame = OpenCV BGR image
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode



# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class DetectionResult:
    """
    Output of ASLDetector.process_frame().
    This is what gets passed downstream to extractor.py
    """
    # Raw landmarks (None if not detected)
    left_hand: Optional[list[Landmark]] = None       # 21 landmarks
    right_hand: Optional[list[Landmark]] = None      # 21 landmarks
    pose: Optional[list[Landmark]] = None            # None (not used in hands-only mode)
    face: Optional[list[Landmark]] = None            # None

    # Annotated frame with skeleton drawn on it
    annotated_frame: Optional[np.ndarray] = None

    # Detection flags
    left_hand_detected: bool = False
    right_hand_detected: bool = False
    pose_detected: bool = False

    # Raw mediapipe result (for advanced use)
    raw_result: object = field(default=None, repr=False)

    def is_valid(self) -> bool:
        """Returns True if at least one hand is detected."""
        return self.left_hand_detected or self.right_hand_detected

    def dominant_hand(self) -> Optional[list[Landmark]]:
        """Returns right hand if available, else left hand."""
        return self.right_hand if self.right_hand_detected else self.left_hand


# ─────────────────────────────────────────────
# Detector Class
# ─────────────────────────────────────────────

class ASLDetector:
    """
    Wraps MediaPipe HandLandmarker for ASL landmark extraction.
    Updated to use mediapipe Tasks API (mediapipe 0.10+).

    Args:
        min_detection_confidence (float): Confidence threshold for detection.
        min_tracking_confidence (float): Confidence threshold for tracking.
        model_complexity (int): Unused, kept for API compatibility.
        include_face (bool): Unused — kept for API compatibility.
        draw_landmarks (bool): Whether to draw skeleton on the annotated frame.
    """

    # Path to the hand landmarker model bundle
    MODEL_PATH = str(Path(__file__).parent.parent / "models" / "hand_landmarker.task")

    HAND_LANDMARKS = {
        "wrist": 0,
        "thumb_cmc": 1,  "thumb_mcp": 2,  "thumb_ip": 3,   "thumb_tip": 4,
        "index_mcp": 5,  "index_pip": 6,  "index_dip": 7,  "index_tip": 8,
        "middle_mcp": 9, "middle_pip": 10, "middle_dip": 11, "middle_tip": 12,
        "ring_mcp": 13,  "ring_pip": 14,  "ring_dip": 15,  "ring_tip": 16,
        "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20,
    }

    POSE_LANDMARKS = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13,    "right_elbow": 14,
        "left_wrist": 15,    "right_wrist": 16,
        "left_hip": 23,      "right_hip": 24,
    }

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        include_face: bool = False,
        draw_landmarks: bool = True,
    ):
        self.draw_landmarks = draw_landmarks
        self.include_face = include_face

        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        print("[ASLDetector] Initialized MediaPipe HandLandmarker successfully.")

    # ─────────────────────────────────────────
    # Main Method
    # ─────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Process a single BGR frame and return detected landmarks.

        Args:
            frame: OpenCV BGR image (np.ndarray)

        Returns:
            DetectionResult with landmarks and annotated frame
        """
        if frame is None or frame.size == 0:
            raise ValueError("[ASLDetector] Received empty frame.")

        # Tasks API requires RGB mediapipe.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection = self._landmarker.detect(mp_image)

        result = DetectionResult(raw_result=detection)

        # ── Extract Hand Landmarks ──
        if detection.hand_landmarks and detection.handedness:
            for hand_landmarks_list, handedness_list in zip(
                detection.hand_landmarks,
                detection.handedness,
            ):
                label = handedness_list[0].category_name  # "Left" or "Right"
                landmarks = [
                    Landmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=getattr(lm, "visibility", 1.0),
                    )
                    for lm in hand_landmarks_list
                ]

                if label == "Right":
                    result.right_hand = landmarks
                    result.right_hand_detected = True
                else:
                    result.left_hand = landmarks
                    result.left_hand_detected = True

        result.annotated_frame = frame.copy()
        return result

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def get_landmark_by_name(
        self,
        result: DetectionResult,
        name: str,
        hand: str = "right",
    ) -> Optional[Landmark]:
        """Get a specific landmark by name."""
        if name in self.HAND_LANDMARKS:
            idx = self.HAND_LANDMARKS[name]
            hand_data = result.right_hand if hand == "right" else result.left_hand
            if hand_data:
                return hand_data[idx]
        return None

    def release(self):
        """Release MediaPipe resources. Call when done."""
        self._landmarker.close()
        print("[ASLDetector] Resources released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# ─────────────────────────────────────────────
# Quick Test (run this file directly to verify)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting webcam test... Press 'q' to quit.\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        exit()

    with ASLDetector(model_complexity=1, draw_landmarks=True) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural feel
            result = detector.process_frame(frame)

            # ── Status Overlay ──
            status_lines = [
                f"Right Hand: {'Y' if result.right_hand_detected else 'N'}",
                f"Left Hand:  {'Y' if result.left_hand_detected else 'N'}",
            ]

            for i, line in enumerate(status_lines):
                color = (0, 255, 0) if "Y" in line else (0, 0, 255)
                cv2.putText(
                    result.annotated_frame,
                    line,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            # ── Show index fingertip coords as example ──
            tip = detector.get_landmark_by_name(result, "index_tip", hand="right")
            if tip:
                cv2.putText(
                    result.annotated_frame,
                    f"Index Tip: ({tip.x:.2f}, {tip.y:.2f}, {tip.z:.2f})",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    1,
                )

            cv2.imshow("ASL Detector - Test", result.annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Test complete.")