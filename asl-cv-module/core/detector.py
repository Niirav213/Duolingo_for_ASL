"""
core/detector.py
----------------
MediaPipe Hands wrapper for ASL landmark detection.
Uses mp.solutions.hands for better detection of hand-only images
and webcam frames where full body is not visible.

Usage:
    detector = ASLDetector()
    result = detector.process_frame(frame)  # frame = OpenCV BGR image
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


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
    Passed downstream to extractor.py
    """
    left_hand: Optional[list[Landmark]] = None        # 21 landmarks
    right_hand: Optional[list[Landmark]] = None       # 21 landmarks
    pose: Optional[list[Landmark]] = None             # None (not used in hands-only mode)
    face: Optional[list[Landmark]] = None             # None
    annotated_frame: Optional[np.ndarray] = None
    left_hand_detected: bool = False
    right_hand_detected: bool = False
    pose_detected: bool = False
    raw_result: object = field(default=None, repr=False)

    def is_valid(self) -> bool:
        return self.left_hand_detected or self.right_hand_detected

    def dominant_hand(self) -> Optional[list[Landmark]]:
        return self.right_hand if self.right_hand_detected else self.left_hand


# ─────────────────────────────────────────────
# Detector Class
# ─────────────────────────────────────────────

class ASLDetector:
    """
    Wraps MediaPipe Hands for ASL landmark extraction.
    Uses hands-only detector which works for both:
    - Close-up hand images (Kaggle dataset)
    - Webcam frames (real-time use)

    Instantiate ONCE and reuse across frames.
    """

    HAND_LANDMARKS = {
        "wrist": 0,
        "thumb_cmc": 1,  "thumb_mcp": 2,  "thumb_ip": 3,   "thumb_tip": 4,
        "index_mcp": 5,  "index_pip": 6,  "index_dip": 7,  "index_tip": 8,
        "middle_mcp": 9, "middle_pip": 10,"middle_dip": 11,"middle_tip": 12,
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
        static_image_mode: bool = False,
    ):
        self.draw_landmarks = draw_landmarks
        self.include_face = include_face

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

        print("[ASLDetector] Initialized MediaPipe Hands successfully.")

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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        mp_result = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        result = DetectionResult(raw_result=mp_result)

        if mp_result.multi_hand_landmarks and mp_result.multi_handedness:
            for hand_landmarks, handedness in zip(
                mp_result.multi_hand_landmarks,
                mp_result.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                landmarks = self._parse_landmarks(hand_landmarks.landmark)

                if label == "Right":
                    result.right_hand = landmarks
                    result.right_hand_detected = True
                else:
                    result.left_hand = landmarks
                    result.left_hand_detected = True

        result.annotated_frame = self._draw(frame.copy(), mp_result) if self.draw_landmarks else frame.copy()
        return result

    def _parse_landmarks(self, raw_landmarks) -> list[Landmark]:
        return [
            Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=1.0)
            for lm in raw_landmarks
        ]

    def _draw(self, frame: np.ndarray, mp_result) -> np.ndarray:
        if mp_result.multi_hand_landmarks:
            for hand_landmarks in mp_result.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_drawing_styles.get_default_hand_landmarks_style(),
                    self._mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame

    def get_landmark_by_name(
        self,
        result: DetectionResult,
        name: str,
        hand: str = "right",
    ) -> Optional[Landmark]:
        if name in self.HAND_LANDMARKS:
            idx = self.HAND_LANDMARKS[name]
            hand_data = result.right_hand if hand == "right" else result.left_hand
            if hand_data:
                return hand_data[idx]
        return None

    def release(self):
        self._hands.close()
        print("[ASLDetector] Resources released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# ─────────────────────────────────────────────
# Quick Test
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

            frame = cv2.flip(frame, 1)
            result = detector.process_frame(frame)

            status_lines = [
                f"Right Hand: {'✓' if result.right_hand_detected else '✗'}",
                f"Left Hand:  {'✓' if result.left_hand_detected else '✗'}",
            ]

            for i, line in enumerate(status_lines):
                color = (0, 255, 0) if "✓" in line else (0, 0, 255)
                cv2.putText(result.annotated_frame, line, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("ASL Detector - Test", result.annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Test complete.")