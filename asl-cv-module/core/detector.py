"""
core/detector.py
----------------
MediaPipe Holistic wrapper for ASL landmark detection.
This is the entry point of the entire CV pipeline.

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
    This is what gets passed downstream to extractor.py
    """
    # Raw landmarks (None if not detected)
    left_hand: Optional[list[Landmark]] = None       # 21 landmarks
    right_hand: Optional[list[Landmark]] = None      # 21 landmarks
    pose: Optional[list[Landmark]] = None            # 33 landmarks
    face: Optional[list[Landmark]] = None            # 468 landmarks (optional)

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
    Wraps MediaPipe Holistic for ASL landmark extraction.

    Designed to be instantiated ONCE and reused across frames.
    Loading MediaPipe is expensive — never instantiate per-frame.

    Args:
        min_detection_confidence (float): Confidence threshold for initial detection.
        min_tracking_confidence (float): Confidence threshold for tracking across frames.
        model_complexity (int): 0 = lite (fastest), 1 = full, 2 = heavy (most accurate).
        include_face (bool): Whether to extract face mesh landmarks (adds overhead).
        draw_landmarks (bool): Whether to draw skeleton on the annotated frame.
    """

    # MediaPipe landmark indices for key body points
    POSE_LANDMARKS = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
    }

    HAND_LANDMARKS = {
        "wrist": 0,
        "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
        "index_mcp": 5, "index_pip": 6, "index_dip": 7, "index_tip": 8,
        "middle_mcp": 9, "middle_pip": 10, "middle_dip": 11, "middle_tip": 12,
        "ring_mcp": 13, "ring_pip": 14, "ring_dip": 15, "ring_tip": 16,
        "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20,
    }

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        model_complexity: int = 1,
        include_face: bool = False,
        draw_landmarks: bool = True,
    ):
        self.include_face = include_face
        self.draw_landmarks = draw_landmarks

        self._mp_holistic = mp.solutions.holistic
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._holistic = self._mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

        print("[ASLDetector] Initialized MediaPipe Holistic successfully.")

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

        # MediaPipe requires RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # performance optimization

        mp_result = self._holistic.process(rgb_frame)

        rgb_frame.flags.writeable = True

        result = DetectionResult(raw_result=mp_result)

        # ── Extract Hand Landmarks ──
        if mp_result.left_hand_landmarks:
            result.left_hand = self._parse_landmarks(mp_result.left_hand_landmarks.landmark)
            result.left_hand_detected = True

        if mp_result.right_hand_landmarks:
            result.right_hand = self._parse_landmarks(mp_result.right_hand_landmarks.landmark)
            result.right_hand_detected = True

        # ── Extract Pose Landmarks ──
        if mp_result.pose_landmarks:
            result.pose = self._parse_landmarks(mp_result.pose_landmarks.landmark)
            result.pose_detected = True

        # ── Extract Face Landmarks (optional) ──
        if self.include_face and mp_result.face_landmarks:
            result.face = self._parse_landmarks(mp_result.face_landmarks.landmark)

        # ── Draw Skeleton on Frame ──
        if self.draw_landmarks:
            result.annotated_frame = self._draw(frame.copy(), mp_result)
        else:
            result.annotated_frame = frame.copy()

        return result

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _parse_landmarks(self, raw_landmarks) -> list[Landmark]:
        """Convert MediaPipe landmark objects to our Landmark dataclass."""
        return [
            Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=getattr(lm, "visibility", 1.0),
            )
            for lm in raw_landmarks
        ]

    def _draw(self, frame: np.ndarray, mp_result) -> np.ndarray:
        """Draw skeleton overlays on the frame."""

        # Draw right hand
        if mp_result.right_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                mp_result.right_hand_landmarks,
                self._mp_holistic.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style(),
            )

        # Draw left hand
        if mp_result.left_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                mp_result.left_hand_landmarks,
                self._mp_holistic.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style(),
            )

        # Draw upper body pose
        if mp_result.pose_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                mp_result.pose_landmarks,
                self._mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_drawing.DrawingSpec(
                    color=(80, 110, 10), thickness=1, circle_radius=1
                ),
                connection_drawing_spec=self._mp_drawing.DrawingSpec(
                    color=(80, 256, 121), thickness=1, circle_radius=1
                ),
            )

        return frame

    def get_landmark_by_name(
        self,
        result: DetectionResult,
        name: str,
        hand: str = "right",
    ) -> Optional[Landmark]:
        """
        Convenience method to get a specific landmark by name.

        Args:
            result: DetectionResult from process_frame()
            name: Landmark name (e.g. 'index_tip', 'wrist', 'nose')
            hand: 'right' or 'left' for hand landmarks

        Returns:
            Landmark or None
        """
        # Check hand landmarks
        if name in self.HAND_LANDMARKS:
            idx = self.HAND_LANDMARKS[name]
            hand_data = result.right_hand if hand == "right" else result.left_hand
            if hand_data:
                return hand_data[idx]

        # Check pose landmarks
        if name in self.POSE_LANDMARKS:
            idx = self.POSE_LANDMARKS[name]
            if result.pose:
                return result.pose[idx]

        return None

    def release(self):
        """Release MediaPipe resources. Call when done."""
        self._holistic.close()
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
                f"Right Hand: {'✓' if result.right_hand_detected else '✗'}",
                f"Left Hand:  {'✓' if result.left_hand_detected else '✗'}",
                f"Pose:       {'✓' if result.pose_detected else '✗'}",
            ]

            for i, line in enumerate(status_lines):
                color = (0, 255, 0) if "✓" in line else (0, 0, 255)
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
                    (10, 130),
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
