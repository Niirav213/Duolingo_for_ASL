"""
core/extractor.py
-----------------
Feature engineering layer for ASL landmark data.
Takes a DetectionResult from detector.py and extracts
meaningful features (joint angles, normalized positions,
orientation vectors) that the PyTorch model can learn from.

Pipeline position:
    detector.py → [extractor.py] → models/static.py or dynamic.py

Usage:
    extractor = ASLFeatureExtractor()
    features = extractor.extract(detection_result)
    # features.vector → numpy array ready for PyTorch model
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Import our DetectionResult and Landmark from detector
from detector import DetectionResult, Landmark


# ─────────────────────────────────────────────
# Output Data Structure
# ─────────────────────────────────────────────

@dataclass
class FeatureVector:
    """
    Output of ASLFeatureExtractor.extract().
    This is what gets passed to the PyTorch model.
    """

    # ── Per-hand joint angles (degrees) ──
    right_hand_angles: Optional[np.ndarray] = None   # shape (15,) — 3 angles × 5 fingers
    left_hand_angles: Optional[np.ndarray] = None    # shape (15,)

    # ── Normalized landmark positions ──
    right_hand_normalized: Optional[np.ndarray] = None  # shape (21 × 3,) = (63,)
    left_hand_normalized: Optional[np.ndarray] = None   # shape (63,)

    # ── Hand orientation (palm normal vector) ──
    right_hand_orientation: Optional[np.ndarray] = None  # shape (3,)
    left_hand_orientation: Optional[np.ndarray] = None   # shape (3,)

    # ── Hand position relative to body ──
    right_hand_position: Optional[np.ndarray] = None    # shape (3,) relative to nose
    left_hand_position: Optional[np.ndarray] = None     # shape (3,)

    # ── Finger extension states (0.0 = curled, 1.0 = extended) ──
    right_finger_extensions: Optional[np.ndarray] = None  # shape (5,)
    left_finger_extensions: Optional[np.ndarray] = None   # shape (5,)

    # ── Final flattened vector for model input ──
    vector: Optional[np.ndarray] = None   # shape (206,) — all features concatenated
    is_valid: bool = False

    def __repr__(self):
        return (
            f"FeatureVector("
            f"valid={self.is_valid}, "
            f"vector_shape={self.vector.shape if self.vector is not None else None})"
        )


# ─────────────────────────────────────────────
# Extractor Class
# ─────────────────────────────────────────────

class ASLFeatureExtractor:
    """
    Converts DetectionResult landmarks into a feature vector
    suitable for input into a PyTorch classification model.

    Feature breakdown (total = 206 features):
    ┌─────────────────────────────────┬────────┐
    │ Feature                         │  Size  │
    ├─────────────────────────────────┼────────┤
    │ Right hand joint angles         │   15   │
    │ Left hand joint angles          │   15   │
    │ Right hand normalized positions │   63   │
    │ Left hand normalized positions  │   63   │
    │ Right hand orientation          │    3   │
    │ Left hand orientation           │    3   │
    │ Right hand body position        │    3   │
    │ Left hand body position         │    3   │
    │ Right finger extensions         │    5   │
    │ Left finger extensions          │    5   │
    ├─────────────────────────────────┼────────┤
    │ TOTAL                           │  178   │
    └─────────────────────────────────┴────────┘

    Note: Missing hands are zero-padded so vector size is always fixed.
    Fixed size is critical — PyTorch models require consistent input shape.
    """

    # Finger joint triplets for angle calculation
    # Each tuple = (base, middle, tip) landmark indices
    # Angle is computed AT the middle joint
    FINGER_ANGLE_TRIPLETS = [
        # Thumb
        (1, 2, 3),   # CMC → MCP → IP
        (2, 3, 4),   # MCP → IP → TIP
        # Index
        (5, 6, 7),   # MCP → PIP → DIP
        (6, 7, 8),   # PIP → DIP → TIP
        # Middle
        (9, 10, 11),
        (10, 11, 12),
        # Ring
        (13, 14, 15),
        (14, 15, 16),
        # Pinky
        (17, 18, 19),
        (18, 19, 20),
        # Finger splay (MCP spread angles)
        (5, 0, 9),   # Index spread from wrist
        (9, 0, 13),  # Middle spread
        (13, 0, 17), # Ring spread
        (5, 0, 17),  # Index to pinky spread
        (0, 5, 9),   # Wrist-index-middle triangle
    ]

    # Finger tip and base indices for extension calculation
    FINGER_TIPS = [4, 8, 12, 16, 20]    # thumb, index, middle, ring, pinky
    FINGER_MCPS = [2, 5, 9, 13, 17]     # corresponding base joints

    def __init__(self, zero_pad_missing: bool = True):
        """
        Args:
            zero_pad_missing: If True, missing hands are filled with zeros.
                              Keep True for model input consistency.
        """
        self.zero_pad_missing = zero_pad_missing

    # ─────────────────────────────────────────
    # Main Method
    # ─────────────────────────────────────────

    def extract(self, detection: DetectionResult) -> FeatureVector:
        """
        Extract all features from a DetectionResult.

        Args:
            detection: Output from ASLDetector.process_frame()

        Returns:
            FeatureVector with .vector ready for PyTorch model
        """
        fv = FeatureVector()

        if not detection.is_valid():
            # No hands detected — return zero vector
            if self.zero_pad_missing:
                fv.vector = np.zeros(178, dtype=np.float32)
            return fv

        # ── Right Hand Features ──
        if detection.right_hand_detected:
            lms = detection.right_hand
            fv.right_hand_angles = self._compute_finger_angles(lms)
            fv.right_hand_normalized = self._normalize_landmarks(lms)
            fv.right_hand_orientation = self._compute_palm_orientation(lms)
            fv.right_finger_extensions = self._compute_finger_extensions(lms)

            if detection.pose_detected:
                fv.right_hand_position = self._hand_relative_to_body(
                    lms, detection.pose
                )

        # ── Left Hand Features ──
        if detection.left_hand_detected:
            lms = detection.left_hand
            fv.left_hand_angles = self._compute_finger_angles(lms)
            fv.left_hand_normalized = self._normalize_landmarks(lms)
            fv.left_hand_orientation = self._compute_palm_orientation(lms)
            fv.left_finger_extensions = self._compute_finger_extensions(lms)

            if detection.pose_detected:
                fv.left_hand_position = self._hand_relative_to_body(
                    lms, detection.pose
                )

        # ── Assemble Final Vector ──
        fv.vector = self._assemble_vector(fv)
        fv.is_valid = True

        return fv

    # ─────────────────────────────────────────
    # Feature Computation Methods
    # ─────────────────────────────────────────

    def _compute_finger_angles(self, landmarks: list[Landmark]) -> np.ndarray:
        """
        Compute joint angles for all finger triplets.

        Each angle is computed at the MIDDLE joint of a triplet (a, b, c).
        Returns array of shape (15,) in degrees [0, 180].

        Example: triplet (5, 6, 7) computes angle at PIP joint of index finger.
        Fully extended finger ≈ 180°, fully curled ≈ 60-90°.
        """
        angles = []
        pts = [lm.to_array() for lm in landmarks]

        for (a, b, c) in self.FINGER_ANGLE_TRIPLETS:
            angle = self._angle_at_joint(pts[a], pts[b], pts[c])
            angles.append(angle)

        return np.array(angles, dtype=np.float32)

    def _normalize_landmarks(self, landmarks: list[Landmark]) -> np.ndarray:
        """
        Normalize all 21 hand landmarks relative to the wrist (landmark 0).

        Why: Raw x,y,z values depend on where the hand is in the frame.
        After normalization, the features describe hand SHAPE, not position.
        This makes the model work regardless of where in frame the hand is.

        Returns array of shape (63,) = 21 landmarks × 3 coords.
        """
        pts = np.array([lm.to_array() for lm in landmarks])  # (21, 3)

        # Use wrist as origin
        wrist = pts[0]
        pts = pts - wrist

        # Scale by the distance from wrist to middle finger MCP (landmark 9)
        # This normalizes for hand size differences between users
        scale = np.linalg.norm(pts[9]) + 1e-6  # avoid division by zero
        pts = pts / scale

        return pts.flatten().astype(np.float32)  # (63,)

    def _compute_palm_orientation(self, landmarks: list[Landmark]) -> np.ndarray:
        """
        Compute the palm normal vector (which direction the palm faces).

        Uses cross product of two vectors across the palm:
        - Vector 1: wrist → index MCP (landmark 0 → 5)
        - Vector 2: wrist → pinky MCP (landmark 0 → 17)

        The cross product gives a vector perpendicular to the palm surface.
        This tells us if the palm faces toward/away from camera, left/right, up/down.

        Returns unit vector of shape (3,).
        """
        pts = [lm.to_array() for lm in landmarks]

        wrist = pts[0]
        index_mcp = pts[5]
        pinky_mcp = pts[17]

        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist

        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal) + 1e-6
        return (normal / norm).astype(np.float32)

    def _compute_finger_extensions(self, landmarks: list[Landmark]) -> np.ndarray:
        """
        Compute how extended each finger is (0.0 = fully curled, 1.0 = fully extended).

        Method: Compare distance from fingertip to wrist vs.
        distance from finger MCP base to wrist.
        If tip is farther than base → finger is extended.

        Returns array of shape (5,) for [thumb, index, middle, ring, pinky].
        """
        pts = [lm.to_array() for lm in landmarks]
        wrist = pts[0]
        extensions = []

        for tip_idx, mcp_idx in zip(self.FINGER_TIPS, self.FINGER_MCPS):
            tip_dist = np.linalg.norm(pts[tip_idx] - wrist)
            mcp_dist = np.linalg.norm(pts[mcp_idx] - wrist)
            # Ratio > 1 means tip is farther than base = extended
            ratio = tip_dist / (mcp_dist + 1e-6)
            # Clamp to [0, 1]
            extensions.append(float(np.clip(ratio - 1.0, 0.0, 1.0)))

        return np.array(extensions, dtype=np.float32)

    def _hand_relative_to_body(
        self,
        hand_landmarks: list[Landmark],
        pose_landmarks: list[Landmark],
    ) -> np.ndarray:
        """
        Compute hand wrist position relative to nose (pose landmark 0).

        Why: Many ASL signs are defined by WHERE the hand is relative to the body.
        For example, the sign for 'MOTHER' is at the chin, 'FATHER' at forehead.
        Raw position is useless — we need body-relative position.

        Returns normalized offset vector of shape (3,).
        """
        # Wrist of the hand (landmark 0 of hand)
        hand_wrist = hand_landmarks[0].to_array()

        # Nose position from pose (landmark 0)
        nose = pose_landmarks[0].to_array()

        # Shoulder width for scale normalization
        left_shoulder = pose_landmarks[11].to_array()
        right_shoulder = pose_landmarks[12].to_array()
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6

        # Offset normalized by shoulder width
        offset = (hand_wrist - nose) / shoulder_width
        return offset.astype(np.float32)

    # ─────────────────────────────────────────
    # Vector Assembly
    # ─────────────────────────────────────────

    def _assemble_vector(self, fv: FeatureVector) -> np.ndarray:
        """
        Concatenate all features into a single flat numpy array.
        Missing features are zero-padded to maintain fixed size.

        Final vector layout (178 total):
        [0:15]   right hand angles
        [15:30]  left hand angles
        [30:93]  right hand normalized positions
        [93:156] left hand normalized positions
        [156:159] right hand orientation
        [159:162] left hand orientation
        [162:165] right hand body position
        [165:168] left hand body position
        [168:173] right finger extensions
        [173:178] left finger extensions
        """
        def safe(arr, size):
            """Return arr if not None, else zeros of given size."""
            return arr if arr is not None else np.zeros(size, dtype=np.float32)

        parts = [
            safe(fv.right_hand_angles, 15),
            safe(fv.left_hand_angles, 15),
            safe(fv.right_hand_normalized, 63),
            safe(fv.left_hand_normalized, 63),
            safe(fv.right_hand_orientation, 3),
            safe(fv.left_hand_orientation, 3),
            safe(fv.right_hand_position, 3),
            safe(fv.left_hand_position, 3),
            safe(fv.right_finger_extensions, 5),
            safe(fv.left_finger_extensions, 5),
        ]

        return np.concatenate(parts).astype(np.float32)

    # ─────────────────────────────────────────
    # Math Helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _angle_at_joint(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> float:
        """
        Compute angle (in degrees) at joint B, formed by points A-B-C.

        Args:
            a: 3D point — one end of the angle
            b: 3D point — the joint (vertex of the angle)
            c: 3D point — other end of the angle

        Returns:
            Angle in degrees [0, 180]
        """
        ba = a - b  # vector from B to A
        bc = c - b  # vector from B to C

        # Cosine rule
        cosine = np.dot(ba, bc) / (
            np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
        )
        # Clamp to [-1, 1] to avoid arccos domain errors from floating point
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))


# ─────────────────────────────────────────────
# Quick Test (run directly to verify)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import cv2
    from detector import ASLDetector

    print("Starting extractor test... Press 'q' to quit.\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        exit()

    extractor = ASLFeatureExtractor()

    with ASLDetector(model_complexity=1, draw_landmarks=True) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            detection = detector.process_frame(frame)
            features = extractor.extract(detection)

            # ── Display feature info on frame ──
            lines = [
                f"Valid: {features.is_valid}",
                f"Vector shape: {features.vector.shape if features.vector is not None else 'N/A'}",
            ]

            if features.right_hand_angles is not None:
                lines.append(
                    f"Index PIP angle: {features.right_hand_angles[2]:.1f} deg"
                )
                lines.append(
                    f"Index extended: {features.right_finger_extensions[1]:.2f}"
                )

            if features.right_hand_orientation is not None:
                o = features.right_hand_orientation
                lines.append(f"Palm normal: ({o[0]:.2f}, {o[1]:.2f}, {o[2]:.2f})")

            for i, line in enumerate(lines):
                cv2.putText(
                    detection.annotated_frame,
                    line,
                    (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("ASL Extractor - Test", detection.annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nSample feature vector (first 20 values):")
    print(features.vector[:20] if features.vector is not None else "No features extracted")
    print(f"\nTotal feature vector size: {features.vector.shape if features.vector is not None else 0}")
