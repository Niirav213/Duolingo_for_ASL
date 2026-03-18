"""
api/pipeline.py
---------------
Orchestrates the full CV pipeline in one place.
Loads all models ONCE at startup and wires:
    detector → extractor → classifier → scorer → feedback

This is the single entry point called by router.py
"""

import cv2
import base64
import numpy as np
import torch
from pathlib import Path
import json

from core.detector import ASLDetector, Landmark
from core.extractor import ASLFeatureExtractor
from core.scorer import ASLScorer
from feedback.generator import FeedbackGenerator
from models.static import StaticSignClassifier
from models.dynamic import DynamicSignClassifier
from api.schemas import AnalyzeFrameResponse, JointScores

MODEL_REGISTRY = {
    "static":  "models/checkpoints/static_sign_best.pt",
    "dynamic": "models/checkpoints/dynamic_sign.pt",
}


class ASLPipeline:
    """Full ASL CV pipeline — instantiated ONCE at server startup."""

    def __init__(
        self,
        load_static: bool = True,
        load_dynamic: bool = True,
        static_labels: list[str] = None,
        dynamic_labels: list[str] = None,
    ):
        print("[ASLPipeline] Initializing...")

        self.detector     = ASLDetector(model_complexity=0, draw_landmarks=False)
        self.extractor    = ASLFeatureExtractor()
        self.scorer       = ASLScorer()
        self.feedback_gen = FeedbackGenerator()

        # ── Load label map ──
        label_map_path = Path("data/datasets/label_map.json")
        if label_map_path.exists():
            with open(label_map_path) as f:
                label_map = json.load(f)
            loaded_labels = [label_map[str(i)] for i in range(len(label_map))]
        else:
            loaded_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        # ── Static classifier ──
        self.static_classifier = None
        if load_static:
            labels_to_use = static_labels or loaded_labels
            self.static_classifier = StaticSignClassifier(
                num_classes=len(labels_to_use),
                labels=labels_to_use,
            )
            static_path = MODEL_REGISTRY["static"]
            if Path(static_path).exists():
                self.static_classifier.load(static_path)
            else:
                print(f"[ASLPipeline] WARNING: No static checkpoint at {static_path}.")

        # ── Dynamic classifier ──
        self.dynamic_classifier = None
        if load_dynamic:
            self.dynamic_classifier = DynamicSignClassifier(
                num_classes=len(dynamic_labels) if dynamic_labels else 100,
                labels=dynamic_labels,
                seq_len=30,
            )
            dynamic_path = MODEL_REGISTRY["dynamic"]
            if Path(dynamic_path).exists():
                self.dynamic_classifier.load(dynamic_path)
            else:
                print(f"[ASLPipeline] WARNING: No dynamic checkpoint at {dynamic_path}.")

        print("[ASLPipeline] Ready.")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def analyze_frame(
        self,
        frame_base64: str,
        target_sign: str,
        mode: str = "static",
        include_landmarks: bool = False,
    ) -> AnalyzeFrameResponse:
        """Full pipeline: base64 frame → JSON response."""

        # ── 1. Decode + flip frame ──
        frame = self._decode_frame(frame_base64)
        frame = cv2.flip(frame, 1)

        # ── 2. Detect landmarks ──
        detection = self.detector.process_frame(frame)

        if not detection.is_valid():
            return self._no_detection_response(target_sign)

        # ── 3. Mirror left hand → treat as right hand ──
        # The model was trained only on right hand data (Kaggle dataset).
        # If only a left hand is detected, flip its x landmarks so the
        # feature vector matches the right-hand training distribution.
        if detection.left_hand_detected and not detection.right_hand_detected:
            detection.right_hand = [
                Landmark(x=1.0 - lm.x, y=lm.y, z=lm.z)
                for lm in detection.left_hand
            ]
            detection.right_hand_detected = True
            detection.left_hand = None
            detection.left_hand_detected = False

        # ── 4. Extract features ──
        features = self.extractor.extract(detection)

        # ── 5. Classify ──
        detected_sign, confidence = "", 0.0
        if mode == "static" and self.static_classifier:
            detected_sign, confidence = self.static_classifier.predict(features.vector)
        elif mode == "dynamic" and self.dynamic_classifier:
            detected_sign, confidence = self.dynamic_classifier.predict(features.vector)

        # ── 6. Score ──
        score_result = self.scorer.score(features, target_sign)

        # ── 7. Feedback ──
        feedback = self.feedback_gen.generate(score_result)

        # ── 8. Build response ──
        joint_scores = JointScores(**{
            k: v for k, v in score_result.joint_scores.items()
            if k in JointScores.model_fields
        })
        joint_scores.position    = score_result.position_score
        joint_scores.orientation = score_result.orientation_score

        landmarks = None
        if include_landmarks and detection.right_hand:
            landmarks = {
                "right_hand": [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in detection.right_hand
                ],
                "left_hand": [],
            }

        return AnalyzeFrameResponse(
            hand_detected=True,
            detected_sign=detected_sign,
            confidence=confidence,
            overall_score=score_result.overall_score,
            is_correct=score_result.is_correct,
            joint_scores=joint_scores,
            messages=feedback.messages,
            praise=feedback.praise,
            emoji=feedback.emoji,
            joint_colors=feedback.joint_colors,
            landmarks=landmarks,
        )

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _decode_frame(self, frame_base64: str) -> np.ndarray:
        """Decode base64 image string to OpenCV BGR numpy array."""
        if "," in frame_base64:
            frame_base64 = frame_base64.split(",")[1]
        img_bytes = base64.b64decode(frame_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("[ASLPipeline] Failed to decode frame.")
        return frame

    def _no_detection_response(self, target_sign: str) -> AnalyzeFrameResponse:
        return AnalyzeFrameResponse(
            hand_detected=False,
            detected_sign="",
            confidence=0.0,
            overall_score=0.0,
            is_correct=False,
            joint_scores=JointScores(),
            messages=["No hand detected. Make sure your hand is visible in the camera."],
            praise="",
            emoji="🤔",
            joint_colors={},
        )

    def release(self):
        self.detector.release()
        print("[ASLPipeline] Released all resources.")