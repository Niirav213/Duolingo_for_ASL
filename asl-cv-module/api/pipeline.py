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
import pickle

from core.detector import ASLDetector
from core.extractor import ASLFeatureExtractor
from core.scorer import ASLScorer
from feedback.generator import FeedbackGenerator, FeedbackResult
from models.static import StaticSignClassifier
from models.dynamic import DynamicSignClassifier
from api.schemas import AnalyzeFrameResponse, JointScores

# ── Model registry — edit to swap models ──
MODEL_REGISTRY = {
    "static": "models/checkpoints/static_sign_best.pt",
    "dynamic": "models/checkpoints/dynamic_sign.pt",
}


class ASLPipeline:
    """
    Full ASL CV pipeline — instantiated ONCE at server startup.

    Call analyze_frame() for every incoming frame.
    """

    def __init__(
    self,
    load_static: bool = True,
    load_dynamic: bool = True,
    static_labels: list[str] = None,
    dynamic_labels: list[str] = None,
    ):
        print("[ASLPipeline] Initializing...")

        # ── Core components ──
        self.detector = ASLDetector(model_complexity=1, draw_landmarks=False, static_image_mode=True)
        self.extractor = ASLFeatureExtractor()
        self.scorer = ASLScorer()
        self.feedback_gen = FeedbackGenerator()

        # ── Load label map from file ──
        import json
        from pathlib import Path
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
                print(f"[ASLPipeline] WARNING: No static checkpoint found at {static_path}.")

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
                print(f"[ASLPipeline] WARNING: No dynamic checkpoint found at {dynamic_path}.")
        #scalar loading
        

    def analyze_frame(
        self,
        frame_base64: str,
        target_sign: str,
        mode: str = "static",
        include_landmarks: bool = False,
    ) -> AnalyzeFrameResponse:
        """
        Full pipeline: base64 frame → JSON response.

        Args:
            frame_base64: base64-encoded JPEG/PNG image string
            target_sign: the ASL sign the user is attempting
            mode: "static" for letters, "dynamic" for word signs
            include_landmarks: whether to include raw landmarks in response

        Returns:
            AnalyzeFrameResponse — ready to serialize to JSON
        """
        # ── 1. Decode frame ──
        frame = self._decode_frame(frame_base64)
        frame = cv2.flip(frame, 1)

        # ── 2. Detect landmarks ──
        detection = self.detector.process_frame(frame)

        if not detection.is_valid():
            return self._no_detection_response(target_sign)

        # ── 3. Extract features ──
        features = self.extractor.extract(detection)

        # ── 4. Classify sign ──
        detected_sign, confidence = "", 0.0


        if mode == "static" and self.static_classifier:
            detected_sign, confidence = self.static_classifier.predict(features.vector)
        elif mode == "dynamic" and self.dynamic_classifier:
            detected_sign, confidence = self.dynamic_classifier.predict(features.vector)

        # ── 5. Score against target ──
        score_result = self.scorer.score(features, target_sign)

        # ── 6. Generate feedback ──
        feedback = self.feedback_gen.generate(score_result)

        # ── 7. Build response ──
        joint_scores = JointScores(**{
            k: v for k, v in score_result.joint_scores.items()
            if k in JointScores.model_fields
        })
        joint_scores.position = score_result.position_score
        joint_scores.orientation = score_result.orientation_score

        landmarks = None
        if include_landmarks and detection.right_hand:
            landmarks = {
                "right_hand": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in detection.right_hand],
                "left_hand": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in detection.left_hand] if detection.left_hand else [],
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

    def _decode_frame(self, frame_base64: str) -> np.ndarray:
        """Decode base64 image string to OpenCV BGR numpy array."""
        # Strip data URL prefix if present (e.g. "data:image/jpeg;base64,...")
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
