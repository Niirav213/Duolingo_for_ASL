"""
ml-service/app/pipeline.py
--------------------------
Replaces the dummy ONNX pipeline with the real asl-cv-module pipeline.

The asl-cv-module folder must be placed next to ml-service:
    project/
        asl-cv-module/      ← your CV module
        ml-service/         ← this backend service
        backend/
        frontend/

This file is a drop-in replacement for the original pipeline.py.
No changes needed anywhere else in ml-service.
"""

import sys
import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ── Add asl-cv-module to path ──
ASL_MODULE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "asl-cv-module"
if not ASL_MODULE_PATH.exists():
    raise ImportError(
        f"[Pipeline] asl-cv-module not found at {ASL_MODULE_PATH}.\n"
        "Make sure asl-cv-module/ is a sibling folder of ml-service/."
    )
sys.path.insert(0, str(ASL_MODULE_PATH))

from core.detector import ASLDetector, Landmark
from core.extractor import ASLFeatureExtractor
from core.scorer import ASLScorer
from feedback.generator import FeedbackGenerator
from models.static import StaticSignClassifier


class MediaPipePipeline:
    """
    Drop-in replacement for the original ONNX pipeline.
    Uses the real asl-cv-module detector + classifier internally.

    Keeps the same interface (load_model / predict) so ml-service/app/main.py
    needs zero changes.
    """

    def __init__(self, model_path: str = "app/models/gesture_model.onnx"):
        # model_path is ignored — we use our own checkpoint
        self.model_loaded = False
        self.detector   = None
        self.extractor  = None
        self.classifier = None
        self.scorer     = None
        self.feedback   = None

        self.class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        # Paths relative to asl-cv-module root
        self._ckpt_path      = ASL_MODULE_PATH / "models" / "checkpoints" / "static_sign_best.pt"
        self._label_map_path = ASL_MODULE_PATH / "data" / "datasets" / "label_map.json"

    def load_model(self):
        """Load all pipeline components. Called once at startup."""
        try:
            logger.info("[Pipeline] Loading asl-cv-module components...")

            self.detector  = ASLDetector(model_complexity=0, draw_landmarks=False)
            self.extractor = ASLFeatureExtractor()
            self.scorer    = ASLScorer(
                references_dir=str(ASL_MODULE_PATH / "data" / "references" / "letters")
            )
            self.feedback  = FeedbackGenerator(
                templates_path=str(ASL_MODULE_PATH / "feedback" / "templates" / "signs.json")
            )

            # Load label map
            if self._label_map_path.exists():
                with open(self._label_map_path) as f:
                    label_map = json.load(f)
                labels = [label_map[str(i)] for i in range(len(label_map))]
            else:
                labels = self.class_labels

            # Load classifier
            self.classifier = StaticSignClassifier(
                num_classes=len(labels),
                labels=labels,
            )
            if not self._ckpt_path.exists():
                logger.warning(f"[Pipeline] Checkpoint not found: {self._ckpt_path}")
                self.model_loaded = False
                return

            self.classifier.load(str(self._ckpt_path))
            self.model_loaded = True
            logger.info("[Pipeline] asl-cv-module loaded successfully.")

        except Exception as e:
            logger.error(f"[Pipeline] Failed to load: {e}")
            self.model_loaded = False

    def predict(self, image: np.ndarray, target_sign: str = "") -> Dict[str, Any]:
        """
        Run the full pipeline on a BGR image.

        Args:
            image:       OpenCV BGR numpy array
            target_sign: Optional — if provided, scores and feedback are included

        Returns dict with:
            class        → detected sign letter e.g. "A"
            confidence   → float 0.0-1.0
            landmarks    → list of [x,y,z] per joint or None
            score        → overall correctness score 0-100 (0 if no target_sign)
            is_correct   → bool
            messages     → list of feedback strings
            joint_colors → dict of joint → "green"/"orange"/"red"
        """
        if not self.model_loaded:
            return self._fallback()

        try:
            # ── Flip so right hand orientation matches training data ──
            image = cv2.flip(image, 1)

            # ── Detect ──
            detection = self.detector.process_frame(image)

            if not detection.is_valid():
                return {
                    "class":        "NONE",
                    "confidence":   0.0,
                    "landmarks":    None,
                    "score":        0.0,
                    "is_correct":   False,
                    "messages":     ["No hand detected"],
                    "joint_colors": {},
                }

            # ── Mirror left hand → right hand ──
            if detection.left_hand_detected and not detection.right_hand_detected:
                detection.right_hand = [
                    Landmark(x=1.0 - lm.x, y=lm.y, z=lm.z)
                    for lm in detection.left_hand
                ]
                detection.right_hand_detected = True
                detection.left_hand = None
                detection.left_hand_detected = False

            # ── Extract features ──
            features = self.extractor.extract(detection)

            # ── Classify ──
            predicted_class, confidence = self.classifier.predict(features.vector)

            # ── Score + feedback (only if target provided) ──
            score        = 0.0
            is_correct   = False
            messages     = []
            joint_colors = {}

            if target_sign:
                score_result = self.scorer.score(features, target_sign.upper())
                fb           = self.feedback.generate(score_result)
                score        = score_result.overall_score
                is_correct   = score_result.is_correct
                messages     = fb.messages
                joint_colors = fb.joint_colors

            # ── Landmarks (raw x,y,z list for frontend skeleton rendering) ──
            landmarks = None
            hand = detection.right_hand or detection.left_hand
            if hand:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand]

            return {
                "class":        predicted_class,
                "confidence":   float(confidence),
                "landmarks":    landmarks,
                "score":        float(score),
                "is_correct":   is_correct,
                "messages":     messages,
                "joint_colors": joint_colors,
            }

        except Exception as e:
            logger.error(f"[Pipeline] Prediction error: {e}")
            return self._fallback()

    def _fallback(self) -> Dict[str, Any]:
        return {
            "class":        "UNKNOWN",
            "confidence":   0.0,
            "landmarks":    None,
            "score":        0.0,
            "is_correct":   False,
            "messages":     ["Model not loaded"],
            "joint_colors": {},
        }

    def __del__(self):
        if self.detector:
            try:
                self.detector.release()
            except Exception:
                pass