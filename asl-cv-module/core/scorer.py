"""
core/scorer.py
--------------
Compares user's feature vector against reference expert poses
and produces a per-joint correctness score + overall score.

Pipeline position:
    extractor.py → [scorer.py] → feedback/generator.py
"""

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from core.extractor import FeatureVector


@dataclass
class ScoreResult:
    """
    Output of ASLScorer.score().
    Passed to feedback/generator.py
    """
    overall_score: float = 0.0                      # 0-100
    joint_scores: dict[str, float] = field(default_factory=dict)   # per joint 0-1
    angle_deviations: dict[str, float] = field(default_factory=dict)
    position_score: float = 0.0                     # hand location correctness
    orientation_score: float = 0.0                  # palm direction correctness
    extension_scores: dict[str, float] = field(default_factory=dict)
    is_correct: bool = False                        # True if overall_score >= threshold
    target_sign: str = ""

    def __repr__(self):
        return f"ScoreResult(sign='{self.target_sign}', score={self.overall_score:.1f}, correct={self.is_correct})"


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Maps angle index → human readable joint name
ANGLE_NAMES = [
    "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip",
    "middle_mcp", "middle_pip",
    "ring_mcp", "ring_pip",
    "pinky_mcp", "pinky_pip",
    "splay_index", "splay_middle", "splay_ring", "splay_full", "splay_triangle",
]


class ASLScorer:
    """
    Scores user's hand pose against a reference expert pose.

    Loads reference JSON files from data/references/letters/
    and computes deviation between user features and reference.
    """

    def __init__(
        self,
        references_dir: str = "data/references/letters",
        angle_tolerance: float = 20.0,
        position_tolerance: float = 0.3,
        correct_threshold: float = 75.0,
    ):
        """
        Args:
            references_dir: Path to folder containing sign JSON files.
            angle_tolerance: Max angle deviation (degrees) for full score.
            position_tolerance: Max position deviation for full score.
            correct_threshold: Overall score % to count as correct.
        """
        self.references_dir = Path(references_dir)
        self.angle_tolerance = angle_tolerance
        self.position_tolerance = position_tolerance
        self.correct_threshold = correct_threshold
        self._cache: dict[str, dict] = {}

    def score(self, features: FeatureVector, target_sign: str) -> ScoreResult:
        """
        Score user feature vector against a target sign.

        Args:
            features: FeatureVector from ASLFeatureExtractor.extract()
            target_sign: Sign label e.g. 'A', 'B', 'HELLO'

        Returns:
            ScoreResult with per-joint scores and overall score
        """
        result = ScoreResult(target_sign=target_sign)

        if not features.is_valid:
            return result

        reference = self._load_reference(target_sign)
        if reference is None:
            return result

        ref_angles = np.array(reference.get("right_hand_angles", [0.0] * 15), dtype=np.float32)
        ref_position = np.array(reference.get("right_hand_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        ref_orientation = np.array(reference.get("right_hand_orientation", [0.0, 0.0, 1.0]), dtype=np.float32)
        ref_extensions = np.array(reference.get("right_finger_extensions", [0.5] * 5), dtype=np.float32)

        # ── Score joint angles ──
        if features.right_hand_angles is not None:
            for i, name in enumerate(ANGLE_NAMES):
                deviation = abs(float(features.right_hand_angles[i]) - float(ref_angles[i]))
                score = max(0.0, 1.0 - deviation / self.angle_tolerance)
                result.joint_scores[name] = round(score, 3)
                result.angle_deviations[name] = round(deviation, 2)

        # ── Score hand position ──
        if features.right_hand_position is not None:
            pos_deviation = np.linalg.norm(features.right_hand_position - ref_position)
            result.position_score = round(
                max(0.0, 1.0 - pos_deviation / self.position_tolerance), 3
            )

        # ── Score palm orientation ──
        if features.right_hand_orientation is not None:
            dot = np.dot(features.right_hand_orientation, ref_orientation)
            result.orientation_score = round(float(np.clip((dot + 1) / 2, 0.0, 1.0)), 3)

        # ── Score finger extensions ──
        if features.right_finger_extensions is not None:
            for i, name in enumerate(FINGER_NAMES):
                deviation = abs(float(features.right_finger_extensions[i]) - float(ref_extensions[i]))
                result.extension_scores[name] = round(max(0.0, 1.0 - deviation), 3)

        # ── Overall score (weighted average) ──
        angle_avg = np.mean(list(result.joint_scores.values())) if result.joint_scores else 0.0
        ext_avg = np.mean(list(result.extension_scores.values())) if result.extension_scores else 0.0

        result.overall_score = round(float(
            0.45 * angle_avg +
            0.20 * result.position_score +
            0.20 * result.orientation_score +
            0.15 * ext_avg
        ) * 100, 1)

        result.is_correct = result.overall_score >= self.correct_threshold
        return result

    def _load_reference(self, sign: str) -> Optional[dict]:
        """Load reference JSON for a sign, with in-memory caching."""
        if sign in self._cache:
            return self._cache[sign]

        path = self.references_dir / f"{sign.upper()}.json"
        if not path.exists():
            print(f"[ASLScorer] No reference found for sign '{sign}' at {path}")
            return None

        with open(path) as f:
            data = json.load(f)

        self._cache[sign] = data
        return data

    def save_reference(self, features: FeatureVector, sign: str):
        """
        Save a feature vector as the reference for a sign.
        Use this when recording expert poses.
        """
        self.references_dir.mkdir(parents=True, exist_ok=True)
        path = self.references_dir / f"{sign.upper()}.json"

        data = {
            "sign": sign.upper(),
            "right_hand_angles": features.right_hand_angles.tolist() if features.right_hand_angles is not None else [],
            "right_hand_position": features.right_hand_position.tolist() if features.right_hand_position is not None else [0, 0, 0],
            "right_hand_orientation": features.right_hand_orientation.tolist() if features.right_hand_orientation is not None else [0, 0, 1],
            "right_finger_extensions": features.right_finger_extensions.tolist() if features.right_finger_extensions is not None else [],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self._cache[sign] = data
        print(f"[ASLScorer] Saved reference for sign '{sign}' → {path}")
