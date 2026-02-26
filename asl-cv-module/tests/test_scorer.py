"""
tests/test_scorer.py
"""

import json
import pytest
import numpy as np
from pathlib import Path
from core.extractor import FeatureVector
from core.scorer import ASLScorer


def make_reference_file(tmp_path, sign="A"):
    ref = {
        "sign": sign,
        "right_hand_angles": [120.0] * 15,
        "right_hand_position": [0.1, -0.3, 0.0],
        "right_hand_orientation": [0.0, 0.0, 1.0],
        "right_finger_extensions": [0.1, 0.0, 0.0, 0.0, 0.0],
    }
    path = tmp_path / f"{sign}.json"
    path.write_text(json.dumps(ref))
    return str(tmp_path)


def make_feature_vector(angles=None, position=None, orientation=None, extensions=None):
    fv = FeatureVector(
        right_hand_angles=np.array(angles or [120.0] * 15, dtype=np.float32),
        right_hand_position=np.array(position or [0.1, -0.3, 0.0], dtype=np.float32),
        right_hand_orientation=np.array(orientation or [0.0, 0.0, 1.0], dtype=np.float32),
        right_finger_extensions=np.array(extensions or [0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        vector=np.zeros(178, dtype=np.float32),
        is_valid=True,
    )
    return fv


def test_perfect_score(tmp_path):
    refs_dir = make_reference_file(tmp_path)
    scorer = ASLScorer(references_dir=refs_dir)
    fv = make_feature_vector()
    result = scorer.score(fv, "A")
    assert result.overall_score == pytest.approx(100.0, abs=1.0)
    assert result.is_correct is True


def test_missing_reference_returns_zero(tmp_path):
    scorer = ASLScorer(references_dir=str(tmp_path))
    fv = make_feature_vector()
    result = scorer.score(fv, "Z")
    assert result.overall_score == 0.0


def test_invalid_features_returns_zero(tmp_path):
    refs_dir = make_reference_file(tmp_path)
    scorer = ASLScorer(references_dir=refs_dir)
    fv = FeatureVector(is_valid=False, vector=np.zeros(178, dtype=np.float32))
    result = scorer.score(fv, "A")
    assert result.overall_score == 0.0


def test_wrong_angles_lowers_score(tmp_path):
    refs_dir = make_reference_file(tmp_path)
    scorer = ASLScorer(references_dir=refs_dir, angle_tolerance=20.0)
    # Far from reference (120 degrees → 30 degrees)
    fv = make_feature_vector(angles=[30.0] * 15)
    result = scorer.score(fv, "A")
    assert result.overall_score < 50.0


def test_joint_scores_in_range(tmp_path):
    refs_dir = make_reference_file(tmp_path)
    scorer = ASLScorer(references_dir=refs_dir)
    fv = make_feature_vector()
    result = scorer.score(fv, "A")
    for score in result.joint_scores.values():
        assert 0.0 <= score <= 1.0


def test_save_and_load_reference(tmp_path):
    scorer = ASLScorer(references_dir=str(tmp_path))
    fv = make_feature_vector()
    scorer.save_reference(fv, "X")
    assert (tmp_path / "X.json").exists()
    loaded = scorer._load_reference("X")
    assert loaded["sign"] == "X"
