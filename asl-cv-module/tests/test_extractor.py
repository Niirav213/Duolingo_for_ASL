"""
tests/test_extractor.py
"""

import numpy as np
import pytest
from core.detector import DetectionResult, Landmark
from core.extractor import ASLFeatureExtractor, FeatureVector


def make_hand_landmarks(n=21):
    """Create dummy hand landmarks in a roughly hand-like arrangement."""
    np.random.seed(42)
    return [Landmark(x=np.random.rand(), y=np.random.rand(), z=np.random.rand() * 0.1) for _ in range(n)]


def make_pose_landmarks(n=33):
    return [Landmark(x=np.random.rand(), y=np.random.rand(), z=0.0) for _ in range(n)]


def make_valid_detection():
    detection = DetectionResult()
    detection.right_hand = make_hand_landmarks(21)
    detection.right_hand_detected = True
    detection.pose = make_pose_landmarks(33)
    detection.pose_detected = True
    return detection


def test_invalid_detection_returns_zero_vector():
    extractor = ASLFeatureExtractor()
    detection = DetectionResult()   # no hands
    fv = extractor.extract(detection)
    assert fv.is_valid is False
    assert fv.vector is not None
    assert fv.vector.shape == (178,)
    assert np.all(fv.vector == 0)


def test_valid_detection_returns_correct_shape():
    extractor = ASLFeatureExtractor()
    detection = make_valid_detection()
    fv = extractor.extract(detection)
    assert fv.is_valid is True
    assert fv.vector.shape == (178,)


def test_feature_vector_dtype():
    extractor = ASLFeatureExtractor()
    detection = make_valid_detection()
    fv = extractor.extract(detection)
    assert fv.vector.dtype == np.float32


def test_angles_shape():
    extractor = ASLFeatureExtractor()
    detection = make_valid_detection()
    fv = extractor.extract(detection)
    assert fv.right_hand_angles.shape == (15,)


def test_normalized_landmarks_shape():
    extractor = ASLFeatureExtractor()
    detection = make_valid_detection()
    fv = extractor.extract(detection)
    assert fv.right_hand_normalized.shape == (63,)


def test_orientation_is_unit_vector():
    extractor = ASLFeatureExtractor()
    detection = make_valid_detection()
    fv = extractor.extract(detection)
    norm = np.linalg.norm(fv.right_hand_orientation)
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_finger_extensions_in_range():
    extractor = ASLFeatureExtractor()
    detection = make_valid_detection()
    fv = extractor.extract(detection)
    assert np.all(fv.right_finger_extensions >= 0.0)
    assert np.all(fv.right_finger_extensions <= 1.0)


def test_angle_at_joint_straight():
    """Straight line should give ~180 degrees."""
    a = np.array([0.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, -1.0, 0.0])
    angle = ASLFeatureExtractor._angle_at_joint(a, b, c)
    assert angle == pytest.approx(180.0, abs=1.0)


def test_angle_at_joint_right_angle():
    """Perpendicular vectors should give 90 degrees."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    angle = ASLFeatureExtractor._angle_at_joint(a, b, c)
    assert angle == pytest.approx(90.0, abs=1.0)
