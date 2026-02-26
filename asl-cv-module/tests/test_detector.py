"""
tests/test_detector.py
"""

import numpy as np
import pytest
from core.detector import ASLDetector, DetectionResult, Landmark


def make_dummy_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_landmark_to_array():
    lm = Landmark(x=0.1, y=0.2, z=0.3)
    arr = lm.to_array()
    assert arr.shape == (3,)
    assert arr[0] == pytest.approx(0.1)


def test_detection_result_is_valid_no_hands():
    result = DetectionResult()
    assert result.is_valid() is False


def test_detection_result_is_valid_with_hand():
    result = DetectionResult(right_hand_detected=True)
    assert result.is_valid() is True


def test_detection_result_dominant_hand_right_preferred():
    lm = [Landmark(0, 0, 0)] * 21
    result = DetectionResult(
        right_hand=lm, right_hand_detected=True,
        left_hand=lm,  left_hand_detected=True,
    )
    assert result.dominant_hand() is result.right_hand


def test_empty_frame_raises():
    detector = ASLDetector(draw_landmarks=False)
    with pytest.raises(ValueError):
        detector.process_frame(np.array([]))
    detector.release()


def test_process_frame_returns_detection_result():
    detector = ASLDetector(draw_landmarks=False, model_complexity=0)
    frame = make_dummy_frame()
    result = detector.process_frame(frame)
    assert isinstance(result, DetectionResult)
    assert result.annotated_frame is not None
    detector.release()


def test_context_manager():
    with ASLDetector(draw_landmarks=False, model_complexity=0) as detector:
        frame = make_dummy_frame()
        result = detector.process_frame(frame)
        assert result is not None
