"""
models/base.py
--------------
Abstract base class for all ASL sign classifiers.
All models must implement predict() and load().
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseSignClassifier(ABC):

    @abstractmethod
    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """
        Predict the sign from a feature vector.

        Args:
            feature_vector: numpy array of shape (178,) from extractor

        Returns:
            (predicted_sign, confidence) e.g. ("A", 0.94)
        """
        pass

    @abstractmethod
    def load(self, checkpoint_path: str):
        """Load model weights from a checkpoint file."""
        pass

    @abstractmethod
    def save(self, checkpoint_path: str):
        """Save model weights to a checkpoint file."""
        pass
