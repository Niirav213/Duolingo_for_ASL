"""MediaPipe + ONNX inference pipeline for gesture detection."""
import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
from typing import Dict, Any
import os


class MediaPipePipeline:
    """Gesture detection pipeline using MediaPipe and ONNX."""

    def __init__(self, model_path: str = "app/models/gesture_model.onnx"):
        """Initialize the pipeline."""
        self.model_path = model_path
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.session = None
        self.model_loaded = False

        # ASL alphabet (26 letters + space)
        self.class_labels = [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z", "SPACE"
        ]

    def load_model(self):
        """Load ONNX model and initialize MediaPipe hands."""
        try:
            # Initialize MediaPipe hands detector
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Load ONNX model if it exists
            if os.path.exists(self.model_path):
                self.session = ort.InferenceSession(self.model_path)
                self.model_loaded = True
                print(f"✓ Model loaded from {self.model_path}")
            else:
                print(f"⚠ Model not found at {self.model_path} - using mock predictions")
                self.model_loaded = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False

    def extract_landmarks(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract hand landmarks from image using MediaPipe."""
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        landmarks = None
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y, landmark.z])
                landmarks.append(hand_data)

        return {
            "landmarks": landmarks,
            "has_hands": results.multi_hand_landmarks is not None,
            "num_hands": len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        }

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict gesture from image."""
        if image is None:
            return {
                "class": "UNKNOWN",
                "confidence": 0.0,
                "landmarks": None
            }

        # Resize image to expected size
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)

        # Extract landmarks
        landmark_data = self.extract_landmarks(image)

        # Run inference
        if self.session:
            try:
                input_name = self.session.get_inputs()[0].name
                output_name = self.session.get_outputs()[0].name
                probabilities = self.session.run([output_name], {input_name: image_batch})[0]

                # Get prediction
                predicted_idx = np.argmax(probabilities[0])
                confidence = float(probabilities[0][predicted_idx])
                predicted_class = self.class_labels[predicted_idx]

                return {
                    "class": predicted_class,
                    "confidence": confidence,
                    "landmarks": landmark_data.get("landmarks"),
                    "all_probabilities": probabilities[0].tolist()
                }
            except Exception as e:
                print(f"Inference error: {e}")
                return {
                    "class": "ERROR",
                    "confidence": 0.0,
                    "landmarks": None
                }
        else:
            # Return mock prediction if no model loaded
            return {
                "class": self.class_labels[np.random.randint(0, len(self.class_labels))],
                "confidence": 0.5 + 0.5 * np.random.random(),
                "landmarks": landmark_data.get("landmarks")
            }

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Resize to 224x224 (common ImageNet size)
        image = cv2.resize(image, (224, 224))

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def __del__(self):
        """Cleanup resources."""
        if self.hands:
            self.hands.close()
