"""Training script for ASL gesture recognition MLP model."""
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mediapipe as mp
from pathlib import Path


class AslGestureModel(nn.Module):
    """Neural network for ASL gesture recognition."""

    def __init__(self, input_size: int = 63, num_classes: int = 27):
        """Initialize model.
        Input size: 63 (21 hand landmarks * 3 coordinates)
        """
        super(AslGestureModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass."""
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


class DataPreprocessor:
    """Preprocess raw gesture images to landmarks."""

    def __init__(self):
        """Initialize preprocessor."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def extract_landmarks(self, image_path: str) -> np.ndarray:
        """Extract hand landmarks from image."""
        image = cv2.imread(image_path)
        if image is None:
            return None

        image = cv2.resize(image, (224, 224))
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

            return np.array(landmarks, dtype=np.float32)

        return None

    def prepare_dataset(self, data_dir: str) -> tuple:
        """Prepare dataset from raw images."""
        X, y = [], []
        class_labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        class_map = {label: idx for idx, label in enumerate(class_labels)}

        print("Extracting landmarks...")
        for gesture_dir in class_labels:
            gesture_path = os.path.join(data_dir, gesture_dir)
            class_idx = class_map[gesture_dir]

            image_files = [f for f in os.listdir(gesture_path) if f.endswith(('.jpg', '.png'))]
            print(f"  Processing {gesture_dir}: {len(image_files)} images")

            for image_file in image_files:
                image_path = os.path.join(gesture_path, image_file)
                landmarks = self.extract_landmarks(image_path)

                if landmarks is not None:
                    X.append(landmarks)
                    y.append(class_idx)

        return np.array(X), np.array(y), class_labels


def train_model(
    data_dir: str = "../data/raw",
    output_model: str = "../app/models/gesture_model.pt",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """Train the ASL gesture recognition model."""
    print("=" * 50)
    print("ASL Gesture Recognition Model Training")
    print("=" * 50)

    # Prepare data
    preprocessor = DataPreprocessor()
    X, y, class_labels = preprocessor.prepare_dataset(data_dir)

    print(f"\nDataset shape: {X.shape}")
    print(f"Classes: {class_labels}")
    print(f"Samples per class: {np.bincount(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AslGestureModel(input_size=63, num_classes=len(class_labels)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nTraining on {device}...")
    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = 100 * correct / total
        avg_loss = train_loss / len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save model
            os.makedirs(os.path.dirname(output_model) or ".", exist_ok=True)
            torch.save(model.state_dict(), output_model)
            print(f"  ✓ Model saved with accuracy: {accuracy:.2f}%")

    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {output_model}")


if __name__ == "__main__":
    train_model()
