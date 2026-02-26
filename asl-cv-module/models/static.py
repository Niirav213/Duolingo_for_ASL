"""
models/static.py
----------------
PyTorch MLP classifier for STATIC ASL signs (letters A-Z, numbers).
Static signs are single-frame — no motion required.

Input:  178-dim feature vector from extractor.py
Output: (sign_label, confidence)

Training:
    python models/static.py  ← runs training loop if called directly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from .base import BaseSignClassifier


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────

class StaticSignNet(nn.Module):
    """
    3-layer MLP for static ASL sign classification.

    Architecture:
        Input(178) → Linear(512) → BN → ReLU → Dropout
                   → Linear(256) → BN → ReLU → Dropout
                   → Linear(128) → BN → ReLU → Dropout
                   → Linear(num_classes) → Softmax
    """

    def __init__(self, input_size: int = 178, num_classes: int = 26):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─────────────────────────────────────────────
# Classifier Wrapper
# ─────────────────────────────────────────────

class StaticSignClassifier(BaseSignClassifier):
    """
    Wraps StaticSignNet for inference.
    Handles loading, device management, and label mapping.
    """

    # Default A-Z labels. Extend for numbers or custom signs.
    DEFAULT_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(
        self,
        num_classes: int = 26,
        labels: list[str] = None,
        device: str = None,
    ):
        self.labels = labels or self.DEFAULT_LABELS
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = StaticSignNet(input_size=178, num_classes=num_classes).to(self.device)
        self.model.eval()

        print(f"[StaticSignClassifier] Ready on device: {self.device}")

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """
        Predict ASL letter from a single feature vector.

        Args:
            feature_vector: shape (178,) numpy array

        Returns:
            (predicted_label, confidence) e.g. ("A", 0.97)
        """
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)  # (1, 178)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, idx = torch.max(probs, dim=1)

        label = self.labels[idx.item()]
        return label, round(confidence.item(), 4)

    def predict_top_k(self, feature_vector: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k predictions with confidence scores."""
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(probs, k, dim=1)

        return [
            (self.labels[i.item()], round(p.item(), 4))
            for i, p in zip(top_idxs[0], top_probs[0])
        ]

    def load(self, checkpoint_path: str):
        """Load trained weights from .pt file."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"[StaticSignClassifier] Checkpoint not found: {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"[StaticSignClassifier] Loaded weights from {path}")

    def save(self, checkpoint_path: str):
        """Save model weights to .pt file."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[StaticSignClassifier] Saved weights to {path}")

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        save_path: str = "models/checkpoints/static_sign.pt",
    ):
        """
        Train the model from numpy arrays.

        Args:
            X_train: shape (N, 178)
            y_train: shape (N,) — integer class labels
            X_val, y_val: optional validation set
            epochs: number of training epochs
            lr: learning rate
            save_path: where to save the best model
        """
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device)

        best_val_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            # Mini-batch training
            perm = torch.randperm(len(X_t))
            epoch_loss = 0.0

            for i in range(0, len(X_t), batch_size):
                idx = perm[i:i + batch_size]
                batch_x, batch_y = X_t[idx], y_t[idx]

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            # Validation
            if X_val is not None and y_val is not None:
                val_acc = self._evaluate(X_val, y_val)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save(save_path)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.3f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

        print(f"[StaticSignClassifier] Training complete. Best val acc: {best_val_acc:.3f}")

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return float(np.mean(preds == y))
