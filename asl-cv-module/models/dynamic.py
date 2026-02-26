"""
models/dynamic.py
-----------------
PyTorch LSTM classifier for DYNAMIC ASL signs (words/phrases with motion).
Dynamic signs require a SEQUENCE of frames to recognize.

Input:  Sequence of feature vectors, shape (seq_len, 178)
Output: (sign_label, confidence)

Default sequence length: 30 frames (~2 seconds at 15fps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from pathlib import Path
from .base import BaseSignClassifier


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────

class DynamicSignNet(nn.Module):
    """
    Bidirectional LSTM for dynamic ASL sign classification.

    Architecture:
        Input sequence (seq_len, 178)
        → BiLSTM(128) × 2 layers
        → Attention pooling
        → Linear(256) → ReLU → Dropout
        → Linear(num_classes) → Softmax
    """

    def __init__(self, input_size: int = 178, hidden_size: int = 128, num_classes: int = 100, num_layers: int = 2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Attention layer — learns which frames matter most
        self.attention = nn.Linear(hidden_size * 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 178)
        lstm_out, _ = self.lstm(x)                        # (batch, seq_len, 256)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (attn_weights * lstm_out).sum(dim=1)   # (batch, 256)

        return self.classifier(context)


# ─────────────────────────────────────────────
# Sequence Buffer
# ─────────────────────────────────────────────

class SequenceBuffer:
    """
    Rolling buffer that collects feature vectors frame by frame.
    When full, provides a sequence ready for the LSTM.
    """

    def __init__(self, seq_len: int = 30, feature_size: int = 178):
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.buffer = deque(maxlen=seq_len)

    def add(self, feature_vector: np.ndarray):
        self.buffer.append(feature_vector.copy())

    def is_ready(self) -> bool:
        return len(self.buffer) == self.seq_len

    def get_sequence(self) -> np.ndarray:
        """Returns sequence of shape (seq_len, feature_size)."""
        return np.array(self.buffer, dtype=np.float32)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# Classifier Wrapper
# ─────────────────────────────────────────────

class DynamicSignClassifier(BaseSignClassifier):
    """
    Wraps DynamicSignNet for real-time inference.
    Manages the rolling sequence buffer internally.
    """

    def __init__(
        self,
        num_classes: int = 100,
        labels: list[str] = None,
        seq_len: int = 30,
        device: str = None,
    ):
        self.labels = labels or [str(i) for i in range(num_classes)]
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DynamicSignNet(
            input_size=178,
            hidden_size=128,
            num_classes=num_classes,
        ).to(self.device)
        self.model.eval()

        self.buffer = SequenceBuffer(seq_len=seq_len)
        print(f"[DynamicSignClassifier] Ready on device: {self.device} | seq_len: {seq_len}")

    def add_frame(self, feature_vector: np.ndarray):
        """Add a single frame's feature vector to the rolling buffer."""
        self.buffer.add(feature_vector)

    def predict(self, feature_vector: np.ndarray = None) -> tuple[str, float]:
        """
        Predict from current buffer contents.
        Optionally pass a feature_vector to add before predicting.

        Returns ("", 0.0) if buffer not full yet.
        """
        if feature_vector is not None:
            self.add_frame(feature_vector)

        if not self.buffer.is_ready():
            return "", 0.0

        self.model.eval()
        with torch.no_grad():
            seq = self.buffer.get_sequence()                           # (30, 178)
            tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)  # (1, 30, 178)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, idx = torch.max(probs, dim=1)

        return self.labels[idx.item()], round(confidence.item(), 4)

    def predict_top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k predictions from current buffer."""
        if not self.buffer.is_ready():
            return []

        self.model.eval()
        with torch.no_grad():
            seq = self.buffer.get_sequence()
            tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(probs, k, dim=1)

        return [
            (self.labels[i.item()], round(p.item(), 4))
            for i, p in zip(top_idxs[0], top_probs[0])
        ]

    def load(self, checkpoint_path: str):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"[DynamicSignClassifier] Checkpoint not found: {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"[DynamicSignClassifier] Loaded weights from {path}")

    def save(self, checkpoint_path: str):
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[DynamicSignClassifier] Saved weights to {path}")

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
        save_path: str = "models/checkpoints/dynamic_sign.pt",
    ):
        """
        Train the LSTM model.

        Args:
            X_train: shape (N, seq_len, 178) — sequences of frames
            y_train: shape (N,) — integer class labels
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device)

        best_val_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(X_t))
            epoch_loss = 0.0

            for i in range(0, len(X_t), batch_size):
                idx = perm[i:i + batch_size]
                batch_x, batch_y = X_t[idx], y_t[idx]

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

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

        print(f"[DynamicSignClassifier] Training complete. Best val acc: {best_val_acc:.3f}")

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return float(np.mean(preds == y))
