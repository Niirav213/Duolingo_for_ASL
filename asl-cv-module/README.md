# ASL CV Module 🤟

Computer vision pipeline for real-time American Sign Language detection and feedback. Part of a Duolingo-style ASL learning web application.

## What This Module Does

Takes a webcam frame → returns sign recognition + per-joint correctness score + human-readable feedback.

```
Camera Frame
    ↓
MediaPipe Holistic     → 21+21+33 landmarks
    ↓
Feature Extraction     → joint angles, positions, orientations (178 features)
    ↓
PyTorch Classifier     → "This looks like sign A" (94% confidence)
    ↓
Scoring Engine         → per-joint deviation from reference expert pose
    ↓
Feedback Generator     → "Curl your thumb more" + green/orange/red joint colors
    ↓
FastAPI Response       → JSON sent to frontend
```

## Project Structure

```
asl-cv-module/
├── core/
│   ├── detector.py          # MediaPipe Holistic wrapper
│   ├── extractor.py         # Feature engineering (angles, orientation)
│   └── scorer.py            # Joint-level correctness scoring
├── models/
│   ├── base.py              # Abstract classifier base class
│   ├── static.py            # PyTorch MLP for letters (A-Z)
│   ├── dynamic.py           # PyTorch LSTM for word signs
│   └── checkpoints/         # Saved model weights (gitignored)
├── feedback/
│   ├── generator.py         # Converts scores → feedback messages
│   └── templates/signs.json # Sign-specific correction hints
├── data/
│   ├── references/letters/  # Expert pose JSON per sign
│   ├── datasets/            # Training data (gitignored)
│   └── scripts/
│       ├── collect.py       # Record reference poses from webcam
│       └── preprocess.py    # Prepare training data
├── api/
│   ├── pipeline.py          # Orchestrates the full pipeline
│   ├── router.py            # FastAPI endpoints
│   └── schemas.py           # Request/response JSON contract
├── tests/                   # Pytest test suite
├── API.md                   # Full API documentation for teammates
├── Dockerfile
└── requirements.txt
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/asl-cv-module
cd asl-cv-module
pip install -r requirements.txt

# 2. Record reference poses (do this for each sign)
python data/scripts/collect.py --sign A
python data/scripts/collect.py --sign B
# ... repeat for all signs

# 3. Start the API server
uvicorn api.router:app --reload --port 8000

# 4. Test it's working
curl http://localhost:8000/health
```

## Team Integration

See [API.md](API.md) for the full JSON contract.

**One-line summary for teammates:**
```
POST /analyze  →  { overall_score, messages, joint_colors, emoji, ... }
```

## Development

```bash
# Run tests
pytest tests/ -v

# Train static model (after collecting data)
python -c "
from models.static import StaticSignClassifier
import numpy as np

X_train = np.load('data/datasets/X_train.npy')
y_train = np.load('data/datasets/y_train.npy')
X_val   = np.load('data/datasets/X_val.npy')
y_val   = np.load('data/datasets/y_val.npy')

clf = StaticSignClassifier(num_classes=26)
clf.train_model(X_train, y_train, X_val, y_val, epochs=50)
"
```

## Tech Stack

| Component | Technology |
|---|---|
| Landmark detection | MediaPipe Holistic |
| Static sign model | PyTorch MLP |
| Dynamic sign model | PyTorch BiLSTM + Attention |
| API server | FastAPI + Uvicorn |
| Deployment | Docker + ONNX (planned) |

## Contributing

This module is part of a larger open-source ASL learning platform. Contributions welcome — see open issues for tasks.
