# ASL CV Module — API Reference

This document defines the JSON contract between the CV module and the rest of the team.

---

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### `POST /analyze`

Analyze a single webcam frame for ASL sign detection and correctness scoring.

**Request**
```json
{
  "frame_base64": "<base64-encoded JPEG or PNG>",
  "target_sign": "A",
  "mode": "static",
  "include_landmarks": false
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `frame_base64` | string | ✅ | Base64-encoded image. Accepts `data:image/jpeg;base64,...` prefix. |
| `target_sign` | string | ✅ | The ASL sign the user is attempting (e.g. `"A"`, `"B"`, `"HELLO"`) |
| `mode` | string | ❌ | `"static"` (default) for letters, `"dynamic"` for word signs |
| `include_landmarks` | bool | ❌ | If `true`, returns raw joint coordinates (for custom rendering) |

**Response**
```json
{
  "hand_detected": true,
  "detected_sign": "A",
  "confidence": 0.94,

  "overall_score": 82.5,
  "is_correct": true,

  "joint_scores": {
    "thumb_mcp": 0.91,
    "thumb_ip": 0.45,
    "index_mcp": 0.88,
    "index_pip": 0.92,
    "middle_mcp": 0.87,
    "middle_pip": 0.90,
    "ring_mcp": 0.88,
    "ring_pip": 0.91,
    "pinky_mcp": 0.85,
    "pinky_pip": 0.90,
    "position": 0.78,
    "orientation": 0.83
  },

  "messages": [
    "Curl your thumb more at the tip.",
    "Adjust your hand position relative to your face/body."
  ],
  "praise": "Nice work! Almost perfect.",
  "emoji": "✅",

  "joint_colors": {
    "thumb_mcp": "green",
    "thumb_ip": "red",
    "index_mcp": "green",
    "index_pip": "green",
    "position": "orange",
    "orientation": "green"
  },

  "landmarks": null
}
```

| Field | Type | Description |
|---|---|---|
| `hand_detected` | bool | Whether any hand was found in the frame |
| `detected_sign` | string | What the model classified the sign as |
| `confidence` | float | Model confidence 0.0–1.0 |
| `overall_score` | float | Correctness score 0–100 |
| `is_correct` | bool | `true` if `overall_score >= 75` |
| `joint_scores` | object | Per-joint score 0.0–1.0 |
| `messages` | string[] | Up to 3 correction instructions |
| `praise` | string | Positive reinforcement message |
| `emoji` | string | UI emoji: 🌟 ✅ 👍 🤔 ❌ |
| `joint_colors` | object | `"green"` / `"orange"` / `"red"` per joint for skeleton overlay |
| `landmarks` | object/null | Raw x,y,z per joint (only if `include_landmarks: true`) |

---

### `GET /health`

Check if the service is running and models are loaded.

**Response**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

---

### `GET /signs`

List all supported ASL signs.

**Response**
```json
{
  "static": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
             "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
             "W", "X", "Y", "Z"],
  "dynamic": []
}
```

---

## Frontend Integration Example

```javascript
// Capture frame from webcam canvas
const canvas = document.getElementById('webcam-canvas');
const base64Frame = canvas.toDataURL('image/jpeg', 0.8);

const response = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    frame_base64: base64Frame,
    target_sign: currentSign,   // e.g. "A"
    mode: 'static',
  })
});

const result = await response.json();

// Update UI
scoreDisplay.textContent = result.overall_score;
emojiDisplay.textContent = result.emoji;
feedbackList.innerHTML = result.messages.map(m => `<li>${m}</li>`).join('');

// Color-code skeleton joints
for (const [joint, color] of Object.entries(result.joint_colors)) {
  renderJoint(joint, color);  // your skeleton renderer
}
```

---

## Running the Service

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn api.router:app --reload --port 8000

# Or with Docker
docker build -t asl-cv .
docker run -p 8000:8000 asl-cv
```
