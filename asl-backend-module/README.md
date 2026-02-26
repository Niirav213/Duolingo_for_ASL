# ASL Learning Platform - Complete Project Structure

## Overview
A full-stack web application for learning American Sign Language (ASL) featuring:
- Real-time gesture recognition using MediaPipe and ONNX
- Interactive lessons and quizzes
- User progress tracking with XP and streaks
- WebSocket support for live gesture detection
- Responsive React + Tailwind frontend
- FastAPI backend with async database operations
- Docker containerization for easy deployment

## Project Structure

### Backend (`/backend`)
FastAPI application with async support

**Core Components:**
- `app/main.py` - FastAPI app initialization with middleware
- `app/core/` - Configuration, security, and dependency injection
- `app/db/` - Database session and models
- `app/models/` - SQLAlchemy ORM models (User, Progress, Sessions)
- `app/api/` - REST endpoints (auth, game, WebSocket)
- `app/services/` - Business logic services
- `app/schemas/` - Pydantic validation models

**Key Endpoints:**
```
POST   /api/v1/auth/register          → Register new user
POST   /api/v1/auth/login             → Login user
POST   /api/v1/auth/refresh           → Refresh access token
WS     /api/v1/ws/gesture/{token}     → Real-time gesture detection
POST   /api/v1/game/session/start     → Start game session
POST   /api/v1/game/session/{id}/end  → End game session
GET    /api/v1/game/stats             → Get user statistics
GET    /api/v1/game/streak            → Get user streak info
```

**Database Models:**
- User: User authentication and profiles
- UserProgress: Lesson completion and XP tracking
- Streak: Daily streak tracking
- GameSession: Individual game session records

### ML Service (`/ml-service`)
Isolated FastAPI service for gesture detection

**Components:**
- `app/main.py` - FastAPI endpoints
- `app/pipeline.py` - MediaPipe + ONNX inference pipeline
- `training/collect_data.py` - Data collection script
- `training/train_static.py` - MLP model training
- `models/` - ONNX model files

**Prediction Endpoint:**
```
POST /predict
{
  "image_data": "base64_encoded_image"
}
→ {
  "predicted_class": "A",
  "confidence": 0.95,
  "landmarks": [[x,y,z], ...]
}
```

### Frontend (`/frontend`)
React + Vite + Tailwind CSS SPA

**Pages:**
- Home - Dashboard with stats, lessons, and leaderboard
- Lesson - Interactive gesture learning with camera
- Quiz - Test knowledge with gesture recognition

**Components:**
- GestureCamera - Real-time video with WebSocket integration
- XPBar - Level and XP progress visualization

**State Management:** Zustand stores for auth and game state

### Configuration Files

**Docker Compose** (`docker-compose.yml`)
Services:
- PostgreSQL (database)
- Redis (cache/broker)
- Backend (FastAPI)
- ML Service (gesture detection)
- Frontend (Nginx)
- Celery Worker (async tasks)

**Environment** (`.env`)
Database, security, and service configuration

## Setup & Installation

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+ (for frontend dev)

### Quick Start with Docker

```bash
# 1. Clone repository
git clone <repo-url>
cd duolingo_asl

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start all services
docker-compose up -d

# 4. Access services
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- ML Service: http://localhost:8001
```

### Local Development

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**ML Service:**
```bash
cd ml-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Database Migrations

```bash
cd backend

# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1
```

## Training the ML Model

```bash
cd ml-service/training

# 1. Collect Data
python collect_data.py
# Follow prompts to capture gesture samples

# 2. Train Model
python train_static.py
# Model saves to ../app/models/gesture_model.pt
```

## API Documentation

Full interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Authentication

Uses JWT tokens with:
- Access Token: 30 minutes expiry
- Refresh Token: 7 days expiry
- Password hashing: bcrypt

Request headers:
```
Authorization: Bearer <access_token>
```

## Real-time Gesture Detection

WebSocket flow:
1. Client connects: `ws://localhost:8000/api/v1/ws/gesture/{token}`
2. Send frame as base64 image
3. Receive prediction with confidence score
4. Auto-closes after inactivity

Example:
```javascript
const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/gesture/${token}`);
ws.send(JSON.stringify({
  image_data: base64Image,
  lesson_id: 1,
  expected_sign: "A"
}));
ws.onmessage = (e) => {
  const { detected_sign, confidence, correct } = JSON.parse(e.data);
};
```

## Performance Optimization

- **Async Database**: SQLAlchemy async with connection pooling
- **Redis Caching**: User stats and leaderboard caching
- **ONNX Runtime**: Optimized ML inference
- **WebSocket**: Efficient real-time communication
- **Image Compression**: Frontend compression before upload

## Deployment

### Production Build

```bash
# Frontend
npm run build
# Output: dist/

# Backend
# Ensure DEBUG=False in .env
# Replace SQLite with PostgreSQL
# Set secure SECRET_KEY

# Docker Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Environment Variables (Production)
```
DEBUG=False
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/db
SECRET_KEY=<generate-secure-key>
ALLOWED_HOSTS=yourdomain.com
CORS_ORIGINS=https://yourdomain.com
```

## Technologies Used

**Backend:**
- FastAPI - Modern web framework
- SQLAlchemy - ORM
- AsyncIO - Async operations
- Pydantic - Data validation
- JWT - Authentication
- WebSockets - Real-time communication

**ML:**
- MediaPipe - Hand pose detection
- ONNX Runtime - Model inference
- PyTorch - Training framework
- OpenCV - Image processing
- scikit-learn - ML utilities

**Frontend:**
- React 18 - UI framework
- Vite - Build tool
- Tailwind CSS - Styling
- Zustand - State management
- Axios - HTTP client
- React Router - Navigation

**DevOps:**
- Docker & Docker Compose
- Nginx - Reverse proxy
- PostgreSQL - Database
- Redis - Cache

## Contributing

1. Create feature branch: `git checkout -b feature/name`
2. Commit changes: `git commit -m 'Add feature'`
3. Push to branch: `git push origin feature/name`
4. Submit pull request

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: [project-issues]
- Documentation: [docs-link]
- Email: support@aslplatform.com

---

**Happy Learning! 🤟**
