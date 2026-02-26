"""
ASL Platform - Quick Start Guide

## Local Development Setup

### 1. Backend Setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app/main.py

### 2. ML Service Setup
cd ml-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app/main.py

### 3. Frontend Setup
cd frontend
npm install
npm run dev

## Services Running:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- ML Service: http://localhost:8001
- API Docs: http://localhost:8000/docs

## Docker Compose Setup

docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down

## Database Setup

# Create migration
cd backend
alembic revision --autogenerate -m "Initial migration"

# Apply migration
alembic upgrade head

## Testing

cd backend
pytest
pytest --cov=app tests/

## Training ML Model

cd ml-service/training
python collect_data.py
python train_static.py

## Default Credentials

Username: testuser
Password: password123

## Useful Commands

# Backend shell
python -c "from app.main import app; print(app)"

# Database shell (SQLite)
sqlite3 test.db

# Redis CLI
redis-cli

# Check API health
curl http://localhost:8000/health

## Troubleshooting

### Port Already in Use
lsof -i :8000  # Check what's using port
kill -9 <PID>  # Kill process

### Database Issues
rm test.db  # Delete SQLite database
python app/main.py  # Restart - creates new DB

### WebSocket Connection Failed
- Check auth token is valid
- Verify ML service is running
- Check CORS settings

### GPU not detected for ML
- Install NVIDIA drivers
- Install CUDA toolkit
- Update requirements with GPU packages

## Performance Tips

- Use Redis for caching
- Enable async database pooling
- Compress images before upload
- Use CDN for static assets

## Production Checklist

- [ ] Change SECRET_KEY
- [ ] Set DEBUG=False
- [ ] Use PostgreSQL (not SQLite)
- [ ] Enable HTTPS
- [ ] Configure CORS properly
- [ ] Set up error logging
- [ ] Configure backups
- [ ] Load test the application
- [ ] Set up monitoring
- [ ] Configure CI/CD
"""
