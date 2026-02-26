# 🚀 ASL Platform - Startup Guide

## System Requirements

- Python 3.11+
- Node.js 18+
- npm or yarn
- Git

## Quick Start (Windows)

### Option 1: Using Batch Script (Recommended)
```cmd
setup.bat
```

This will automatically:
1. Create Python virtual environments
2. Install all dependencies
3. Initialize the database with demo user

### Option 2: Manual Setup

**1. Backend Setup**
```cmd
cd backend
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python setup_db.py
uvicorn app.main:app --reload
```

**2. ML Service Setup** (New Terminal)
```cmd
cd ml-service
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python app/main.py
```

**3. Frontend Setup** (New Terminal)
```cmd
cd frontend
npm install
npm run dev
```

## Quick Start (Mac/Linux)

### Option 1: Using Shell Script (Recommended)
```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

**1. Backend Setup**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup_db.py
uvicorn app.main:app --reload
```

**2. ML Service Setup** (New Terminal)
```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

**3. Frontend Setup** (New Terminal)
```bash
cd frontend
npm install
npm run dev
```

## Docker Setup

**Prerequisites:**
- Docker
- Docker Compose

**Start All Services:**
```bash
docker-compose up -d
```

**Check Services:**
```bash
docker-compose ps
```

**View Logs:**
```bash
docker-compose logs -f backend
docker-compose logs -f ml-service
docker-compose logs -f frontend
```

**Stop Services:**
```bash
docker-compose down
```

## Access Points

After starting the services:

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:5173 | Web Application |
| Backend | http://localhost:8000 | API Server |
| API Docs | http://localhost:8000/docs | Swagger Documentation |
| ML Service | http://localhost:8001 | Gesture Detection API |
| Health Check | http://localhost:8000/health | Backend Health Status |

## Default Credentials

Use these to log in for the first time:

```
Username: demo
Password: demo123
```

**Create New Account:**
- Click "Sign up here" on the login page
- Fill in the registration form
- Note: Username and email must be unique

## Troubleshooting

### Backend Won't Start

**Error: "Address already in use"**
```bash
# Find and kill process using port 8000
lsof -i :8000
kill -9 <PID>
```

**Error: "Module not found"**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Error: "Database locked"**
```bash
# Delete the database file and restart
rm test.db
python setup_db.py
```

### ML Service Won't Start

**Error: "No module named 'mediapipe'"**
```bash
# This is a known issue - reinstall with specific version
pip install --upgrade mediapipe
```

**Model not found warning:**
- This is normal! The system will use mock predictions
- The model file will be created when you train it

### Frontend Won't Start

**Error: "Port 5173 already in use"**
```bash
# Use a different port
npm run dev -- --port 5174
```

**Error: "Cannot find module"**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Database Connection Issues

**Check database:**
```bash
# For SQLite
sqlite3 test.db ".tables"

# For PostgreSQL
psql -U user -d asl_db -c "\dt"
```

**Reset database:**
```bash
rm test.db
python setup_db.py
```

## Common Tasks

### Create Admin User
```bash
# See setup_db.py and modify to create admin
cd backend
python setup_db.py
```

### View API Documentation
Open: http://localhost:8000/docs

### Train ML Model
```bash
cd ml-service/training
python collect_data.py
python train_static.py
```

### Run Tests
```bash
cd backend
pytest
```

### Check Backend Health
```bash
curl http://localhost:8000/health
```

## Environment Configuration

Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
- `DEBUG=False` for production
- `SECRET_KEY` - Change to a long random string
- `DATABASE_URL` - Configure database connection

## Development Tips

1. **Frontend Changes:** Auto-reload enabled at `http://localhost:5173`
2. **Backend Changes:** Use `--reload` flag with uvicorn for auto-reload
3. **ML Service Changes:** Restart the service manually
4. **Database:** Check migrations with `alembic history`
5. **Logs:** View detailed logs in terminal output

## Production Deployment

**Before deploying:**

1. Set `DEBUG=False` in `.env`
2. Generate secure `SECRET_KEY`:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
3. Use PostgreSQL instead of SQLite
4. Enable HTTPS
5. Configure CORS properly
6. Set up monitoring and logging
7. Run security checks

**Using Docker:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Support & Debugging

- **API Issues:** Check `http://localhost:8000/docs`
- **Frontend Issues:** Check browser console (F12)
- **Database Issues:** Check SQLite file or PostgreSQL logs
- **ML Issues:** Check console output for model loading
- **WebSocket Issues:** Check browser console for connection errors

## Next Steps

1. ✅ Complete on
