@echo off
REM ASL Platform - Windows Startup Script

echo.
echo ===================================
echo    ASL Learning Platform Starter
echo ===================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    pause
    exit /b 1
)

REM Setup Backend
echo.
echo [1/3] Setting up Backend...
cd backend
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -q -r requirements.txt
python setup_db.py
echo Backend setup complete!
cd ..

REM Setup ML Service
echo.
echo [2/3] Setting up ML Service...
cd ml-service
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -q -r requirements.txt
echo ML Service setup complete!
cd ..

REM Setup Frontend
echo.
echo [3/3] Setting up Frontend...
cd frontend
call npm install -q
echo Frontend setup complete!
cd ..

echo.
echo ===================================
echo     Setup Complete!
echo ===================================
echo.
echo To start the application, run:
echo   - Backend: cd backend ^& python -m venv venv ^& venv\Scripts\activate.bat ^& uvicorn app.main:app --reload
echo   - ML Service: cd ml-service ^& python -m venv venv ^& venv\Scripts\activate.bat ^& python app/main.py
echo   - Frontend: cd frontend ^& npm run dev
echo.
echo Or use Docker:
echo   docker-compose up -d
echo.
pause
