@echo off
REM ASL Platform - Quick Start Script

echo.
echo ==================================================
echo   ASL Learning Platform - Quick Start
echo ==================================================
echo.

REM Check Python
echo [1] Checking Python...
C:/Users/kisho/Desktop/duolingo_asl/.venv/Scripts/python.exe --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Python found
) else (
    echo ✗ Python not found
    pause
    exit /b 1
)

REM Check Node.js
echo [2] Checking Node.js...
"C:\Program Files\nodejs\node.exe" --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Node.js found
) else (
    echo ✗ Node.js not found
    echo Please install from: https://nodejs.org/
    pause
    exit /b 1
)

REM Check Backend
echo [3] Checking Backend...
cd backend
C:/Users/kisho/Desktop/duolingo_asl/.venv/Scripts/python.exe -c "from app.main import app" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Backend ready
) else (
    echo ✗ Backend setup failed
    cd ..
    pause
    exit /b 1
)
cd ..

REM Check ML Service
echo [4] Checking ML Service...
cd ml-service
C:/Users/kisho/Desktop/duolingo_asl/.venv/Scripts/python.exe -c "from app.main import app" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ ML Service ready
) else (
    echo ✗ ML Service setup failed
    cd ..
    pause
    exit /b 1
)
cd ..

REM Check Frontend
echo [5] Checking Frontend...
if exist frontend\node_modules (
    echo ✓ Frontend ready
) else (
    echo ✗ Frontend dependencies not installed
    pause
    exit /b 1
)

echo.
echo ==================================================
echo   ✓ All Systems Ready!
echo ==================================================
echo.
echo NEXT STEPS - Open 3 Command Prompts and run:
echo.
echo Terminal 1 ^(Backend^):
echo cd backend
echo C:/Users/kisho/Desktop/duolingo_asl/.venv/Scripts/python.exe setup_db.py
echo C:/Users/kisho/Desktop/duolingo_asl/.venv/Scripts/python.exe -m uvicorn app.main:app --reload
echo.
echo Terminal 2 ^(ML Service^):
echo cd ml-service
echo C:/Users/kisho/Desktop/duolingo_asl/.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8001
echo.
echo Terminal 3 ^(Frontend^):
echo cd frontend
echo $env:Path += ";C:\Program Files\nodejs"
echo npm run dev
echo.
echo Then visit: http://localhost:5173
echo Login: demo / demo123
echo.
pause
