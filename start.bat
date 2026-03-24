@echo off
title ASL Platform Launcher
echo ============================================
echo        ASL Learning Platform Launcher
echo ============================================
echo.

echo [1/3] Starting Backend (port 8000)...
start "ASL Backend" cmd /k "cd /d c:\Users\kisho\Duolingo_for_ASL\asl-backend-module\backend && C:/Users/kisho/Desktop/duolingo_asl/.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo [2/3] Starting CV Module (port 8002)...
start "ASL CV Module" cmd /k "cd /d c:\Users\kisho\Duolingo_for_ASL\asl-cv-module && C:/Users/kisho/Desktop/duolingo_asl/.venv/Scripts/python.exe -m uvicorn api.router:app --reload --port 8002"

timeout /t 5 /nobreak >nul

echo [3/3] Starting Frontend (port 5173)...
start "ASL Frontend" cmd /k "cd /d c:\Users\kisho\Duolingo_for_ASL\ASL_Frontend && npm run dev"

echo.
echo ============================================
echo  All services starting in separate windows!
echo ============================================
echo.
echo   Frontend:   http://localhost:5173
echo   Backend:    http://localhost:8000/docs
echo   CV Module:  http://localhost:8002/docs
echo.
echo  Wait ~10 seconds for CV module to load,
echo  then open http://localhost:5173 in browser.
echo.
pause
