#!/bin/bash

echo ""
echo "==================================="
echo "   ASL Learning Platform Starter"
echo "==================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null
then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null
then
    echo "ERROR: Node.js is not installed"
    exit 1
fi

# Setup Backend
echo ""
echo "[1/3] Setting up Backend..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
python setup_db.py
echo "Backend setup complete!"
cd ..

# Setup ML Service
echo ""
echo "[2/3] Setting up ML Service..."
cd ml-service
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
echo "ML Service setup complete!"
cd ..

# Setup Frontend
echo ""
echo "[3/3] Setting up Frontend..."
cd frontend
npm install -q
echo "Frontend setup complete!"
cd ..

echo ""
echo "==================================="
echo "     Setup Complete!"
echo "==================================="
echo ""
echo "To start the application, run:"
echo "  - Backend: cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
echo "  - ML Service: cd ml-service && source venv/bin/activate && python app/main.py"
echo "  - Frontend: cd frontend && npm run dev"
echo ""
echo "Or use Docker:"
echo "  docker-compose up -d"
echo ""
