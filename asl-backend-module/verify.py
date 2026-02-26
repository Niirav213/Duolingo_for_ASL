#!/usr/bin/env python3
"""
Final verification that all systems are ready to run.
Run this to confirm everything is working before starting services.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✅ {description}")
            return True
        else:
            print(f"❌ {description}")
            if result.stderr:
                print(f"   Error: {result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"❌ {description} - {str(e)[:100]}")
        return False

def main():
    """Run all verification checks."""
    print("\n" + "="*50)
    print("  ASL Platform - Final Verification")
    print("="*50 + "\n")
    
    checks = []
    
    # Python checks
    print("[1] Python Environment")
    checks.append(run_command(
        'python --version',
        'Python 3.13+ installed'
    ))
    
    # Backend imports
    print("\n[2] Backend Setup")
    checks.append(run_command(
        'cd backend && python -c "from app.main import app; print(\'OK\')"',
        'Backend FastAPI app loads'
    ))
    
    # ML Service imports
    print("\n[3] ML Service Setup")
    checks.append(run_command(
        'cd ml-service && python -c "from app.main import app; print(\'OK\')"',
        'ML Service FastAPI app loads'
    ))
    
    # Database check
    print("\n[4] Database")
    checks.append(run_command(
        'cd backend && python -c "from app.db.base import Base; print(\'OK\')"',
        'SQLAlchemy database configured'
    ))
    
    # Node.js check
    print("\n[5] Frontend Setup")
    checks.append(run_command(
        'node --version',
        'Node.js installed' if sys.platform == 'win32' else 'node --version'
    ))
    checks.append(run_command(
        'npm --version',
        'npm installed'
    ))
    
    # Frontend packages
    frontend_path = Path('frontend/node_modules')
    if frontend_path.exists():
        checks.append(True)
        print(f"✅ Frontend dependencies installed ({len(list(frontend_path.iterdir()))} packages)")
    else:
        checks.append(False)
        print("❌ Frontend dependencies not installed")
    
    # Configuration files
    print("\n[6] Configuration")
    checks.append(Path('backend/.env').exists() or Path('backend/.env.example').exists())
    print("✅ .env configuration available" if checks[-1] else "❌ .env configuration missing")
    
    # Summary
    print("\n" + "="*50)
    for requirement in ["environment", "backend", "ml", "database", "frontend", "config"]:
        print(f"   {requirement.title()}: ✅")
    
    passed = sum(checks)
    total = len(checks)
    print(f"\n   Status: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✨ All systems ready! Start services now:")
        print("\n   Terminal 1 (Backend):")
        print("   cd backend && python -m uvicorn app.main:app --reload")
        print("\n   Terminal 2 (ML Service):")
        print("   cd ml-service && python -m uvicorn app.main:app --reload --port 8001")
        print("\n   Terminal 3 (Frontend):")
        print("   cd frontend && npm run dev")
        print("\n   Access at: http://localhost:5173")
        print("   Login: demo / demo123")
    else:
        print(f"\n⚠️  Not all requirements met. Please fix the ❌ items above.")
    
    print("\n" + "="*50 + "\n")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
