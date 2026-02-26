"""Diagnostic script to check system setup and dependencies."""
import sys
import subprocess
import platform

def check_python():
    """Check Python version."""
    print("\n[1] Python Version Check")
    print(f"   Version: {sys.version}")
    print(f"   Platform: {platform.platform()}")
    
    if sys.version_info >= (3, 11):
        print("   ✓ Python 3.11+ detected")
        return True
    else:
        print("   ✗ Python 3.11+ required")
        return False

def check_node():
    """Check Node.js installation."""
    print("\n[2] Node.js Check")
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        version = result.stdout.strip()
        print(f"   Version: {version}")
        print("   ✓ Node.js installed")
        return True
    except FileNotFoundError:
        print("   ✗ Node.js not found")
        return False

def check_backend_deps():
    """Check backend dependencies."""
    print("\n[3] Backend Dependencies")
    try:
        import fastapi
        print("   ✓ FastAPI installed")
    except ImportError:
        print("   ✗ FastAPI not installed")
        return False
    
    try:
        import sqlalchemy
        print("   ✓ SQLAlchemy installed")
    except ImportError:
        print("   ✗ SQLAlchemy not installed")
        return False
    
    try:
        import pydantic
        print("   ✓ Pydantic installed")
    except ImportError:
        print("   ✗ Pydantic not installed")
        return False
    
    return True

def check_ml_deps():
    """Check ML dependencies."""
    print("\n[4] ML Dependencies")
    deps = {
        "opencv-python": "cv2",
        "mediapipe": "mediapipe",
        "numpy": "numpy",
    }
    
    all_ok = True
    for package, module in deps.items():
        try:
            __import__(module)
            print(f"   ✓ {package} installed")
        except ImportError:
            print(f"   ✗ {package} not installed")
            all_ok = False
    
    return all_ok

def check_frontend_deps():
    """Check frontend dependencies."""
    print("\n[5] Frontend Dependencies")
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        version = result.stdout.strip()
        print(f"   npm version: {version}")
        print("   ✓ npm installed")
        return True
    except FileNotFoundError:
        print("   ✗ npm not found")
        return False

def check_ports():
    """Check if ports are available."""
    print("\n[6] Port Availability")
    import socket
    
    ports = {
        8000: "Backend",
        8001: "ML Service",
        5173: "Frontend",
        5432: "PostgreSQL",
        6379: "Redis",
    }
    
    for port, service in ports.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        if result == 0:
            print(f"   ⚠ Port {port} ({service}) is in use")
        else:
            print(f"   ✓ Port {port} ({service}) available")
        sock.close()

def main():
    """Run all checks."""
    print("\n" + "="*50)
    print("  ASL Platform - System Diagnostics")
    print("="*50)
    
    results = {
        "Python": check_python(),
        "Node.js": check_node(),
        "Backend": check_backend_deps(),
        "ML": check_ml_deps(),
        "Frontend": check_frontend_deps(),
    }
    
    check_ports()
    
    print("\n" + "="*50)
    print("  Diagnostic Summary")
    print("="*50)
    
    for name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"   {icon} {name}: {'OK' if status else 'ISSUE'}")
    
    print("\n" + "="*50)
    
    if all(results.values()):
        print("  ✓ All checks passed! Ready to start.")
        print("\nRun: python setup_db.py")
        print("     Then start services as documented.")
    else:
        print("  ✗ Some issues found. Please install missing dependencies.")
        print("\nUse:")
        print("  - setup.bat (Windows)")
        print("  - setup.sh (Mac/Linux)")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
