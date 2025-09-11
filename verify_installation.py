#!/usr/bin/env python
"""
Golf Swing Analysis System v4.2
Installation Verification Script
"""

import sys
import importlib
from typing import List, Tuple
import warnings

# Suppress warnings during import check
warnings.filterwarnings('ignore')

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, 'Not installed'

def main():
    print("=" * 60)
    print("Golf Swing Analysis System v4.2")
    print("Installation Verification")
    print("=" * 60)
    print()
    
    # Core packages to check
    packages = [
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('Flask', 'flask'),
        ('Pillow', 'PIL'),
        ('scikit-learn', 'sklearn'),
        ('plotly', 'plotly'),
        ('seaborn', 'seaborn'),
        ('openpyxl', 'openpyxl'),
        ('xlsxwriter', 'xlsxwriter'),
        ('pytest', 'pytest'),
        ('tqdm', 'tqdm'),
        ('pyserial', 'serial'),
        ('scikit-image', 'skimage'),
    ]
    
    optional_packages = [
        ('PyQt5', 'PyQt5'),
        ('numba', 'numba'),
        ('joblib', 'joblib'),
        ('imageio', 'imageio'),
        ('Flask-SocketIO', 'flask_socketio'),
        ('Flask-CORS', 'flask_cors'),
    ]
    
    print("Checking Core Packages:")
    print("-" * 40)
    
    all_core_installed = True
    for package_name, import_name in packages:
        installed, version = check_package(package_name, import_name)
        status = "[OK]" if installed else "[MISSING]"
        if not installed:
            all_core_installed = False
        print(f"{status:10} {package_name:20} {version:15}")
    
    print()
    print("Checking Optional Packages:")
    print("-" * 40)
    
    for package_name, import_name in optional_packages:
        installed, version = check_package(package_name, import_name)
        status = "[OK]" if installed else "[OPTIONAL]"
        print(f"{status:10} {package_name:20} {version:15}")
    
    print()
    print("=" * 60)
    
    # System information
    print("System Information:")
    print("-" * 40)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check for CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        print(f"CUDA Available: {cuda_available}")
        print(f"CUDA Version: {cuda_version}")
    except ImportError:
        try:
            import cupy
            print(f"CuPy Available: True")
            print(f"CuPy Version: {cupy.__version__}")
        except ImportError:
            print("GPU Acceleration: Not configured (optional)")
    
    print()
    print("=" * 60)
    
    if all_core_installed:
        print("[SUCCESS] All core packages are installed successfully!")
        print("The system is ready for use.")
        
        # Test basic imports
        print()
        print("Testing basic functionality...")
        try:
            import cv2
            import numpy as np
            
            # Create a simple test array
            test_array = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Try basic OpenCV operation
            gray = cv2.cvtColor(test_array, cv2.COLOR_BGR2GRAY)
            
            print("[SUCCESS] Basic OpenCV operations work correctly!")
            
        except Exception as e:
            print(f"[WARNING] Basic operations test failed: {e}")
    else:
        print("[ERROR] Some core packages are missing!")
        print("Please run: pip install -r requirements.txt")
    
    print()
    print("To start the system:")
    print("  - Main analyzer: python scripts/run_main_analyzer.py")
    print("  - Web dashboard: python scripts/run_web_dashboard.py")
    print("  - Kiosk system: python scripts/run_kiosk.py")
    print()
    
    return 0 if all_core_installed else 1

if __name__ == "__main__":
    sys.exit(main())