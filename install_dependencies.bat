@echo off
echo ============================================
echo Golf Swing Analysis System v4.2
echo Dependencies Installation Script
echo ============================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
    echo.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Installing core dependencies...
echo This may take several minutes depending on your internet connection.
echo.

REM Install in groups to avoid timeout and memory issues
echo [1/5] Installing core scientific packages...
pip install numpy==1.24.3 scipy==1.11.4

echo.
echo [2/5] Installing computer vision packages...
pip install opencv-python==4.8.1.78 opencv-contrib-python==4.8.1.78 Pillow==10.1.0 scikit-image==0.22.0

echo.
echo [3/5] Installing data analysis packages...
pip install pandas==2.1.3 matplotlib==3.8.1 seaborn==0.13.0 plotly==5.18.0

echo.
echo [4/5] Installing web and ML packages...
pip install Flask==3.0.0 Flask-SocketIO==5.3.5 Flask-CORS==4.0.0 scikit-learn==1.3.2

echo.
echo [5/5] Installing utilities and tools...
pip install openpyxl==3.1.2 xlsxwriter==3.1.9 pytest==7.4.3 tqdm==4.66.1 pyserial==3.5

echo.
echo ============================================
echo Optional packages (install manually if needed):
echo - PyQt5==5.15.10 (for alternative GUI)
echo - numba==0.58.1 (for JIT compilation)
echo - cupy-cuda12x (for GPU acceleration with CUDA)
echo ============================================

echo.
echo Installation complete!
echo.
echo To activate the environment, run: venv\Scripts\activate
echo To verify installation, run: python -m pytest tests/
echo.
pause