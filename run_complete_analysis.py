#!/usr/bin/env python3
"""
골프공 3D 분석 마스터 스크립트
전체 분석 프로세스를 순차적으로 실행
"""
import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """헤더 출력"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")

def run_script(script_name, description):
    """Python 스크립트 실행"""
    print_header(description)
    print(f"Running: {script_name}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✓ {script_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_name}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        return False

def check_dependencies():
    """필요한 라이브러리 확인"""
    print_header("Checking Dependencies")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✓ All dependencies are installed!")
    return True

def check_data_directories():
    """데이터 디렉토리 확인"""
    print_header("Checking Data Directories")
    
    required_dirs = [
        "data2/Calibration_image_1025",
        "data2/driver"
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
            
            # 파일 개수 확인
            if "Calibration" in dir_path:
                bmp_files = list(Path(dir_path).glob("*.bmp"))
                print(f"  - Found {len(bmp_files)} calibration images")
            else:
                shot_dirs = [d for d in os.listdir(dir_path) 
                           if os.path.isdir(os.path.join(dir_path, d)) and d.isdigit()]
                print(f"  - Found {len(shot_dirs)} shot directories")
        else:
            print(f"✗ {dir_path} does NOT exist")
            all_exist = False
    
    if not all_exist:
        print("\n[ERROR] Required data directories are missing!")
        return False
    
    print("\n✓ All data directories are ready!")
    return True

def main():
    """메인 함수"""
    print_header("GOLF BALL 3D ANALYSIS MASTER SCRIPT")
    
    print("This script will run the complete golf ball 3D analysis pipeline:")
    print("  1. Precise Vertical Stereo Calibration")
    print("  2. Golf Ball 3D Physics Analysis")
    print("  3. Analysis Report Generation")
    print()
    
    # 1. 의존성 확인
    if not check_dependencies():
        print("\n[ERROR] Please install missing dependencies before continuing.")
        return
    
    # 2. 데이터 디렉토리 확인
    if not check_data_directories():
        print("\n[ERROR] Please ensure all required data directories exist.")
        return
    
    # 3. 캘리브레이션 실행
    success = run_script(
        "precise_vertical_stereo_calibration.py",
        "STEP 1: PRECISE VERTICAL STEREO CALIBRATION"
    )
    
    if not success:
        print("\n[ERROR] Calibration failed. Stopping.")
        return
    
    # 캘리브레이션 결과 확인
    if not os.path.exists("precise_vertical_stereo_calibration.json"):
        print("\n[ERROR] Calibration result file not found!")
        return
    
    print("\n✓ Calibration results saved!")
    
    # 4. 3D 분석 실행
    success = run_script(
        "golf_ball_3d_physics_analyzer.py",
        "STEP 2: GOLF BALL 3D PHYSICS ANALYSIS"
    )
    
    if not success:
        print("\n[ERROR] 3D analysis failed. Stopping.")
        return
    
    # 분석 결과 확인
    if not os.path.exists("golf_ball_3d_analysis_results.json"):
        print("\n[ERROR] Analysis result file not found!")
        return
    
    print("\n✓ Analysis results saved!")
    
    # 5. 리포트 생성 실행
    success = run_script(
        "generate_analysis_report.py",
        "STEP 3: ANALYSIS REPORT GENERATION"
    )
    
    if not success:
        print("\n[ERROR] Report generation failed.")
        return
    
    # 최종 결과 확인
    print_header("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    
    print("Generated files:")
    print("  ✓ precise_vertical_stereo_calibration.json - Calibration parameters")
    print("  ✓ golf_ball_3d_analysis_results.json - 3D tracking results")
    print("  ✓ analysis_report.md - Comprehensive analysis report")
    print("  ✓ analysis_plots/ - Visualization plots")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Review analysis_report.md for detailed results")
    print("  2. Check analysis_plots/ for visualizations")
    print("  3. Compare with real measurement data")
    print("  4. Fine-tune calibration if needed")
    print("=" * 80)

if __name__ == "__main__":
    main()
