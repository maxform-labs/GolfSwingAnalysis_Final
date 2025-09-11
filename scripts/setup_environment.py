#!/usr/bin/env python3
"""
환경 설정 스크립트
초기 설치 및 환경 구성
"""

import sys
import os
import subprocess
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.system_config import SystemConfig

def install_dependencies():
    """의존성 설치"""
    requirements_file = project_root / "requirements.txt"
    
    if requirements_file.exists():
        print("📦 Python 패키지 설치 중...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                      check=True)
        print("✅ Python 패키지 설치 완료")
    else:
        print("⚠️  requirements.txt 파일을 찾을 수 없습니다.")

def create_directories():
    """디렉토리 생성"""
    print("📁 필요한 디렉토리 생성 중...")
    SystemConfig.create_directories()
    print("✅ 디렉토리 생성 완료")

def check_gpu_support():
    """GPU 지원 확인"""
    try:
        import cv2
        print(f"📊 OpenCV 버전: {cv2.__version__}")
        
        # CUDA 지원 확인
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"🚀 CUDA 지원: {cv2.cuda.getCudaEnabledDeviceCount()}개 디바이스")
        else:
            print("⚠️  CUDA 지원 없음 (CPU 모드로 실행)")
            
    except ImportError:
        print("⚠️  OpenCV가 설치되지 않았습니다.")

def create_config_files():
    """설정 파일 생성"""
    print("⚙️  설정 파일 확인 중...")
    
    # .env 파일 생성
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ .env 파일 생성 완료")

def main():
    """환경 설정 메인 함수"""
    print("🏌️  골프 스윙 분석 시스템 v4.2 환경 설정")
    print("=" * 50)
    
    try:
        # 1. 디렉토리 생성
        create_directories()
        
        # 2. 의존성 설치
        install_dependencies()
        
        # 3. GPU 지원 확인
        check_gpu_support()
        
        # 4. 설정 파일 생성
        create_config_files()
        
        print("\n" + "=" * 50)
        print("🎉 환경 설정 완료!")
        print("\n📋 다음 명령어로 시스템을 실행하세요:")
        print("  • 메인 분석기: python scripts/run_main_analyzer.py")
        print("  • 키오스크: python scripts/run_kiosk.py")
        print("  • 웹 대시보드: python scripts/run_web_dashboard.py")
        
    except Exception as e:
        print(f"❌ 환경 설정 오류: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())