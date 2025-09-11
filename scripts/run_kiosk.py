#!/usr/bin/env python3
"""
키오스크 시스템 실행 스크립트
터치스크린 인터페이스 기반 골프 분석 키오스크
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.interfaces.kiosk_system import KioskSystem
from config.system_config import SystemConfig

def main():
    """키오스크 실행 함수"""
    try:
        # 설정 로드
        config = SystemConfig()
        
        print(f"🖥️  골프 스윙 분석 키오스크 v{config.app_version} 시작")
        print(f"📱 터치스크린 인터페이스 활성화")
        
        # 키오스크 초기화 및 실행
        kiosk = KioskSystem(config)
        kiosk.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  키오스크를 안전하게 종료합니다...")
    except Exception as e:
        print(f"❌ 키오스크 오류: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())