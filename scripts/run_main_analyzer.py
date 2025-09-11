#!/usr/bin/env python3
"""
메인 분석기 실행 스크립트
골프 스윙 분석 시스템 v4.2 - 820fps 실시간 분석
"""

import sys
import os
import argparse
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.main_analyzer import GolfSwingAnalyzer, SystemConfig
from config.system_config import SystemConfig as GlobalConfig

def setup_environment():
    """환경 설정"""
    # 필요한 디렉토리 생성
    GlobalConfig.create_directories()
    
    # 로깅 설정 (필요시)
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='골프 스윙 분석 시스템 v4.2')
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    parser.add_argument('--gpu', action='store_true', help='GPU 가속 사용', default=True)
    args = parser.parse_args()
    
    # 환경 설정
    setup_environment()
    
    try:
        # 설정 로드
        if args.config:
            # 사용자 설정 파일 로드 (구현 필요)
            config = GlobalConfig()
        else:
            config = GlobalConfig()
        
        # GPU 설정
        config.gpu_acceleration = args.gpu
        
        print(f"🏌️  골프 스윙 분석 시스템 v{config.app_version} 시작")
        print(f"⚡ 카메라: {config.camera_fps}fps, {config.resolution[0]}x{config.resolution[1]}")
        print(f"🎯 목표 정확도: {config.accuracy_target*100}%")
        print(f"🚀 GPU 가속: {'활성화' if config.gpu_acceleration else '비활성화'}")
        
        # 분석기 초기화 및 실행
        analyzer = GolfSwingAnalyzer(config)
        analyzer.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  시스템을 안전하게 종료합니다...")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())