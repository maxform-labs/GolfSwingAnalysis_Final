#!/usr/bin/env python3
"""
웹 대시보드 실행 스크립트
Flask 기반 웹 인터페이스 (포트 5000)
"""

import sys
import os
import argparse
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.interfaces.web_dashboard import WebDashboard
from config.system_config import SystemConfig

def main():
    """웹 대시보드 실행 함수"""
    parser = argparse.ArgumentParser(description='골프 스윙 분석 웹 대시보드')
    parser.add_argument('--port', type=int, default=5000, help='서버 포트')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='서버 호스트')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    args = parser.parse_args()
    
    try:
        # 설정 로드
        config = SystemConfig()
        
        print(f"🌐 골프 스윙 분석 웹 대시보드 v{config.app_version} 시작")
        print(f"🔗 서버 주소: http://{args.host}:{args.port}")
        print(f"🎯 820fps 실시간 모니터링 대시보드")
        
        # 웹 대시보드 초기화 및 실행
        dashboard = WebDashboard(config)
        dashboard.run(host=args.host, port=args.port, debug=args.debug)
        
    except KeyboardInterrupt:
        print("\n⏹️  웹 서버를 안전하게 종료합니다...")
    except Exception as e:
        print(f"❌ 웹 서버 오류: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())