#!/usr/bin/env python3
"""
배치 처리 스크립트
이미지 처리, 분석, 검증을 일괄 처리
"""

import sys
import os
import argparse
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.processing.format_converter import convert_bmp_to_jpg
from src.processing.path_manager import rename_korean_folders
from src.processing.image_enhancement.ultra_enhancer import UltraImageEnhancer
from src.processing.image_enhancement.fast_enhancer import FastImageEnhancer
from config.system_config import SystemConfig

def main():
    """배치 처리 실행 함수"""
    parser = argparse.ArgumentParser(description='골프 분석 배치 처리')
    parser.add_argument('--task', type=str, required=True, 
                       choices=['convert', 'rename', 'enhance-ultra', 'enhance-fast', 'all'],
                       help='실행할 작업')
    parser.add_argument('--input', type=str, help='입력 디렉토리')
    parser.add_argument('--output', type=str, help='출력 디렉토리')
    args = parser.parse_args()
    
    try:
        config = SystemConfig()
        
        print(f"🔄 골프 분석 배치 처리 v{config.app_version}")
        print(f"📋 작업: {args.task}")
        
        if args.task == 'convert' or args.task == 'all':
            print("🔄 BMP → JPG 변환 중...")
            converted, skipped, errors = convert_bmp_to_jpg(
                args.input or 'shot-image',
                args.output or 'data/images/shot-image-jpg'
            )
            print(f"✅ 변환 완료: {converted}개, 건너뜀: {skipped}개, 오류: {len(errors)}개")
        
        if args.task == 'rename' or args.task == 'all':
            print("🔄 한국어 폴더명 → 영어 변환 중...")
            renamed_count, errors = rename_korean_folders()
            print(f"✅ 변환 완료: {renamed_count}개 폴더, 오류: {len(errors)}개")
        
        if args.task == 'enhance-ultra' or args.task == 'all':
            print("🔄 Ultra 이미지 향상 중...")
            enhancer = UltraImageEnhancer()
            enhanced_count, error_count = enhancer.process_all_shot_images(
                args.input or 'data/images/shot-image-jpg',
                args.output or 'data/images/shot-image-treated'
            )
            print(f"✅ 향상 완료: {enhanced_count}개, 오류: {error_count}개")
        
        if args.task == 'enhance-fast':
            print("🔄 Fast 이미지 향상 중...")
            enhancer = FastImageEnhancer()
            enhanced_count, error_count = enhancer.process_all_shot_images(
                args.input or 'data/images/shot-image-jpg',
                args.output or 'data/images/shot-image-treated'
            )
            print(f"✅ 향상 완료: {enhanced_count}개, 오류: {error_count}개")
        
        print("🎉 배치 처리 완료!")
        
    except Exception as e:
        print(f"❌ 배치 처리 오류: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())