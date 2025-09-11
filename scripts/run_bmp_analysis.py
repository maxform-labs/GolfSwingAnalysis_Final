#!/usr/bin/env python3
"""
BMP Direct Analysis Script
BMP 파일을 직접 분석하여 딤플을 검출하는 스크립트
"""

import sys
import os
import argparse
from pathlib import Path
import logging

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.spin_analysis.no_dimple_spin_final import FinalNoDimpleSpinAnalyzer

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bmp_analysis.log'),
            logging.StreamHandler()
        ]
    )

def find_bmp_files(base_path: str, pattern: str = "*.bmp") -> list:
    """
    BMP 파일 찾기
    
    Args:
        base_path: 검색할 기본 경로
        pattern: 파일 패턴
        
    Returns:
        BMP 파일 경로 리스트
    """
    bmp_files = []
    base_path = Path(base_path)
    
    if base_path.exists():
        for bmp_file in base_path.rglob(pattern):
            bmp_files.append(str(bmp_file))
    
    return sorted(bmp_files)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='BMP 딤플 분석 시스템')
    parser.add_argument('--input', type=str, required=True, 
                       help='BMP 파일이 있는 폴더 경로')
    parser.add_argument('--club', type=str, default='driver', 
                       choices=['driver', '7iron'], help='클럽 타입')
    parser.add_argument('--max-files', type=int, default=10, 
                       help='최대 처리할 파일 수')
    parser.add_argument('--output', type=str, default='bmp_analysis_result.json',
                       help='결과 저장 파일명')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    parser.add_argument('--enhanced', action='store_true', help='개선된 가시성 모드 (JPG 처리 방식 적용)')
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # BMP 파일 찾기
        logger.info(f"BMP 파일 검색 중: {args.input}")
        bmp_files = find_bmp_files(args.input)
        
        if not bmp_files:
            logger.error(f"BMP 파일을 찾을 수 없습니다: {args.input}")
            return 1
        
        # 파일 수 제한
        if len(bmp_files) > args.max_files:
            bmp_files = bmp_files[:args.max_files]
            logger.info(f"처리할 파일 수를 {args.max_files}개로 제한했습니다.")
        
        logger.info(f"발견된 BMP 파일: {len(bmp_files)}개")
        
        # 분석기 초기화
        analyzer = FinalNoDimpleSpinAnalyzer(enable_bmp_analysis=True)
        
        # 분석 실행
        if args.enhanced:
            logger.info("개선된 BMP 딤플 분석 시작...")
            result = analyzer.analyze_bmp_with_improved_visibility(bmp_files, args.club)
        else:
            logger.info("BMP 딤플 분석 시작...")
            result = analyzer.analyze_bmp_sequence(bmp_files, args.club)
        
        # 결과 출력
        print("\n" + "="*60)
        print("🏌️ BMP 딤플 분석 결과")
        print("="*60)
        print(f"클럽 타입: {args.club}")
        print(f"처리된 파일: {result.get('bmp_files_processed', 0)}개")
        print(f"딤플 분석 사용: {'예' if result.get('dimple_analysis_used', False) else '아니오'}")
        print(f"개선된 가시성: {'예' if result.get('improved_visibility', False) else '아니오'}")
        print(f"Total Spin: {result.get('total_spin', 0)} rpm")
        print(f"Backspin: {result.get('backspin', 0)} rpm")
        print(f"Sidespin: {result.get('sidespin', 0)} rpm")
        print(f"신뢰도: {result.get('confidence', 0):.1%}")
        print(f"분석 방법: {result.get('method', 'unknown')}")
        print("="*60)
        
        # 결과 저장
        import json
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"결과 저장 완료: {output_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
