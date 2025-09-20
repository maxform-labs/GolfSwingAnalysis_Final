#!/usr/bin/env python3
"""
v7.0 딤플 분석기 실행 스크립트
BMP 직접 처리 및 혁신적 딤플 분석
"""

import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.algorithms.spin_analysis.dimple_analyzer_v7 import DimpleAnalyzerV7, process_directory_v7

def main():
    """메인 실행 함수"""
    
    # 입출력 경로 설정
    input_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1")
    output_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-v7-enhanced/driver/no_marker_ball-1")
    
    print("="*80)
    print("딤플 분석 시스템 v7.0 - BMP 직접 처리")
    print("개발팀: maxform")
    print("="*80)
    
    # 디렉토리 처리
    result = process_directory_v7(input_dir, output_dir)
    
    # 결과 출력
    if 'error' not in result:
        print(f"\n처리 완료!")
        print(f"- 총 파일: {result['total_files']}")
        print(f"- 처리된 배치: {result['batches_processed']}")
        print(f"- 출력 경로: {result['output_dir']}")
    else:
        print(f"\n오류 발생: {result['error']}")
    
    return result

if __name__ == "__main__":
    main()