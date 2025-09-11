#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMP Direct Processing Script for Golf Swing Analysis
BMP 파일을 직접 처리하여 딤플 검출 기반 스핀 분석을 수행하는 스크립트
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 직접 import로 의존성 문제 해결
try:
    from src.utils.bmp_loader import create_bmp_loader
    from src.processing.image_enhancement.dimple_enhancer import DimpleEnhancer
except ImportError:
    # 직접 경로로 모듈 로드
    import importlib.util
    
    # BMP Loader 직접 로드
    bmp_loader_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'utils', 'bmp_loader.py')
    spec = importlib.util.spec_from_file_location("bmp_loader", bmp_loader_path)
    bmp_loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bmp_loader_module)
    create_bmp_loader = bmp_loader_module.create_bmp_loader
    
    # Dimple Enhancer 직접 로드
    dimple_enhancer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'processing', 'image_enhancement', 'dimple_enhancer.py')
    spec = importlib.util.spec_from_file_location("dimple_enhancer", dimple_enhancer_path)
    dimple_enhancer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dimple_enhancer_module)
    DimpleEnhancer = dimple_enhancer_module.DimpleEnhancer

class BMPDirectProcessor:
    """BMP 파일 직접 처리를 위한 통합 시스템"""
    
    def __init__(self, enable_cache: bool = True, debug: bool = False):
        """
        BMP 직접 처리기 초기화
        
        Args:
            enable_cache: 메모리 캐시 사용 여부
            debug: 디버그 모드
        """
        self.bmp_loader = create_bmp_loader(enable_cache)
        self.dimple_enhancer = DimpleEnhancer()
        # Spin analyzer는 현재 사용하지 않음 (의존성 문제)
        # self.spin_analyzer = FinalNoDimpleSpinAnalyzer(enable_bmp_analysis=True)
        self.debug = debug
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 통계 정보
        self.stats = {
            'processed_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_dimples_detected': 0,
            'processing_time': 0.0
        }
    
    def detect_ball_region(self, image: np.ndarray) -> Optional[tuple]:
        """
        골프공 영역 검출 (1440x300 어두운 이미지 최적화)
        
        Args:
            image: 입력 이미지
            
        Returns:
            (center_x, center_y, radius) 또는 None
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 매우 어두운 이미지를 위한 전처리
        # 1. 히스토그램 균등화로 대비 향상
        enhanced = cv2.equalizeHist(gray)
        
        # 2. 가우시안 블러 (더 강한 블러)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
        
        # 빠른 처리를 위한 최적화된 파라미터 (상위 2개만)
        param_sets = [
            # 1440x300 해상도에 최적화된 파라미터
            # (dp, minDist, param1, param2, minR, maxR)
            (2, 20, 40, 20, 15, 50),      # 매우 민감하게 (어두운 이미지용)
            (1, 30, 50, 25, 15, 60),      # 중간 민감도 (백업)
        ]
        
        best_circle = None
        best_score = 0
        
        for dp, minDist, param1, param2, minR, maxR in param_sets:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                param1=param1, param2=param2, minRadius=minR, maxRadius=maxR
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # 이미지 경계 내부인지 확인
                    if (r < x < gray.shape[1] - r and r < y < gray.shape[0] - r):
                        # 골프공 영역의 특징 점수 계산
                        roi = gray[max(0, y-r):min(gray.shape[0], y+r), 
                                  max(0, x-r):min(gray.shape[1], x+r)]
                        
                        if roi.size > 0:
                            # 원형도와 대비를 고려한 점수
                            contrast_score = np.std(roi) / 255.0
                            size_score = min(1.0, r / 30.0)  # 적절한 크기
                            position_score = 1.0 - abs(y - gray.shape[0]/2) / (gray.shape[0]/2)  # 중앙 선호
                            
                            total_score = contrast_score * 0.5 + size_score * 0.3 + position_score * 0.2
                            
                            if total_score > best_score:
                                best_score = total_score
                                best_circle = (x, y, r)
                                
                                if self.debug:
                                    self.logger.debug(f"골프공 후보: 중심=({x},{y}), 반지름={r}, 점수={total_score:.3f}")
        
        if best_circle and best_score > 0.1:  # 최소 점수 기준
            if self.debug:
                self.logger.debug(f"최종 선택: 중심=({best_circle[0]},{best_circle[1]}), 반지름={best_circle[2]}, 점수={best_score:.3f}")
            return best_circle
        
        return None
    
    def process_single_image(self, image_path: str) -> Dict:
        """
        단일 이미지 처리
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            처리 결과 딕셔너리
        """
        start_time = datetime.now()
        result = {
            'file': image_path,
            'success': False,
            'ball_detected': False,
            'dimples_detected': 0,
            'dimple_quality': 0.0,
            'enhanced_image': None,
            'error': None
        }
        
        try:
            # BMP 파일 로드 (선명도 유지)
            image = self.bmp_loader.load_bmp_safely(image_path)
            if image is None:
                result['error'] = 'Failed to load BMP file'
                return result
            
            # 골프공 영역 검출
            ball_region = self.detect_ball_region(image)
            if ball_region is None:
                result['error'] = 'Ball region not detected'
                return result
            
            result['ball_detected'] = True
            center_x, center_y, radius = ball_region
            
            # 딤플 검출 및 향상
            dimples = self.dimple_enhancer.detect_dimple_patterns(
                image, (center_x, center_y), radius
            )
            
            result['dimples_detected'] = len(dimples)
            if dimples:
                result['dimple_quality'] = np.mean([d['quality'] for d in dimples])
            
            # 이미지 향상 (딤플 가시성)
            enhanced = self.dimple_enhancer.enhance_dimple_visibility(image)
            result['enhanced_image'] = enhanced
            
            # 통계 업데이트
            self.stats['total_dimples_detected'] += len(dimples)
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"이미지 처리 실패 {image_path}: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats['processing_time'] += processing_time
        
        return result
    
    def process_directory(self, input_dir: str, output_dir: str, max_files: Optional[int] = None) -> Dict:
        """
        디렉토리의 모든 BMP 파일 처리
        
        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리
            max_files: 최대 처리 파일 수 (None이면 전체)
            
        Returns:
            처리 결과 통계
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 출력 디렉토리 생성
        output_path.mkdir(parents=True, exist_ok=True)
        
        # BMP 파일 목록 수집
        bmp_files = list(input_path.rglob("*.bmp"))
        if max_files:
            bmp_files = bmp_files[:max_files]
        
        self.logger.info(f"총 {len(bmp_files)}개의 BMP 파일을 처리합니다...")
        
        processed_results = []
        
        for i, bmp_file in enumerate(bmp_files):
            self.logger.info(f"처리 중... [{i+1}/{len(bmp_files)}] {bmp_file.name}")
            
            # 파일 처리
            result = self.process_single_image(str(bmp_file))
            processed_results.append(result)
            
            self.stats['processed_files'] += 1
            if result['success']:
                self.stats['successful_files'] += 1
                
                # 향상된 이미지 저장
                if result['enhanced_image'] is not None:
                    # 원본 구조 유지하며 출력 경로 생성
                    relative_path = bmp_file.relative_to(input_path)
                    output_file = output_path / relative_path.with_suffix('.bmp')
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # BMP 형식으로 저장 (선명도 유지)
                    cv2.imwrite(str(output_file), result['enhanced_image'])
                    
                    if self.debug:
                        self.logger.debug(f"저장완료: {output_file}")
                        self.logger.debug(f"딤플 검출: {result['dimples_detected']}개, "
                                        f"품질: {result['dimple_quality']:.3f}")
            else:
                self.stats['failed_files'] += 1
                self.logger.warning(f"처리 실패: {bmp_file.name} - {result['error']}")
        
        # 최종 통계
        total_time = self.stats['processing_time']
        avg_time_per_file = total_time / len(bmp_files) if bmp_files else 0
        success_rate = (self.stats['successful_files'] / len(bmp_files) * 100) if bmp_files else 0
        
        summary = {
            'total_files': len(bmp_files),
            'processed_files': self.stats['processed_files'],
            'successful_files': self.stats['successful_files'],
            'failed_files': self.stats['failed_files'],
            'success_rate': success_rate,
            'total_dimples_detected': self.stats['total_dimples_detected'],
            'avg_dimples_per_image': self.stats['total_dimples_detected'] / self.stats['successful_files'] if self.stats['successful_files'] > 0 else 0,
            'total_processing_time': total_time,
            'avg_time_per_file': avg_time_per_file,
            'memory_usage': self.bmp_loader.get_cache_info()
        }
        
        # 결과 로깅
        self.logger.info("=== BMP 직접 처리 완료 ===")
        self.logger.info(f"총 파일: {summary['total_files']}")
        self.logger.info(f"성공: {summary['successful_files']}")
        self.logger.info(f"실패: {summary['failed_files']}")
        self.logger.info(f"성공률: {summary['success_rate']:.1f}%")
        self.logger.info(f"총 딤플 검출: {summary['total_dimples_detected']}개")
        self.logger.info(f"평균 딤플/이미지: {summary['avg_dimples_per_image']:.1f}개")
        self.logger.info(f"총 처리시간: {summary['total_processing_time']:.1f}초")
        self.logger.info(f"평균 처리시간: {summary['avg_time_per_file']:.3f}초/파일")
        
        return summary

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='BMP 직접 처리 스크립트')
    parser.add_argument('--input', '-i', required=True,
                        help='입력 디렉토리 (BMP 파일 포함)')
    parser.add_argument('--output', '-o', required=True,
                        help='출력 디렉토리')
    parser.add_argument('--max-files', type=int,
                        help='최대 처리 파일 수 (디버그용)')
    parser.add_argument('--no-cache', action='store_true',
                        help='메모리 캐시 비활성화')
    parser.add_argument('--debug', action='store_true',
                        help='디버그 모드 활성화')
    
    args = parser.parse_args()
    
    # 경로 검증
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"오류: 입력 디렉토리가 존재하지 않습니다: {input_path}")
        sys.exit(1)
    
    # BMP 직접 처리기 초기화
    processor = BMPDirectProcessor(
        enable_cache=not args.no_cache,
        debug=args.debug
    )
    
    # 처리 실행
    try:
        summary = processor.process_directory(
            str(input_path),
            args.output,
            args.max_files
        )
        
        print("\n=== 최종 결과 ===")
        print(f"처리 완료: {summary['successful_files']}/{summary['total_files']} "
              f"({summary['success_rate']:.1f}%)")
        print(f"딤플 검출: {summary['total_dimples_detected']}개")
        print(f"처리 시간: {summary['total_processing_time']:.1f}초")
        
    except KeyboardInterrupt:
        print("\n처리가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()