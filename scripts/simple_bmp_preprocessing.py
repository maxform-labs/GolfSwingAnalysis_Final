#!/usr/bin/env python3
"""
Simple BMP Batch Preprocessing Script
개선된 알고리즘으로 모든 BMP 파일을 전처리하여 저장
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import argparse
from datetime import datetime

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bmp_preprocessing.log'),
            logging.StreamHandler()
        ]
    )

def find_bmp_files(base_path: str) -> List[str]:
    """
    모든 BMP 파일 찾기
    
    Args:
        base_path: 검색할 기본 경로
        
    Returns:
        BMP 파일 경로 리스트
    """
    bmp_files = []
    base_path = Path(base_path)
    
    if base_path.exists():
        for bmp_file in base_path.rglob("*.bmp"):
            bmp_files.append(str(bmp_file))
    
    return sorted(bmp_files)

def apply_jpg_processing_methods(image: np.ndarray) -> np.ndarray:
    """
    기존 JPG 처리 방식을 BMP에 적용
    
    Args:
        image: 입력 이미지
        
    Returns:
        JPG 처리 방식이 적용된 이미지
    """
    # 1. 감마 보정 (기존 JPG 처리의 핵심)
    gamma = 1.7  # JPG 처리에서 사용된 감마 값
    gamma_corrected = np.power(image / 255.0, 1/gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # 2. 대비 향상
    alpha = 1.2  # 대비 증가
    beta = 30    # 밝기 조정
    enhanced = cv2.convertScaleAbs(gamma_corrected, alpha=alpha, beta=beta)
    
    # 3. 노이즈 제거
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 4. 딤플 검출용 엣지 강화
    sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.uint8(sobel / sobel.max() * 255)
    
    # 5. 원본과 엣지 강화 이미지 결합
    final = cv2.addWeighted(enhanced, 0.7, sobel, 0.3, 0)
    
    return final

def enhance_dimple_visibility_advanced(image: np.ndarray) -> np.ndarray:
    """
    BMP 딤플 가시성 고급 향상 (기존 JPG 처리 방식 적용)
    
    Args:
        image: 입력 이미지
        
    Returns:
        딤플이 강조된 이미지
    """
    # 1. 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 2. 히스토그램 분석으로 밝기 조정
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # 3. 밝은 이미지 감지 및 감마 보정
    mean_brightness = np.mean(gray)
    if mean_brightness > 150:  # 너무 밝은 이미지
        # 감마 보정 (어둡게)
        gamma = 0.7
        gray = np.power(gray / 255.0, gamma) * 255.0
        gray = gray.astype(np.uint8)
    
    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 5. 언샤프 마스킹 (선명도 향상)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    # 6. 딤플 특화 필터 적용
    dimple_kernel = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  2,  2,  2, -1],
        [-1,  2,  8,  2, -1],
        [-1,  2,  2,  2, -1],
        [-1, -1, -1, -1, -1]
    ])
    enhanced = cv2.filter2D(enhanced, -1, dimple_kernel)
    
    # 7. 이진화를 통한 딤플 강조
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 8. 모폴로지 연산으로 딤플 정제
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return enhanced

def preprocess_bmp_image(image: np.ndarray) -> np.ndarray:
    """
    BMP 이미지 전처리 (개선된 알고리즘 적용)
    
    Args:
        image: 원본 BMP 이미지
        
    Returns:
        전처리된 이미지
    """
    # 1. JPG 처리 방식 적용
    jpg_style_processed = apply_jpg_processing_methods(image)
    
    # 2. 고급 딤플 강조
    enhanced = enhance_dimple_visibility_advanced(jpg_style_processed)
    
    return enhanced

def load_bmp_safely(file_path: str) -> np.ndarray:
    """
    BMP 파일을 안전하게 로드
    
    Args:
        file_path: BMP 파일 경로
        
    Returns:
        로드된 이미지 배열 또는 None
    """
    try:
        # OpenCV로 BMP 파일 로드
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            return None
        
        # 색상 모드 확인 및 변환
        if len(image.shape) == 3:
            # BGR 이미지
            return image
        elif len(image.shape) == 2:
            # 그레이스케일 이미지
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            return None
            
    except Exception as e:
        logging.error(f"BMP 로드 실패 {file_path}: {e}")
        return None

def save_processed_image(processed_image: np.ndarray, output_path: str) -> bool:
    """
    전처리된 이미지 저장
    
    Args:
        processed_image: 전처리된 이미지
        output_path: 출력 경로
        
    Returns:
        저장 성공 여부
    """
    try:
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # BMP 형식으로 저장 (원본 품질 유지)
        success = cv2.imwrite(output_path, processed_image)
        
        if success:
            # 파일 크기 확인
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
            else:
                return False
        else:
            return False
            
    except Exception as e:
        logging.error(f"이미지 저장 실패 {output_path}: {e}")
        return False

def process_bmp_batch(input_dir: str, output_dir: str, 
                     max_files: int = None) -> Dict[str, int]:
    """
    BMP 파일 배치 전처리
    
    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        max_files: 최대 처리할 파일 수 (None이면 모든 파일)
        
    Returns:
        처리 결과 통계
    """
    logger = logging.getLogger(__name__)
    
    # BMP 파일 찾기
    logger.info(f"BMP 파일 검색 중: {input_dir}")
    bmp_files = find_bmp_files(input_dir)
    
    if not bmp_files:
        logger.error(f"BMP 파일을 찾을 수 없습니다: {input_dir}")
        return {"error": "BMP 파일을 찾을 수 없습니다"}
    
    # 파일 수 제한
    if max_files and len(bmp_files) > max_files:
        bmp_files = bmp_files[:max_files]
        logger.info(f"처리할 파일 수를 {max_files}개로 제한했습니다.")
    
    logger.info(f"발견된 BMP 파일: {len(bmp_files)}개")
    
    # 통계 변수
    stats = {
        "total_files": len(bmp_files),
        "processed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": []
    }
    
    # 배치 처리 시작
    logger.info("BMP 배치 전처리 시작...")
    start_time = datetime.now()
    
    for i, bmp_file in enumerate(bmp_files, 1):
        try:
            logger.info(f"처리 중 ({i}/{len(bmp_files)}): {os.path.basename(bmp_file)}")
            
            # 1. BMP 파일 로드
            image = load_bmp_safely(bmp_file)
            if image is None:
                logger.warning(f"이미지 로드 실패: {bmp_file}")
                stats["failed"] += 1
                stats["errors"].append(f"로드 실패: {bmp_file}")
                continue
            
            # 2. 이미지 전처리
            processed_image = preprocess_bmp_image(image)
            
            # 3. 출력 경로 생성
            relative_path = os.path.relpath(bmp_file, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            
            # 4. 이미지 저장
            if save_processed_image(processed_image, output_path):
                stats["processed"] += 1
                logger.info(f"저장 완료: {output_path}")
            else:
                stats["failed"] += 1
                stats["errors"].append(f"저장 실패: {bmp_file}")
                logger.error(f"저장 실패: {bmp_file}")
            
            # 5. 진행률 표시
            if i % 10 == 0:
                progress = (i / len(bmp_files)) * 100
                logger.info(f"진행률: {progress:.1f}% ({i}/{len(bmp_files)})")
                
        except Exception as e:
            logger.error(f"처리 중 오류 발생 {bmp_file}: {e}")
            stats["failed"] += 1
            stats["errors"].append(f"처리 오류: {bmp_file} - {str(e)}")
    
    # 처리 완료
    end_time = datetime.now()
    processing_time = end_time - start_time
    
    logger.info("="*60)
    logger.info("BMP 배치 전처리 완료")
    logger.info("="*60)
    logger.info(f"총 파일 수: {stats['total_files']}")
    logger.info(f"처리 성공: {stats['processed']}")
    logger.info(f"처리 실패: {stats['failed']}")
    logger.info(f"처리 시간: {processing_time}")
    logger.info(f"평균 처리 시간: {processing_time.total_seconds() / len(bmp_files):.2f}초/파일")
    
    if stats["errors"]:
        logger.warning(f"오류 발생: {len(stats['errors'])}개")
        for error in stats["errors"][:5]:  # 처음 5개 오류만 표시
            logger.warning(f"  - {error}")
        if len(stats["errors"]) > 5:
            logger.warning(f"  ... 및 {len(stats['errors']) - 5}개 추가 오류")
    
    return stats

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='BMP 배치 전처리 시스템')
    parser.add_argument('--input', type=str, 
                       default='data/images/shot-image-original',
                       help='입력 BMP 파일 디렉토리')
    parser.add_argument('--output', type=str, 
                       default='data/images/shot-image-bmp-treated-2',
                       help='출력 디렉토리')
    parser.add_argument('--max-files', type=int, default=None,
                       help='최대 처리할 파일 수 (테스트용)')
    parser.add_argument('--debug', action='store_true', 
                       help='디버그 모드')
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 입력 디렉토리 확인
        if not os.path.exists(args.input):
            logger.error(f"입력 디렉토리가 존재하지 않습니다: {args.input}")
            return 1
        
        # 출력 디렉토리 생성
        os.makedirs(args.output, exist_ok=True)
        logger.info(f"출력 디렉토리 생성: {args.output}")
        
        # 배치 처리 실행
        stats = process_bmp_batch(args.input, args.output, args.max_files)
        
        if "error" in stats:
            logger.error(f"처리 실패: {stats['error']}")
            return 1
        
        # 결과 요약
        success_rate = (stats["processed"] / stats["total_files"]) * 100
        logger.info(f"처리 성공률: {success_rate:.1f}%")
        
        if success_rate >= 90:
            logger.info("✅ 배치 전처리가 성공적으로 완료되었습니다!")
            return 0
        else:
            logger.warning("⚠️ 일부 파일 처리에 실패했습니다.")
            return 1
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"배치 처리 중 오류 발생: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()
