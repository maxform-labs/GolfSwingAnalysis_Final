#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 v7.0 딤플 분석기 - 어두운 이미지 처리 최적화
단계별 개선: 히스토그램 평활화, 동적 감마 조정, 적응형 밝기 향상
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class ImprovedV7Processor:
    """개선된 v7.0 딤플 분석기 - 어두운 이미지 최적화"""
    
    def __init__(self, debug: bool = True):
        self.debug = debug
        
        # 개선된 파라미터
        self.params = {
            'dark_threshold': 50,      # 어두운 이미지 판단 기준
            'bright_threshold': 200,   # 밝은 이미지 판단 기준
            'target_brightness': 128,  # 목표 밝기
            'min_contrast': 50,        # 최소 대비
        }
        
        if self.debug:
            print("=== 개선된 v7.0 딤플 분석기 초기화 ===")
            print("특징: 어두운 이미지 자동 감지 및 적응형 처리")
    
    def analyze_image_brightness(self, img: np.ndarray) -> Dict:
        """
        이미지 밝기 분석
        
        Returns:
            밝기 통계 정보
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        # 히스토그램 분석
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # 어두운 픽셀 비율 (0-50)
        dark_pixels = np.sum(hist[:50]) / gray.size
        # 밝은 픽셀 비율 (200-255)
        bright_pixels = np.sum(hist[200:]) / gray.size
        
        return {
            'mean': mean_brightness,
            'std': std_brightness,
            'min': min_val,
            'max': max_val,
            'dark_ratio': dark_pixels,
            'bright_ratio': bright_pixels,
            'contrast': max_val - min_val
        }
    
    def calculate_dynamic_gamma(self, brightness_stats: Dict) -> float:
        """
        동적 감마 값 계산
        
        Args:
            brightness_stats: 밝기 분석 결과
            
        Returns:
            최적 감마 값
        """
        mean_brightness = brightness_stats['mean']
        
        if mean_brightness < 30:
            # 매우 어두운 이미지: 강한 밝기 증가
            gamma = 0.3
        elif mean_brightness < 50:
            # 어두운 이미지: 중간 밝기 증가
            gamma = 0.5
        elif mean_brightness < 80:
            # 약간 어두운 이미지: 약한 밝기 증가
            gamma = 0.7
        elif mean_brightness > 180:
            # 밝은 이미지: 어둡게
            gamma = 1.5
        elif mean_brightness > 150:
            # 약간 밝은 이미지
            gamma = 1.2
        else:
            # 적당한 밝기
            gamma = 1.0
        
        if self.debug:
            print(f"  동적 감마: {gamma:.1f} (평균 밝기: {mean_brightness:.1f})")
        
        return gamma
    
    def step1_histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """
        단계 1: 히스토그램 평활화 우선 적용
        """
        if self.debug:
            print("단계 1: 히스토그램 평활화 적용")
        
        # BGR to YUV
        if len(img.shape) == 3:
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            
            # Y 채널에만 히스토그램 평활화
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            
            # YUV to BGR
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            result = cv2.equalizeHist(img)
        
        return result
    
    def step2_dynamic_gamma_correction(self, img: np.ndarray, gamma: float = None) -> np.ndarray:
        """
        단계 2: 동적 감마 보정
        """
        if self.debug:
            print("단계 2: 동적 감마 보정")
        
        # 밝기 분석
        stats = self.analyze_image_brightness(img)
        
        # 감마 값 결정
        if gamma is None:
            gamma = self.calculate_dynamic_gamma(stats)
        
        # 감마 보정 적용
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        
        if len(img.shape) == 3:
            result = cv2.LUT(img, table)
        else:
            result = cv2.LUT(img, table)
        
        return result
    
    def step3_adaptive_brightness_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        단계 3: 적응형 밝기 향상 (CLAHE + 추가 처리)
        """
        if self.debug:
            print("단계 3: 적응형 밝기 향상")
        
        # LAB 색공간으로 변환
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = img.copy()
        
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # 추가 밝기 조정
        current_mean = np.mean(l_enhanced)
        if current_mean < self.params['target_brightness']:
            # 밝기 부족 시 추가 조정
            alpha = self.params['target_brightness'] / current_mean
            alpha = min(alpha, 2.0)  # 최대 2배까지만
            l_enhanced = cv2.convertScaleAbs(l_enhanced, alpha=alpha, beta=0)
        
        # 결과 합성
        if len(img.shape) == 3:
            result = cv2.merge([l_enhanced, a, b])
            result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        else:
            result = l_enhanced
        
        return result
    
    def step4_dimple_enhancement(self, img: np.ndarray, ball_center: Tuple[int, int], ball_radius: int) -> np.ndarray:
        """
        단계 4: 딤플 강조 처리 (v7.0 알고리즘)
        """
        if self.debug:
            print("단계 4: 딤플 강조 처리")
        
        x, y = ball_center
        r = ball_radius
        
        # 볼 영역 추출
        roi_size = int(r * 2.5)
        x1 = max(0, x - roi_size//2)
        y1 = max(0, y - roi_size//2)
        x2 = min(img.shape[1], x + roi_size//2)
        y2 = min(img.shape[0], y + roi_size//2)
        
        ball_roi = img[y1:y2, x1:x2].copy()
        
        if ball_roi.size == 0:
            return img
        
        # 3배 확대
        zoomed = cv2.resize(ball_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # 그레이스케일 변환
        gray_zoomed = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY) if len(zoomed.shape) == 3 else zoomed
        
        # 다중 필터링
        # 1. 언샤프 마스킹
        gaussian = cv2.GaussianBlur(gray_zoomed, (0, 0), 2.0)
        unsharp = cv2.addWeighted(gray_zoomed, 1.5, gaussian, -0.5, 0)
        
        # 2. 라플라시안
        laplacian = cv2.Laplacian(unsharp, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # 3. 형태학적 처리
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(unsharp, cv2.MORPH_TOPHAT, kernel)
        
        # 결합
        enhanced = cv2.addWeighted(unsharp, 0.5, laplacian, 0.25, 0)
        enhanced = cv2.addWeighted(enhanced, 0.75, tophat, 0.25, 0)
        
        # 원본 크기로 복원
        resized = cv2.resize(enhanced, (x2-x1, y2-y1))
        
        # 결과 적용
        result = img.copy()
        if len(result.shape) == 3:
            result[y1:y2, x1:x2] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        else:
            result[y1:y2, x1:x2] = resized
        
        return result
    
    def detect_ball(self, img: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """볼 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # 먼저 밝기 향상
        gray = cv2.equalizeHist(gray)
        
        # Hough Circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]
            return (x, y), r
        
        # 기본값
        h, w = gray.shape
        return (w//2, h//2), 30
    
    def process_complete_pipeline(self, img_path: Union[str, Path], output_path: Union[str, Path]) -> Dict:
        """
        완전한 처리 파이프라인
        """
        img_path = Path(img_path)
        output_path = Path(output_path)
        
        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            return {'success': False, 'error': 'Failed to load image'}
        
        print(f"\n처리 중: {img_path.name}")
        
        # 원본 밝기 분석
        original_stats = self.analyze_image_brightness(img)
        print(f"  원본 - 평균 밝기: {original_stats['mean']:.1f}, 대비: {original_stats['contrast']:.1f}")
        
        # 단계별 처리
        
        # 단계 1: 히스토그램 평활화
        step1_result = self.step1_histogram_equalization(img)
        
        # 단계 2: 동적 감마 보정
        step2_result = self.step2_dynamic_gamma_correction(step1_result)
        
        # 단계 3: 적응형 밝기 향상
        step3_result = self.step3_adaptive_brightness_enhancement(step2_result)
        
        # 볼 검출
        ball_center, ball_radius = self.detect_ball(step3_result)
        print(f"  볼 검출: 중심 {ball_center}, 반지름 {ball_radius}")
        
        # 단계 4: 딤플 강조
        final_result = self.step4_dimple_enhancement(step3_result, ball_center, ball_radius)
        
        # 최종 밝기 분석
        final_stats = self.analyze_image_brightness(final_result)
        print(f"  최종 - 평균 밝기: {final_stats['mean']:.1f}, 대비: {final_stats['contrast']:.1f}")
        
        # 저장
        success = cv2.imwrite(str(output_path), final_result)
        
        if success:
            print(f"  [OK] 저장: {output_path.name}")
            return {
                'success': True,
                'input': str(img_path),
                'output': str(output_path),
                'ball_center': ball_center,
                'ball_radius': ball_radius,
                'brightness_improvement': final_stats['mean'] - original_stats['mean'],
                'contrast_improvement': final_stats['contrast'] - original_stats['contrast']
            }
        else:
            return {'success': False, 'error': 'Failed to save image'}


def main():
    """메인 실행 함수"""
    
    # 경로 설정
    input_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1")
    output_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-improved-v7/driver/no_marker_ball-1")
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 프로세서 초기화
    processor = ImprovedV7Processor(debug=True)
    
    print("="*80)
    print("개선된 v7.0 딤플 분석기 - 어두운 이미지 최적화")
    print("="*80)
    print(f"입력: {input_dir}")
    print(f"출력: {output_dir}")
    print("="*80)
    
    # 특정 문제 이미지들 우선 처리
    test_files = [
        "1_1.bmp",
        "Gamma_1_4.bmp", 
        "Gamma_1_6.bmp",
        "2_1.bmp"
    ]
    
    results = []
    
    for filename in test_files:
        img_path = input_dir / filename
        if img_path.exists():
            output_path = output_dir / f"{img_path.stem}_improved.png"
            result = processor.process_complete_pipeline(img_path, output_path)
            results.append(result)
    
    # 결과 요약
    print("\n" + "="*80)
    print("처리 결과 요약")
    print("="*80)
    
    successful = sum(1 for r in results if r.get('success', False))
    print(f"성공: {successful}/{len(results)}")
    
    if results:
        avg_brightness_improvement = np.mean([r.get('brightness_improvement', 0) for r in results if r.get('success')])
        avg_contrast_improvement = np.mean([r.get('contrast_improvement', 0) for r in results if r.get('success')])
        
        print(f"평균 밝기 개선: {avg_brightness_improvement:+.1f}")
        print(f"평균 대비 개선: {avg_contrast_improvement:+.1f}")
    
    # 결과 저장
    result_file = output_dir / "improvement_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장: {result_file}")
    print("="*80)


if __name__ == "__main__":
    main()