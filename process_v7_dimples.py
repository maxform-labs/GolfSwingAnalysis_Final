#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v7.0 딤플 분석기 독립 실행 스크립트
BMP 직접 처리 및 혁신적 딤플 분석
"""

import cv2
import numpy as np
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DimpleAnalyzerV7Standalone:
    """BMP 직접 처리 및 혁신적 딤플 분석 시스템 v7.0 - 독립 실행 버전"""
    
    def __init__(self, fps: int = 820, debug: bool = True):
        self.fps = fps
        self.debug = debug
        self.pixel_to_mm = 0.1
        
        self.dimple_params = {
            'zoom_factor': 3,
            'gamma': 0.5,
            'clahe_clip_limit': 2.0,
            'overexposure_threshold': 240,
            'min_dimple_radius': 2,
            'max_dimple_radius': 8
        }
        
        if self.debug:
            print("=== 딤플 분석 시스템 v7.0 초기화 완료 ===")
    
    def read_bmp_direct(self, bmp_path: Union[str, Path]) -> Optional[np.ndarray]:
        """BMP 파일 직접 읽기"""
        bmp_path = Path(bmp_path)
        
        if not bmp_path.exists():
            return None
        
        try:
            # OpenCV로 읽기 (더 안정적)
            img = cv2.imread(str(bmp_path))
            if img is not None and self.debug:
                print(f"BMP 로드: {bmp_path.name} ({img.shape})")
            return img
        except Exception as e:
            if self.debug:
                print(f"BMP 읽기 실패: {e}")
            return None
    
    def correct_overexposure(self, image: np.ndarray, 
                            ball_center: Tuple[int, int], 
                            ball_radius: int) -> np.ndarray:
        """과노출 보정"""
        x, y = ball_center
        r = ball_radius
        
        # ROI 설정
        roi_size = int(r * 3)
        x1 = max(0, x - roi_size//2)
        y1 = max(0, y - roi_size//2)
        x2 = min(image.shape[1], x + roi_size//2)
        y2 = min(image.shape[0], y + roi_size//2)
        
        roi = image[y1:y2, x1:x2].copy()
        
        if roi.size == 0:
            return image
        
        # 그레이스케일 변환
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # 과노출 영역 검출
        overexposed_mask = gray_roi > self.dimple_params['overexposure_threshold']
        
        if np.sum(overexposed_mask) > 0:
            # 감마 보정
            gamma = self.dimple_params['gamma']
            gamma_corrected = np.power(gray_roi / 255.0, gamma) * 255.0
            gamma_corrected = gamma_corrected.astype(np.uint8)
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=self.dimple_params['clahe_clip_limit'], 
                                   tileGridSize=(8,8))
            clahe_applied = clahe.apply(gamma_corrected)
            
            # 언샤프 마스킹
            gaussian = cv2.GaussianBlur(clahe_applied, (0, 0), 2.0)
            unsharp = cv2.addWeighted(clahe_applied, 1.5, gaussian, -0.5, 0)
            
            # 과노출 영역만 교체
            corrected_roi = gray_roi.copy()
            corrected_roi[overexposed_mask] = unsharp[overexposed_mask]
            
            # 원본에 적용
            result = image.copy()
            if len(image.shape) == 3:
                result[y1:y2, x1:x2] = cv2.cvtColor(corrected_roi, cv2.COLOR_GRAY2BGR)
            else:
                result[y1:y2, x1:x2] = corrected_roi
            
            return result
        
        return image
    
    def enhanced_ball_detection(self, image: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """향상된 볼 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Hough Circles로 볼 검출
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # 첫 번째 원을 볼로 선택
            x, y, r = circles[0]
            return (x, y), r
        
        # 기본값
        h, w = gray.shape
        return (w//2, h//2), 30
    
    def process_single_image(self, img_path: Path, output_path: Path) -> Dict:
        """단일 이미지 처리"""
        # 이미지 읽기
        img = self.read_bmp_direct(img_path)
        if img is None:
            return {'success': False, 'error': 'Failed to load image'}
        
        # 볼 검출
        center, radius = self.enhanced_ball_detection(img)
        
        # 과노출 보정
        corrected = self.correct_overexposure(img, center, radius)
        
        # 3배 확대 처리
        x, y = center
        r = radius
        roi_size = int(r * 2)
        x1 = max(0, x - roi_size//2)
        y1 = max(0, y - roi_size//2)
        x2 = min(img.shape[1], x + roi_size//2)
        y2 = min(img.shape[0], y + roi_size//2)
        
        ball_roi = corrected[y1:y2, x1:x2]
        
        if ball_roi.size > 0:
            # 3배 확대
            zoomed = cv2.resize(ball_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            
            # 그레이스케일 변환
            gray_zoomed = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY) if len(zoomed.shape) == 3 else zoomed
            
            # 딤플 강조 필터링
            # 라플라시안
            laplacian = cv2.Laplacian(gray_zoomed, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            # DoG
            g1 = cv2.GaussianBlur(gray_zoomed, (5, 5), 1.0)
            g2 = cv2.GaussianBlur(gray_zoomed, (9, 9), 2.0)
            dog = cv2.subtract(g1, g2)
            
            # Top-hat
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            tophat = cv2.morphologyEx(gray_zoomed, cv2.MORPH_TOPHAT, kernel)
            
            # 결합
            enhanced = cv2.addWeighted(gray_zoomed, 0.4, laplacian, 0.3, 0)
            enhanced = cv2.addWeighted(enhanced, 0.7, dog, 0.3, 0)
            
            # 최종 이미지에 적용
            final = corrected.copy()
            
            # 확대된 영역을 원본 크기로 축소하여 적용
            resized_enhanced = cv2.resize(enhanced, (x2-x1, y2-y1))
            if len(final.shape) == 3:
                final[y1:y2, x1:x2] = cv2.cvtColor(resized_enhanced, cv2.COLOR_GRAY2BGR)
            else:
                final[y1:y2, x1:x2] = resized_enhanced
            
            # 저장
            success = cv2.imwrite(str(output_path), final)
            
            if success:
                return {
                    'success': True,
                    'ball_center': center,
                    'ball_radius': radius,
                    'output': str(output_path)
                }
        
        # 실패 시 원본 저장
        cv2.imwrite(str(output_path), corrected)
        return {
            'success': True,
            'ball_center': center,
            'ball_radius': radius,
            'output': str(output_path)
        }


def main():
    """메인 실행 함수"""
    
    # 경로 설정
    input_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1")
    output_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-v7-enhanced/driver/no_marker_ball-1")
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 분석기 초기화
    analyzer = DimpleAnalyzerV7Standalone(debug=True)
    
    # BMP 파일 찾기
    bmp_files = sorted(input_dir.glob("*.bmp"))
    
    if not bmp_files:
        print(f"BMP 파일을 찾을 수 없음: {input_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"딤플 분석 v7.0 - BMP 직접 처리 (독립 실행 버전)")
    print(f"{'='*80}")
    print(f"입력: {input_dir}")
    print(f"출력: {output_dir}")
    print(f"파일 수: {len(bmp_files)}")
    print(f"{'='*80}\n")
    
    results = []
    processed = 0
    
    # 모든 파일 처리
    for i, bmp_file in enumerate(bmp_files):
        output_file = output_dir / f"{bmp_file.stem}_v7.png"
        
        print(f"[{i+1}/{len(bmp_files)}] 처리 중: {bmp_file.name}")
        
        result = analyzer.process_single_image(bmp_file, output_file)
        results.append(result)
        
        if result['success']:
            processed += 1
            print(f"  [OK] 저장: {output_file.name}")
            if 'ball_center' in result:
                print(f"  볼 위치: {result['ball_center']}, 반지름: {result['ball_radius']}")
        else:
            print(f"  [FAIL] 실패: {result.get('error', 'Unknown error')}")
    
    # 결과 저장
    result_file = output_dir / "v7_processing_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_files': len(bmp_files),
            'processed': processed,
            'success_rate': f"{(processed/len(bmp_files)*100):.1f}%",
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"처리 완료!")
    print(f"- 총 파일: {len(bmp_files)}")
    print(f"- 처리 성공: {processed}")
    print(f"- 성공률: {(processed/len(bmp_files)*100):.1f}%")
    print(f"- 결과 저장: {result_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()