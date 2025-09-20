#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모든 이미지에 개선된 v7.0 처리 적용
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class BatchImprovedProcessor:
    """배치 처리용 개선된 프로세서"""
    
    def __init__(self):
        self.params = {
            'target_brightness': 128,
        }
    
    def analyze_brightness(self, img: np.ndarray) -> float:
        """밝기 분석"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return np.mean(gray)
    
    def calculate_gamma(self, brightness: float) -> float:
        """동적 감마 계산"""
        if brightness < 30:
            return 0.3
        elif brightness < 50:
            return 0.5
        elif brightness < 80:
            return 0.7
        else:
            return 1.0
    
    def process_image(self, img_path: Path, output_path: Path) -> Dict:
        """이미지 처리"""
        # 로드
        img = cv2.imread(str(img_path))
        if img is None:
            return {'success': False}
        
        original_brightness = self.analyze_brightness(img)
        
        # 단계 1: 히스토그램 평활화
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        step1 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # 단계 2: 동적 감마 보정
        gamma = self.calculate_gamma(original_brightness)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        step2 = cv2.LUT(step1, table)
        
        # 단계 3: CLAHE
        lab = cv2.cvtColor(step2, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # 밝기 조정
        current_mean = np.mean(l)
        if current_mean < self.params['target_brightness']:
            alpha = min(self.params['target_brightness'] / current_mean, 2.0)
            l = cv2.convertScaleAbs(l, alpha=alpha, beta=0)
        
        step3 = cv2.merge([l, a, b])
        step3 = cv2.cvtColor(step3, cv2.COLOR_LAB2BGR)
        
        # 볼 검출
        gray = cv2.cvtColor(step3, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            x, y, r = np.round(circles[0, 0]).astype("int")
        else:
            h, w = gray.shape
            x, y, r = w//2, h//2, 30
        
        # 단계 4: 볼 영역 딤플 강조
        roi_size = int(r * 2.5)
        x1 = max(0, x - roi_size//2)
        y1 = max(0, y - roi_size//2)
        x2 = min(step3.shape[1], x + roi_size//2)
        y2 = min(step3.shape[0], y + roi_size//2)
        
        ball_roi = step3[y1:y2, x1:x2].copy()
        
        if ball_roi.size > 0:
            # 3배 확대 후 처리
            zoomed = cv2.resize(ball_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray_zoomed = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
            
            # 언샤프 마스킹
            gaussian = cv2.GaussianBlur(gray_zoomed, (0, 0), 2.0)
            unsharp = cv2.addWeighted(gray_zoomed, 1.5, gaussian, -0.5, 0)
            
            # 라플라시안
            laplacian = cv2.Laplacian(unsharp, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            # Top-hat
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            tophat = cv2.morphologyEx(unsharp, cv2.MORPH_TOPHAT, kernel)
            
            # 결합
            enhanced = cv2.addWeighted(unsharp, 0.5, laplacian, 0.25, 0)
            enhanced = cv2.addWeighted(enhanced, 0.75, tophat, 0.25, 0)
            
            # 원본 크기로 복원하여 적용
            resized = cv2.resize(enhanced, (x2-x1, y2-y1))
            step3[y1:y2, x1:x2] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        
        # 저장
        final_brightness = self.analyze_brightness(step3)
        success = cv2.imwrite(str(output_path), step3)
        
        return {
            'success': success,
            'ball_center': (x, y),
            'ball_radius': r,
            'brightness_improvement': final_brightness - original_brightness,
            'gamma_used': gamma
        }

def main():
    """전체 파일 처리"""
    
    input_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1")
    output_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-improved-v7-final/driver/no_marker_ball-1")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = BatchImprovedProcessor()
    
    # BMP 파일 찾기
    bmp_files = sorted(input_dir.glob("*.bmp"))
    
    print("="*80)
    print("개선된 v7.0 전체 파일 처리")
    print("="*80)
    print(f"입력: {input_dir}")
    print(f"출력: {output_dir}")
    print(f"파일 수: {len(bmp_files)}")
    print("="*80)
    
    successful = 0
    total_brightness_improvement = 0
    
    for i, bmp_file in enumerate(bmp_files):
        output_file = output_dir / f"{bmp_file.stem}_final.png"
        
        print(f"[{i+1}/{len(bmp_files)}] {bmp_file.name}", end=" ... ")
        
        result = processor.process_image(bmp_file, output_file)
        
        if result['success']:
            successful += 1
            total_brightness_improvement += result['brightness_improvement']
            print(f"OK (밝기: +{result['brightness_improvement']:.1f}, 감마: {result['gamma_used']:.1f})")
        else:
            print("FAILED")
    
    # 결과
    avg_improvement = total_brightness_improvement / successful if successful > 0 else 0
    
    print("\n" + "="*80)
    print("처리 완료!")
    print("="*80)
    print(f"성공: {successful}/{len(bmp_files)} ({successful/len(bmp_files)*100:.1f}%)")
    print(f"평균 밝기 개선: +{avg_improvement:.1f}")
    print("="*80)

if __name__ == "__main__":
    main()