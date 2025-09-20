#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 BMP Optimizer 테스트
"""

import sys
sys.path.append('.')

# 직접 import
from src.processing.image_enhancement.bmp_dimple_optimizer import BMPDimpleOptimizer
import cv2
import numpy as np
from pathlib import Path

def test_optimizer():
    """간단한 테스트"""
    print("="*80)
    print("BMP Dimple Optimizer v2.0 테스트")
    print("="*80)
    
    # 옵티마이저 생성
    optimizer = BMPDimpleOptimizer()
    print("✅ Optimizer 생성 성공")
    
    # 샘플 BMP 찾기
    base_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images")
    
    # BMP 원본 폴더 확인
    original_dir = base_dir / "shot-image-original"
    if not original_dir.exists():
        print(f"❌ 원본 폴더가 없습니다: {original_dir}")
        return
        
    # 샘플 BMP 파일 찾기
    sample_bmps = list(original_dir.rglob("*.bmp"))[:3]
    
    if not sample_bmps:
        print("❌ BMP 파일을 찾을 수 없습니다")
        return
        
    print(f"\n샘플 파일 {len(sample_bmps)}개 발견")
    
    # 출력 디렉토리
    output_dir = base_dir / "shot-image-bmp-optimized-test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for bmp_path in sample_bmps:
        print(f"\n처리 중: {bmp_path.name}")
        
        # 출력 경로
        output_path = output_dir / bmp_path.relative_to(original_dir).with_suffix('.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 처리
            result = optimizer.process_bmp_for_dimples(str(bmp_path), str(output_path))
            
            # 결과 확인
            if output_path.exists():
                print(f"  ✅ 성공: {output_path}")
                
                # 품질 비교
                original = cv2.imread(str(bmp_path), cv2.IMREAD_GRAYSCALE)
                processed = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
                
                if original is not None and processed is not None:
                    # 선명도 비교
                    original_sharpness = cv2.Laplacian(original, cv2.CV_64F).var()
                    processed_sharpness = cv2.Laplacian(processed, cv2.CV_64F).var()
                    
                    print(f"  선명도: {original_sharpness:.1f} → {processed_sharpness:.1f}")
                    
                    # 대비 비교
                    original_contrast = original.max() - original.min()
                    processed_contrast = processed.max() - processed.min()
                    
                    print(f"  대비: {original_contrast} → {processed_contrast}")
                    
        except Exception as e:
            print(f"  ❌ 오류: {e}")
    
    print("\n" + "="*80)
    print(f"테스트 완료. 결과: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    test_optimizer()