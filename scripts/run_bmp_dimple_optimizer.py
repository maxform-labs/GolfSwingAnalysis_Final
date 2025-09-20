#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMP Dimple Optimizer 실행 스크립트
Gamma 렌즈 이미지 최적화 처리
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import cv2
import numpy as np
from src.processing.image_enhancement.bmp_dimple_optimizer import BMPDimpleOptimizer
import logging
import argparse
from datetime import datetime

def analyze_image_quality(image_path: str) -> dict:
    """
    이미지 품질 분석
    
    Args:
        image_path: 이미지 경로
        
    Returns:
        품질 지표 딕셔너리
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return {}
        
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # 품질 지표 계산
    metrics = {
        'mean_brightness': np.mean(gray),
        'std_brightness': np.std(gray),
        'contrast': gray.max() - gray.min(),
        'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'edge_density': np.mean(cv2.Canny(gray, 50, 150)) * 100,
        'histogram_entropy': -np.sum(np.histogram(gray, 256)[0] * np.log2(np.histogram(gray, 256)[0] + 1e-7))
    }
    
    return metrics

def compare_processing_methods(base_dir: str):
    """
    세 가지 처리 방법 비교
    
    Args:
        base_dir: 기본 디렉토리
    """
    methods = {
        'JPG 변환 후 처리': 'shot-image-treated',
        'BMP 직접 처리 v1': 'shot-image-bmp-treated',
        'BMP 직접 처리 v2': 'shot-image-bmp-treated-2'
    }
    
    print("\n" + "="*80)
    print("이미지 처리 방법 비교 분석")
    print("="*80)
    
    for method_name, folder_name in methods.items():
        folder_path = Path(base_dir) / 'data/images' / folder_name
        
        if not folder_path.exists():
            print(f"\n❌ {method_name}: 폴더가 존재하지 않습니다 ({folder_path})")
            continue
            
        print(f"\n📁 {method_name} ({folder_name})")
        print("-"*40)
        
        # 샘플 이미지 분석
        sample_images = list(folder_path.rglob("*1_1.*"))[:3]
        
        if not sample_images:
            print("  샘플 이미지를 찾을 수 없습니다")
            continue
            
        total_metrics = {}
        for img_path in sample_images:
            metrics = analyze_image_quality(str(img_path))
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = []
                total_metrics[key].append(value)
        
        # 평균 계산 및 출력
        for key, values in total_metrics.items():
            avg_value = np.mean(values)
            print(f"  {key:20}: {avg_value:.2f}")

def process_with_new_optimizer(input_dir: str, output_dir: str, sample_only: bool = False):
    """
    새로운 BMP Dimple Optimizer로 처리
    
    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        sample_only: 샘플만 처리할지 여부
    """
    print("\n" + "="*80)
    print("BMP Dimple Optimizer v2.0 처리 시작")
    print("="*80)
    
    optimizer = BMPDimpleOptimizer()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 처리할 파일 목록
    if sample_only:
        # 각 폴더에서 샘플 3개씩만
        bmp_files = []
        for folder in input_path.rglob("*"):
            if folder.is_dir():
                folder_bmps = list(folder.glob("*.bmp"))[:3]
                bmp_files.extend(folder_bmps)
    else:
        bmp_files = list(input_path.rglob("*.bmp"))
    
    print(f"처리할 파일 수: {len(bmp_files)}")
    
    success_count = 0
    error_count = 0
    
    for i, bmp_file in enumerate(bmp_files, 1):
        relative_path = bmp_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix('.png')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"[{i}/{len(bmp_files)}] 처리 중: {relative_path}")
            
            # 처리 전 품질
            before_metrics = analyze_image_quality(str(bmp_file))
            
            # 최적화 처리
            result = optimizer.process_bmp_for_dimples(str(bmp_file), str(output_file))
            
            # 처리 후 품질
            after_metrics = analyze_image_quality(str(output_file))
            
            # 개선도 출력
            print(f"  ✅ 성공")
            print(f"  품질 개선:")
            print(f"    - 선명도: {before_metrics.get('sharpness', 0):.1f} → {after_metrics.get('sharpness', 0):.1f}")
            print(f"    - 대비: {before_metrics.get('contrast', 0):.1f} → {after_metrics.get('contrast', 0):.1f}")
            print(f"    - 엣지: {before_metrics.get('edge_density', 0):.1f}% → {after_metrics.get('edge_density', 0):.1f}%")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            error_count += 1
    
    print("\n" + "="*80)
    print(f"처리 완료: 성공 {success_count}, 실패 {error_count}")
    print(f"출력 디렉토리: {output_path}")
    print("="*80)

def visualize_comparison(base_dir: str, sample_path: str):
    """
    처리 방법 시각적 비교
    
    Args:
        base_dir: 기본 디렉토리
        sample_path: 샘플 이미지 경로 (상대 경로)
    """
    import matplotlib.pyplot as plt
    
    methods = {
        'Original BMP': f'data/images/shot-image-original/{sample_path}',
        'JPG Treated': f'data/images/shot-image-treated/{sample_path.replace(".bmp", ".jpg")}',
        'BMP Treated v1': f'data/images/shot-image-bmp-treated/{sample_path}',
        'BMP Treated v2': f'data/images/shot-image-bmp-treated-2/{sample_path}',
        'New Optimizer': f'data/images/shot-image-bmp-optimized/{sample_path.replace(".bmp", ".png")}'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (method_name, rel_path) in enumerate(methods.items()):
        if idx >= len(axes):
            break
            
        full_path = Path(base_dir) / rel_path
        
        if full_path.exists():
            img = cv2.imread(str(full_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img_rgb)
                axes[idx].set_title(method_name)
                axes[idx].axis('off')
                
                # 이미지 일부 확대 (딤플 영역)
                h, w = img.shape[:2]
                zoom_area = img_rgb[h//2-50:h//2+50, w//2-50:w//2+50]
                
                # 확대 영역 표시
                from matplotlib.patches import Rectangle
                rect = Rectangle((w//2-50, h//2-50), 100, 100, 
                               linewidth=2, edgecolor='r', facecolor='none')
                axes[idx].add_patch(rect)
        else:
            axes[idx].text(0.5, 0.5, f'Not found\n{rel_path}', 
                         ha='center', va='center')
            axes[idx].axis('off')
    
    # 마지막 subplot에 확대 영역 표시
    if idx < len(axes) - 1:
        axes[-1].imshow(zoom_area)
        axes[-1].set_title("Zoomed Area (Center)")
        axes[-1].axis('off')
    
    plt.tight_layout()
    
    # 저장
    output_path = Path(base_dir) / 'data/results' / f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"\n비교 이미지 저장: {output_path}")
    
    plt.show()

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="BMP Dimple Optimizer v2.0")
    parser.add_argument("--mode", choices=['analyze', 'process', 'compare', 'visualize'], 
                       default='process',
                       help="실행 모드")
    parser.add_argument("--input", "-i", 
                       default="C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original",
                       help="입력 디렉토리")
    parser.add_argument("--output", "-o", 
                       default="C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-bmp-optimized",
                       help="출력 디렉토리")
    parser.add_argument("--sample", action="store_true", 
                       help="샘플만 처리 (각 폴더당 3개)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="상세 출력")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    base_dir = "C:/src/GolfSwingAnalysis_Final_ver8"
    
    if args.mode == 'analyze':
        # 현재 처리 방법 분석
        compare_processing_methods(base_dir)
        
    elif args.mode == 'process':
        # 새로운 최적화 처리
        process_with_new_optimizer(args.input, args.output, args.sample)
        
    elif args.mode == 'compare':
        # 모든 방법 비교
        compare_processing_methods(base_dir)
        print("\n새로운 최적화 처리 시작...")
        process_with_new_optimizer(args.input, args.output, sample_only=True)
        
    elif args.mode == 'visualize':
        # 시각적 비교
        sample = "7iron/logo_ball-1/1_1.bmp"
        visualize_comparison(base_dir, sample)

if __name__ == "__main__":
    main()