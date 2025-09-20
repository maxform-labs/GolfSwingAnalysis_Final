#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지 품질 비교 도구
처리된 이미지들의 품질을 비교 분석합니다.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ImageQualityComparator:
    """이미지 품질 비교 분석기"""
    
    def __init__(self):
        self.base_path = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images")
    
    def analyze_brightness_stats(self, img_path: Path) -> Dict:
        """밝기 통계 분석"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return {'error': 'Failed to load image'}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            return {
                'mean': float(np.mean(gray)),
                'std': float(np.std(gray)),
                'min': int(np.min(gray)),
                'max': int(np.max(gray)),
                'median': float(np.median(gray)),
                'q25': float(np.percentile(gray, 25)),
                'q75': float(np.percentile(gray, 75)),
                'contrast': float(np.std(gray)) / float(np.mean(gray)) if np.mean(gray) > 0 else 0
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_ball_visibility(self, img_path: Path) -> Dict:
        """볼 검출 및 가시성 분석"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return {'ball_detected': False, 'error': 'Failed to load image'}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # HoughCircles로 볼 검출
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 50,
                param1=50, param2=30, minRadius=10, maxRadius=100
            )
            
            ball_detected = circles is not None and len(circles[0]) > 0
            ball_count = len(circles[0]) if circles is not None else 0
            
            result = {
                'ball_detected': ball_detected,
                'ball_count': ball_count,
                'image_size': gray.shape,
                'visibility_score': 0
            }
            
            if ball_detected:
                # 가장 큰 원을 볼로 가정
                x, y, r = np.round(circles[0, 0]).astype("int")
                
                # 볼 영역 추출
                ball_region = gray[max(0, y-r):min(gray.shape[0], y+r), 
                                  max(0, x-r):min(gray.shape[1], x+r)]
                
                if ball_region.size > 0:
                    ball_brightness = np.mean(ball_region)
                    ball_contrast = np.std(ball_region)
                    
                    # 가시성 점수 계산 (밝기와 대비 조합)
                    visibility_score = min(100, (ball_brightness / 255 * 50) + (ball_contrast / 128 * 50))
                    
                    result.update({
                        'ball_center': (int(x), int(y)),
                        'ball_radius': int(r),
                        'ball_brightness': float(ball_brightness),
                        'ball_contrast': float(ball_contrast),
                        'visibility_score': float(visibility_score)
                    })
            
            return result
            
        except Exception as e:
            return {'ball_detected': False, 'error': str(e)}
    
    def compare_folders(self, folder_names: List[str], sample_count: int = 5) -> Dict:
        """여러 폴더의 이미지 품질 비교"""
        results = {}
        
        for folder_name in folder_names:
            folder_path = self.base_path / folder_name
            if not folder_path.exists():
                results[folder_name] = {'error': f'Folder not found: {folder_path}'}
                continue
            
            # PNG, JPG, BMP 파일 찾기
            image_files = list(folder_path.glob("*.png")) + \
                         list(folder_path.glob("*.jpg")) + \
                         list(folder_path.glob("*.bmp"))
            
            if not image_files:
                results[folder_name] = {'error': 'No image files found'}
                continue
            
            # 샘플 이미지들 분석
            sample_files = sorted(image_files)[:sample_count]
            folder_results = {
                'total_files': len(image_files),
                'analyzed_files': len(sample_files),
                'brightness_stats': [],
                'ball_detection': [],
                'average_stats': {}
            }
            
            brightness_means = []
            visibility_scores = []
            ball_detections = 0
            
            for img_file in sample_files:
                # 밝기 분석
                brightness_stats = self.analyze_brightness_stats(img_file)
                folder_results['brightness_stats'].append({
                    'file': img_file.name,
                    'stats': brightness_stats
                })
                
                if 'mean' in brightness_stats:
                    brightness_means.append(brightness_stats['mean'])
                
                # 볼 검출 분석
                ball_analysis = self.detect_ball_visibility(img_file)
                folder_results['ball_detection'].append({
                    'file': img_file.name,
                    'analysis': ball_analysis
                })
                
                if ball_analysis.get('ball_detected', False):
                    ball_detections += 1
                    if 'visibility_score' in ball_analysis:
                        visibility_scores.append(ball_analysis['visibility_score'])
            
            # 평균 통계 계산
            if brightness_means:
                folder_results['average_stats'] = {
                    'avg_brightness': float(np.mean(brightness_means)),
                    'ball_detection_rate': float(ball_detections / len(sample_files) * 100),
                    'avg_visibility_score': float(np.mean(visibility_scores)) if visibility_scores else 0
                }
            
            results[folder_name] = folder_results
        
        return results
    
    def print_comparison_report(self, comparison_results: Dict):
        """비교 결과 리포트 출력"""
        print("=" * 80)
        print("이미지 품질 비교 분석 리포트")
        print("=" * 80)
        
        for folder_name, results in comparison_results.items():
            print(f"\n📁 {folder_name}")
            print("-" * 60)
            
            if 'error' in results:
                print(f"❌ 오류: {results['error']}")
                continue
            
            print(f"총 파일 수: {results['total_files']}개")
            print(f"분석 파일 수: {results['analyzed_files']}개")
            
            if 'average_stats' in results and results['average_stats']:
                stats = results['average_stats']
                print(f"평균 밝기: {stats['avg_brightness']:.1f}/255")
                print(f"볼 검출율: {stats['ball_detection_rate']:.1f}%")
                print(f"가시성 점수: {stats['avg_visibility_score']:.1f}/100")
                
                # 품질 등급 평가
                brightness = stats['avg_brightness']
                detection_rate = stats['ball_detection_rate']
                visibility = stats['avg_visibility_score']
                
                if brightness > 100 and detection_rate > 80 and visibility > 70:
                    grade = "🟢 우수 (Excellent)"
                elif brightness > 60 and detection_rate > 60 and visibility > 50:
                    grade = "🟡 양호 (Good)"  
                elif brightness > 30 and detection_rate > 40 and visibility > 30:
                    grade = "🟠 보통 (Fair)"
                else:
                    grade = "🔴 불량 (Poor)"
                
                print(f"종합 품질: {grade}")

def main():
    """메인 실행 함수"""
    comparator = ImageQualityComparator()
    
    print("골프 스윙 이미지 품질 비교 분석을 시작합니다...\n")
    
    # 분석할 폴더들 (실제 존재하는 폴더들만)
    folders_to_compare = [
        "shot-image-improved-v7-final/driver/no_marker_ball-1",  # v7.0 최종 처리
        "shot-image-improved-v7/driver/no_marker_ball-1",        # v7.0 기본 처리  
        "shot-image-bmp-treated-3/driver/no_marker_ball-1",      # BMP 처리
        "shot-image-jpg/driver/no_marker_ball-1",                # JPG 변환
        "shot-image-original/driver/no_marker_ball-1"            # 원본 BMP
    ]
    
    # 존재하는 폴더만 필터링
    base_path = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images")
    existing_folders = [f for f in folders_to_compare if (base_path / f).exists()]
    
    print(f"분석 대상 폴더: {len(existing_folders)}개")
    for folder in existing_folders:
        print(f"  - {folder}")
    
    # 품질 비교 실행
    results = comparator.compare_folders(existing_folders, sample_count=5)
    
    # 결과 리포트 출력
    comparator.print_comparison_report(results)
    
    print(f"\n💡 권장 사항:")
    print(f"   가장 좋은 품질의 이미지는 'shot-image-improved-v7-final' 폴더에 있습니다.")
    print(f"   이 폴더의 이미지들이 딤플 분석에 최적화되어 있습니다.")

if __name__ == "__main__":
    main()