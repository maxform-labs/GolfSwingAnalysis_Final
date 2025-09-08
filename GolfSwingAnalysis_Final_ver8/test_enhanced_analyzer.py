#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
테스트용 향상된 분석기 - 일부 프레임만 처리
"""

from enhanced_golf_analyzer import EnhancedGolfAnalyzer
import os
from datetime import datetime

def test_analysis():
    """테스트 분석 - 처음 20프레임만"""
    print("=== 향상된 골프 분석 테스트 ===")
    
    analyzer = EnhancedGolfAnalyzer()
    
    # 이미지 디렉토리
    image_dir = "C:/src/GolfSwingAnalysis_Final_ver8/shot-image-jpg/7iron_no_marker_ball_shot1"
    
    # 처음 20개 파일만
    jpg_files = []
    for file in sorted(os.listdir(image_dir)):
        if file.lower().endswith('.jpg'):
            jpg_files.append(os.path.join(image_dir, file))
            if len(jpg_files) >= 20:  # 20개만
                break
                
    print(f"테스트: {len(jpg_files)}개 이미지 처리")
    
    results = []
    for i, img_path in enumerate(jpg_files, 1):
        print(f"프레임 {i}/{len(jpg_files)} 처리 중...")
        
        # 모든 프레임 디버그 이미지 저장
        result = analyzer.analyze_frame(img_path, i, save_debug=True)
        results.append(result)
        
        # 검출 결과 즉시 출력
        if result['ball_data']:
            bd = result['ball_data']
            print(f"  → 볼 검출! 위치:({bd.x:.1f}, {bd.y:.1f}), 방법:{bd.detection_method}, 상태:{bd.motion_state}")
        if result['club_data']:
            cd = result['club_data']
            print(f"  → 클럽 검출! 위치:({cd.x:.1f}, {cd.y:.1f})")
    
    # 통계
    ball_detections = sum(1 for r in results if r['ball_data'])
    club_detections = sum(1 for r in results if r['club_data'])
    
    print(f"\n=== 테스트 결과 ===")
    print(f"볼 검출: {ball_detections}/{len(results)} ({100*ball_detections/len(results):.1f}%)")
    print(f"클럽 검출: {club_detections}/{len(results)} ({100*club_detections/len(results):.1f}%)")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"C:/src/GolfSwingAnalysis_Final_ver8/test_enhanced_{timestamp}.xlsx"
    analyzer.export_to_excel(results, output_path)
    
    print(f"\n결과 파일: {output_path}")
    print("디버그 이미지: C:/src/GolfSwingAnalysis_Final_ver8/enhanced_debug/")

if __name__ == "__main__":
    test_analysis()