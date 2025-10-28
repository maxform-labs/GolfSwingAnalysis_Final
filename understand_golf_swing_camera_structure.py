#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
골프 스윙 카메라 촬영 구조 이해
실제 카메라 높이와 촬영 구조 분석
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_golf_swing_camera_structure():
    """골프 스윙 카메라 촬영 구조 분석"""
    
    print("=== 골프 스윙 카메라 촬영 구조 분석 ===")
    
    # 실제 카메라 높이 (사용자 제공)
    bottom_camera_height = 550  # mm (50-60cm 중간값)
    top_camera_height = 950      # mm (90-100cm 중간값)
    
    print(f"실제 카메라 높이:")
    print(f"  하단 카메라: {bottom_camera_height}mm ({bottom_camera_height/100:.1f}cm)")
    print(f"  상단 카메라: {top_camera_height}mm ({top_camera_height/100:.1f}cm)")
    print(f"  높이 차이: {top_camera_height - bottom_camera_height}mm")
    print()
    
    # 골프 스윙 촬영 구조 분석
    print("골프 스윙 촬영 구조 분석:")
    print("-" * 50)
    
    # 1. 골프 스윙 영역
    swing_area_length = 2000  # mm (2m)
    swing_area_width = 1000   # mm (1m)
    
    print(f"1. 골프 스윙 영역:")
    print(f"  길이: {swing_area_length}mm (2m)")
    print(f"  너비: {swing_area_width}mm (1m)")
    print()
    
    # 2. 카메라 배치
    # 하단 카메라: 골프공 위치에서 측면
    # 상단 카메라: 골프공 위치에서 위쪽
    camera_distance = 500  # mm (카메라와 골프공 거리)
    
    print(f"2. 카메라 배치:")
    print(f"  카메라-골프공 거리: {camera_distance}mm")
    print(f"  하단 카메라: 측면에서 촬영")
    print(f"  상단 카메라: 위쪽에서 촬영")
    print()
    
    # 3. 촬영 각도 분석
    # 하단 카메라: 수평에서 약간 위쪽
    # 상단 카메라: 위에서 아래쪽
    bottom_angle = np.arctan2(top_camera_height - bottom_camera_height, camera_distance) * 180 / np.pi
    top_angle = np.arctan2(bottom_camera_height, camera_distance) * 180 / np.pi
    
    print(f"3. 촬영 각도:")
    print(f"  하단 카메라 각도: {bottom_angle:.1f}도 (수평에서 위쪽)")
    print(f"  상단 카메라 각도: {top_angle:.1f}도 (위에서 아래쪽)")
    print()
    
    # 4. 스테레오 비전 구조
    # 두 카메라가 서로 다른 각도에서 같은 지점을 촬영
    stereo_angle = bottom_angle + top_angle
    
    print(f"4. 스테레오 비전 구조:")
    print(f"  스테레오 각도: {stereo_angle:.1f}도")
    print(f"  베이스라인: {np.sqrt(camera_distance**2 + (top_camera_height - bottom_camera_height)**2):.0f}mm")
    print()
    
    return {
        'bottom_height': bottom_camera_height,
        'top_height': top_camera_height,
        'camera_distance': camera_distance,
        'bottom_angle': bottom_angle,
        'top_angle': top_angle,
        'stereo_angle': stereo_angle
    }

def create_golf_swing_camera_structure_diagram():
    """골프 스윙 카메라 구조 다이어그램 생성"""
    
    print("=== 골프 스윙 카메라 구조 다이어그램 생성 ===")
    
    # 분석 결과
    analysis = analyze_golf_swing_camera_structure()
    
    # 시각화
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 측면 뷰 (X-Z 평면)
    ax1 = plt.subplot(2, 2, 1)
    
    # 골프 스윙 영역
    swing_area = plt.Rectangle((0, 0), 2000, 1000, 
                              fill=False, edgecolor='gray', linestyle='--', 
                              linewidth=2, alpha=0.7, label='Golf Swing Area')
    ax1.add_patch(swing_area)
    
    # 카메라 위치
    ax1.scatter(500, analysis['bottom_height'], s=200, c='blue', marker='s', 
                label='Bottom Camera (측면)', zorder=5)
    ax1.scatter(500, analysis['top_height'], s=200, c='red', marker='s', 
                label='Top Camera (위쪽)', zorder=5)
    
    # 골프공 위치
    ax1.scatter(1000, 500, s=150, c='orange', marker='o', 
                label='Golf Ball', zorder=5)
    
    # 촬영 각도 선
    ax1.plot([500, 1000], [analysis['bottom_height'], 500], 
             'b-', linewidth=2, alpha=0.7, label='Bottom Camera View')
    ax1.plot([500, 1000], [analysis['top_height'], 500], 
             'r-', linewidth=2, alpha=0.7, label='Top Camera View')
    
    ax1.set_xlim(0, 2500)
    ax1.set_ylim(0, 1200)
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Z Position (mm)')
    ax1.set_title('Golf Swing Camera Structure (Side View)\nX-Z Plane')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. 위에서 본 뷰 (X-Y 평면)
    ax2 = plt.subplot(2, 2, 2)
    
    # 골프 스윙 영역
    swing_area_top = plt.Rectangle((0, 0), 2000, 1000, 
                                  fill=False, edgecolor='gray', linestyle='--', 
                                  linewidth=2, alpha=0.7, label='Golf Swing Area')
    ax2.add_patch(swing_area_top)
    
    # 카메라 위치 (위에서 본 뷰)
    ax2.scatter(500, 500, s=200, c='blue', marker='s', 
                label='Bottom Camera', zorder=5)
    ax2.scatter(500, 500, s=200, c='red', marker='s', 
                label='Top Camera', zorder=5)
    
    # 골프공 위치
    ax2.scatter(1000, 500, s=150, c='orange', marker='o', 
                label='Golf Ball', zorder=5)
    
    # 촬영 각도 선
    ax2.plot([500, 1000], [500, 500], 
             'b-', linewidth=2, alpha=0.7, label='Bottom Camera View')
    ax2.plot([500, 1000], [500, 500], 
             'r-', linewidth=2, alpha=0.7, label='Top Camera View')
    
    ax2.set_xlim(0, 2500)
    ax2.set_ylim(0, 1000)
    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel('Y Position (mm)')
    ax2.set_title('Golf Swing Camera Structure (Top View)\nX-Y Plane')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 3. 3D 뷰
    ax3 = plt.subplot(2, 2, 3, projection='3d')
    
    # 카메라 위치 (3D)
    ax3.scatter(500, 500, analysis['bottom_height'], s=200, c='blue', marker='s', 
                label='Bottom Camera', zorder=5)
    ax3.scatter(500, 500, analysis['top_height'], s=200, c='red', marker='s', 
                label='Top Camera', zorder=5)
    
    # 골프공 위치
    ax3.scatter(1000, 500, 500, s=150, c='orange', marker='o', 
                label='Golf Ball', zorder=5)
    
    # 촬영 각도 선
    ax3.plot([500, 1000], [500, 500], [analysis['bottom_height'], 500], 
             'b-', linewidth=2, alpha=0.7, label='Bottom Camera View')
    ax3.plot([500, 1000], [500, 500], [analysis['top_height'], 500], 
             'r-', linewidth=2, alpha=0.7, label='Top Camera View')
    
    ax3.set_xlim(0, 2500)
    ax3.set_ylim(0, 1000)
    ax3.set_zlim(0, 1200)
    ax3.set_xlabel('X Position (mm)')
    ax3.set_ylabel('Y Position (mm)')
    ax3.set_zlabel('Z Position (mm)')
    ax3.set_title('Golf Swing Camera Structure (3D View)')
    ax3.legend(loc='upper right')
    
    # 4. 촬영 각도 분석
    ax4 = plt.subplot(2, 2, 4)
    
    angles = ['Bottom\nCamera', 'Top\nCamera', 'Stereo\nAngle']
    angle_values = [analysis['bottom_angle'], analysis['top_angle'], analysis['stereo_angle']]
    colors = ['blue', 'red', 'green']
    
    bars = ax4.bar(angles, angle_values, color=colors, alpha=0.7)
    ax4.set_ylabel('Angle (degrees)')
    ax4.set_title('Camera Viewing Angles')
    ax4.grid(True, alpha=0.3)
    
    for bar, angle in zip(bars, angle_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{angle:.1f}°', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 저장
    output_path = '../ball_detect/golf_swing_camera_structure.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"골프 스윙 카메라 구조 다이어그램 저장: {output_path}")
    
    return output_path

def create_golf_swing_analysis_report():
    """골프 스윙 촬영 구조 분석 보고서 생성"""
    
    # 분석 결과
    analysis = analyze_golf_swing_camera_structure()
    
    report_content = f"""# 골프 스윙 카메라 촬영 구조 분석 보고서

## 📊 실제 카메라 설정

### 카메라 높이
- **하단 카메라**: {analysis['bottom_height']}mm ({analysis['bottom_height']/100:.1f}cm)
- **상단 카메라**: {analysis['top_height']}mm ({analysis['top_height']/100:.1f}cm)
- **높이 차이**: {analysis['top_height'] - analysis['bottom_height']}mm

### 촬영 구조
- **카메라-골프공 거리**: {analysis['camera_distance']}mm
- **하단 카메라**: 측면에서 촬영 (수평에서 위쪽)
- **상단 카메라**: 위쪽에서 촬영 (위에서 아래쪽)

## 📐 촬영 각도 분석

### 카메라별 촬영 각도
- **하단 카메라 각도**: {analysis['bottom_angle']:.1f}도 (수평에서 위쪽)
- **상단 카메라 각도**: {analysis['top_angle']:.1f}도 (위에서 아래쪽)
- **스테레오 각도**: {analysis['stereo_angle']:.1f}도

### 스테레오 비전 구조
- **베이스라인**: {np.sqrt(analysis['camera_distance']**2 + (analysis['top_height'] - analysis['bottom_height'])**2):.0f}mm
- **촬영 방식**: 서로 다른 각도에서 같은 지점 촬영
- **3D 복원**: 스테레오 비전으로 3D 좌표 계산

## 🎯 골프 스윙 촬영 목적

### 촬영 영역
- **길이**: 2000mm (2m)
- **너비**: 1000mm (1m)
- **목적**: 골프공의 3D 궤적 추적

### 촬영 구조의 장점
1. **다각도 촬영**: 서로 다른 각도에서 촬영
2. **3D 복원**: 스테레오 비전으로 정확한 3D 좌표
3. **궤적 추적**: 골프공의 움직임을 3D로 추적
4. **정확도**: 높은 정확도의 3D 측정

## 📁 생성된 파일
- **구조 다이어그램**: ball_detect/golf_swing_camera_structure.png
- **분석 보고서**: 이 문서

## ✅ 결론
골프 스윙 촬영을 위한 스테레오 카메라 구조입니다.
서로 다른 높이와 각도에서 촬영하여 골프공의 3D 궤적을 정확하게 추적할 수 있습니다.

---
*구조 분석 완료: 2025-10-20*
"""
    
    # 보고서 저장
    with open('../ball_detect/golf_swing_camera_structure_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("골프 스윙 촬영 구조 분석 보고서 저장: ../ball_detect/golf_swing_camera_structure_report.md")
    return '../ball_detect/golf_swing_camera_structure_report.md'

if __name__ == "__main__":
    print("골프 스윙 카메라 촬영 구조 분석 시작...")
    
    # 골프 스윙 카메라 구조 분석
    analysis = analyze_golf_swing_camera_structure()
    
    # 골프 스윙 카메라 구조 다이어그램
    diagram_path = create_golf_swing_camera_structure_diagram()
    
    # 골프 스윙 촬영 구조 분석 보고서
    report_path = create_golf_swing_analysis_report()
    
    print(f"\n=== 골프 스윙 카메라 촬영 구조 분석 완료 ===")
    print(f"구조 다이어그램: {diagram_path}")
    print(f"분석 보고서: {report_path}")
    print(f"\n골프 스윙 촬영 구조:")
    print(f"  - 하단 카메라: {analysis['bottom_height']}mm (측면 촬영)")
    print(f"  - 상단 카메라: {analysis['top_height']}mm (위쪽 촬영)")
    print(f"  - 촬영 각도: {analysis['stereo_angle']:.1f}도")
    print(f"  - 목적: 골프공 3D 궤적 추적")
    print(f"  - 구조: 스테레오 비전")
