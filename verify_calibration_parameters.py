#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
캘리브레이션 파라미터 재검증
현재 캘리브레이션 파라미터의 정확성 검증
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json

def load_calibration_data():
    """캘리브레이션 데이터 로드"""
    
    with open('manual_calibration_470mm.json', 'r', encoding='utf-8') as f:
        calibration_data = json.load(f)
    
    return calibration_data

def analyze_calibration_parameters():
    """캘리브레이션 파라미터 분석"""
    
    print("=== 캘리브레이션 파라미터 재검증 ===")
    
    # 캘리브레이션 데이터 로드
    calibration_data = load_calibration_data()
    
    print("현재 캘리브레이션 파라미터:")
    print("-" * 50)
    print(f"  - 초점거리: {calibration_data['focal_length']}px")
    print(f"  - 베이스라인: {calibration_data['baseline']}mm")
    print(f"  - 이미지 크기: {calibration_data['image_size']}")
    print(f"  - 카메라1 주점: ({calibration_data['camera_matrix_1'][0][2]}, {calibration_data['camera_matrix_1'][1][2]})")
    print(f"  - 카메라2 주점: ({calibration_data['camera_matrix_2'][0][2]}, {calibration_data['camera_matrix_2'][1][2]})")
    print(f"  - 회전 행렬: {calibration_data['rotation_matrix']}")
    print(f"  - 이동 벡터: {calibration_data['translation_vector']}")
    
    # 파라미터 검증
    print("\n파라미터 검증:")
    print("-" * 50)
    
    # 1. 초점거리 검증
    focal_length = calibration_data['focal_length']
    image_width = calibration_data['image_size'][0]
    image_height = calibration_data['image_size'][1]
    
    print(f"1. 초점거리 검증:")
    print(f"   - 현재 초점거리: {focal_length}px")
    print(f"   - 이미지 크기: {image_width}x{image_height}")
    print(f"   - 초점거리/이미지폭 비율: {focal_length/image_width:.3f}")
    print(f"   - 초점거리/이미지높이 비율: {focal_length/image_height:.3f}")
    
    if focal_length > image_width or focal_length > image_height:
        print("   WARNING: 초점거리가 이미지 크기보다 큽니다!")
    
    # 2. 주점 검증
    principal_point_x1 = calibration_data['camera_matrix_1'][0][2]
    principal_point_y1 = calibration_data['camera_matrix_1'][1][2]
    principal_point_x2 = calibration_data['camera_matrix_2'][0][2]
    principal_point_y2 = calibration_data['camera_matrix_2'][1][2]
    
    print(f"\n2. 주점 검증:")
    print(f"   - 카메라1 주점: ({principal_point_x1}, {principal_point_y1})")
    print(f"   - 카메라2 주점: ({principal_point_x2}, {principal_point_y2})")
    print(f"   - 이미지 중심: ({image_width/2}, {image_height/2})")
    
    if abs(principal_point_x1 - image_width/2) > image_width * 0.1:
        print("   WARNING: 카메라1 주점이 이미지 중심에서 멉니다!")
    if abs(principal_point_y1 - image_height/2) > image_height * 0.1:
        print("   WARNING: 카메라1 주점이 이미지 중심에서 멉니다!")
    
    # 3. 베이스라인 검증
    baseline = calibration_data['baseline']
    print(f"\n3. 베이스라인 검증:")
    print(f"   - 현재 베이스라인: {baseline}mm")
    print(f"   - 실제 측정값: 470mm")
    print(f"   - 오차: {abs(baseline - 470):.1f}mm")
    
    if abs(baseline - 470) > 10:
        print("   WARNING: 베이스라인 오차가 큽니다!")
    
    # 4. 회전/이동 검증
    rotation_matrix = np.array(calibration_data['rotation_matrix'])
    translation_vector = np.array(calibration_data['translation_vector'])
    
    print(f"\n4. 회전/이동 검증:")
    print(f"   - 회전 행렬: {rotation_matrix}")
    print(f"   - 이동 벡터: {translation_vector}")
    print(f"   - 이동 벡터 크기: {np.linalg.norm(translation_vector):.1f}mm")
    
    # 5. 해상도 불일치 검증
    print(f"\n5. 해상도 불일치 검증:")
    print(f"   - 캘리브레이션 이미지: 1440x1080")
    print(f"   - 샷 데이터 이미지: 1440x300")
    print(f"   - Y축 스케일 팩터: {300/1080:.3f}")
    print(f"   - 스케일된 초점거리: {focal_length * 300/1080:.1f}px")
    print(f"   - 스케일된 주점 Y: {principal_point_y1 * 300/1080:.1f}px")
    
    return calibration_data

def suggest_calibration_improvements(calibration_data):
    """캘리브레이션 개선 제안"""
    
    print("\n=== 캘리브레이션 개선 제안 ===")
    
    # 해상도 스케일링 적용
    scale_factor_y = 300 / 1080  # 0.278
    
    print("1. 해상도 스케일링 적용:")
    print(f"   - Y축 스케일 팩터: {scale_factor_y:.3f}")
    print(f"   - 스케일된 초점거리: {calibration_data['focal_length'] * scale_factor_y:.1f}px")
    print(f"   - 스케일된 주점 Y: {calibration_data['camera_matrix_1'][1][2] * scale_factor_y:.1f}px")
    
    # 개선된 캘리브레이션 파라미터 생성
    improved_calibration = calibration_data.copy()
    improved_calibration['focal_length'] = calibration_data['focal_length'] * scale_factor_y
    improved_calibration['camera_matrix_1'][1][1] = improved_calibration['focal_length']
    improved_calibration['camera_matrix_1'][1][2] = calibration_data['camera_matrix_1'][1][2] * scale_factor_y
    improved_calibration['camera_matrix_2'][1][1] = improved_calibration['focal_length']
    improved_calibration['camera_matrix_2'][1][2] = calibration_data['camera_matrix_2'][1][2] * scale_factor_y
    improved_calibration['image_size'] = [1440, 300]
    improved_calibration['scaling_applied'] = True
    improved_calibration['scale_factor_y'] = scale_factor_y
    
    print("\n2. 개선된 캘리브레이션 파라미터:")
    print(f"   - 초점거리: {improved_calibration['focal_length']:.1f}px")
    print(f"   - 이미지 크기: {improved_calibration['image_size']}")
    print(f"   - 카메라1 주점: ({improved_calibration['camera_matrix_1'][0][2]}, {improved_calibration['camera_matrix_1'][1][2]})")
    print(f"   - 카메라2 주점: ({improved_calibration['camera_matrix_2'][0][2]}, {improved_calibration['camera_matrix_2'][1][2]})")
    
    # 개선된 캘리브레이션 저장
    with open('improved_calibration_470mm.json', 'w', encoding='utf-8') as f:
        json.dump(improved_calibration, f, indent=2, ensure_ascii=False)
    
    print(f"\n개선된 캘리브레이션 저장: improved_calibration_470mm.json")
    
    return improved_calibration

def test_improved_calibration():
    """개선된 캘리브레이션 테스트"""
    
    print("\n=== 개선된 캘리브레이션 테스트 ===")
    
    # 개선된 캘리브레이션 로드
    with open('improved_calibration_470mm.json', 'r', encoding='utf-8') as f:
        improved_calibration = json.load(f)
    
    # 테스트 시차 계산
    test_disparities = [10, 20, 30, 40, 50]  # 픽셀
    baseline = improved_calibration['baseline']
    focal_length = improved_calibration['focal_length']
    
    print("시차-깊이 관계 테스트:")
    print("-" * 50)
    print("시차(px) | 깊이(mm) | 깊이(m)")
    print("-" * 50)
    
    for disparity in test_disparities:
        depth = (baseline * focal_length) / disparity
        print(f"{disparity:8d} | {depth:8.1f} | {depth/1000:.2f}")
    
    # 실제 골프공 거리 검증
    print(f"\n실제 골프공 거리 검증:")
    print(f"  - 실제 거리: 500mm")
    print(f"  - 필요한 시차: {baseline * focal_length / 500:.1f}px")
    print(f"  - 시차 범위: 10-50px")
    print(f"  - 예상 깊이 범위: {baseline * focal_length / 50:.1f}mm ~ {baseline * focal_length / 10:.1f}mm")
    
    return improved_calibration

def create_calibration_verification_visualization(calibration_data, improved_calibration):
    """캘리브레이션 검증 시각화"""
    
    print("\n=== 캘리브레이션 검증 시각화 생성 ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 시차-깊이 관계
    disparities = np.linspace(5, 100, 100)
    baseline = calibration_data['baseline']
    focal_length = calibration_data['focal_length']
    improved_focal_length = improved_calibration['focal_length']
    
    depths_original = (baseline * focal_length) / disparities
    depths_improved = (baseline * improved_focal_length) / disparities
    
    axes[0,0].plot(disparities, depths_original, 'r-', label='Original', linewidth=2)
    axes[0,0].plot(disparities, depths_improved, 'b-', label='Improved', linewidth=2)
    axes[0,0].axhline(y=500, color='g', linestyle='--', label='Actual Distance (500mm)')
    axes[0,0].set_xlabel('Disparity (px)')
    axes[0,0].set_ylabel('Depth (mm)')
    axes[0,0].set_title('Disparity-Depth Relationship')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 초점거리 비교
    methods = ['Original', 'Improved']
    focal_lengths = [focal_length, improved_focal_length]
    axes[0,1].bar(methods, focal_lengths, color=['red', 'blue'])
    axes[0,1].set_ylabel('Focal Length (px)')
    axes[0,1].set_title('Focal Length Comparison')
    for i, v in enumerate(focal_lengths):
        axes[0,1].text(i, v + 10, f'{v:.1f}', ha='center')
    
    # 주점 비교
    principal_points = [
        (calibration_data['camera_matrix_1'][0][2], calibration_data['camera_matrix_1'][1][2]),
        (improved_calibration['camera_matrix_1'][0][2], improved_calibration['camera_matrix_1'][1][2])
    ]
    
    axes[1,0].plot([p[0] for p in principal_points], [p[1] for p in principal_points], 'ro-', markersize=8)
    axes[1,0].set_xlabel('Principal Point X (px)')
    axes[1,0].set_ylabel('Principal Point Y (px)')
    axes[1,0].set_title('Principal Point Comparison')
    axes[1,0].grid(True)
    
    # 이미지 크기 비교
    image_sizes = [
        calibration_data['image_size'],
        improved_calibration['image_size']
    ]
    
    axes[1,1].bar(['Original', 'Improved'], [size[0] for size in image_sizes], color=['red', 'blue'], alpha=0.7, label='Width')
    axes[1,1].bar(['Original', 'Improved'], [size[1] for size in image_sizes], color=['orange', 'cyan'], alpha=0.7, label='Height')
    axes[1,1].set_ylabel('Image Size (px)')
    axes[1,1].set_title('Image Size Comparison')
    axes[1,1].legend()
    
    plt.tight_layout()
    
    # 저장
    output_path = '../ball_detect/calibration_verification_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"캘리브레이션 검증 시각화 저장: {output_path}")
    
    return output_path

if __name__ == "__main__":
    print("캘리브레이션 파라미터 재검증 시작...")
    
    # 캘리브레이션 파라미터 분석
    calibration_data = analyze_calibration_parameters()
    
    # 캘리브레이션 개선 제안
    improved_calibration = suggest_calibration_improvements(calibration_data)
    
    # 개선된 캘리브레이션 테스트
    test_improved_calibration()
    
    # 캘리브레이션 검증 시각화
    viz_path = create_calibration_verification_visualization(calibration_data, improved_calibration)
    
    print(f"\n=== 캘리브레이션 파라미터 재검증 완료 ===")
    print(f"검증 시각화: {viz_path}")
    print(f"개선된 캘리브레이션: improved_calibration_470mm.json")
