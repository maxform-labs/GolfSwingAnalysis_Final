#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POV 중심 위치를 캘리브레이션 이미지에 표시하는 스크립트
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

def load_calibration_data():
    """캘리브레이션 데이터 로드"""
    with open('manual_calibration_470mm.json', 'r', encoding='utf-8') as f:
        calibration_data = json.load(f)
    return calibration_data

def calculate_pov_center_in_image(calibration_data):
    """이미지 좌표계에서 POV 중심 위치 계산"""
    
    # 카메라 파라미터
    camera_matrix_1 = np.array(calibration_data['camera_matrix_1'])
    camera_matrix_2 = np.array(calibration_data['camera_matrix_2'])
    translation_vector = np.array(calibration_data['translation_vector'])
    
    # 3D 공간에서의 카메라 위치
    cam1_pos_3d = np.array([0, 0, 0])  # 기준점
    cam2_pos_3d = translation_vector   # [247, 400, 0]
    
    # POV 중심 (3D 공간)
    pov_center_3d = (cam1_pos_3d + cam2_pos_3d) / 2  # [123.5, 200.0, 0.0]
    
    # 각 카메라에서 POV 중심의 이미지 좌표 계산
    # 카메라1 (기준점이므로 단순 변환)
    pov_cam1_3d = pov_center_3d - cam1_pos_3d  # [123.5, 200.0, 0.0]
    
    # 카메라2 (상대적 위치)
    pov_cam2_3d = pov_center_3d - cam2_pos_3d  # [-123.5, -200.0, 0.0]
    
    # 이미지 좌표로 변환 (단순화된 투영)
    # 실제로는 카메라 내부 파라미터를 사용해야 하지만, 
    # 여기서는 이미지 중심 기준으로 근사 계산
    
    image_size = calibration_data['image_size']  # [1440, 300]
    focal_length = calibration_data['focal_length']  # 1440
    
    # 카메라1에서의 이미지 좌표 (POV가 카메라 앞에 있다고 가정)
    # Z 거리를 1000mm로 가정 (실제 골프공 권장 거리)
    z_distance = 1000.0  # mm
    
    # 카메라1 이미지 좌표
    pov_cam1_x = (pov_cam1_3d[0] / z_distance) * focal_length + image_size[0] / 2
    pov_cam1_y = (pov_cam1_3d[1] / z_distance) * focal_length + image_size[1] / 2
    
    # 카메라2 이미지 좌표
    pov_cam2_x = (pov_cam2_3d[0] / z_distance) * focal_length + image_size[0] / 2
    pov_cam2_y = (pov_cam2_3d[1] / z_distance) * focal_length + image_size[1] / 2
    
    return {
        'cam1': (int(pov_cam1_x), int(pov_cam1_y)),
        'cam2': (int(pov_cam2_x), int(pov_cam2_y)),
        'pov_3d': pov_center_3d,
        'z_distance': z_distance
    }

def mark_pov_center_on_image(image_path, pov_coords, output_path):
    """이미지에 POV 중심 위치 표시"""
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return False
    
    # 카메라 구분 (파일명에서)
    if 'Cam1' in image_path:
        pov_x, pov_y = pov_coords['cam1']
        camera_name = "Cam1"
    elif 'Cam2' in image_path:
        pov_x, pov_y = pov_coords['cam2']
        camera_name = "Cam2"
    else:
        print(f"카메라 정보를 찾을 수 없습니다: {image_path}")
        return False
    
    # 이미지 크기 확인 및 좌표 조정
    height, width = image.shape[:2]
    
    # 좌표가 이미지 범위 내에 있는지 확인
    if 0 <= pov_x < width and 0 <= pov_y < height:
        # 빨간색 원으로 POV 중심 표시
        cv2.circle(image, (pov_x, pov_y), 15, (0, 0, 255), 3)  # 빨간색 원
        cv2.circle(image, (pov_x, pov_y), 5, (0, 0, 255), -1)   # 빨간색 점
        
        # POV 라벨 추가
        label = f"POV Center ({camera_name})"
        cv2.putText(image, label, (pov_x + 20, pov_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 좌표 정보 추가
        coord_text = f"({pov_x}, {pov_y})"
        cv2.putText(image, coord_text, (pov_x + 20, pov_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        print(f"POV 중심 표시: {camera_name} - ({pov_x}, {pov_y})")
    else:
        print(f"POV 좌표가 이미지 범위를 벗어남: {camera_name} - ({pov_x}, {pov_y})")
        # 이미지 중심에 표시
        pov_x, pov_y = width // 2, height // 2
        cv2.circle(image, (pov_x, pov_y), 15, (0, 0, 255), 3)
        cv2.circle(image, (pov_x, pov_y), 5, (0, 0, 255), -1)
        label = f"POV Center ({camera_name}) - Approximate"
        cv2.putText(image, label, (pov_x + 20, pov_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 이미지 저장
    cv2.imwrite(output_path, image)
    print(f"저장 완료: {output_path}")
    return True

def process_calibration_images():
    """캘리브레이션 이미지들에 POV 중심 표시"""
    
    # 캘리브레이션 데이터 로드
    calibration_data = load_calibration_data()
    
    # POV 중심 좌표 계산
    pov_coords = calculate_pov_center_in_image(calibration_data)
    
    print("=== POV 중심 위치 정보 ===")
    print(f"3D POV 중심: {pov_coords['pov_3d']}")
    print(f"가정된 Z 거리: {pov_coords['z_distance']}mm")
    print(f"Cam1 이미지 좌표: {pov_coords['cam1']}")
    print(f"Cam2 이미지 좌표: {pov_coords['cam2']}")
    print()
    
    # pov_center 폴더 생성
    output_dir = Path('../pov_center')
    output_dir.mkdir(exist_ok=True)
    
    # 캘리브레이션 이미지 폴더
    calibration_dir = Path('data/calibrations')
    
    # 처리할 이미지 파일들 (5개 선택)
    image_files = [
        '2025-09-23_14-03-38_Cam1.bmp',
        '2025-09-23_14-03-38_Cam2.bmp',
        '2025-09-23_14-04-03_Cam1.bmp',
        '2025-09-23_14-04-03_Cam2.bmp',
        '2025-09-23_14-04-08_Cam1.bmp'
    ]
    
    processed_count = 0
    
    for image_file in image_files:
        input_path = calibration_dir / image_file
        output_path = output_dir / f"pov_marked_{image_file}"
        
        if input_path.exists():
            print(f"처리 중: {image_file}")
            if mark_pov_center_on_image(str(input_path), pov_coords, str(output_path)):
                processed_count += 1
        else:
            print(f"파일을 찾을 수 없습니다: {input_path}")
    
    print(f"\n=== 처리 완료 ===")
    print(f"총 {processed_count}개 이미지 처리 완료")
    print(f"결과 저장 위치: {output_dir.absolute()}")
    
    return processed_count

def create_pov_visualization():
    """POV 중심 위치 시각화 생성"""
    
    # 캘리브레이션 데이터 로드
    calibration_data = load_calibration_data()
    pov_coords = calculate_pov_center_in_image(calibration_data)
    
    # 시각화 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 3D 공간에서의 POV 중심
    ax1.scatter(0, 0, color='blue', s=100, label='Cam1')
    ax1.scatter(247, 400, color='green', s=100, label='Cam2')
    ax1.scatter(123.5, 200, color='red', s=150, label='POV Center')
    
    # 카메라 간 연결선
    ax1.plot([0, 247], [0, 400], 'k--', alpha=0.5, label='Baseline (470mm)')
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('3D Space: Camera Positions and POV Center')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 이미지 좌표에서의 POV 중심
    image_size = calibration_data['image_size']
    
    # 카메라1 이미지
    ax2.scatter(pov_coords['cam1'][0], pov_coords['cam1'][1], 
               color='red', s=150, label='POV Center (Cam1)')
    ax2.set_xlim(0, image_size[0])
    ax2.set_ylim(image_size[1], 0)  # 이미지 좌표계 (Y축 뒤집기)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_title('Image Coordinates: POV Center Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    output_path = '../pov_center/pov_center_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"POV 시각화 저장: {output_path}")
    
    return output_path

if __name__ == "__main__":
    print("POV 중심 위치 표시 시작...")
    
    # 캘리브레이션 이미지 처리
    processed_count = process_calibration_images()
    
    # POV 시각화 생성
    viz_path = create_pov_visualization()
    
    print(f"\n=== 최종 결과 ===")
    print(f"처리된 이미지: {processed_count}개")
    print(f"시각화 파일: {viz_path}")
    print("모든 파일이 '../pov_center/' 폴더에 저장되었습니다.")
