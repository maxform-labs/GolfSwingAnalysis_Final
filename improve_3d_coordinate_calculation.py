#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D 좌표 계산 공식 수정
시차 계산 개선 및 3D 좌표 계산 정확도 향상
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

def detect_golf_ball_advanced(image_path):
    """고급 골프공 검출"""
    
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 다중 스케일 분석
    scales = [0.8, 1.0, 1.2, 1.5]
    best_detection = None
    best_score = 0
    
    for scale in scales:
        if scale != 1.0:
            h, w = gray.shape
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_gray = cv2.resize(gray, (new_w, new_h))
        else:
            scaled_gray = gray.copy()
        
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(scaled_gray)
        
        # 다중 임계값 처리
        thresholds = [160, 180, 200, 220]
        
        for thresh_val in thresholds:
            white_mask = cv2.inRange(enhanced, thresh_val, 255)
            
            # 모폴로지 연산
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            
            # 컨투어 분석
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50 or area > 8000:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.5:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    if 5 <= radius <= 60:
                        center_x, center_y = scaled_gray.shape[1] // 2, scaled_gray.shape[0] // 2
                        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        position_score = max(0, 1 - distance_from_center / (scaled_gray.shape[1] * 0.6))
                        
                        score = circularity * area * position_score * (1 + circularity)
                        
                        if score > best_score:
                            best_score = score
                            if scale != 1.0:
                                x = x / scale
                                y = y / scale
                                radius = radius / scale
                            best_detection = (int(x), int(y), int(radius))
    
    if best_detection is not None:
        return (best_detection[0], best_detection[1]), best_detection[2], img
    else:
        return None, None, img

def calculate_3d_position_improved(ball_pos_cam1, ball_pos_cam2, calibration_data):
    """개선된 3D 위치 계산"""
    
    if ball_pos_cam1 is None or ball_pos_cam2 is None:
        return None
    
    # 캘리브레이션 파라미터
    focal_length = calibration_data['focal_length']
    baseline = calibration_data['baseline']
    principal_point_x = calibration_data['camera_matrix_1'][0][2]
    principal_point_y = calibration_data['camera_matrix_1'][1][2]
    
    # 개선된 시차 계산 (부호 고려)
    disparity = ball_pos_cam1[0] - ball_pos_cam2[0]
    
    if abs(disparity) < 1:  # 시차가 너무 작으면 무시
        return None
    
    # 깊이 계산
    depth = (baseline * focal_length) / abs(disparity)
    
    # 깊이 범위 검증 (0.5m ~ 5m)
    if depth < 500 or depth > 5000:
        return None
    
    # 3D 좌표 계산 (개선된 공식)
    # X 좌표: 카메라 1 기준
    x_3d = (ball_pos_cam1[0] - principal_point_x) * depth / focal_length
    
    # Y 좌표: 카메라 1 기준
    y_3d = (ball_pos_cam1[1] - principal_point_y) * depth / focal_length
    
    # Z 좌표: 깊이
    z_3d = depth
    
    return np.array([x_3d, y_3d, z_3d])

def calculate_3d_position_alternative(ball_pos_cam1, ball_pos_cam2, calibration_data):
    """대안 3D 위치 계산 (카메라 2 기준)"""
    
    if ball_pos_cam1 is None or ball_pos_cam2 is None:
        return None
    
    # 캘리브레이션 파라미터
    focal_length = calibration_data['focal_length']
    baseline = calibration_data['baseline']
    principal_point_x = calibration_data['camera_matrix_2'][0][2]
    principal_point_y = calibration_data['camera_matrix_2'][1][2]
    
    # 시차 계산 (카메라 2 기준)
    disparity = ball_pos_cam2[0] - ball_pos_cam1[0]
    
    if abs(disparity) < 1:
        return None
    
    # 깊이 계산
    depth = (baseline * focal_length) / abs(disparity)
    
    # 깊이 범위 검증
    if depth < 500 or depth > 5000:
        return None
    
    # 3D 좌표 계산 (카메라 2 기준)
    x_3d = (ball_pos_cam2[0] - principal_point_x) * depth / focal_length
    y_3d = (ball_pos_cam2[1] - principal_point_y) * depth / focal_length
    z_3d = depth
    
    return np.array([x_3d, y_3d, z_3d])

def calculate_3d_position_stereo(ball_pos_cam1, ball_pos_cam2, calibration_data):
    """스테레오 비전 3D 위치 계산"""
    
    if ball_pos_cam1 is None or ball_pos_cam2 is None:
        return None
    
    # 캘리브레이션 파라미터
    focal_length = calibration_data['focal_length']
    baseline = calibration_data['baseline']
    principal_point_x1 = calibration_data['camera_matrix_1'][0][2]
    principal_point_y1 = calibration_data['camera_matrix_1'][1][2]
    principal_point_x2 = calibration_data['camera_matrix_2'][0][2]
    principal_point_y2 = calibration_data['camera_matrix_2'][1][2]
    
    # 시차 계산
    disparity = ball_pos_cam1[0] - ball_pos_cam2[0]
    
    if abs(disparity) < 1:
        return None
    
    # 깊이 계산
    depth = (baseline * focal_length) / abs(disparity)
    
    # 깊이 범위 검증
    if depth < 500 or depth > 5000:
        return None
    
    # 3D 좌표 계산 (두 카메라 평균)
    x_3d_1 = (ball_pos_cam1[0] - principal_point_x1) * depth / focal_length
    y_3d_1 = (ball_pos_cam1[1] - principal_point_y1) * depth / focal_length
    
    x_3d_2 = (ball_pos_cam2[0] - principal_point_x2) * depth / focal_length
    y_3d_2 = (ball_pos_cam2[1] - principal_point_y2) * depth / focal_length
    
    # 평균 계산
    x_3d = (x_3d_1 + x_3d_2) / 2
    y_3d = (y_3d_1 + y_3d_2) / 2
    z_3d = depth
    
    return np.array([x_3d, y_3d, z_3d])

def analyze_3d_coordinate_calculation():
    """3D 좌표 계산 분석"""
    
    print("=== 3D 좌표 계산 공식 수정 ===")
    
    # 캘리브레이션 데이터 로드
    calibration_data = load_calibration_data()
    print(f"캘리브레이션 데이터 로드 완료")
    
    # 이미지 디렉토리
    image_dir = 'data/swingData/5Iron_0930/1'
    
    # 골프공 위치 검출
    cam1_positions = []
    cam2_positions = []
    
    print("\n골프공 위치 검출:")
    print("-" * 50)
    
    for i in range(1, 11):
        # Cam1 검출
        cam1_path = os.path.join(image_dir, f'Gamma_1_{i}.bmp')
        ball_pos_cam1, radius_cam1, _ = detect_golf_ball_advanced(cam1_path)
        
        # Cam2 검출
        cam2_path = os.path.join(image_dir, f'Gamma_2_{i}.bmp')
        ball_pos_cam2, radius_cam2, _ = detect_golf_ball_advanced(cam2_path)
        
        if ball_pos_cam1 is not None and ball_pos_cam2 is not None:
            cam1_positions.append(ball_pos_cam1)
            cam2_positions.append(ball_pos_cam2)
            print(f"  Frame {i}: Cam1({ball_pos_cam1}), Cam2({ball_pos_cam2})")
        else:
            cam1_positions.append(None)
            cam2_positions.append(None)
            print(f"  Frame {i}: 검출 실패")
    
    # 3D 위치 계산 비교
    print("\n3D 위치 계산 비교:")
    print("-" * 50)
    
    positions_original = []
    positions_improved = []
    positions_alternative = []
    positions_stereo = []
    
    for i in range(10):
        if cam1_positions[i] is not None and cam2_positions[i] is not None:
            # 원본 방식
            disparity_orig = abs(cam1_positions[i][0] - cam2_positions[i][0])
            if disparity_orig > 0:
                depth_orig = (calibration_data['baseline'] * calibration_data['focal_length']) / disparity_orig
                if 500 <= depth_orig <= 5000:
                    x_orig = (cam1_positions[i][0] - calibration_data['camera_matrix_1'][0][2]) * depth_orig / calibration_data['focal_length']
                    y_orig = (cam1_positions[i][1] - calibration_data['camera_matrix_1'][1][2]) * depth_orig / calibration_data['focal_length']
                    positions_original.append(np.array([x_orig, y_orig, depth_orig]))
                else:
                    positions_original.append(None)
            else:
                positions_original.append(None)
            
            # 개선된 방식
            pos_improved = calculate_3d_position_improved(cam1_positions[i], cam2_positions[i], calibration_data)
            positions_improved.append(pos_improved)
            
            # 대안 방식
            pos_alternative = calculate_3d_position_alternative(cam1_positions[i], cam2_positions[i], calibration_data)
            positions_alternative.append(pos_alternative)
            
            # 스테레오 방식
            pos_stereo = calculate_3d_position_stereo(cam1_positions[i], cam2_positions[i], calibration_data)
            positions_stereo.append(pos_stereo)
            
            print(f"  Frame {i+1}:")
            if positions_original[i] is not None:
                print(f"    원본: ({positions_original[i][0]:.1f}, {positions_original[i][1]:.1f}, {positions_original[i][2]:.1f}) mm")
            if positions_improved[i] is not None:
                print(f"    개선: ({positions_improved[i][0]:.1f}, {positions_improved[i][1]:.1f}, {positions_improved[i][2]:.1f}) mm")
            if positions_alternative[i] is not None:
                print(f"    대안: ({positions_alternative[i][0]:.1f}, {positions_alternative[i][1]:.1f}, {positions_alternative[i][2]:.1f}) mm")
            if positions_stereo[i] is not None:
                print(f"    스테레오: ({positions_stereo[i][0]:.1f}, {positions_stereo[i][1]:.1f}, {positions_stereo[i][2]:.1f}) mm")
            print()
        else:
            positions_original.append(None)
            positions_improved.append(None)
            positions_alternative.append(None)
            positions_stereo.append(None)
    
    # 3D 좌표 계산 시각화
    create_3d_coordinate_visualization(positions_original, positions_improved, positions_alternative, positions_stereo)
    
    return {
        'positions_original': positions_original,
        'positions_improved': positions_improved,
        'positions_alternative': positions_alternative,
        'positions_stereo': positions_stereo
    }

def create_3d_coordinate_visualization(positions_original, positions_improved, positions_alternative, positions_stereo):
    """3D 좌표 계산 시각화"""
    
    print("\n=== 3D 좌표 계산 시각화 생성 ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # X 좌표 비교
    frames = list(range(1, 11))
    x_orig = [pos[0] if pos is not None else None for pos in positions_original]
    x_imp = [pos[0] if pos is not None else None for pos in positions_improved]
    x_alt = [pos[0] if pos is not None else None for pos in positions_alternative]
    x_stereo = [pos[0] if pos is not None else None for pos in positions_stereo]
    
    axes[0,0].plot(frames, x_orig, 'ro-', label='Original', markersize=6)
    axes[0,0].plot(frames, x_imp, 'bo-', label='Improved', markersize=6)
    axes[0,0].plot(frames, x_alt, 'go-', label='Alternative', markersize=6)
    axes[0,0].plot(frames, x_stereo, 'mo-', label='Stereo', markersize=6)
    axes[0,0].set_xlabel('Frame')
    axes[0,0].set_ylabel('X Coordinate (mm)')
    axes[0,0].set_title('X Coordinate Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Y 좌표 비교
    y_orig = [pos[1] if pos is not None else None for pos in positions_original]
    y_imp = [pos[1] if pos is not None else None for pos in positions_improved]
    y_alt = [pos[1] if pos is not None else None for pos in positions_alternative]
    y_stereo = [pos[1] if pos is not None else None for pos in positions_stereo]
    
    axes[0,1].plot(frames, y_orig, 'ro-', label='Original', markersize=6)
    axes[0,1].plot(frames, y_imp, 'bo-', label='Improved', markersize=6)
    axes[0,1].plot(frames, y_alt, 'go-', label='Alternative', markersize=6)
    axes[0,1].plot(frames, y_stereo, 'mo-', label='Stereo', markersize=6)
    axes[0,1].set_xlabel('Frame')
    axes[0,1].set_ylabel('Y Coordinate (mm)')
    axes[0,1].set_title('Y Coordinate Comparison')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Z 좌표 비교
    z_orig = [pos[2] if pos is not None else None for pos in positions_original]
    z_imp = [pos[2] if pos is not None else None for pos in positions_improved]
    z_alt = [pos[2] if pos is not None else None for pos in positions_alternative]
    z_stereo = [pos[2] if pos is not None else None for pos in positions_stereo]
    
    axes[1,0].plot(frames, z_orig, 'ro-', label='Original', markersize=6)
    axes[1,0].plot(frames, z_imp, 'bo-', label='Improved', markersize=6)
    axes[1,0].plot(frames, z_alt, 'go-', label='Alternative', markersize=6)
    axes[1,0].plot(frames, z_stereo, 'mo-', label='Stereo', markersize=6)
    axes[1,0].set_xlabel('Frame')
    axes[1,0].set_ylabel('Z Coordinate (mm)')
    axes[1,0].set_title('Z Coordinate (Depth) Comparison')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 3D 궤적 비교
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    
    # 원본 궤적
    valid_orig = [pos for pos in positions_original if pos is not None]
    if valid_orig:
        x_coords = [pos[0] for pos in valid_orig]
        y_coords = [pos[1] for pos in valid_orig]
        z_coords = [pos[2] for pos in valid_orig]
        ax.plot(x_coords, y_coords, z_coords, 'ro-', label='Original')
    
    # 개선된 궤적
    valid_imp = [pos for pos in positions_improved if pos is not None]
    if valid_imp:
        x_coords = [pos[0] for pos in valid_imp]
        y_coords = [pos[1] for pos in valid_imp]
        z_coords = [pos[2] for pos in valid_imp]
        ax.plot(x_coords, y_coords, z_coords, 'bo-', label='Improved')
    
    # 대안 궤적
    valid_alt = [pos for pos in positions_alternative if pos is not None]
    if valid_alt:
        x_coords = [pos[0] for pos in valid_alt]
        y_coords = [pos[1] for pos in valid_alt]
        z_coords = [pos[2] for pos in valid_alt]
        ax.plot(x_coords, y_coords, z_coords, 'go-', label='Alternative')
    
    # 스테레오 궤적
    valid_stereo = [pos for pos in positions_stereo if pos is not None]
    if valid_stereo:
        x_coords = [pos[0] for pos in valid_stereo]
        y_coords = [pos[1] for pos in valid_stereo]
        z_coords = [pos[2] for pos in valid_stereo]
        ax.plot(x_coords, y_coords, z_coords, 'mo-', label='Stereo')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Trajectory Comparison')
    ax.legend()
    
    plt.tight_layout()
    
    # 저장
    output_path = '../ball_detect/3d_coordinate_calculation_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D 좌표 계산 분석 시각화 저장: {output_path}")
    
    return output_path

if __name__ == "__main__":
    print("3D 좌표 계산 공식 수정 시작...")
    
    # 3D 좌표 계산 분석
    results = analyze_3d_coordinate_calculation()
    
    print(f"\n=== 3D 좌표 계산 분석 완료 ===")
    
    # 각 방식별 성공률 계산
    success_rates = {}
    for method, positions in results.items():
        success_count = sum(1 for pos in positions if pos is not None)
        success_rate = success_count / len(positions) * 100
        success_rates[method] = success_rate
        print(f"{method}: {success_rate:.1f}% 성공률")
    
    # 추천 방식 선택
    best_method = max(success_rates, key=success_rates.get)
    print(f"\n추천 방식: {best_method} ({success_rates[best_method]:.1f}% 성공률)")
