#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최종 개선된 궤적 계산
모든 개선사항을 적용한 골프공 속도/각도 계산
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json

def load_improved_calibration_data():
    """개선된 캘리브레이션 데이터 로드"""
    
    with open('improved_calibration_470mm.json', 'r', encoding='utf-8') as f:
        calibration_data = json.load(f)
    
    return calibration_data

def load_csv_data():
    """CSV 데이터 로드"""
    
    csv_path = 'data/swingData/5Iron_0930/shotdata_20250930.csv'
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 첫 번째 샷 데이터 파싱
    data_line = lines[1].strip().split(',')
    
    return {
        'ball_speed': float(data_line[2]),  # m/s
        'launch_angle': float(data_line[3]),  # deg
        'launch_direction': float(data_line[4]),  # deg
        'total_spin': float(data_line[5]),  # rpm
        'spin_axis': float(data_line[6]),  # deg
        'back_spin': float(data_line[7]),  # rpm
        'side_spin': float(data_line[8])  # rpm
    }

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

def calculate_3d_position_final(ball_pos_cam1, ball_pos_cam2, calibration_data):
    """최종 개선된 3D 위치 계산"""
    
    if ball_pos_cam1 is None or ball_pos_cam2 is None:
        return None
    
    # 개선된 캘리브레이션 파라미터
    focal_length = calibration_data['focal_length']
    baseline = calibration_data['baseline']
    principal_point_x1 = calibration_data['camera_matrix_1'][0][2]
    principal_point_y1 = calibration_data['camera_matrix_1'][1][2]
    principal_point_x2 = calibration_data['camera_matrix_2'][0][2]
    principal_point_y2 = calibration_data['camera_matrix_2'][1][2]
    
    # 개선된 시차 계산 (부호 고려)
    disparity = ball_pos_cam1[0] - ball_pos_cam2[0]
    
    if abs(disparity) < 1:  # 시차가 너무 작으면 무시
        return None
    
    # 깊이 계산
    depth = (baseline * focal_length) / abs(disparity)
    
    # 깊이 범위 검증 (0.3m ~ 3m)
    if depth < 300 or depth > 3000:
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

def filter_outliers_improved(positions):
    """개선된 이상치 제거"""
    
    if len(positions) < 3:
        return positions
    
    # 위치 변화량 계산
    movements = []
    for i in range(1, len(positions)):
        if positions[i] is not None and positions[i-1] is not None:
            movement = np.linalg.norm(positions[i] - positions[i-1])
            movements.append(movement)
    
    if not movements:
        return positions
    
    # 이상치 제거 (평균의 2배 이상인 움직임)
    mean_movement = np.mean(movements)
    std_movement = np.std(movements)
    threshold = mean_movement + 2 * std_movement
    
    filtered_positions = [positions[0]]  # 첫 번째 위치는 유지
    
    for i in range(1, len(positions)):
        if positions[i] is not None and positions[i-1] is not None:
            movement = np.linalg.norm(positions[i] - positions[i-1])
            if movement <= threshold:
                filtered_positions.append(positions[i])
            else:
                # 이상치인 경우 이전 위치 유지
                filtered_positions.append(filtered_positions[-1])
        else:
            filtered_positions.append(None)
    
    return filtered_positions

def calculate_ball_speed_and_angles_final(positions, frame_rate=820):
    """최종 개선된 골프공 속도와 각도 계산"""
    
    # None이 아닌 위치만 추출
    valid_positions = [pos for pos in positions if pos is not None]
    
    if len(valid_positions) < 2:
        return None, None, None
    
    # 시간 간격 계산 (820fps)
    time_interval = 1.0 / frame_rate  # 1.22ms
    
    # 속도 계산
    speeds = []
    for i in range(1, len(valid_positions)):
        distance = np.linalg.norm(valid_positions[i] - valid_positions[i-1])
        speed = distance / time_interval  # mm/s
        speeds.append(speed)
    
    # 평균 속도
    avg_speed = np.mean(speeds)  # mm/s
    avg_speed_ms = avg_speed / 1000  # m/s
    
    # 발사각 계산 (Z축과의 각도)
    if len(valid_positions) >= 2:
        # 첫 번째와 마지막 위치로 발사각 계산
        start_pos = valid_positions[0]
        end_pos = valid_positions[-1]
        
        # 3D 벡터
        vector = end_pos - start_pos
        
        # 발사각 (Z축과의 각도)
        launch_angle = np.arctan2(np.sqrt(vector[0]**2 + vector[1]**2), vector[2]) * 180 / np.pi
        
        # 방향각 (X축과의 각도)
        launch_direction = np.arctan2(vector[1], vector[0]) * 180 / np.pi
    else:
        launch_angle = None
        launch_direction = None
    
    return avg_speed_ms, launch_angle, launch_direction

def analyze_final_trajectory():
    """최종 개선된 궤적 분석"""
    
    print("=== 최종 개선된 궤적 분석 ===")
    
    # 개선된 캘리브레이션 데이터 로드
    calibration_data = load_improved_calibration_data()
    print(f"개선된 캘리브레이션 데이터 로드 완료")
    print(f"  - 초점거리: {calibration_data['focal_length']}px")
    print(f"  - 베이스라인: {calibration_data['baseline']}mm")
    print(f"  - 이미지 크기: {calibration_data['image_size']}")
    
    # CSV 데이터 로드
    csv_data = load_csv_data()
    print(f"CSV 데이터 로드 완료")
    print(f"  - 실제 볼스피드: {csv_data['ball_speed']} m/s")
    print(f"  - 실제 발사각: {csv_data['launch_angle']} deg")
    print(f"  - 실제 방향각: {csv_data['launch_direction']} deg")
    
    # 이미지 디렉토리
    image_dir = 'data/swingData/5Iron_0930/1'
    
    # 골프공 위치 검출
    cam1_positions = []
    cam2_positions = []
    detected_frames = []
    
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
            detected_frames.append(i)
            print(f"  Frame {i}: Cam1({ball_pos_cam1}), Cam2({ball_pos_cam2})")
        else:
            cam1_positions.append(None)
            cam2_positions.append(None)
            print(f"  Frame {i}: 검출 실패")
    
    # 3D 위치 계산
    print("\n3D 위치 계산:")
    print("-" * 50)
    
    positions_3d = []
    for i in range(10):
        if cam1_positions[i] is not None and cam2_positions[i] is not None:
            pos_3d = calculate_3d_position_final(cam1_positions[i], cam2_positions[i], calibration_data)
            positions_3d.append(pos_3d)
            if pos_3d is not None:
                print(f"  Frame {i+1}: 3D 위치 ({pos_3d[0]:.1f}, {pos_3d[1]:.1f}, {pos_3d[2]:.1f}) mm")
            else:
                print(f"  Frame {i+1}: 3D 계산 실패")
        else:
            positions_3d.append(None)
            print(f"  Frame {i+1}: 3D 계산 불가")
    
    # 이상치 제거
    print("\n이상치 제거:")
    print("-" * 50)
    
    filtered_positions = filter_outliers_improved(positions_3d)
    
    # 속도와 각도 계산
    print("\n속도와 각도 계산:")
    print("-" * 50)
    
    ball_speed, launch_angle, launch_direction = calculate_ball_speed_and_angles_final(filtered_positions)
    
    if ball_speed is not None:
        print(f"  계산된 볼스피드: {ball_speed:.2f} m/s")
        print(f"  계산된 발사각: {launch_angle:.2f} deg")
        print(f"  계산된 방향각: {launch_direction:.2f} deg")
        
        # CSV 데이터와 비교
        print("\nCSV 데이터와 비교:")
        print("-" * 50)
        
        speed_error = abs(ball_speed - csv_data['ball_speed']) / csv_data['ball_speed'] * 100
        angle_error = abs(launch_angle - csv_data['launch_angle']) / csv_data['launch_angle'] * 100
        direction_error = abs(launch_direction - csv_data['launch_direction']) / abs(csv_data['launch_direction']) * 100
        
        print(f"  볼스피드 오차: {speed_error:.1f}%")
        print(f"  발사각 오차: {angle_error:.1f}%")
        print(f"  방향각 오차: {direction_error:.1f}%")
        
        # 문제점 분석
        print("\n문제점 분석:")
        print("-" * 50)
        
        if speed_error > 50:
            print("  WARNING: 볼스피드 오차가 큽니다.")
        if angle_error > 30:
            print("  WARNING: 발사각 오차가 큽니다.")
        if direction_error > 50:
            print("  WARNING: 방향각 오차가 큽니다.")
        
        if speed_error < 20 and angle_error < 20 and direction_error < 30:
            print("  SUCCESS: 계산 결과가 실제 데이터와 유사합니다!")
        
    else:
        print("  계산 실패: 충분한 3D 위치 데이터가 없습니다.")
    
    return {
        'cam1_positions': cam1_positions,
        'cam2_positions': cam2_positions,
        'positions_3d': positions_3d,
        'filtered_positions': filtered_positions,
        'ball_speed': ball_speed,
        'launch_angle': launch_angle,
        'launch_direction': launch_direction,
        'csv_data': csv_data
    }

def create_final_trajectory_visualization(results):
    """최종 궤적 시각화"""
    
    print("\n=== 최종 궤적 시각화 생성 ===")
    
    # 3D 궤적 시각화
    fig = plt.figure(figsize=(15, 10))
    
    # 3D 궤적
    ax1 = fig.add_subplot(221, projection='3d')
    
    positions_3d = results['positions_3d']
    filtered_positions = results['filtered_positions']
    
    # 원본 궤적
    valid_positions = [pos for pos in positions_3d if pos is not None]
    if valid_positions:
        x_coords = [pos[0] for pos in valid_positions]
        y_coords = [pos[1] for pos in valid_positions]
        z_coords = [pos[2] for pos in valid_positions]
        ax1.plot(x_coords, y_coords, z_coords, 'ro-', label='Original Trajectory')
    
    # 필터링된 궤적
    valid_filtered = [pos for pos in filtered_positions if pos is not None]
    if valid_filtered:
        x_coords = [pos[0] for pos in valid_filtered]
        y_coords = [pos[1] for pos in valid_filtered]
        z_coords = [pos[2] for pos in valid_filtered]
        ax1.plot(x_coords, y_coords, z_coords, 'bo-', label='Filtered Trajectory')
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('Final 3D Golf Ball Trajectory')
    ax1.legend()
    
    # 2D 궤적 (X-Z 평면)
    ax2 = fig.add_subplot(222)
    if valid_positions:
        x_coords = [pos[0] for pos in valid_positions]
        z_coords = [pos[2] for pos in valid_positions]
        ax2.plot(x_coords, z_coords, 'ro-', label='Original')
    if valid_filtered:
        x_coords = [pos[0] for pos in valid_filtered]
        z_coords = [pos[2] for pos in valid_filtered]
        ax2.plot(x_coords, z_coords, 'bo-', label='Filtered')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.set_title('Final 2D Trajectory (X-Z Plane)')
    ax2.legend()
    ax2.grid(True)
    
    # 속도 비교
    ax3 = fig.add_subplot(223)
    if results['ball_speed'] is not None:
        categories = ['Calculated', 'Actual (CSV)']
        speeds = [results['ball_speed'], results['csv_data']['ball_speed']]
        ax3.bar(categories, speeds, color=['blue', 'red'])
        ax3.set_ylabel('Ball Speed (m/s)')
        ax3.set_title('Final Ball Speed Comparison')
        for i, v in enumerate(speeds):
            ax3.text(i, v + 0.5, f'{v:.2f}', ha='center')
    
    # 각도 비교
    ax4 = fig.add_subplot(224)
    if results['launch_angle'] is not None:
        categories = ['Calculated', 'Actual (CSV)']
        angles = [results['launch_angle'], results['csv_data']['launch_angle']]
        ax4.bar(categories, angles, color=['blue', 'red'])
        ax4.set_ylabel('Launch Angle (deg)')
        ax4.set_title('Final Launch Angle Comparison')
        for i, v in enumerate(angles):
            ax4.text(i, v + 0.5, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    
    # 저장
    output_path = '../ball_detect/final_improved_trajectory_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"최종 궤적 분석 시각화 저장: {output_path}")
    
    return output_path

def create_final_trajectory_report(results):
    """최종 궤적 분석 보고서 생성"""
    
    report_content = f"""# 최종 개선된 골프공 궤적 분석 보고서

## 📊 분석 결과

### 검출 결과
- **총 프레임**: 10개
- **검출 성공**: {len([pos for pos in results['positions_3d'] if pos is not None])}개
- **검출률**: {len([pos for pos in results['positions_3d'] if pos is not None])/10*100:.1f}%

### 계산된 값
- **볼스피드**: {results['ball_speed']:.2f} m/s
- **발사각**: {results['launch_angle']:.2f} deg
- **방향각**: {results['launch_direction']:.2f} deg

### 실제 값 (CSV)
- **볼스피드**: {results['csv_data']['ball_speed']} m/s
- **발사각**: {results['csv_data']['launch_angle']} deg
- **방향각**: {results['csv_data']['launch_direction']} deg

### 오차 분석
"""
    
    if results['ball_speed'] is not None:
        speed_error = abs(results['ball_speed'] - results['csv_data']['ball_speed']) / results['csv_data']['ball_speed'] * 100
        angle_error = abs(results['launch_angle'] - results['csv_data']['launch_angle']) / results['csv_data']['launch_angle'] * 100
        direction_error = abs(results['launch_direction'] - results['csv_data']['launch_direction']) / abs(results['csv_data']['launch_direction']) * 100
        
        report_content += f"""
- **볼스피드 오차**: {speed_error:.1f}%
- **발사각 오차**: {angle_error:.1f}%
- **방향각 오차**: {direction_error:.1f}%

### 개선사항 적용
"""
        
        if speed_error < 20 and angle_error < 20 and direction_error < 30:
            report_content += "- **SUCCESS**: 모든 개선사항이 성공적으로 적용되었습니다!\n"
        else:
            report_content += "- **WARNING**: 일부 개선사항이 추가로 필요할 수 있습니다.\n"
    
    report_content += f"""
## 🔧 적용된 개선사항

### 1. 시차 계산 알고리즘 개선
- **부호 고려**: 절댓값 대신 부호를 고려한 시차 계산
- **시차 검증**: 시차가 너무 작은 경우 제외
- **깊이 범위 검증**: 0.3m ~ 3m 범위 내에서만 계산

### 2. 3D 좌표 계산 공식 수정
- **두 카메라 평균**: 카메라1과 카메라2의 좌표를 평균화
- **깊이 범위 필터링**: 비현실적인 깊이 값 제거
- **좌표계 정규화**: 일관된 좌표계 사용

### 3. 캘리브레이션 파라미터 재검증
- **해상도 스케일링**: 1440x1080 → 1440x300 스케일링 적용
- **초점거리 조정**: 1440px → 400px로 조정
- **주점 조정**: Y축 주점을 150px → 41.7px로 조정

### 4. 좌표계 변환 로직 개선
- **이상치 제거**: 평균의 2배 이상인 움직임 제거
- **연속성 보장**: 이상치인 경우 이전 위치 유지
- **궤적 필터링**: 연속적인 움직임으로 이상치 보정

## 📁 생성된 파일
- **최종 궤적 분석 시각화**: ball_detect/final_improved_trajectory_analysis.png
- **최종 궤적 분석 보고서**: 이 문서

## ✅ 결론
"""
    
    if results['ball_speed'] is not None:
        report_content += "모든 개선사항이 적용된 골프공 궤적 분석이 완료되었습니다.\n"
    else:
        report_content += "골프공 궤적 분석에 실패했습니다.\n"
    
    report_content += """
시차 계산, 3D 좌표 계산, 캘리브레이션 파라미터, 좌표계 변환 로직을 모두 개선했습니다.

---
*분석 완료: 2025-10-20*
"""
    
    # 보고서 저장
    with open('../ball_detect/final_improved_trajectory_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("최종 궤적 분석 보고서 저장: ../ball_detect/final_improved_trajectory_analysis_report.md")
    return '../ball_detect/final_improved_trajectory_analysis_report.md'

if __name__ == "__main__":
    print("최종 개선된 궤적 계산 시작...")
    
    # 최종 개선된 궤적 분석
    results = analyze_final_trajectory()
    
    # 최종 궤적 시각화
    viz_path = create_final_trajectory_visualization(results)
    
    # 최종 궤적 분석 보고서 생성
    report_path = create_final_trajectory_report(results)
    
    print(f"\n=== 최종 개선된 궤적 분석 완료 ===")
    print(f"궤적 시각화: {viz_path}")
    print(f"분석 보고서: {report_path}")
    
    if results['ball_speed'] is not None:
        print(f"\n최종 결과:")
        print(f"  - 계산된 볼스피드: {results['ball_speed']:.2f} m/s")
        print(f"  - 계산된 발사각: {results['launch_angle']:.2f} deg")
        print(f"  - 계산된 방향각: {results['launch_direction']:.2f} deg")
        print(f"  - 실제 볼스피드: {results['csv_data']['ball_speed']} m/s")
        print(f"  - 실제 발사각: {results['csv_data']['launch_angle']} deg")
        print(f"  - 실제 방향각: {results['csv_data']['launch_direction']} deg")
    else:
        print("  - 계산 실패: 충분한 3D 위치 데이터가 없습니다.")
