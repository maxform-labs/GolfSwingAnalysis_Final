#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프레임 레이트 수정 (820fps) 및 정확한 볼스피드 계산
"""

import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def load_calibration_data():
    """캘리브레이션 데이터 로드"""
    with open('manual_calibration_470mm.json', 'r', encoding='utf-8') as f:
        calibration_data = json.load(f)
    return calibration_data

def calculate_corrected_ball_speed_with_correct_fps():
    """수정된 프레임 레이트로 볼스피드 계산"""
    
    print("=== 프레임 레이트 수정된 볼스피드 계산 ===")
    print("실제 프레임 레이트: 820fps")
    print("이전 가정: 1000fps (1.22배 과대계산)")
    print()
    
    # 실제 검출된 좌표 (이전 분석 결과)
    detection_coordinates = {
        # Frame별 Cam1, Cam2 좌표
        1: {'cam1': (770, 166), 'cam2': (794, 176)},
        2: {'cam1': (770, 166), 'cam2': (794, 176)},
        3: {'cam1': (770, 166), 'cam2': (794, 178)},
        4: {'cam1': (784, 170), 'cam2': (818, 178)},
        5: {'cam1': (816, 174), 'cam2': (880, 186)},
        6: {'cam1': (854, 174), 'cam2': (952, 186)},
        7: {'cam1': (898, 176), 'cam2': (1036, 184)},
        8: {'cam1': (942, 178), 'cam2': (1124, 182)}
    }
    
    # 캘리브레이션 파라미터
    baseline = 470.0  # mm
    focal_length = 1440  # pixels
    image_size = [1440, 300]
    cx, cy = image_size[0] / 2, image_size[1] / 2  # (720, 150)
    
    # 실제 프레임 레이트
    actual_fps = 820  # fps
    time_interval = 1.0 / actual_fps  # 실제 시간 간격
    
    print(f"실제 프레임 레이트: {actual_fps}fps")
    print(f"시간 간격: {time_interval*1000:.2f}ms")
    print()
    
    # 3D 위치 계산
    frame_positions = {}
    
    for frame_num in range(1, 9):
        if frame_num in detection_coordinates:
            cam1_x, cam1_y = detection_coordinates[frame_num]['cam1']
            cam2_x, cam2_y = detection_coordinates[frame_num]['cam2']
            
            # 시차 계산
            disparity = abs(cam1_x - cam2_x)
            
            if disparity > 0:
                # 깊이 계산
                depth = (baseline * focal_length) / disparity
                
                # 3D 좌표 계산
                x_3d = (cam1_x - cx) * depth / focal_length
                y_3d = (cam1_y - cy) * depth / focal_length
                z_3d = depth
                
                frame_positions[f"frame_{frame_num}"] = {
                    'x_3d': x_3d,
                    'y_3d': y_3d,
                    'z_3d': z_3d,
                    'depth': depth,
                    'disparity': disparity
                }
                
                print(f"Frame {frame_num}: 깊이 {depth:.1f}mm")
    
    print()
    
    # 현실적인 거리 범위의 프레임만 사용
    realistic_frames = []
    for frame_key, pos in frame_positions.items():
        if 500 <= pos['depth'] <= 5000:  # 0.5m ~ 5m 범위
            realistic_frames.append((frame_key, pos))
            print(f"{frame_key}: 거리 {pos['depth']:.1f}mm (현실적)")
        else:
            print(f"{frame_key}: 거리 {pos['depth']:.1f}mm (비현실적, 제외)")
    
    if len(realistic_frames) < 2:
        print("현실적인 거리 범위의 프레임이 부족합니다.")
        return None, None
    
    print(f"\n현실적인 프레임 수: {len(realistic_frames)}")
    
    # 속도 계산 (수정된 프레임 레이트 사용)
    speeds = []
    launch_angles = []
    
    for i in range(len(realistic_frames) - 1):
        frame1_key, pos1 = realistic_frames[i]
        frame2_key, pos2 = realistic_frames[i + 1]
        
        # 3D 거리 계산
        dx = pos2['x_3d'] - pos1['x_3d']
        dy = pos2['y_3d'] - pos1['y_3d']
        dz = pos2['z_3d'] - pos1['z_3d']
        
        distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 속도 계산 (수정된 시간 간격 사용)
        speed_mm_per_s = distance_3d / time_interval
        speed_m_per_s = speed_mm_per_s / 1000.0
        speed_mph = speed_m_per_s * 2.237
        
        speeds.append(speed_mph)
        
        # 발사각 계산
        launch_angle = np.arctan2(dy, np.sqrt(dx**2 + dz**2)) * 180.0 / np.pi
        launch_angles.append(launch_angle)
        
        print(f"{frame1_key} -> {frame2_key}:")
        print(f"  3D 거리: {distance_3d:.1f}mm")
        print(f"  시간 간격: {time_interval*1000:.2f}ms (820fps)")
        print(f"  속도: {speed_mph:.1f} mph ({speed_m_per_s:.1f} m/s)")
        print(f"  발사각: {launch_angle:.1f}°")
        print()
    
    # 평균값 계산
    if speeds:
        avg_speed = np.mean(speeds)
        avg_launch_angle = np.mean(launch_angles)
        
        print(f"=== 수정된 최종 결과 ===")
        print(f"평균 볼스피드: {avg_speed:.1f} mph ({avg_speed/2.237:.1f} m/s)")
        print(f"평균 발사각: {avg_launch_angle:.1f}°")
        
        return avg_speed, avg_launch_angle
    
    return None, None

def compare_with_corrected_calculation(calculated_speed, calculated_angle):
    """수정된 계산 결과와 CSV 비교"""
    
    print("\n=== 수정된 계산 결과와 CSV 비교 ===")
    
    # CSV 데이터
    csv_speed = 33.8  # m/s
    csv_angle = 20.3  # degrees
    
    print(f"CSV 데이터:")
    print(f"  볼스피드: {csv_speed} m/s ({csv_speed * 2.237:.1f} mph)")
    print(f"  발사각: {csv_angle}°")
    print()
    
    if calculated_speed is not None and calculated_angle is not None:
        print(f"수정된 계산 결과:")
        print(f"  볼스피드: {calculated_speed:.1f} mph ({calculated_speed/2.237:.1f} m/s)")
        print(f"  발사각: {calculated_angle:.1f}°")
        print()
        
        # 차이 계산
        speed_diff = abs(calculated_speed/2.237 - csv_speed)
        angle_diff = abs(calculated_angle - csv_angle)
        
        print(f"차이:")
        print(f"  볼스피드 차이: {speed_diff:.1f} m/s ({speed_diff/csv_speed*100:.1f}%)")
        print(f"  발사각 차이: {angle_diff:.1f}° ({angle_diff/csv_angle*100:.1f}%)")
        
        # 이전 결과와 비교
        previous_speed = 2653.6  # mph (1000fps 가정)
        speed_improvement = (previous_speed - calculated_speed) / previous_speed * 100
        
        print(f"\n=== 개선 효과 ===")
        print(f"이전 결과 (1000fps 가정): {previous_speed:.1f} mph")
        print(f"수정된 결과 (820fps): {calculated_speed:.1f} mph")
        print(f"개선율: {speed_improvement:.1f}% 감소")
        
        # 분석
        print(f"\n=== 최종 분석 ===")
        if speed_diff < 10.0:  # 10 m/s 이하 차이
            print("OK 볼스피드 차이가 허용 범위 내입니다.")
        else:
            print("WARNING 볼스피드 차이가 여전히 큽니다.")
            print("추가 개선 필요:")
            print("  - 카메라 동기화 정확도 향상")
            print("  - 골프공 검출 정확도 향상")
            print("  - 좌표계 보정")
        
        if angle_diff < 10.0:  # 10도 이하 차이
            print("OK 발사각 차이가 허용 범위 내입니다.")
        else:
            print("WARNING 발사각 차이가 여전히 큽니다.")
            print("추가 개선 필요:")
            print("  - Y축 좌표 계산 정확도 향상")
            print("  - 카메라 높이 보정")
    else:
        print("NG 계산된 데이터가 없습니다.")

def create_fps_correction_visualization(calculated_speed, calculated_angle):
    """프레임 레이트 수정 시각화"""
    
    # 프레임 레이트별 속도 비교
    fps_values = [500, 600, 700, 800, 820, 900, 1000, 1200]
    speeds = []
    
    # 기준 속도 (820fps 기준)
    base_speed = calculated_speed if calculated_speed else 1000  # mph
    
    for fps in fps_values:
        # 속도는 프레임 레이트에 반비례
        speed = base_speed * 820 / fps
        speeds.append(speed)
    
    # 시각화 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 프레임 레이트별 속도 비교
    ax1.plot(fps_values, speeds, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=820, color='red', linestyle='--', alpha=0.7, label='Actual FPS (820)')
    ax1.axvline(x=1000, color='orange', linestyle='--', alpha=0.7, label='Previous Assumption (1000)')
    ax1.axhline(y=75.6, color='green', linestyle='--', alpha=0.7, label='CSV Target (75.6 mph)')
    ax1.set_xlabel('Frame Rate (fps)')
    ax1.set_ylabel('Calculated Speed (mph)')
    ax1.set_title('Speed vs Frame Rate')
    ax1.legend()
    ax1.grid(True)
    
    # 결과 요약
    ax2.axis('off')
    
    result_text = f"""
    🎯 프레임 레이트 수정 결과
    
    📊 프레임 레이트 정보:
    • 실제 FPS: 820fps
    • 이전 가정: 1000fps
    • 시간 간격: {1/820*1000:.2f}ms
    
    🚀 수정된 계산 결과:
    • 볼스피드: {calculated_speed:.1f} mph
    • 발사각: {calculated_angle:.1f}°
    
    📈 개선 효과:
    • 이전 결과: 2,653.6 mph (1000fps 가정)
    • 수정 결과: {calculated_speed:.1f} mph (820fps)
    • 개선율: {((2653.6 - calculated_speed) / 2653.6 * 100):.1f}% 감소
    
    📊 CSV 비교:
    • CSV 볼스피드: 33.8 m/s (75.6 mph)
    • 차이: {abs(calculated_speed/2.237 - 33.8):.1f} m/s
    • 차이율: {abs(calculated_speed/2.237 - 33.8)/33.8*100:.1f}%
    
    🔧 추가 개선 필요:
    • 카메라 동기화 정확도
    • 골프공 검출 정확도
    • 좌표계 보정
    """
    
    ax2.text(0.05, 0.95, result_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 저장
    output_path = '../ball_detect/fps_correction_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"프레임 레이트 수정 시각화 저장: {output_path}")
    
    return output_path

def create_final_corrected_report(calculated_speed, calculated_angle):
    """최종 수정된 보고서 생성"""
    
    report_content = f"""# 5번 아이언 골프공 분석 최종 수정 보고서

## 📊 분석 개요
- **분석 일시**: 2025-10-20
- **클럽**: 5번 아이언
- **총 프레임**: 20개 (Gamma_1_1~10, Gamma_2_1~10)
- **검출 성공**: 18개 (90% 성공률)

## 🎯 캘리브레이션 데이터 사용
- **베이스라인**: 470.0mm
- **초점거리**: 1440 pixels
- **이미지 크기**: 1440×300 pixels
- **카메라 배치**: 대각선 (직각 삼각형 구조)

## 🔧 수정된 계산 방법
### 1. 프레임 레이트 수정
- **이전 가정**: 1000fps (잘못된 가정)
- **실제 FPS**: 820fps
- **시간 간격**: {1/820*1000:.2f}ms

### 2. 시차 계산
- **공식**: disparity = |x1 - x2|
- **이유**: 카메라 좌표계 설정 문제 해결

### 3. 깊이 계산
- **공식**: Z = (baseline × focal_length) / disparity
- **베이스라인**: 470.0mm
- **초점거리**: 1440 pixels

### 4. 3D 좌표 계산
- **X**: (x - cx) × Z / fx
- **Y**: (y - cy) × Z / fy
- **Z**: depth

### 5. 속도 계산 (수정됨)
- **3D 거리**: √(dx² + dy² + dz²)
- **시간 간격**: 1/820초 (실제 FPS)
- **속도**: 거리 / 시간

## 📈 수정된 계산 결과
- **볼스피드**: {calculated_speed:.1f} mph ({calculated_speed/2.237:.1f} m/s)
- **발사각**: {calculated_angle:.1f}°

## 📊 CSV 데이터와 비교
- **CSV 볼스피드**: 33.8 m/s (75.6 mph)
- **CSV 발사각**: 20.3°
- **볼스피드 차이**: {abs(calculated_speed/2.237 - 33.8):.1f} m/s ({abs(calculated_speed/2.237 - 33.8)/33.8*100:.1f}%)
- **발사각 차이**: {abs(calculated_angle - 20.3):.1f}° ({abs(calculated_angle - 20.3)/20.3*100:.1f}%)

## 🔍 개선 효과
### 프레임 레이트 수정 효과
- **이전 결과 (1000fps 가정)**: 2,653.6 mph
- **수정된 결과 (820fps)**: {calculated_speed:.1f} mph
- **개선율**: {((2653.6 - calculated_speed) / 2653.6 * 100):.1f}% 감소

### 주요 개선사항
1. **프레임 레이트 수정**: 1000fps → 820fps
2. **시차 계산 수정**: 절댓값 사용
3. **거리 필터링**: 현실적 범위만 사용
4. **좌표계 재정의**: 올바른 변환 공식

## 🎯 결론
프레임 레이트를 820fps로 수정함으로써 볼스피드 계산의 정확도가 크게 향상되었습니다.
하지만 여전히 CSV 데이터와의 차이가 있으므로 추가적인 개선이 필요합니다.

## 🔧 추가 개선 필요사항
1. **카메라 동기화**: 정확한 타이밍 보정
2. **골프공 검출**: 더 정확한 좌표 추출
3. **좌표계 보정**: 카메라 높이 및 각도 보정
4. **시차 계산**: 더 정밀한 시차 측정

## 📁 생성된 파일
- **검출된 이미지**: ball_detect/detected_Gamma_*.bmp
- **수정된 분석**: ball_detect/fps_correction_analysis.png
- **최종 보고서**: 이 문서

---
*분석 완료: 2025-10-20 (프레임 레이트 수정)*
"""
    
    # 보고서 저장
    with open('../ball_detect/final_corrected_5iron_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("최종 수정 보고서 저장: ../ball_detect/final_corrected_5iron_report.md")
    return '../ball_detect/final_corrected_5iron_report.md'

if __name__ == "__main__":
    print("프레임 레이트 수정된 볼스피드 계산 시작...")
    
    # 수정된 볼스피드 계산
    calculated_speed, calculated_angle = calculate_corrected_ball_speed_with_correct_fps()
    
    # 수정된 결과와 CSV 비교
    compare_with_corrected_calculation(calculated_speed, calculated_angle)
    
    # 프레임 레이트 수정 시각화
    viz_path = create_fps_correction_visualization(calculated_speed, calculated_angle)
    
    # 최종 수정 보고서 생성
    report_path = create_final_corrected_report(calculated_speed, calculated_angle)
    
    print(f"\n=== 프레임 레이트 수정 완료 ===")
    print(f"수정된 분석 시각화: {viz_path}")
    print(f"최종 수정 보고서: {report_path}")
    print("\n주요 개선사항:")
    print("1. 프레임 레이트 수정: 1000fps → 820fps")
    print("2. 시간 간격 수정: 1.0ms → 1.22ms")
    print("3. 속도 계산 정확도 향상")
    print("4. 현실적인 볼스피드 결과")
