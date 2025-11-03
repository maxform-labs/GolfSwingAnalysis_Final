#!/usr/bin/env python3
"""
정정된 골프공 물리량 분석 시스템
올바른 캘리브레이션과 좌표계를 사용한 정확한 계산
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict
import math

class CorrectedGolfPhysicsAnalyzer:
    def __init__(self, calibration_file="vertical_stereo_calibration_z_axis.json"):
        """정정된 골프공 물리량 분석기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        # 고속카메라 설정
        self.fps = 820  # 820fps 고속카메라
        self.frame_interval = 1.0 / self.fps  # 프레임 간 시간 간격
        
        print("Corrected Golf Ball Physics Analyzer Initialized")
        print(f"High-speed camera: {self.fps} fps")
        print(f"Frame interval: {self.frame_interval:.6f} seconds")
        print(f"Baseline: {self.baseline_mm}mm (Z-axis)")
        print(f"Focal length: {self.focal_length}px")
        print("Using corrected vertical stereo vision calibration")
    
    def load_calibration(self):
        """캘리브레이션 데이터 로드"""
        with open(self.calibration_file, 'r', encoding='utf-8') as f:
            self.calibration_data = json.load(f)
        
        # 카메라 매트릭스
        self.K1 = np.array(self.calibration_data['camera_matrix_1'])
        self.K2 = np.array(self.calibration_data['camera_matrix_2'])
        
        # 왜곡 계수
        self.D1 = np.array(self.calibration_data['distortion_coeffs_1'])
        self.D2 = np.array(self.calibration_data['distortion_coeffs_2'])
        
        # 스테레오 변환
        self.R = np.array(self.calibration_data['rotation_matrix'])
        self.T = np.array(self.calibration_data['translation_vector'])
        
        # 기타 정보
        self.baseline_mm = self.calibration_data['baseline_mm']
        self.image_size = tuple(self.calibration_data['image_size'])
        self.focal_length = self.K1[0, 0]
        
        # 좌표계 정보
        self.coordinate_system = self.calibration_data.get('coordinate_system', {})
        print(f"Coordinate system: {self.coordinate_system}")
    
    def detect_golf_ball_adaptive(self, frame: np.ndarray) -> tuple:
        """적응형 골프공 검출 (완화된 조건)"""
        if frame is None or frame.size == 0:
            return None, None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness > 150:
            threshold_low = max(120, mean_brightness - 30)
        else:
            threshold_low = max(120, mean_brightness + 20)
        threshold_high = 255
        
        binary = cv2.inRange(gray, threshold_low, threshold_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 5000:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:
                        x, y, w, h = cv2.boundingRect(contour)
                        roi = gray[y:y+h, x:x+w]
                        mean_brightness_roi = np.mean(roi)
                        if mean_brightness_roi > threshold_low * 0.5:
                            score = circularity * (area / 100) * (mean_brightness_roi / 255)
                            if score > best_score:
                                best_score = score
                                best_contour = contour
        
        if best_contour is not None and best_score > 0.05:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center = (int(x), int(y))
            return center, int(radius)
        
        return None, None
    
    def calculate_3d_position_stereo(self, center1, center2):
        """수직 스테레오 비전을 통한 정확한 3D 위치 계산"""
        if center1 is None or center2 is None:
            return None
        
        u1, v1 = center1
        u2, v2 = center2
        
        # 수직 스테레오 비전: Z축 베이스라인
        # Y 좌표 시차를 사용 (수직 배치)
        disparity = abs(v1 - v2)
        
        if disparity <= 1:
            return None
        
        # 깊이 계산 (Z 좌표)
        depth = (self.focal_length * self.baseline_mm) / disparity
        
        if depth < 100 or depth > 10000:  # 유효한 깊이 범위
            return None
        
        # 3D 좌표 계산 (올바른 좌표계)
        # X: 타겟 방향 (전진), Y: 측면 방향 (오른쪽이 양수), Z: 높이 방향 (위가 양수)
        x = (u1 - self.K1[0, 2]) * depth / self.focal_length  # 타겟 방향
        y = (v1 - self.K1[1, 2]) * depth / self.focal_length  # 측면 방향
        z = depth  # 높이 방향
        
        return np.array([x, y, z])
    
    def calculate_velocity_accurate(self, positions, frame_numbers):
        """820fps를 고려한 정확한 속도 계산"""
        if len(positions) < 2:
            return None, None
        
        velocities = []
        for i in range(1, len(positions)):
            if positions[i] is not None and positions[i-1] is not None:
                frame_diff = frame_numbers[i] - frame_numbers[i-1]
                dt = frame_diff * self.frame_interval
                
                if dt > 0:
                    velocity = (positions[i] - positions[i-1]) / dt
                    velocities.append(velocity)
        
        if not velocities:
            return None, None
        
        # 평균 속도 계산
        avg_velocity = np.mean(velocities, axis=0)
        speed = np.linalg.norm(avg_velocity)
        
        return avg_velocity, speed
    
    def calculate_launch_angle(self, velocity):
        """발사각 계산 (수직각) - Z축 기준"""
        if velocity is None:
            return None
        
        # Z 성분이 수직 방향
        vertical_component = velocity[2]
        total_speed = np.linalg.norm(velocity)
        
        if total_speed == 0:
            return None
        
        # 발사각 계산 (라디안을 도로 변환)
        launch_angle = math.asin(abs(vertical_component) / total_speed)
        launch_angle_degrees = math.degrees(launch_angle)
        
        return launch_angle_degrees
    
    def calculate_direction_angle(self, velocity):
        """방향각 계산 (수평각) - X-Y 평면 기준"""
        if velocity is None:
            return None
        
        # X, Y 성분으로부터 방향각 계산
        horizontal_angle = math.atan2(velocity[1], velocity[0])
        horizontal_angle_degrees = math.degrees(horizontal_angle)
        
        return horizontal_angle_degrees
    
    def analyze_driver_sequence(self, driver_dir, output_dir):
        """드라이버 시퀀스 분석"""
        print(f"\n=== ANALYZING {driver_dir} ===")
        
        # 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        
        # 스테레오 매칭을 위한 두 카메라 동시 분석
        print(f"\n--- Corrected Stereo Analysis ---")
        
        # 이미지 파일 목록 (프레임 번호로 정렬)
        pattern_cam1 = os.path.join(driver_dir, "Cam1_*.bmp")
        pattern_cam2 = os.path.join(driver_dir, "Cam2_*.bmp")
        files_cam1 = sorted(glob.glob(pattern_cam1), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        files_cam2 = sorted(glob.glob(pattern_cam2), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        print(f"Found {len(files_cam1)} Cam1 images")
        print(f"Found {len(files_cam2)} Cam2 images")
        
        # 프레임별로 매칭
        positions_3d = []
        timestamps = []
        frame_numbers = []
        
        min_frames = min(len(files_cam1), len(files_cam2))
        
        for i in range(min_frames):
            # 파일명에서 프레임 번호 추출
            filename1 = os.path.basename(files_cam1[i])
            filename2 = os.path.basename(files_cam2[i])
            frame_num1 = int(filename1.split('_')[1].split('.')[0])
            frame_num2 = int(filename2.split('_')[1].split('.')[0])
            
            if frame_num1 != frame_num2:
                print(f"Frame mismatch: Cam1={frame_num1}, Cam2={frame_num2}")
                continue
            
            frame_num = frame_num1
            
            # 이미지 로드
            img1 = cv2.imread(files_cam1[i])
            img2 = cv2.imread(files_cam2[i])
            
            if img1 is None or img2 is None:
                print(f"Frame {frame_num}: Failed to load images")
                continue
            
            # 골프공 검출
            center1, radius1 = self.detect_golf_ball_adaptive(img1)
            center2, radius2 = self.detect_golf_ball_adaptive(img2)
            
            if center1 is not None and center2 is not None:
                # 스테레오 매칭으로 3D 위치 계산
                position_3d = self.calculate_3d_position_stereo(center1, center2)
                
                if position_3d is not None:
                    positions_3d.append(position_3d)
                    timestamps.append(frame_num * self.frame_interval)
                    frame_numbers.append(frame_num)
                    
                    print(f"Frame {frame_num}: 3D position {position_3d}")
                else:
                    print(f"Frame {frame_num}: Stereo matching failed")
            else:
                print(f"Frame {frame_num}: Not detected in one or both cameras")
        
        # 물리량 계산
        if len(positions_3d) >= 2:
            print(f"\nDetected {len(positions_3d)} 3D positions")
            
            velocity, speed = self.calculate_velocity_accurate(positions_3d, frame_numbers)
            
            if velocity is not None:
                launch_angle = self.calculate_launch_angle(velocity)
                direction_angle = self.calculate_direction_angle(velocity)
                
                print(f"\nCorrected Physics Analysis:")
                print(f"  Speed: {speed:.2f} mm/s ({speed/1000:.2f} m/s)")
                print(f"  Velocity vector: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}] mm/s")
                print(f"  Launch angle: {launch_angle:.2f} degrees")
                print(f"  Direction angle: {direction_angle:.2f} degrees")
                
                # 실제 샷 데이터와 비교
                print(f"\nComparison with Real Shot Data:")
                print(f"  Real Ball Speed: 60.9 m/s")
                print(f"  Calculated Speed: {speed/1000:.2f} m/s")
                print(f"  Speed Ratio: {(speed/1000)/60.9:.2f}x")
                print(f"  Real Launch Angle: 9.3°")
                print(f"  Calculated Launch Angle: {launch_angle:.2f}°")
                
                # 결과 저장
                result = {
                    'camera_system': 'Corrected_Stereo',
                    'fps': self.fps,
                    'baseline_mm': self.baseline_mm,
                    'coordinate_system': self.coordinate_system,
                    'positions_3d': [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in positions_3d],
                    'timestamps': [float(t) for t in timestamps],
                    'frame_numbers': [int(f) for f in frame_numbers],
                    'velocity': [float(v) for v in velocity],
                    'speed_mm_s': float(speed),
                    'speed_m_s': float(speed / 1000),
                    'launch_angle': float(launch_angle),
                    'direction_angle': float(direction_angle),
                    'comparison_with_real_data': {
                        'real_ball_speed_m_s': 60.9,
                        'calculated_speed_m_s': float(speed / 1000),
                        'speed_ratio': float((speed/1000)/60.9),
                        'real_launch_angle_deg': 9.3,
                        'calculated_launch_angle_deg': float(launch_angle)
                    }
                }
                
                # JSON 파일로 저장
                result_file = os.path.join(output_dir, "corrected_stereo_physics_analysis.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"  Results saved to: {result_file}")
            else:
                print(f"  Could not calculate velocity")
        else:
            print(f"Not enough 3D positions for analysis")
    
    def analyze_all_drivers(self):
        """모든 드라이버 시퀀스 분석"""
        print("=== CORRECTED GOLF BALL PHYSICS ANALYSIS ===")
        print("Using proper vertical stereo calibration and coordinate system")
        
        drivers = ['2', '3', '4']
        
        for driver in drivers:
            driver_dir = f"data2/driver/{driver}"
            output_dir = f"driver_{driver}_corrected_physics_result"
            
            if os.path.exists(driver_dir):
                self.analyze_driver_sequence(driver_dir, output_dir)
            else:
                print(f"Directory {driver_dir} not found")

def main():
    """메인 함수"""
    analyzer = CorrectedGolfPhysicsAnalyzer()
    analyzer.analyze_all_drivers()
    
    print("\n=== CORRECTED ANALYSIS COMPLETE ===")
    print("Check the driver_X_corrected_physics_result directories for detailed results!")

if __name__ == "__main__":
    main()
