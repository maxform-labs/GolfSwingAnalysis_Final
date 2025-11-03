#!/usr/bin/env python3
"""
발사각 보정 분석기
골프공의 실제 움직임 방향과 계산된 좌표계 간의 보정
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

class CorrectedAngleAnalyzer:
    def __init__(self, calibration_file="vertical_stereo_calibration_z_axis.json"):
        """발사각 보정 분석기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        # 고속카메라 설정
        self.fps = 820
        self.frame_interval = 1.0 / self.fps
        
        print("Corrected Angle Analyzer Initialized")
        print(f"High-speed camera: {self.fps} fps")
        print(f"Baseline: {self.baseline_mm}mm (Z-axis)")
        print(f"Focal length: {self.focal_length}px")
    
    def load_calibration(self):
        """캘리브레이션 데이터 로드"""
        with open(self.calibration_file, 'r', encoding='utf-8') as f:
            self.calibration_data = json.load(f)
        
        self.K1 = np.array(self.calibration_data['camera_matrix_1'])
        self.K2 = np.array(self.calibration_data['camera_matrix_2'])
        self.D1 = np.array(self.calibration_data['distortion_coeffs_1'])
        self.D2 = np.array(self.calibration_data['distortion_coeffs_2'])
        self.R = np.array(self.calibration_data['rotation_matrix'])
        self.T = np.array(self.calibration_data['translation_vector'])
        
        self.baseline_mm = self.calibration_data['baseline_mm']
        self.image_size = tuple(self.calibration_data['image_size'])
        self.focal_length = self.K1[0, 0]
        self.coordinate_system = self.calibration_data.get('coordinate_system', {})
    
    def detect_golf_ball_adaptive(self, frame: np.ndarray) -> tuple:
        """적응형 골프공 검출"""
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
            if 50 < area < 10000:
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
        """스테레오 매칭을 통한 3D 위치 계산"""
        if center1 is None or center2 is None:
            return None
        
        u1, v1 = center1
        u2, v2 = center2
        
        # Y 좌표 시차 사용 (수직 스테레오)
        disparity = abs(v1 - v2)
        
        if disparity <= 1:
            return None
        
        depth = (self.focal_length * self.baseline_mm) / disparity
        
        if depth < 100 or depth > 10000:
            return None
        
        # 3D 좌표 계산
        x = (u1 - self.K1[0, 2]) * depth / self.focal_length
        y = (v1 - self.K1[1, 2]) * depth / self.focal_length
        z = depth
        
        return np.array([x, y, z])
    
    def analyze_trajectory_direction(self, positions_3d):
        """궤적 방향 분석"""
        if len(positions_3d) < 2:
            return None
        
        # 궤적의 주요 방향 벡터 계산
        trajectory_vectors = []
        for i in range(1, len(positions_3d)):
            if positions_3d[i] is not None and positions_3d[i-1] is not None:
                vector = positions_3d[i] - positions_3d[i-1]
                trajectory_vectors.append(vector)
        
        if not trajectory_vectors:
            return None
        
        # 평균 궤적 방향
        avg_trajectory = np.mean(trajectory_vectors, axis=0)
        avg_trajectory = avg_trajectory / np.linalg.norm(avg_trajectory)  # 정규화
        
        return avg_trajectory
    
    def calculate_coordinate_correction(self, positions_3d, real_launch_angle=9.3):
        """좌표계 보정 계산"""
        if len(positions_3d) < 2:
            return None, None
        
        # 궤적 방향 분석
        trajectory_direction = self.analyze_trajectory_direction(positions_3d)
        if trajectory_direction is None:
            return None, None
        
        print(f"Original trajectory direction: {trajectory_direction}")
        
        # 현재 좌표계에서의 발사각 계산
        current_launch_angle = math.degrees(math.asin(abs(trajectory_direction[2])))
        print(f"Current calculated launch angle: {current_launch_angle:.2f}°")
        print(f"Real launch angle: {real_launch_angle}°")
        
        # 보정 각도 계산
        angle_correction = real_launch_angle - current_launch_angle
        print(f"Angle correction needed: {angle_correction:.2f}°")
        
        # 좌표계 회전 행렬 계산
        # Z축을 중심으로 회전하여 발사각 보정
        correction_angle_rad = math.radians(angle_correction)
        
        # X-Z 평면에서 회전 (발사각 보정)
        rotation_matrix = np.array([
            [math.cos(correction_angle_rad), 0, math.sin(correction_angle_rad)],
            [0, 1, 0],
            [-math.sin(correction_angle_rad), 0, math.cos(correction_angle_rad)]
        ])
        
        return rotation_matrix, angle_correction
    
    def apply_coordinate_correction(self, positions_3d, rotation_matrix):
        """좌표계 보정 적용"""
        corrected_positions = []
        for pos in positions_3d:
            if pos is not None:
                corrected_pos = rotation_matrix @ pos
                corrected_positions.append(corrected_pos)
            else:
                corrected_positions.append(None)
        
        return corrected_positions
    
    def calculate_corrected_velocity(self, corrected_positions, frame_numbers):
        """보정된 좌표계에서 속도 계산"""
        if len(corrected_positions) < 2:
            return None, None
        
        velocities = []
        for i in range(1, len(corrected_positions)):
            if corrected_positions[i] is not None and corrected_positions[i-1] is not None:
                frame_diff = frame_numbers[i] - frame_numbers[i-1]
                dt = frame_diff * self.frame_interval
                
                if dt > 0:
                    velocity = (corrected_positions[i] - corrected_positions[i-1]) / dt
                    velocities.append(velocity)
        
        if not velocities:
            return None, None
        
        avg_velocity = np.mean(velocities, axis=0)
        speed = np.linalg.norm(avg_velocity)
        
        return avg_velocity, speed
    
    def calculate_corrected_launch_angle(self, velocity):
        """보정된 발사각 계산"""
        if velocity is None:
            return None
        
        vertical_component = velocity[2]
        total_speed = np.linalg.norm(velocity)
        
        if total_speed == 0:
            return None
        
        launch_angle = math.asin(abs(vertical_component) / total_speed)
        launch_angle_degrees = math.degrees(launch_angle)
        
        return launch_angle_degrees
    
    def calculate_corrected_direction_angle(self, velocity):
        """보정된 방향각 계산"""
        if velocity is None:
            return None
        
        horizontal_angle = math.atan2(velocity[1], velocity[0])
        horizontal_angle_degrees = math.degrees(horizontal_angle)
        
        return horizontal_angle_degrees
    
    def analyze_calibration_with_correction(self, calibration_dir, output_dir):
        """보정을 적용한 캘리브레이션 분석"""
        print(f"\n=== ANALYZING CALIBRATION WITH ANGLE CORRECTION ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 이미지 파일 목록
        pattern_cam1 = os.path.join(calibration_dir, "Cam1_*.bmp")
        pattern_cam2 = os.path.join(calibration_dir, "Cam2_*.bmp")
        files_cam1 = sorted(glob.glob(pattern_cam1), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        files_cam2 = sorted(glob.glob(pattern_cam2), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        print(f"Found {len(files_cam1)} Cam1 calibration images")
        print(f"Found {len(files_cam2)} Cam2 calibration images")
        
        # 프레임별로 매칭
        positions_3d = []
        timestamps = []
        frame_numbers = []
        
        min_frames = min(len(files_cam1), len(files_cam2))
        
        for i in range(min_frames):
            filename1 = os.path.basename(files_cam1[i])
            filename2 = os.path.basename(files_cam2[i])
            frame_num1 = int(filename1.split('_')[1].split('.')[0])
            frame_num2 = int(filename2.split('_')[1].split('.')[0])
            
            if frame_num1 != frame_num2:
                continue
            
            frame_num = frame_num1
            
            img1 = cv2.imread(files_cam1[i])
            img2 = cv2.imread(files_cam2[i])
            
            if img1 is None or img2 is None:
                continue
            
            center1, radius1 = self.detect_golf_ball_adaptive(img1)
            center2, radius2 = self.detect_golf_ball_adaptive(img2)
            
            if center1 is not None and center2 is not None:
                position_3d = self.calculate_3d_position_stereo(center1, center2)
                
                if position_3d is not None:
                    positions_3d.append(position_3d)
                    timestamps.append(frame_num * self.frame_interval)
                    frame_numbers.append(frame_num)
        
        if len(positions_3d) >= 2:
            print(f"\nDetected {len(positions_3d)} 3D positions")
            
            # 좌표계 보정 계산
            rotation_matrix, angle_correction = self.calculate_coordinate_correction(positions_3d)
            
            if rotation_matrix is not None:
                # 보정 적용
                corrected_positions = self.apply_coordinate_correction(positions_3d, rotation_matrix)
                
                # 보정된 속도 계산
                corrected_velocity, corrected_speed = self.calculate_corrected_velocity(corrected_positions, frame_numbers)
                
                if corrected_velocity is not None:
                    corrected_launch_angle = self.calculate_corrected_launch_angle(corrected_velocity)
                    corrected_direction_angle = self.calculate_corrected_direction_angle(corrected_velocity)
                    
                    print(f"\nCorrected Physics Analysis:")
                    print(f"  Speed: {corrected_speed:.2f} mm/s ({corrected_speed/1000:.2f} m/s)")
                    print(f"  Velocity vector: [{corrected_velocity[0]:.2f}, {corrected_velocity[1]:.2f}, {corrected_velocity[2]:.2f}] mm/s")
                    print(f"  Launch angle: {corrected_launch_angle:.2f} degrees")
                    print(f"  Direction angle: {corrected_direction_angle:.2f} degrees")
                    print(f"  Angle correction applied: {angle_correction:.2f} degrees")
                    
                    # 실제 샷 데이터와 비교
                    print(f"\nComparison with Real Shot Data:")
                    print(f"  Real Ball Speed: 60.9 m/s")
                    print(f"  Corrected Speed: {corrected_speed/1000:.2f} m/s")
                    print(f"  Speed Ratio: {(corrected_speed/1000)/60.9:.2f}x")
                    print(f"  Real Launch Angle: 9.3°")
                    print(f"  Corrected Launch Angle: {corrected_launch_angle:.2f}°")
                    print(f"  Launch Angle Error: {abs(corrected_launch_angle - 9.3):.2f}°")
                    
                    # 결과 저장
                    result = {
                        'camera_system': 'Corrected_Stereo',
                        'fps': self.fps,
                        'baseline_mm': self.baseline_mm,
                        'coordinate_system': self.coordinate_system,
                        'angle_correction_degrees': float(angle_correction),
                        'rotation_matrix': rotation_matrix.tolist(),
                        'original_positions_3d': [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in positions_3d],
                        'corrected_positions_3d': [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in corrected_positions],
                        'timestamps': [float(t) for t in timestamps],
                        'frame_numbers': [int(f) for f in frame_numbers],
                        'corrected_velocity': [float(v) for v in corrected_velocity],
                        'corrected_speed_mm_s': float(corrected_speed),
                        'corrected_speed_m_s': float(corrected_speed / 1000),
                        'corrected_launch_angle': float(corrected_launch_angle),
                        'corrected_direction_angle': float(corrected_direction_angle),
                        'comparison_with_real_data': {
                            'real_ball_speed_m_s': 60.9,
                            'corrected_speed_m_s': float(corrected_speed / 1000),
                            'speed_ratio': float((corrected_speed/1000)/60.9),
                            'real_launch_angle_deg': 9.3,
                            'corrected_launch_angle_deg': float(corrected_launch_angle),
                            'launch_angle_error_deg': float(abs(corrected_launch_angle - 9.3))
                        }
                    }
                    
                    # JSON 파일로 저장
                    result_file = os.path.join(output_dir, "corrected_angle_physics_analysis.json")
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    print(f"  Results saved to: {result_file}")
                else:
                    print(f"  Could not calculate corrected velocity")
            else:
                print(f"  Could not calculate coordinate correction")
        else:
            print(f"Not enough 3D positions for analysis")

def main():
    """메인 함수"""
    analyzer = CorrectedAngleAnalyzer()
    
    calibration_dir = "data2/Calibration_image_1025"
    output_dir = "corrected_angle_physics_result"
    
    if os.path.exists(calibration_dir):
        analyzer.analyze_calibration_with_correction(calibration_dir, output_dir)
    else:
        print(f"Calibration directory {calibration_dir} not found")
    
    print("\n=== CORRECTED ANGLE ANALYSIS COMPLETE ===")
    print("Check the corrected_angle_physics_result directory for detailed results!")

if __name__ == "__main__":
    main()
