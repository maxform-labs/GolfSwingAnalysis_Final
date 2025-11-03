#!/usr/bin/env python3
"""
좌표계 재정의 분석기
드라이버 이미지에 맞는 새로운 좌표계를 정의하고 최적화
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import math

class CoordinateSystemRedefinitionAnalyzer:
    def __init__(self, calibration_file="vertical_stereo_calibration_z_axis.json"):
        """좌표계 재정의 분석기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        # 스케일 팩터
        self.scale_factor = 3.6  # 1080/300
        
        self.fps = 820
        self.frame_interval = 1.0 / self.fps
        
        print("Coordinate System Redefinition Analyzer Initialized")
        print(f"Scale factor: {self.scale_factor:.3f}")
        print(f"High-speed camera: {self.fps} fps")
        print(f"Baseline: {self.baseline_mm}mm")
    
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
    
    def calculate_3d_position_with_new_coordinates(self, center1, center2, coordinate_system):
        """새로운 좌표계를 사용한 3D 위치 계산"""
        if center1 is None or center2 is None:
            return None
        
        u1, v1 = center1
        u2, v2 = center2
        
        # 시차 계산 (X 또는 Y 방향)
        if coordinate_system['disparity_type'] == 'x':
            disparity = abs(u1 - u2)
        else:  # y
            disparity = abs(v1 - v2)
        
        if disparity <= 1:
            return None
        
        # 깊이 계산
        depth = (self.focal_length * self.baseline_mm) / disparity
        
        if depth < 100 or depth > 10000:
            return None
        
        # 3D 좌표 계산
        x = (u1 - self.K1[0, 2]) * depth / self.focal_length
        y = (v1 - self.K1[1, 2]) * depth / self.focal_length
        z = depth
        
        # 스케일 팩터 적용
        if coordinate_system['apply_scale_factor']:
            y = y * self.scale_factor
        
        return np.array([x, y, z])
    
    def calculate_physics_with_new_coordinates(self, positions_3d, frame_numbers, coordinate_system):
        """새로운 좌표계에서 물리량 계산"""
        if len(positions_3d) < 2:
            return None
        
        # 좌표계 변환
        transform_matrix = np.array(coordinate_system['transform_matrix'])
        transformed_positions = []
        for pos in positions_3d:
            if pos is not None:
                transformed_pos = transform_matrix @ pos
                transformed_positions.append(transformed_pos)
            else:
                transformed_positions.append(None)
        
        # 속도 계산
        velocities = []
        for i in range(1, len(transformed_positions)):
            if transformed_positions[i] is not None and transformed_positions[i-1] is not None:
                frame_diff = frame_numbers[i] - frame_numbers[i-1]
                dt = frame_diff * self.frame_interval
                
                if dt > 0:
                    velocity = (transformed_positions[i] - transformed_positions[i-1]) / dt
                    velocities.append(velocity)
        
        if not velocities:
            return None
        
        avg_velocity = np.mean(velocities, axis=0)
        speed = np.linalg.norm(avg_velocity)
        
        # 발사각 계산
        if coordinate_system['vertical_axis'] == 'y':
            vertical_component = avg_velocity[1]  # Y축이 수직
        elif coordinate_system['vertical_axis'] == 'z':
            vertical_component = avg_velocity[2]  # Z축이 수직
        else:
            vertical_component = avg_velocity[0]  # X축이 수직
        
        if speed > 0:
            launch_angle = math.degrees(math.asin(abs(vertical_component) / speed))
        else:
            launch_angle = 0
        
        # 방향각 계산
        if coordinate_system['vertical_axis'] == 'y':
            horizontal_angle = math.atan2(avg_velocity[2], avg_velocity[0])  # Z, X
        elif coordinate_system['vertical_axis'] == 'z':
            horizontal_angle = math.atan2(avg_velocity[1], avg_velocity[0])  # Y, X
        else:
            horizontal_angle = math.atan2(avg_velocity[2], avg_velocity[1])  # Z, Y
        
        direction_angle = math.degrees(horizontal_angle)
        
        return {
            'velocity': avg_velocity.tolist(),
            'speed_mm_s': float(speed),
            'speed_m_s': float(speed / 1000),
            'launch_angle': float(launch_angle),
            'direction_angle': float(direction_angle),
            'coordinate_system': coordinate_system['name'],
            'disparity_type': coordinate_system['disparity_type'],
            'vertical_axis': coordinate_system['vertical_axis'],
            'scale_factor_applied': coordinate_system['apply_scale_factor']
        }
    
    def test_coordinate_systems(self, test_shot_dir="data2/driver/2", real_data=None):
        """다양한 좌표계 테스트"""
        print(f"\n=== TESTING COORDINATE SYSTEMS ===")
        
        # 테스트할 좌표계들
        coordinate_systems = [
            # 기본 좌표계들
            {
                'name': 'Standard X-Forward Y-Up Z-Right',
                'disparity_type': 'x',
                'vertical_axis': 'y',
                'apply_scale_factor': False,
                'transform_matrix': [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
            },
            {
                'name': 'Standard X-Forward Y-Up Z-Right (Scale)',
                'disparity_type': 'x',
                'vertical_axis': 'y',
                'apply_scale_factor': True,
                'transform_matrix': [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
            },
            {
                'name': 'Y-Disparity X-Forward Y-Up Z-Right',
                'disparity_type': 'y',
                'vertical_axis': 'y',
                'apply_scale_factor': False,
                'transform_matrix': [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
            },
            {
                'name': 'Y-Disparity X-Forward Y-Up Z-Right (Scale)',
                'disparity_type': 'y',
                'vertical_axis': 'y',
                'apply_scale_factor': True,
                'transform_matrix': [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
            },
            # Z축이 수직인 좌표계들
            {
                'name': 'X-Forward Y-Right Z-Up',
                'disparity_type': 'x',
                'vertical_axis': 'z',
                'apply_scale_factor': False,
                'transform_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            },
            {
                'name': 'X-Forward Y-Right Z-Up (Scale)',
                'disparity_type': 'x',
                'vertical_axis': 'z',
                'apply_scale_factor': True,
                'transform_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            },
            # X축이 수직인 좌표계들
            {
                'name': 'Y-Forward Z-Right X-Up',
                'disparity_type': 'y',
                'vertical_axis': 'x',
                'apply_scale_factor': False,
                'transform_matrix': [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
            },
            {
                'name': 'Y-Forward Z-Right X-Up (Scale)',
                'disparity_type': 'y',
                'vertical_axis': 'x',
                'apply_scale_factor': True,
                'transform_matrix': [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
            }
        ]
        
        # 테스트 샷 분석
        pattern_cam1 = os.path.join(test_shot_dir, "Cam1_*.bmp")
        pattern_cam2 = os.path.join(test_shot_dir, "Cam2_*.bmp")
        files_cam1 = sorted(glob.glob(pattern_cam1), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        files_cam2 = sorted(glob.glob(pattern_cam2), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        print(f"Testing with {len(files_cam1)} Cam1 images, {len(files_cam2)} Cam2 images")
        
        # 2D 위치 수집
        centers_2d = []
        frame_numbers_all = []
        
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
                centers_2d.append((center1, center2))
                frame_numbers_all.append(frame_num)
        
        if len(centers_2d) < 2:
            print("Not enough 2D positions for testing")
            return None
        
        print(f"Collected {len(centers_2d)} 2D position pairs")
        
        # 각 좌표계 테스트
        results = []
        for coord_sys in coordinate_systems:
            print(f"\nTesting: {coord_sys['name']}")
            
            # 해당 좌표계로 3D 위치 계산
            positions_3d = []
            for center1, center2 in centers_2d:
                pos_3d = self.calculate_3d_position_with_new_coordinates(center1, center2, coord_sys)
                if pos_3d is not None:
                    positions_3d.append(pos_3d)
            
            if len(positions_3d) >= 2:
                physics = self.calculate_physics_with_new_coordinates(positions_3d, frame_numbers_all, coord_sys)
                
                if physics:
                    print(f"  Speed: {physics['speed_m_s']:.2f} m/s")
                    print(f"  Launch angle: {physics['launch_angle']:.2f}°")
                    print(f"  Direction angle: {physics['direction_angle']:.2f}°")
                    
                    # 실제 데이터와 비교
                    if real_data is not None and len(real_data) > 0:
                        real_speed = real_data.iloc[1]['BallSpeed(m/s)']  # Shot 2
                        real_launch_angle = real_data.iloc[1]['LaunchAngle(deg)']
                        real_direction_angle = real_data.iloc[1]['LaunchDirection(deg)']
                        
                        speed_error = abs(physics['speed_m_s'] - real_speed) / real_speed * 100
                        launch_angle_error = abs(physics['launch_angle'] - real_launch_angle)
                        direction_angle_error = abs(physics['direction_angle'] - real_direction_angle)
                        
                        print(f"  Speed error: {speed_error:.1f}%")
                        print(f"  Launch angle error: {launch_angle_error:.2f}°")
                        print(f"  Direction angle error: {direction_angle_error:.2f}°")
                        
                        results.append({
                            'coordinate_system': coord_sys['name'],
                            'disparity_type': coord_sys['disparity_type'],
                            'vertical_axis': coord_sys['vertical_axis'],
                            'apply_scale_factor': coord_sys['apply_scale_factor'],
                            'speed_m_s': physics['speed_m_s'],
                            'launch_angle': physics['launch_angle'],
                            'direction_angle': physics['direction_angle'],
                            'speed_error_percent': speed_error,
                            'launch_angle_error': launch_angle_error,
                            'direction_angle_error': direction_angle_error,
                            'total_error': speed_error + launch_angle_error + direction_angle_error
                        })
                    else:
                        results.append({
                            'coordinate_system': coord_sys['name'],
                            'disparity_type': coord_sys['disparity_type'],
                            'vertical_axis': coord_sys['vertical_axis'],
                            'apply_scale_factor': coord_sys['apply_scale_factor'],
                            'speed_m_s': physics['speed_m_s'],
                            'launch_angle': physics['launch_angle'],
                            'direction_angle': physics['direction_angle'],
                            'speed_error_percent': 0,
                            'launch_angle_error': 0,
                            'direction_angle_error': 0,
                            'total_error': 0
                        })
                else:
                    print("  Failed to calculate physics")
            else:
                print("  Not enough 3D positions")
        
        return results
    
    def find_optimal_coordinate_system(self, test_results):
        """최적 좌표계 찾기"""
        if not test_results:
            return None
        
        # 총 오차가 가장 작은 좌표계 선택
        best_result = min(test_results, key=lambda x: x['total_error'])
        
        print(f"\n=== OPTIMAL COORDINATE SYSTEM FOUND ===")
        print(f"System: {best_result['coordinate_system']}")
        print(f"Disparity type: {best_result['disparity_type']}")
        print(f"Vertical axis: {best_result['vertical_axis']}")
        print(f"Scale factor applied: {best_result['apply_scale_factor']}")
        print(f"Speed: {best_result['speed_m_s']:.2f} m/s")
        print(f"Launch angle: {best_result['launch_angle']:.2f}°")
        print(f"Direction angle: {best_result['direction_angle']:.2f}°")
        print(f"Total error: {best_result['total_error']:.2f}")
        
        return best_result

def main():
    """메인 함수"""
    analyzer = CoordinateSystemRedefinitionAnalyzer()
    
    # 실제 데이터 로드
    real_data = None
    csv_file = "data2/driver/shotdata_20251020.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = df.dropna()
        real_data = df
        print(f"Loaded real data: {len(real_data)} shots")
    
    # 좌표계 테스트
    test_results = analyzer.test_coordinate_systems("data2/driver/2", real_data)
    
    if test_results:
        # 결과 저장
        df_results = pd.DataFrame(test_results)
        df_results.to_csv("coordinate_system_test_results.csv", index=False)
        print(f"\nTest results saved to: coordinate_system_test_results.csv")
        
        # 최적 좌표계 찾기
        optimal_system = analyzer.find_optimal_coordinate_system(test_results)
        
        if optimal_system:
            print(f"\nOptimal coordinate system identified!")
            print(f"Use this system for further analysis.")
    else:
        print("No valid test results")
    
    print("\n=== COORDINATE SYSTEM REDEFINITION COMPLETE ===")

if __name__ == "__main__":
    main()
