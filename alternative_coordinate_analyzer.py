#!/usr/bin/env python3
"""
대안적 좌표계 분석기
다양한 좌표계 가정으로 발사각 보정 시도
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

class AlternativeCoordinateAnalyzer:
    def __init__(self, calibration_file="vertical_stereo_calibration_z_axis.json"):
        """대안적 좌표계 분석기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        self.fps = 820
        self.frame_interval = 1.0 / self.fps
        
        print("Alternative Coordinate Analyzer Initialized")
        print(f"Testing different coordinate system assumptions")
    
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
    
    def calculate_3d_position_x_disparity(self, center1, center2):
        """X 시차를 사용한 3D 위치 계산"""
        if center1 is None or center2 is None:
            return None
        
        u1, v1 = center1
        u2, v2 = center2
        
        # X 좌표 시차 사용
        disparity = abs(u1 - u2)
        
        if disparity <= 1:
            return None
        
        depth = (self.focal_length * self.baseline_mm) / disparity
        
        if depth < 100 or depth > 10000:
            return None
        
        # 3D 좌표 계산 (X 시차 기반)
        x = (u1 - self.K1[0, 2]) * depth / self.focal_length
        y = (v1 - self.K1[1, 2]) * depth / self.focal_length
        z = depth
        
        return np.array([x, y, z])
    
    def calculate_3d_position_y_disparity(self, center1, center2):
        """Y 시차를 사용한 3D 위치 계산"""
        if center1 is None or center2 is None:
            return None
        
        u1, v1 = center1
        u2, v2 = center2
        
        # Y 좌표 시차 사용
        disparity = abs(v1 - v2)
        
        if disparity <= 1:
            return None
        
        depth = (self.focal_length * self.baseline_mm) / disparity
        
        if depth < 100 or depth > 10000:
            return None
        
        # 3D 좌표 계산 (Y 시차 기반)
        x = (u1 - self.K1[0, 2]) * depth / self.focal_length
        y = (v1 - self.K1[1, 2]) * depth / self.focal_length
        z = depth
        
        return np.array([x, y, z])
    
    def test_coordinate_systems(self, positions_3d, frame_numbers, real_launch_angle=9.3):
        """다양한 좌표계 가정 테스트"""
        if len(positions_3d) < 2:
            return None
        
        # 궤적 방향 분석
        trajectory_vectors = []
        for i in range(1, len(positions_3d)):
            if positions_3d[i] is not None and positions_3d[i-1] is not None:
                vector = positions_3d[i] - positions_3d[i-1]
                trajectory_vectors.append(vector)
        
        if not trajectory_vectors:
            return None
        
        avg_trajectory = np.mean(trajectory_vectors, axis=0)
        avg_trajectory = avg_trajectory / np.linalg.norm(avg_trajectory)
        
        print(f"Original trajectory direction: {avg_trajectory}")
        
        # 다양한 좌표계 가정 테스트
        coordinate_systems = [
            {
                'name': 'Original (X=forward, Y=right, Z=up)',
                'x_axis': [1, 0, 0],
                'y_axis': [0, 1, 0],
                'z_axis': [0, 0, 1]
            },
            {
                'name': 'Alternative 1 (X=right, Y=up, Z=forward)',
                'x_axis': [0, 1, 0],
                'y_axis': [0, 0, 1],
                'z_axis': [1, 0, 0]
            },
            {
                'name': 'Alternative 2 (X=up, Y=forward, Z=right)',
                'x_axis': [0, 0, 1],
                'y_axis': [1, 0, 0],
                'z_axis': [0, 1, 0]
            },
            {
                'name': 'Alternative 3 (X=forward, Y=up, Z=right)',
                'x_axis': [1, 0, 0],
                'y_axis': [0, 0, 1],
                'z_axis': [0, 1, 0]
            }
        ]
        
        best_system = None
        best_error = float('inf')
        
        for system in coordinate_systems:
            print(f"\n--- Testing {system['name']} ---")
            
            # 좌표계 변환
            x_axis = np.array(system['x_axis'])
            y_axis = np.array(system['y_axis'])
            z_axis = np.array(system['z_axis'])
            
            # 변환 행렬
            transform_matrix = np.array([x_axis, y_axis, z_axis])
            
            # 궤적을 새로운 좌표계로 변환
            transformed_trajectory = transform_matrix @ avg_trajectory
            
            # 발사각 계산 (Z축이 수직이라고 가정)
            if abs(transformed_trajectory[2]) <= 1:
                launch_angle = math.degrees(math.asin(abs(transformed_trajectory[2])))
            else:
                launch_angle = 90.0
            
            error = abs(launch_angle - real_launch_angle)
            
            print(f"  Transformed trajectory: {transformed_trajectory}")
            print(f"  Calculated launch angle: {launch_angle:.2f}°")
            print(f"  Error: {error:.2f}°")
            
            if error < best_error:
                best_error = error
                best_system = system
                best_system['transform_matrix'] = transform_matrix
                best_system['launch_angle'] = launch_angle
                best_system['error'] = error
        
        return best_system
    
    def analyze_with_alternative_coordinates(self, calibration_dir, output_dir):
        """대안적 좌표계로 분석"""
        print(f"\n=== ANALYZING WITH ALTERNATIVE COORDINATE SYSTEMS ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 이미지 파일 목록
        pattern_cam1 = os.path.join(calibration_dir, "Cam1_*.bmp")
        pattern_cam2 = os.path.join(calibration_dir, "Cam2_*.bmp")
        files_cam1 = sorted(glob.glob(pattern_cam1), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        files_cam2 = sorted(glob.glob(pattern_cam2), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        print(f"Found {len(files_cam1)} Cam1 calibration images")
        print(f"Found {len(files_cam2)} Cam2 calibration images")
        
        # X 시차와 Y 시차 모두 테스트
        for disparity_type in ['X', 'Y']:
            print(f"\n{'='*60}")
            print(f"TESTING {disparity_type} DISPARITY")
            print(f"{'='*60}")
            
            positions_3d = []
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
                    if disparity_type == 'X':
                        position_3d = self.calculate_3d_position_x_disparity(center1, center2)
                    else:
                        position_3d = self.calculate_3d_position_y_disparity(center1, center2)
                    
                    if position_3d is not None:
                        positions_3d.append(position_3d)
                        frame_numbers.append(frame_num)
            
            if len(positions_3d) >= 2:
                print(f"\nDetected {len(positions_3d)} 3D positions with {disparity_type} disparity")
                
                # 다양한 좌표계 테스트
                best_system = self.test_coordinate_systems(positions_3d, frame_numbers)
                
                if best_system:
                    print(f"\nBEST COORDINATE SYSTEM: {best_system['name']}")
                    print(f"Launch angle error: {best_system['error']:.2f}°")
                    
                    # 결과 저장
                    result = {
                        'disparity_type': disparity_type,
                        'best_coordinate_system': {
                            'name': best_system['name'],
                            'x_axis': best_system['x_axis'],
                            'y_axis': best_system['y_axis'],
                            'z_axis': best_system['z_axis'],
                            'transform_matrix': best_system['transform_matrix'].tolist(),
                            'launch_angle': best_system['launch_angle'],
                            'error': best_system['error']
                        },
                        'positions_3d': [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in positions_3d],
                        'frame_numbers': [int(f) for f in frame_numbers]
                    }
                    
                    result_file = os.path.join(output_dir, f"alternative_coordinates_{disparity_type.lower()}_disparity.json")
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    print(f"Results saved to: {result_file}")
            else:
                print(f"Not enough 3D positions for {disparity_type} disparity")

def main():
    """메인 함수"""
    analyzer = AlternativeCoordinateAnalyzer()
    
    calibration_dir = "data2/Calibration_image_1025"
    output_dir = "alternative_coordinates_result"
    
    if os.path.exists(calibration_dir):
        analyzer.analyze_with_alternative_coordinates(calibration_dir, output_dir)
    else:
        print(f"Calibration directory {calibration_dir} not found")
    
    print("\n=== ALTERNATIVE COORDINATE ANALYSIS COMPLETE ===")
    print("Check the alternative_coordinates_result directory for detailed results!")

if __name__ == "__main__":
    main()
