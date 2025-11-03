#!/usr/bin/env python3
"""
골프공 물리량 분석 시스템 (최종版)
전체 이미지를 사용하여 속도, 발사각, 방향각 계산
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

class GolfBallPhysicsAnalyzer:
    def __init__(self, calibration_file="corrected_stereo_calibration_470mm.json"):
        """골프공 물리량 분석기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        print("Golf Ball Physics Analyzer Initialized")
        print(f"Baseline: {self.baseline_mm}mm")
        print(f"Focal length: {self.focal_length}px")
    
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
    
    def detect_golf_ball_adaptive(self, frame: np.ndarray) -> tuple:
        """적응형 골프공 검출"""
        if frame is None or frame.size == 0:
            return None, None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 이미지별 적응형 임계값 설정
        mean_brightness = np.mean(gray)
        
        # 적응형 임계값 계산
        if mean_brightness > 150:
            threshold_low = max(180, mean_brightness - 30)
        else:
            threshold_low = max(120, mean_brightness + 20)
        threshold_high = 255
        
        # 1. 적응형 이진화
        binary = cv2.inRange(gray, threshold_low, threshold_high)
        
        # 2. 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 3. 컨투어 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if 20 < area < 3000:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.5:
                        x, y, w, h = cv2.boundingRect(contour)
                        roi = gray[y:y+h, x:x+w]
                        mean_brightness_roi = np.mean(roi)
                        
                        if mean_brightness_roi > threshold_low * 0.8:
                            score = circularity * (area / 100) * (mean_brightness_roi / 255)
                            
                            if score > best_score:
                                best_score = score
                                best_contour = contour
        
        if best_contour is not None and best_score > 0.1:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center = (int(x), int(y))
            return center, int(radius)
        
        # HoughCircles 보조 검출
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 
            dp=1, minDist=20,
            param1=30, param2=20,
            minRadius=5, maxRadius=30
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for circle in circles:
                center = (circle[0], circle[1])
                radius = circle[2]
                
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, radius, 255, -1)
                mean_brightness = cv2.mean(gray, mask)[0]
                
                if mean_brightness > threshold_low * 0.8:
                    return center, radius
        
        return None, None
    
    def calculate_3d_position(self, center1, center2):
        """3D 위치 계산 (스테레오 매칭)"""
        if center1 is None or center2 is None:
            return None
        
        u1, v1 = center1
        u2, v2 = center2
        
        # 시차 계산
        disparity = u1 - u2
        
        if disparity <= 0:
            return None
        
        # 깊이 계산
        depth = (self.focal_length * self.baseline_mm) / disparity
        
        if depth < 50 or depth > 5000:  # 유효한 깊이 범위
            return None
        
        # 3D 좌표 계산
        x = (u1 - self.K1[0, 2]) * depth / self.focal_length
        y = (v1 - self.K1[1, 2]) * depth / self.focal_length
        z = depth
        
        return np.array([x, y, z])
    
    def calculate_velocity(self, positions, timestamps):
        """속도 계산"""
        if len(positions) < 2:
            return None, None
        
        velocities = []
        for i in range(1, len(positions)):
            if positions[i] is not None and positions[i-1] is not None:
                dt = timestamps[i] - timestamps[i-1]
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
        """발사각 계산 (수직각)"""
        if velocity is None:
            return None
        
        # 속도 벡터의 Z 성분과 전체 속도의 비율
        vertical_component = velocity[2]
        total_speed = np.linalg.norm(velocity)
        
        if total_speed == 0:
            return None
        
        # 발사각 계산 (라디안을 도로 변환)
        launch_angle = math.asin(vertical_component / total_speed)
        launch_angle_degrees = math.degrees(launch_angle)
        
        return launch_angle_degrees
    
    def calculate_direction_angle(self, velocity):
        """방향각 계산 (수평각)"""
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
        
        # 각 카메라별 분석
        for camera in ['Cam1', 'Cam2']:
            print(f"\n--- {camera} Analysis ---")
            
            # 이미지 파일 목록
            pattern = os.path.join(driver_dir, f"{camera}_*.bmp")
            files = sorted(glob.glob(pattern))
            
            positions = []
            timestamps = []
            frame_numbers = []
            
            for file_path in files:
                filename = os.path.basename(file_path)
                frame_num = int(filename.split('_')[1].split('.')[0])
                
                # 이미지 로드
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Frame {frame_num}: Failed to load image")
                    continue
                
                print(f"Frame {frame_num}: Image size {img.shape}")
                
                # 골프공 검출 (전체 이미지 사용)
                center, radius = self.detect_golf_ball_adaptive(img)
                
                if center is not None:
                    positions.append(center)
                    timestamps.append(frame_num * 0.033)  # 30fps 가정
                    frame_numbers.append(frame_num)
                    
                    print(f"Frame {frame_num}: Detected at {center}")
                else:
                    print(f"Frame {frame_num}: Not detected")
            
            # 3D 위치 계산
            if len(positions) > 0:
                print(f"\n{camera} detected {len(positions)} frames")
                
                # 간단한 3D 위치 추정 (단일 카메라)
                # 실제로는 스테레오 매칭이 필요하지만, 여기서는 Z=1000mm 고정
                z_fixed = 1000.0  # mm
                positions_3d = []
                
                for pos in positions:
                    u, v = pos
                    # 간단한 3D 좌표 계산
                    x = (u - self.K1[0, 2]) * z_fixed / self.focal_length
                    y = (v - self.K1[1, 2]) * z_fixed / self.focal_length
                    positions_3d.append(np.array([x, y, z_fixed]))
                
                # 물리량 계산
                if len(positions_3d) >= 2:
                    velocity, speed = self.calculate_velocity(positions_3d, timestamps)
                    
                    if velocity is not None:
                        launch_angle = self.calculate_launch_angle(velocity)
                        direction_angle = self.calculate_direction_angle(velocity)
                        
                        print(f"\n{camera} Physics Analysis:")
                        print(f"  Speed: {speed:.2f} mm/s")
                        print(f"  Velocity vector: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}] mm/s")
                        print(f"  Launch angle: {launch_angle:.2f} degrees")
                        print(f"  Direction angle: {direction_angle:.2f} degrees")
                        
                        # 결과 저장 (모든 numpy 타입을 Python 기본 타입으로 변환)
                        result = {
                            'camera': camera,
                            'positions_2d': [[int(p[0]), int(p[1])] for p in positions],
                            'positions_3d': [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in positions_3d],
                            'timestamps': [float(t) for t in timestamps],
                            'frame_numbers': [int(f) for f in frame_numbers],
                            'velocity': [float(v) for v in velocity],
                            'speed': float(speed),
                            'launch_angle': float(launch_angle),
                            'direction_angle': float(direction_angle)
                        }
                        
                        # JSON 파일로 저장
                        result_file = os.path.join(output_dir, f"{camera}_physics_analysis.json")
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        
                        print(f"  Results saved to: {result_file}")
                    else:
                        print(f"  Could not calculate velocity")
                else:
                    print(f"  Not enough positions for velocity calculation")
            else:
                print(f"{camera}: No golf balls detected")
    
    def analyze_all_drivers(self):
        """모든 드라이버 시퀀스 분석"""
        print("=== GOLF BALL PHYSICS ANALYSIS ===")
        
        drivers = ['2', '3', '4']
        
        for driver in drivers:
            driver_dir = f"data2/driver/{driver}"
            output_dir = f"driver_{driver}_physics_result"
            
            if os.path.exists(driver_dir):
                self.analyze_driver_sequence(driver_dir, output_dir)
            else:
                print(f"Directory {driver_dir} not found")

def main():
    """메인 함수"""
    analyzer = GolfBallPhysicsAnalyzer()
    analyzer.analyze_all_drivers()
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Check the driver_X_physics_result directories for detailed results!")

if __name__ == "__main__":
    main()
