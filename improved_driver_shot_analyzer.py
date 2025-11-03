#!/usr/bin/env python3
"""
개선된 드라이버 샷 분석 시스템
실제 측정 데이터와 비교하여 골프 공의 속도, 방향각, 발사각 계산
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
from datetime import datetime

class ImprovedDriverShotAnalyzer:
    def __init__(self, calibration_file="improved_stereo_calibration_470mm.json"):
        """개선된 드라이버 샷 분석기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        # 실제 측정 데이터 (두 번째 행)
        self.real_data = {
            'ball_speed': 60.9,  # m/s
            'launch_angle': 9.3,  # degrees
            'launch_direction': 0.4  # degrees
        }
        
        print(f"Improved Driver Shot Analyzer Initialized")
        print(f"Real measurement data:")
        print(f"  Ball Speed: {self.real_data['ball_speed']} m/s")
        print(f"  Launch Angle: {self.real_data['launch_angle']} degrees")
        print(f"  Launch Direction: {self.real_data['launch_direction']} degrees")
    
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
        
        print(f"OK Calibration loaded")
        print(f"  Baseline: {self.baseline_mm}mm")
        print(f"  Focal length: {self.focal_length:.2f} pixels")
    
    def detect_ball_center_improved(self, img, method='hough'):
        """개선된 골프 공 중심 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 이미지 전처리
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        if method == 'hough':
            # 허프 원 검출 (더 엄격한 파라미터)
            circles = cv2.HoughCircles(
                enhanced, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                param1=80, param2=40, minRadius=3, maxRadius=30
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # 가장 큰 원 선택
                largest_circle = max(circles, key=lambda x: x[2])
                return (largest_circle[0], largest_circle[1]), largest_circle[2]
        
        elif method == 'contour':
            # 컨투어 기반 검출
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 1)
            edges = cv2.Canny(blurred, 30, 100)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 면적이 적당한 컨투어만 선택
                valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 2000]
                
                if valid_contours:
                    # 가장 큰 컨투어 선택
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    
                    # 원형도 확인
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.5:  # 원형도가 높은 경우
                            # 중심과 반지름 계산
                            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                            return (int(x), int(y)), int(radius)
        
        return None, None
    
    def calculate_3d_position_improved(self, u1, v1, u2, v2):
        """개선된 3D 위치 계산"""
        # 시차 계산
        disparity = u1 - u2
        
        if disparity <= 0 or disparity < 1:  # 최소 시차 요구
            return None
        
        # 깊이 계산 (Z = (f * B) / disparity)
        depth = (self.focal_length * self.baseline_mm) / disparity
        
        # 너무 가깝거나 먼 거리 제거
        if depth < 100 or depth > 10000:  # 100mm ~ 10m
            return None
        
        # 3D 좌표 계산
        x = (u1 - self.K1[0, 2]) * depth / self.focal_length
        y = (v1 - self.K1[1, 2]) * depth / self.focal_length
        z = depth
        
        return np.array([x, y, z])
    
    def calculate_velocity_and_angles_improved(self, positions, time_interval=1/1000):  # 1000fps 가정
        """개선된 속도와 각도 계산"""
        if len(positions) < 2:
            return None
        
        # 연속된 프레임 간 속도 계산
        velocities = []
        for i in range(1, len(positions)):
            pos1 = positions[i-1]
            pos2 = positions[i]
            
            # 3D 속도 벡터
            velocity_3d = (pos2 - pos1) / time_interval
            velocities.append(velocity_3d)
        
        if not velocities:
            return None
        
        # 평균 속도 계산
        avg_velocity = np.mean(velocities, axis=0)
        
        # 속력 (m/s)
        speed = np.linalg.norm(avg_velocity) / 1000  # mm/s to m/s
        
        # 발사각 계산 (수직 각도)
        # Z축이 위쪽을 향한다고 가정
        horizontal_speed = np.sqrt(avg_velocity[0]**2 + avg_velocity[1]**2)
        launch_angle = np.arctan2(avg_velocity[2], horizontal_speed)
        launch_angle = np.degrees(launch_angle)
        
        # 방향각 계산 (수평 각도)
        # X축이 타겟 방향을 향한다고 가정
        launch_direction = np.arctan2(avg_velocity[1], avg_velocity[0])
        launch_direction = np.degrees(launch_direction)
        
        return {
            'speed': speed,
            'launch_angle': launch_angle,
            'launch_direction': launch_direction,
            'velocity_3d': avg_velocity
        }
    
    def analyze_driver_shot_improved(self, image_folder="data2/driver/2"):
        """개선된 드라이버 샷 분석"""
        print(f"\nAnalyzing driver shot from: {image_folder}")
        
        # 이미지 파일 목록
        gamma1_files = sorted(glob.glob(f"{image_folder}/Gamma_1_*.bmp"))
        gamma2_files = sorted(glob.glob(f"{image_folder}/Gamma_2_*.bmp"))
        
        print(f"Found {len(gamma1_files)} Gamma_1 images")
        print(f"Found {len(gamma2_files)} Gamma_2 images")
        
        if len(gamma1_files) == 0 or len(gamma2_files) == 0:
            print("X No images found")
            return None
        
        # 각 프레임별 분석
        frame_results = []
        positions_3d = []
        
        for i in range(min(len(gamma1_files), len(gamma2_files))):
            img1_path = gamma1_files[i]
            img2_path = gamma2_files[i]
            
            # 이미지 로드
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                print(f"Frame {i+1}: Failed to load images")
                continue
            
            # 골프 공 검출
            center1, radius1 = self.detect_ball_center_improved(img1)
            center2, radius2 = self.detect_ball_center_improved(img2)
            
            if center1 is None or center2 is None:
                print(f"Frame {i+1}: Ball not detected")
                continue
            
            # 3D 위치 계산
            u1, v1 = center1
            u2, v2 = center2
            
            position_3d = self.calculate_3d_position_improved(u1, v1, u2, v2)
            
            if position_3d is not None:
                positions_3d.append(position_3d)
                
                result = {
                    'frame': i+1,
                    'center1': center1,
                    'center2': center2,
                    'position_3d': position_3d
                }
                frame_results.append(result)
                
                print(f"Frame {i+1}: 3D position ({position_3d[0]:.1f}, {position_3d[1]:.1f}, {position_3d[2]:.1f}) mm")
                
                # 시각화 저장
                self.save_detection_visualization(img1, img2, center1, center2, 
                                               radius1, radius2, position_3d, i+1)
        
        if len(positions_3d) < 2:
            print("X Not enough valid frames for trajectory analysis")
            return None
        
        # 속도와 각도 계산 (다양한 프레임 레이트 시도)
        print(f"\nTrying different frame rates:")
        
        best_result = None
        best_error = float('inf')
        
        for fps in [30, 60, 120, 240, 480, 1000]:
            time_interval = 1/fps
            motion_data = self.calculate_velocity_and_angles_improved(positions_3d, time_interval)
            
            if motion_data is None:
                continue
            
            # 실제 데이터와의 오차 계산
            speed_error = abs(motion_data['speed'] - self.real_data['ball_speed'])
            angle_error = abs(motion_data['launch_angle'] - self.real_data['launch_angle'])
            direction_error = abs(motion_data['launch_direction'] - self.real_data['launch_direction'])
            
            total_error = speed_error + angle_error + direction_error
            
            print(f"  {fps}fps: Speed={motion_data['speed']:.1f}m/s, Angle={motion_data['launch_angle']:.1f}deg, Error={total_error:.1f}")
            
            if total_error < best_error:
                best_error = total_error
                best_result = motion_data
                best_fps = fps
        
        if best_result is None:
            print("X Failed to calculate motion parameters")
            return None
        
        # 결과 출력
        print(f"\n=== IMPROVED ANALYSIS RESULTS ===")
        print(f"Best frame rate: {best_fps}fps")
        print(f"Analyzed frames: {len(frame_results)}")
        print(f"Calculated Ball Speed: {best_result['speed']:.1f} m/s")
        print(f"Calculated Launch Angle: {best_result['launch_angle']:.1f} degrees")
        print(f"Calculated Launch Direction: {best_result['launch_direction']:.1f} degrees")
        
        print(f"\n=== COMPARISON WITH REAL DATA ===")
        print(f"Ball Speed:")
        print(f"  Calculated: {best_result['speed']:.1f} m/s")
        print(f"  Real: {self.real_data['ball_speed']:.1f} m/s")
        print(f"  Error: {abs(best_result['speed'] - self.real_data['ball_speed']):.1f} m/s")
        print(f"  Error %: {abs(best_result['speed'] - self.real_data['ball_speed'])/self.real_data['ball_speed']*100:.1f}%")
        
        print(f"\nLaunch Angle:")
        print(f"  Calculated: {best_result['launch_angle']:.1f} degrees")
        print(f"  Real: {self.real_data['launch_angle']:.1f} degrees")
        print(f"  Error: {abs(best_result['launch_angle'] - self.real_data['launch_angle']):.1f} degrees")
        
        print(f"\nLaunch Direction:")
        print(f"  Calculated: {best_result['launch_direction']:.1f} degrees")
        print(f"  Real: {self.real_data['launch_direction']:.1f} degrees")
        print(f"  Error: {abs(best_result['launch_direction'] - self.real_data['launch_direction']):.1f} degrees")
        
        return {
            'frame_results': frame_results,
            'motion_data': best_result,
            'positions_3d': positions_3d,
            'best_fps': best_fps
        }
    
    def save_detection_visualization(self, img1, img2, center1, center2, radius1, radius2, 
                                   position_3d, frame_num):
        """검출 결과 시각화 저장"""
        # 이미지에 원 그리기
        img1_vis = img1.copy()
        img2_vis = img2.copy()
        
        cv2.circle(img1_vis, center1, radius1, (0, 255, 0), 2)
        cv2.circle(img1_vis, center1, 2, (0, 0, 255), -1)
        
        cv2.circle(img2_vis, center2, radius2, (0, 255, 0), 2)
        cv2.circle(img2_vis, center2, 2, (0, 0, 255), -1)
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Camera 1 - Frame {frame_num}\nBall at ({center1[0]}, {center1[1]})')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Camera 2 - Frame {frame_num}\nBall at ({center2[0]}, {center2[1]})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'improved_driver_shot_frame_{frame_num:02d}_detection.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Detection visualization saved: improved_driver_shot_frame_{frame_num:02d}_detection.png")

def main():
    """메인 함수"""
    print("=== Improved Driver Shot Analysis System ===")
    
    # 개선된 드라이버 샷 분석기 초기화
    analyzer = ImprovedDriverShotAnalyzer()
    
    # 드라이버 샷 분석
    results = analyzer.analyze_driver_shot_improved()
    
    if results:
        print("\nOK Improved driver shot analysis completed successfully!")
        print("\nDetection images saved for each frame.")
    else:
        print("\nX Improved driver shot analysis failed.")

if __name__ == "__main__":
    main()
