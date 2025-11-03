#!/usr/bin/env python3
"""
근본적 해결책 분석기
실제 골프공 크기와 거리를 기반으로 한 새로운 접근법
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

class FundamentalSolutionAnalyzer:
    def __init__(self):
        """근본적 해결책 분석기 초기화"""
        self.fps = 820
        self.frame_interval = 1.0 / self.fps
        
        # 골프공 실제 크기 (mm)
        self.golf_ball_diameter_mm = 42.67  # 표준 골프공 지름
        
        # 카메라 간 거리 (mm) - 실제 측정값
        self.baseline_mm = 470.0
        
        print("Fundamental Solution Analyzer Initialized")
        print(f"Golf ball diameter: {self.golf_ball_diameter_mm} mm")
        print(f"Camera baseline: {self.baseline_mm} mm")
        print(f"High-speed camera: {self.fps} fps")
    
    def detect_golf_ball_with_size_estimation(self, frame: np.ndarray) -> tuple:
        """크기 추정을 포함한 골프공 검출"""
        if frame is None or frame.size == 0:
            return None, None, None
            
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
        best_radius = 0
        
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
                                # 반지름 추정
                                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                                best_radius = radius
        
        if best_contour is not None and best_score > 0.05:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center = (int(x), int(y))
            return center, int(radius), best_radius
        
        return None, None, None
    
    def calculate_depth_from_ball_size(self, radius_pixels, focal_length_pixels):
        """골프공 크기를 이용한 깊이 계산"""
        if radius_pixels <= 0 or focal_length_pixels <= 0:
            return None
        
        # 실제 골프공 반지름 (mm)
        real_radius_mm = self.golf_ball_diameter_mm / 2
        
        # 깊이 계산: depth = (focal_length * real_size) / pixel_size
        depth_mm = (focal_length_pixels * real_radius_mm) / radius_pixels
        
        return depth_mm
    
    def estimate_focal_length_from_baseline(self, disparity_pixels):
        """베이스라인과 시차를 이용한 초점거리 추정"""
        if disparity_pixels <= 0:
            return None
        
        # 실제 베이스라인 (mm)
        baseline_mm = self.baseline_mm
        
        # 가정: 1m 거리에서의 시차를 기준으로 초점거리 추정
        assumed_depth_mm = 1000  # 1m
        focal_length_pixels = (disparity_pixels * assumed_depth_mm) / baseline_mm
        
        return focal_length_pixels
    
    def calculate_3d_position_fundamental(self, center1, center2, radius1, radius2):
        """근본적 방법으로 3D 위치 계산"""
        if center1 is None or center2 is None or radius1 is None or radius2 is None:
            return None
        
        u1, v1 = center1
        u2, v2 = center2
        
        # 시차 계산
        disparity = abs(u1 - u2)
        
        if disparity <= 1:
            return None
        
        # 방법 1: 시차 기반 깊이 계산
        # 가정된 초점거리 (픽셀 단위)
        assumed_focal_length = 1000  # 픽셀 단위로 가정
        depth_disparity = (assumed_focal_length * self.baseline_mm) / disparity
        
        # 방법 2: 골프공 크기 기반 깊이 계산
        avg_radius = (radius1 + radius2) / 2
        depth_size = self.calculate_depth_from_ball_size(avg_radius, assumed_focal_length)
        
        # 두 방법의 평균 사용
        if depth_size is not None:
            depth = (depth_disparity + depth_size) / 2
        else:
            depth = depth_disparity
        
        if depth < 100 or depth > 10000:
            return None
        
        # 3D 좌표 계산
        x = (u1 - 720) * depth / assumed_focal_length  # 이미지 중심을 720으로 가정
        y = (v1 - 150) * depth / assumed_focal_length  # 이미지 중심을 150으로 가정
        z = depth
        
        return np.array([x, y, z])
    
    def calculate_physics_fundamental(self, positions_3d, frame_numbers):
        """근본적 방법으로 물리량 계산"""
        if len(positions_3d) < 2:
            return None
        
        # 속도 계산
        velocities = []
        for i in range(1, len(positions_3d)):
            if positions_3d[i] is not None and positions_3d[i-1] is not None:
                frame_diff = frame_numbers[i] - frame_numbers[i-1]
                dt = frame_diff * self.frame_interval
                
                if dt > 0:
                    velocity = (positions_3d[i] - positions_3d[i-1]) / dt
                    velocities.append(velocity)
        
        if not velocities:
            return None
        
        avg_velocity = np.mean(velocities, axis=0)
        speed = np.linalg.norm(avg_velocity)
        
        # 발사각 계산 (Y축이 수직이라고 가정)
        vertical_component = avg_velocity[1]  # Y축이 수직
        if speed > 0:
            launch_angle = math.degrees(math.asin(abs(vertical_component) / speed))
        else:
            launch_angle = 0
        
        # 방향각 계산 (X-Z 평면에서의 각도)
        horizontal_angle = math.atan2(avg_velocity[2], avg_velocity[0])  # Z, X
        direction_angle = math.degrees(horizontal_angle)
        
        return {
            'velocity': avg_velocity.tolist(),
            'speed_mm_s': float(speed),
            'speed_m_s': float(speed / 1000),
            'launch_angle': float(launch_angle),
            'direction_angle': float(direction_angle),
            'method': 'fundamental_size_based'
        }
    
    def analyze_single_shot_fundamental(self, shot_dir, shot_number):
        """근본적 방법으로 단일 샷 분석"""
        print(f"\n--- Analyzing Shot {shot_number} with Fundamental Method ---")
        
        # 이미지 파일 목록
        pattern_cam1 = os.path.join(shot_dir, "Cam1_*.bmp")
        pattern_cam2 = os.path.join(shot_dir, "Cam2_*.bmp")
        files_cam1 = sorted(glob.glob(pattern_cam1), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        files_cam2 = sorted(glob.glob(pattern_cam2), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        print(f"  Found {len(files_cam1)} Cam1 images, {len(files_cam2)} Cam2 images")
        
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
            
            center1, radius1, est_radius1 = self.detect_golf_ball_with_size_estimation(img1)
            center2, radius2, est_radius2 = self.detect_golf_ball_with_size_estimation(img2)
            
            if center1 is not None and center2 is not None:
                position_3d = self.calculate_3d_position_fundamental(center1, center2, est_radius1, est_radius2)
                
                if position_3d is not None:
                    positions_3d.append(position_3d)
                    frame_numbers.append(frame_num)
                    print(f"    Frame {frame_num}: 3D pos = {position_3d}")
        
        if len(positions_3d) >= 2:
            print(f"  Detected {len(positions_3d)} 3D positions with fundamental method")
            
            # 물리량 계산
            physics = self.calculate_physics_fundamental(positions_3d, frame_numbers)
            
            if physics:
                print(f"  Speed: {physics['speed_m_s']:.2f} m/s")
                print(f"  Launch angle: {physics['launch_angle']:.2f}°")
                print(f"  Direction angle: {physics['direction_angle']:.2f}°")
                
                return {
                    'shot_number': shot_number,
                    'success': True,
                    'detected_frames': len(positions_3d),
                    'total_frames': min_frames,
                    'detection_rate': len(positions_3d) / min_frames,
                    'physics': physics
                }
            else:
                print(f"  Could not calculate physics")
                return {
                    'shot_number': shot_number,
                    'success': False,
                    'detected_frames': len(positions_3d),
                    'total_frames': min_frames,
                    'detection_rate': len(positions_3d) / min_frames,
                    'physics': None
                }
        else:
            print(f"  Not enough 3D positions for analysis")
            return {
                'shot_number': shot_number,
                'success': False,
                'detected_frames': len(positions_3d),
                'total_frames': min_frames,
                'detection_rate': len(positions_3d) / min_frames if min_frames > 0 else 0,
                'physics': None
            }
    
    def analyze_all_shots_fundamental(self, driver_dir="data2/driver", output_dir="fundamental_analysis_result"):
        """근본적 방법으로 모든 드라이버 샷 분석"""
        print(f"\n=== ANALYZING ALL DRIVER SHOTS WITH FUNDAMENTAL METHOD ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 데이터가 있는 디렉토리 찾기
        subdirs = [d for d in os.listdir(driver_dir) if os.path.isdir(os.path.join(driver_dir, d))]
        subdirs.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        # 데이터가 있는 디렉토리만 필터링
        data_dirs = []
        for subdir in subdirs:
            subdir_path = os.path.join(driver_dir, subdir)
            files = os.listdir(subdir_path)
            cam_files = [f for f in files if "Cam" in f and not f.endswith('.json')]
            if len(cam_files) > 0:
                data_dirs.append(subdir)
        
        print(f"Found {len(data_dirs)} shots with data: {data_dirs}")
        
        # 실제 측정 데이터 로드
        real_data = self.load_real_shot_data()
        
        # 각 샷 분석
        results = []
        for shot_dir in data_dirs:
            shot_number = int(shot_dir)
            shot_path = os.path.join(driver_dir, shot_dir)
            
            result = self.analyze_single_shot_fundamental(shot_path, shot_number)
            results.append(result)
        
        # 결과 저장
        self.save_fundamental_results(results, real_data, output_dir)
        
        # 비교 리포트 생성
        self.generate_fundamental_comparison_report(results, real_data, output_dir)
        
        return results
    
    def load_real_shot_data(self):
        """실제 측정 데이터 로드"""
        csv_file = "data2/driver/shotdata_20251020.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            # 마지막 빈 행 제거
            df = df.dropna()
            return df
        return None
    
    def save_fundamental_results(self, results, real_data, output_dir):
        """근본적 방법 결과 저장"""
        # JSON 결과 저장
        json_results = []
        for result in results:
            json_result = {
                'shot_number': result['shot_number'],
                'success': result['success'],
                'detected_frames': result['detected_frames'],
                'total_frames': result['total_frames'],
                'detection_rate': result['detection_rate'],
                'method': 'fundamental_size_based'
            }
            
            if result['physics']:
                json_result['physics'] = result['physics']
            
            json_results.append(json_result)
        
        with open(os.path.join(output_dir, "fundamental_analysis.json"), 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # CSV 결과 저장
        csv_data = []
        for i, result in enumerate(results):
            if result['success'] and result['physics']:
                physics = result['physics']
                csv_data.append({
                    'Shot': result['shot_number'],
                    'Success': 'Yes',
                    'Detected_Frames': result['detected_frames'],
                    'Detection_Rate': f"{result['detection_rate']:.2%}",
                    'Speed_m_s': f"{physics['speed_m_s']:.2f}",
                    'Launch_Angle_deg': f"{physics['launch_angle']:.2f}",
                    'Direction_Angle_deg': f"{physics['direction_angle']:.2f}",
                    'Method': physics['method']
                })
            else:
                csv_data.append({
                    'Shot': result['shot_number'],
                    'Success': 'No',
                    'Detected_Frames': result['detected_frames'],
                    'Detection_Rate': f"{result['detection_rate']:.2%}",
                    'Speed_m_s': 'N/A',
                    'Launch_Angle_deg': 'N/A',
                    'Direction_Angle_deg': 'N/A',
                    'Method': 'fundamental_size_based'
                })
        
        df_results = pd.DataFrame(csv_data)
        df_results.to_csv(os.path.join(output_dir, "fundamental_analysis.csv"), index=False)
        
        print(f"\nFundamental Results saved to {output_dir}/")
        print(f"  - fundamental_analysis.json")
        print(f"  - fundamental_analysis.csv")
    
    def generate_fundamental_comparison_report(self, results, real_data, output_dir):
        """근본적 방법 비교 리포트 생성"""
        if real_data is None:
            print("No real data available for comparison")
            return
        
        # 성공한 샷들만 필터링
        successful_shots = [r for r in results if r['success'] and r['physics']]
        
        if not successful_shots:
            print("No successful shots for comparison")
            return
        
        print(f"\n=== FUNDAMENTAL METHOD COMPARISON WITH REAL DATA ===")
        print(f"Successful shots: {len(successful_shots)}/{len(results)}")
        
        # 실제 데이터와 매칭
        comparison_data = []
        for shot in successful_shots:
            shot_num = shot['shot_number']
            physics = shot['physics']
            
            # 실제 데이터에서 해당 샷 찾기 (1-based index)
            if shot_num <= len(real_data):
                real_row = real_data.iloc[shot_num - 1]
                real_speed = real_row['BallSpeed(m/s)']
                real_launch_angle = real_row['LaunchAngle(deg)']
                real_direction_angle = real_row['LaunchDirection(deg)']
                
                comparison_data.append({
                    'Shot': shot_num,
                    'Real_Speed': real_speed,
                    'Fundamental_Speed': physics['speed_m_s'],
                    'Speed_Ratio': physics['speed_m_s'] / real_speed,
                    'Speed_Error_Percent': abs(physics['speed_m_s'] - real_speed) / real_speed * 100,
                    'Real_Launch_Angle': real_launch_angle,
                    'Fundamental_Launch_Angle': physics['launch_angle'],
                    'Launch_Angle_Error': abs(physics['launch_angle'] - real_launch_angle),
                    'Real_Direction_Angle': real_direction_angle,
                    'Fundamental_Direction_Angle': physics['direction_angle'],
                    'Direction_Angle_Error': abs(physics['direction_angle'] - real_direction_angle)
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison.to_csv(os.path.join(output_dir, "fundamental_comparison.csv"), index=False)
            
            # 통계 요약
            avg_speed_ratio = df_comparison['Speed_Ratio'].mean()
            avg_speed_error = df_comparison['Speed_Error_Percent'].mean()
            avg_launch_angle_error = df_comparison['Launch_Angle_Error'].mean()
            avg_direction_angle_error = df_comparison['Direction_Angle_Error'].mean()
            
            print(f"\nFundamental Method Comparison Summary:")
            print(f"  Average Speed Ratio: {avg_speed_ratio:.2f}x")
            print(f"  Average Speed Error: {avg_speed_error:.1f}%")
            print(f"  Average Launch Angle Error: {avg_launch_angle_error:.2f}°")
            print(f"  Average Direction Angle Error: {avg_direction_angle_error:.2f}°")
            
            print(f"\nDetailed Fundamental Method Comparison:")
            print(df_comparison.to_string(index=False))
            
            print(f"\nFundamental method comparison data saved to: {output_dir}/fundamental_comparison.csv")

def main():
    """메인 함수"""
    analyzer = FundamentalSolutionAnalyzer()
    
    driver_dir = "data2/driver"
    output_dir = "fundamental_analysis_result"
    
    if os.path.exists(driver_dir):
        results = analyzer.analyze_all_shots_fundamental(driver_dir, output_dir)
        
        # 요약 출력
        successful_shots = [r for r in results if r['success']]
        print(f"\n=== FUNDAMENTAL METHOD ANALYSIS SUMMARY ===")
        print(f"Total shots analyzed: {len(results)}")
        print(f"Successful analyses: {len(successful_shots)}")
        print(f"Success rate: {len(successful_shots)/len(results):.1%}")
        
        if successful_shots:
            avg_detection_rate = np.mean([r['detection_rate'] for r in successful_shots])
            print(f"Average detection rate: {avg_detection_rate:.1%}")
    else:
        print(f"Driver directory {driver_dir} not found")
    
    print("\n=== FUNDAMENTAL SOLUTION ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()

