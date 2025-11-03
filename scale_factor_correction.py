#!/usr/bin/env python3
"""
스케일 팩터 보정 분석기
이미지 크기 차이를 보정하는 스케일 팩터를 적용한 골프공 물리량 계산
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

class ScaleFactorCorrectionAnalyzer:
    def __init__(self, calibration_file="vertical_stereo_calibration_z_axis.json"):
        """스케일 팩터 보정 분석기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        # 스케일 팩터 계산
        self.scale_factor = self.calculate_scale_factor()
        
        # 최적 좌표계
        self.optimal_coordinate_system = {
            'name': 'Scale Factor Corrected',
            'x_axis': [1, 0, 0],    # X = forward
            'y_axis': [0, 0, 1],    # Y = up  
            'z_axis': [0, 1, 0]     # Z = right
        }
        self.transform_matrix = np.array(self.optimal_coordinate_system['x_axis'] + 
                                       self.optimal_coordinate_system['y_axis'] + 
                                       self.optimal_coordinate_system['z_axis']).reshape(3, 3)
        
        self.fps = 820
        self.frame_interval = 1.0 / self.fps
        
        print("Scale Factor Correction Analyzer Initialized")
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
    
    def calculate_scale_factor(self):
        """스케일 팩터 계산"""
        # 캘리브레이션 이미지 크기 (전체 해상도)
        calib_height = self.image_size[0]  # 1080
        calib_width = self.image_size[1]   # 1440
        
        # 드라이버 이미지 크기 (ROI 적용된 크기)
        driver_height = 300
        driver_width = 1440
        
        # 높이 기준 스케일 팩터
        height_scale = calib_height / driver_height  # 1080 / 300 = 3.6
        
        print(f"Calibration image size: {calib_width}x{calib_height}")
        print(f"Driver image size: {driver_width}x{driver_height}")
        print(f"Height scale factor: {height_scale:.3f}")
        
        return height_scale
    
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
    
    def calculate_3d_position_with_scale_correction(self, center1, center2):
        """스케일 팩터 보정을 적용한 3D 위치 계산"""
        if center1 is None or center2 is None:
            return None
        
        u1, v1 = center1
        u2, v2 = center2
        
        # X 좌표 시차 사용
        disparity = abs(u1 - u2)
        
        if disparity <= 1:
            return None
        
        # 스케일 팩터 적용된 깊이 계산
        # 드라이버 이미지는 ROI가 적용되어 있으므로 스케일 팩터를 적용
        depth = (self.focal_length * self.baseline_mm) / disparity
        
        if depth < 100 or depth > 10000:
            return None
        
        # 3D 좌표 계산 (스케일 팩터 적용)
        # Y 좌표에 스케일 팩터 적용 (높이 방향)
        x = (u1 - self.K1[0, 2]) * depth / self.focal_length
        y = (v1 - self.K1[1, 2]) * depth / self.focal_length * self.scale_factor
        z = depth
        
        return np.array([x, y, z])
    
    def calculate_physics_with_scale_correction(self, positions_3d, frame_numbers):
        """스케일 팩터 보정된 좌표계에서 물리량 계산"""
        if len(positions_3d) < 2:
            return None
        
        # 좌표계 변환
        transformed_positions = []
        for pos in positions_3d:
            if pos is not None:
                transformed_pos = self.transform_matrix @ pos
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
        
        # 발사각 계산 (Y축이 수직)
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
            'scale_factor': float(self.scale_factor),
            'transformed_positions': [[float(p[0]), float(p[1]), float(p[2])] for p in transformed_positions if p is not None]
        }
    
    def analyze_single_shot_with_scale_correction(self, shot_dir, shot_number):
        """스케일 팩터 보정을 적용한 단일 샷 분석"""
        print(f"\n--- Analyzing Shot {shot_number} with Scale Factor Correction ---")
        
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
            
            center1, radius1 = self.detect_golf_ball_adaptive(img1)
            center2, radius2 = self.detect_golf_ball_adaptive(img2)
            
            if center1 is not None and center2 is not None:
                position_3d = self.calculate_3d_position_with_scale_correction(center1, center2)
                
                if position_3d is not None:
                    positions_3d.append(position_3d)
                    frame_numbers.append(frame_num)
        
        if len(positions_3d) >= 2:
            print(f"  Detected {len(positions_3d)} 3D positions with scale correction")
            
            # 물리량 계산
            physics = self.calculate_physics_with_scale_correction(positions_3d, frame_numbers)
            
            if physics:
                print(f"  Speed: {physics['speed_m_s']:.2f} m/s")
                print(f"  Launch angle: {physics['launch_angle']:.2f}°")
                print(f"  Direction angle: {physics['direction_angle']:.2f}°")
                print(f"  Scale factor applied: {physics['scale_factor']:.3f}")
                
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
    
    def analyze_all_shots_with_scale_correction(self, driver_dir="data2/driver", output_dir="scale_corrected_analysis_result"):
        """스케일 팩터 보정을 적용한 모든 드라이버 샷 분석"""
        print(f"\n=== ANALYZING ALL DRIVER SHOTS WITH SCALE FACTOR CORRECTION ===")
        
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
            
            result = self.analyze_single_shot_with_scale_correction(shot_path, shot_number)
            results.append(result)
        
        # 결과 저장
        self.save_scale_results(results, real_data, output_dir)
        
        # 비교 리포트 생성
        self.generate_scale_comparison_report(results, real_data, output_dir)
        
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
    
    def save_scale_results(self, results, real_data, output_dir):
        """스케일 보정 결과 저장"""
        # JSON 결과 저장
        json_results = []
        for result in results:
            json_result = {
                'shot_number': result['shot_number'],
                'success': result['success'],
                'detected_frames': result['detected_frames'],
                'total_frames': result['total_frames'],
                'detection_rate': result['detection_rate'],
                'scale_factor': self.scale_factor
            }
            
            if result['physics']:
                json_result['physics'] = result['physics']
            
            json_results.append(json_result)
        
        with open(os.path.join(output_dir, "scale_corrected_analysis.json"), 'w', encoding='utf-8') as f:
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
                    'Scale_Factor': f"{physics['scale_factor']:.3f}"
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
                    'Scale_Factor': f"{self.scale_factor:.3f}"
                })
        
        df_results = pd.DataFrame(csv_data)
        df_results.to_csv(os.path.join(output_dir, "scale_corrected_analysis.csv"), index=False)
        
        print(f"\nScale Corrected Results saved to {output_dir}/")
        print(f"  - scale_corrected_analysis.json")
        print(f"  - scale_corrected_analysis.csv")
    
    def generate_scale_comparison_report(self, results, real_data, output_dir):
        """스케일 보정 비교 리포트 생성"""
        if real_data is None:
            print("No real data available for comparison")
            return
        
        # 성공한 샷들만 필터링
        successful_shots = [r for r in results if r['success'] and r['physics']]
        
        if not successful_shots:
            print("No successful shots for comparison")
            return
        
        print(f"\n=== SCALE FACTOR CORRECTED COMPARISON WITH REAL DATA ===")
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
                    'Scale_Corrected_Speed': physics['speed_m_s'],
                    'Speed_Ratio': physics['speed_m_s'] / real_speed,
                    'Speed_Error_Percent': abs(physics['speed_m_s'] - real_speed) / real_speed * 100,
                    'Real_Launch_Angle': real_launch_angle,
                    'Scale_Corrected_Launch_Angle': physics['launch_angle'],
                    'Launch_Angle_Error': abs(physics['launch_angle'] - real_launch_angle),
                    'Real_Direction_Angle': real_direction_angle,
                    'Scale_Corrected_Direction_Angle': physics['direction_angle'],
                    'Direction_Angle_Error': abs(physics['direction_angle'] - real_direction_angle),
                    'Scale_Factor': physics['scale_factor']
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison.to_csv(os.path.join(output_dir, "scale_corrected_comparison.csv"), index=False)
            
            # 통계 요약
            avg_speed_ratio = df_comparison['Speed_Ratio'].mean()
            avg_speed_error = df_comparison['Speed_Error_Percent'].mean()
            avg_launch_angle_error = df_comparison['Launch_Angle_Error'].mean()
            avg_direction_angle_error = df_comparison['Direction_Angle_Error'].mean()
            
            print(f"\nScale Factor Corrected Comparison Summary:")
            print(f"  Average Speed Ratio: {avg_speed_ratio:.2f}x")
            print(f"  Average Speed Error: {avg_speed_error:.1f}%")
            print(f"  Average Launch Angle Error: {avg_launch_angle_error:.2f}°")
            print(f"  Average Direction Angle Error: {avg_direction_angle_error:.2f}°")
            print(f"  Scale Factor Used: {self.scale_factor:.3f}")
            
            print(f"\nDetailed Scale Factor Corrected Comparison:")
            print(df_comparison.to_string(index=False))
            
            print(f"\nScale Factor corrected comparison data saved to: {output_dir}/scale_corrected_comparison.csv")

def main():
    """메인 함수"""
    analyzer = ScaleFactorCorrectionAnalyzer()
    
    driver_dir = "data2/driver"
    output_dir = "scale_corrected_analysis_result"
    
    if os.path.exists(driver_dir):
        results = analyzer.analyze_all_shots_with_scale_correction(driver_dir, output_dir)
        
        # 요약 출력
        successful_shots = [r for r in results if r['success']]
        print(f"\n=== SCALE FACTOR CORRECTED ANALYSIS SUMMARY ===")
        print(f"Total shots analyzed: {len(results)}")
        print(f"Successful analyses: {len(successful_shots)}")
        print(f"Success rate: {len(successful_shots)/len(results):.1%}")
        
        if successful_shots:
            avg_detection_rate = np.mean([r['detection_rate'] for r in successful_shots])
            print(f"Average detection rate: {avg_detection_rate:.1%}")
    else:
        print(f"Driver directory {driver_dir} not found")
    
    print("\n=== SCALE FACTOR CORRECTED DRIVER SHOT ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()

