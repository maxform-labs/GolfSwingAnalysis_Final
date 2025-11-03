#!/usr/bin/env python3
"""
ROI 좌표계 보정 분석기
1440x1080 캘리브레이션을 1440x300 ROI에 정확히 적용하는 좌표 변환 시스템
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

class ROICoordinateCorrectionAnalyzer:
    def __init__(self, calibration_file="vertical_stereo_calibration_z_axis.json"):
        """ROI 좌표계 보정 분석기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        # ROI 정보 로드
        self.roi_info = self.load_roi_info()
        
        # 좌표 변환 행렬 계산
        self.setup_coordinate_transformation()
        
        self.fps = 820
        self.frame_interval = 1.0 / self.fps
        
        print("ROI Coordinate Correction Analyzer Initialized")
        print(f"Calibration image size: {self.calibration_image_size}")
        print(f"ROI image size: {self.roi_image_size}")
        print(f"ROI offset: {self.roi_offset}")
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
        self.calibration_image_size = tuple(self.calibration_data['image_size'])  # (1080, 1440)
        self.focal_length = self.K1[0, 0]
    
    def load_roi_info(self):
        """ROI 정보 로드"""
        # 샘플 드라이버 디렉토리에서 ROI 정보 수집
        sample_dirs = ['1', '2', '3', '4', '5']
        roi_cam1_list = []
        roi_cam2_list = []
        
        for shot_dir in sample_dirs:
            shot_path = os.path.join("data2/driver", shot_dir)
            
            roi_cam1_file = os.path.join(shot_path, "roi_cam1.json")
            roi_cam2_file = os.path.join(shot_path, "roi_cam2.json")
            
            if os.path.exists(roi_cam1_file):
                with open(roi_cam1_file, 'r', encoding='utf-8') as f:
                    roi_cam1 = json.load(f)
                    roi_cam1_list.append(roi_cam1)
            
            if os.path.exists(roi_cam2_file):
                with open(roi_cam2_file, 'r', encoding='utf-8') as f:
                    roi_cam2 = json.load(f)
                    roi_cam2_list.append(roi_cam2)
        
        # 공통 ROI 정보 (모든 샷이 동일하다고 가정)
        if roi_cam1_list and roi_cam2_list:
            return {
                'cam1': roi_cam1_list[0],
                'cam2': roi_cam2_list[0]
            }
        else:
            # 기본값 설정
            return {
                'cam1': {'Height': 300, 'Width': 1440, 'XOffset': 0, 'YOffset': 396},
                'cam2': {'Height': 300, 'Width': 1440, 'XOffset': 0, 'YOffset': 372}
            }
    
    def setup_coordinate_transformation(self):
        """좌표 변환 설정"""
        # 캘리브레이션 이미지 크기 (1080, 1440)
        self.calibration_height, self.calibration_width = self.calibration_image_size
        
        # ROI 이미지 크기 (300, 1440)
        self.roi_height = self.roi_info['cam1']['Height']  # 300
        self.roi_width = self.roi_info['cam1']['Width']    # 1440
        
        # ROI 오프셋 (캘리브레이션 이미지 내에서의 ROI 시작점)
        self.roi_offset = {
            'cam1': (self.roi_info['cam1']['XOffset'], self.roi_info['cam1']['YOffset']),  # (0, 396)
            'cam2': (self.roi_info['cam2']['XOffset'], self.roi_info['cam2']['YOffset'])   # (0, 372)
        }
        
        # ROI 이미지 크기
        self.roi_image_size = (self.roi_width, self.roi_height)  # (1440, 300)
        
        print(f"ROI Setup:")
        print(f"  Calibration size: {self.calibration_width}x{self.calibration_height}")
        print(f"  ROI size: {self.roi_width}x{self.roi_height}")
        print(f"  ROI offset Cam1: {self.roi_offset['cam1']}")
        print(f"  ROI offset Cam2: {self.roi_offset['cam2']}")
    
    def transform_coordinates_to_roi(self, point, camera_num):
        """캘리브레이션 좌표계를 ROI 좌표계로 변환"""
        x, y = point
        x_offset, y_offset = self.roi_offset[f'cam{camera_num}']
        
        # ROI 좌표계로 변환: x' = x - x_start, y' = y - y_start
        x_roi = x - x_offset
        y_roi = y - y_offset
        
        return (x_roi, y_roi)
    
    def transform_coordinates_from_roi(self, point, camera_num):
        """ROI 좌표계를 캘리브레이션 좌표계로 변환"""
        x_roi, y_roi = point
        x_offset, y_offset = self.roi_offset[f'cam{camera_num}']
        
        # 캘리브레이션 좌표계로 변환: x = x' + x_start, y = y' + y_start
        x = x_roi + x_offset
        y = y_roi + y_offset
        
        return (x, y)
    
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
    
    def calculate_3d_position_with_roi_correction(self, center1, center2):
        """ROI 좌표계 보정을 적용한 3D 위치 계산"""
        if center1 is None or center2 is None:
            return None
        
        # ROI 좌표를 캘리브레이션 좌표로 변환
        center1_calib = self.transform_coordinates_from_roi(center1, 1)
        center2_calib = self.transform_coordinates_from_roi(center2, 2)
        
        u1, v1 = center1_calib
        u2, v2 = center2_calib
        
        # 시차 계산 (X 방향)
        disparity = abs(u1 - u2)
        
        if disparity <= 1:
            return None
        
        # 깊이 계산
        depth = (self.focal_length * self.baseline_mm) / disparity
        
        if depth < 100 or depth > 10000:
            return None
        
        # 3D 좌표 계산 (캘리브레이션 좌표계에서)
        x = (u1 - self.K1[0, 2]) * depth / self.focal_length
        y = (v1 - self.K1[1, 2]) * depth / self.focal_length
        z = depth
        
        return np.array([x, y, z])
    
    def calculate_physics_with_roi_correction(self, positions_3d, frame_numbers):
        """ROI 보정된 좌표계에서 물리량 계산"""
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
            'method': 'roi_coordinate_correction'
        }
    
    def analyze_single_shot_with_roi_correction(self, shot_dir, shot_number):
        """ROI 좌표계 보정을 적용한 단일 샷 분석"""
        print(f"\n--- Analyzing Shot {shot_number} with ROI Coordinate Correction ---")
        
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
                # 좌표 변환 정보 출력
                center1_calib = self.transform_coordinates_from_roi(center1, 1)
                center2_calib = self.transform_coordinates_from_roi(center2, 2)
                
                print(f"    Frame {frame_num}:")
                print(f"      ROI coords: Cam1{center1}, Cam2{center2}")
                print(f"      Calib coords: Cam1{center1_calib}, Cam2{center2_calib}")
                
                position_3d = self.calculate_3d_position_with_roi_correction(center1, center2)
                
                if position_3d is not None:
                    positions_3d.append(position_3d)
                    frame_numbers.append(frame_num)
                    print(f"      3D position: {position_3d}")
        
        if len(positions_3d) >= 2:
            print(f"  Detected {len(positions_3d)} 3D positions with ROI coordinate correction")
            
            # 물리량 계산
            physics = self.calculate_physics_with_roi_correction(positions_3d, frame_numbers)
            
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
    
    def analyze_all_shots_with_roi_correction(self, driver_dir="data2/driver", output_dir="roi_coordinate_correction_result"):
        """ROI 좌표계 보정을 적용한 모든 드라이버 샷 분석"""
        print(f"\n=== ANALYZING ALL DRIVER SHOTS WITH ROI COORDINATE CORRECTION ===")
        
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
            
            result = self.analyze_single_shot_with_roi_correction(shot_path, shot_number)
            results.append(result)
        
        # 결과 저장
        self.save_roi_correction_results(results, real_data, output_dir)
        
        # 비교 리포트 생성
        self.generate_roi_correction_comparison_report(results, real_data, output_dir)
        
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
    
    def save_roi_correction_results(self, results, real_data, output_dir):
        """ROI 좌표계 보정 결과 저장"""
        # JSON 결과 저장
        json_results = []
        for result in results:
            json_result = {
                'shot_number': result['shot_number'],
                'success': result['success'],
                'detected_frames': result['detected_frames'],
                'total_frames': result['total_frames'],
                'detection_rate': result['detection_rate'],
                'method': 'roi_coordinate_correction',
                'roi_info': self.roi_info
            }
            
            if result['physics']:
                json_result['physics'] = result['physics']
            
            json_results.append(json_result)
        
        with open(os.path.join(output_dir, "roi_coordinate_correction_analysis.json"), 'w', encoding='utf-8') as f:
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
                    'Method': 'roi_coordinate_correction'
                })
        
        df_results = pd.DataFrame(csv_data)
        df_results.to_csv(os.path.join(output_dir, "roi_coordinate_correction_analysis.csv"), index=False)
        
        print(f"\nROI Coordinate Correction Results saved to {output_dir}/")
        print(f"  - roi_coordinate_correction_analysis.json")
        print(f"  - roi_coordinate_correction_analysis.csv")
    
    def generate_roi_correction_comparison_report(self, results, real_data, output_dir):
        """ROI 좌표계 보정 비교 리포트 생성"""
        if real_data is None:
            print("No real data available for comparison")
            return
        
        # 성공한 샷들만 필터링
        successful_shots = [r for r in results if r['success'] and r['physics']]
        
        if not successful_shots:
            print("No successful shots for comparison")
            return
        
        print(f"\n=== ROI COORDINATE CORRECTION COMPARISON WITH REAL DATA ===")
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
                    'ROI_Corrected_Speed': physics['speed_m_s'],
                    'Speed_Ratio': physics['speed_m_s'] / real_speed,
                    'Speed_Error_Percent': abs(physics['speed_m_s'] - real_speed) / real_speed * 100,
                    'Real_Launch_Angle': real_launch_angle,
                    'ROI_Corrected_Launch_Angle': physics['launch_angle'],
                    'Launch_Angle_Error': abs(physics['launch_angle'] - real_launch_angle),
                    'Real_Direction_Angle': real_direction_angle,
                    'ROI_Corrected_Direction_Angle': physics['direction_angle'],
                    'Direction_Angle_Error': abs(physics['direction_angle'] - real_direction_angle)
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison.to_csv(os.path.join(output_dir, "roi_coordinate_correction_comparison.csv"), index=False)
            
            # 통계 요약
            avg_speed_ratio = df_comparison['Speed_Ratio'].mean()
            avg_speed_error = df_comparison['Speed_Error_Percent'].mean()
            avg_launch_angle_error = df_comparison['Launch_Angle_Error'].mean()
            avg_direction_angle_error = df_comparison['Direction_Angle_Error'].mean()
            
            print(f"\nROI Coordinate Correction Comparison Summary:")
            print(f"  Average Speed Ratio: {avg_speed_ratio:.2f}x")
            print(f"  Average Speed Error: {avg_speed_error:.1f}%")
            print(f"  Average Launch Angle Error: {avg_launch_angle_error:.2f}°")
            print(f"  Average Direction Angle Error: {avg_direction_angle_error:.2f}°")
            
            print(f"\nDetailed ROI Coordinate Correction Comparison:")
            print(df_comparison.to_string(index=False))
            
            print(f"\nROI coordinate correction comparison data saved to: {output_dir}/roi_coordinate_correction_comparison.csv")

def main():
    """메인 함수"""
    analyzer = ROICoordinateCorrectionAnalyzer()
    
    driver_dir = "data2/driver"
    output_dir = "roi_coordinate_correction_result"
    
    if os.path.exists(driver_dir):
        results = analyzer.analyze_all_shots_with_roi_correction(driver_dir, output_dir)
        
        # 요약 출력
        successful_shots = [r for r in results if r['success']]
        print(f"\n=== ROI COORDINATE CORRECTION ANALYSIS SUMMARY ===")
        print(f"Total shots analyzed: {len(results)}")
        print(f"Successful analyses: {len(successful_shots)}")
        print(f"Success rate: {len(successful_shots)/len(results):.1%}")
        
        if successful_shots:
            avg_detection_rate = np.mean([r['detection_rate'] for r in successful_shots])
            print(f"Average detection rate: {avg_detection_rate:.1%}")
    else:
        print(f"Driver directory {driver_dir} not found")
    
    print("\n=== ROI COORDINATE CORRECTION ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()

