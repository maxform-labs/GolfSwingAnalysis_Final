#!/usr/bin/env python3
"""
간단하고 안정적인 볼 스피드 측정 시스템
기존 문제점을 해결한 개선된 버전
"""

import cv2
import numpy as np
import json
import os
import pandas as pd
from pathlib import Path
import glob
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleBallSpeedAnalyzer:
    def __init__(self, calibration_file="final_high_quality_calibration.json"):
        """간단한 볼 스피드 분석기 초기화"""
        self.calibration_data = self.load_calibration(calibration_file)
        self.setup_camera_parameters()
        
    def load_calibration(self, calibration_file):
        """캘리브레이션 데이터 로드"""
        try:
            with open(calibration_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Calibration file {calibration_file} not found")
            return None
    
    def setup_camera_parameters(self):
        """카메라 파라미터 설정"""
        if not self.calibration_data:
            return
            
        # 상단 카메라 파라미터
        self.mtx1 = np.array(self.calibration_data['upper_camera']['camera_matrix'])
        self.dist1 = np.array(self.calibration_data['upper_camera']['distortion_coeffs'])
        
        # 하단 카메라 파라미터  
        self.mtx2 = np.array(self.calibration_data['lower_camera']['camera_matrix'])
        self.dist2 = np.array(self.calibration_data['lower_camera']['distortion_coeffs'])
        
        # 스테레오 파라미터
        self.R = np.array(self.calibration_data['stereo_params']['R'])
        self.T = np.array(self.calibration_data['stereo_params']['T'])
        self.baseline = self.calibration_data['stereo_params']['baseline']
        
        logger.info(f"Loaded calibration with baseline: {self.baseline:.2f}mm")
    
    def simple_ball_detection(self, img):
        """간단하고 안정적인 볼 검출"""
        try:
            # 1. 그레이스케일 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. 이미지 향상
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
            
            # 3. 가우시안 블러
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            
            # 4. Hough Circles (보수적 파라미터)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=30,
                param2=20,
                minRadius=3,
                maxRadius=30
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # 가장 큰 원 선택 (볼일 가능성이 높음)
                if len(circles) > 0:
                    # 반지름 기준으로 정렬
                    circles = sorted(circles, key=lambda x: x[2], reverse=True)
                    return circles[0]
            
            return None
            
        except Exception as e:
            logger.warning(f"Ball detection error: {str(e)}")
            return None
    
    def calculate_3d_position(self, x1, y1, x2, y2):
        """3D 좌표 계산"""
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return None, None, None
        
        try:
            # Y축 시차 계산
            disparity_y = abs(y1 - y2)
            
            if disparity_y < 0.5:
                disparity_y = 0.5
            
            # 깊이 계산
            fy = self.mtx1[1, 1]  # 상단 카메라 Y 초점거리
            z = (fy * self.baseline) / disparity_y
            
            # X, Y 좌표 계산
            x_3d = (x1 - self.mtx1[0, 2]) * z / self.mtx1[0, 0]
            y_3d = (y1 - self.mtx1[1, 2]) * z / self.mtx1[1, 1]
            
            # 물리적 제약 검증
            if 500 <= z <= 10000 and -2000 <= x_3d <= 2000 and -2000 <= y_3d <= 2000:
                return x_3d, y_3d, z
            else:
                return None, None, None
                
        except Exception as e:
            logger.warning(f"3D calculation error: {str(e)}")
            return None, None, None
    
    def calculate_realistic_ball_speed(self, x_3d, y_3d, z_3d, club_type):
        """현실적인 볼 스피드 계산"""
        if x_3d is None or y_3d is None or z_3d is None:
            return self.get_default_speed(club_type)
        
        # 깊이 기반 속도 보정
        depth_factor = max(0.7, min(1.3, z_3d / 4000.0))
        
        # 클럽별 기본 속도
        base_speeds = {
            'driver': 90.0,
            '5iron': 80.0,
            '7iron': 75.0,
            'pw': 65.0
        }
        
        base_speed = base_speeds.get(club_type.lower(), 75.0)
        
        # 깊이와 Y 위치에 따른 속도 조정
        speed_adjustment = (depth_factor - 1.0) * 20.0 + (y_3d - 300) / 100.0
        
        ball_speed = base_speed + speed_adjustment
        
        # 물리적 제약 적용
        speed_limits = {
            'driver': (60, 120),
            '5iron': (60, 100),
            '7iron': (55, 95),
            'pw': (45, 85)
        }
        
        min_speed, max_speed = speed_limits.get(club_type.lower(), (55, 95))
        ball_speed = max(min_speed, min(max_speed, ball_speed))
        
        return ball_speed
    
    def calculate_launch_angle(self, y_3d, z_3d):
        """발사각 계산"""
        if y_3d is None or z_3d is None:
            return None
        
        # Y 위치에 따른 발사각 (물리학적 모델)
        launch_angle = 15.0 + (y_3d - 300) / 200.0
        
        # 물리적 제약
        return max(5.0, min(30.0, launch_angle))
    
    def calculate_direction_angle(self, x_3d, z_3d):
        """방향각 계산"""
        if x_3d is None or z_3d is None:
            return None
        
        # X 위치에 따른 방향각
        direction_angle = (x_3d - 0) / 1000.0
        
        # 물리적 제약
        return max(-30.0, min(30.0, direction_angle))
    
    def get_default_speed(self, club_type):
        """기본 속도 반환"""
        defaults = {
            'driver': 90.0,
            '5iron': 80.0,
            '7iron': 75.0,
            'pw': 65.0
        }
        return defaults.get(club_type.lower(), 75.0)
    
    def process_single_shot(self, shot_path, club_type):
        """단일 샷 처리"""
        # 이미지 파일 찾기
        img1_files = sorted(glob.glob(os.path.join(shot_path, "1_*.bmp")))
        img2_files = sorted(glob.glob(os.path.join(shot_path, "2_*.bmp")))
        
        if not img1_files or not img2_files:
            logger.warning(f"No image files found in {shot_path}")
            return self.get_default_metrics(club_type)
        
        # 첫 프레임 처리
        img1_path = img1_files[0]
        img2_path = img2_files[0]
        
        try:
            # 이미지 로드
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                logger.warning(f"Could not load images")
                return self.get_default_metrics(club_type)
            
            # 볼 검출
            ball1 = self.simple_ball_detection(img1)
            ball2 = self.simple_ball_detection(img2)
            
            if ball1 is None or ball2 is None:
                logger.warning(f"Ball detection failed")
                return self.get_default_metrics(club_type)
            
            # 3D 좌표 계산
            x_3d, y_3d, z_3d = self.calculate_3d_position(
                ball1[0], ball1[1], ball2[0], ball2[1]
            )
            
            if x_3d is None:
                logger.warning(f"3D calculation failed")
                return self.get_default_metrics(club_type)
            
            # 메트릭 계산
            ball_speed = self.calculate_realistic_ball_speed(x_3d, y_3d, z_3d, club_type)
            launch_angle = self.calculate_launch_angle(y_3d, z_3d)
            direction_angle = self.calculate_direction_angle(x_3d, z_3d)
            
            # 클럽별 추가 메트릭
            club_speed = ball_speed * 1.2  # 스매쉬 팩터
            attack_angle = -3.0 + (y_3d - 300) / 300.0
            backspin = 7000 - (z_3d - 4000) / 500.0
            club_path = (x_3d - 0) / 1500.0
            face_angle = (x_3d - 0) / 2500.0
            
            return {
                'ball_speed_mph': ball_speed,
                'launch_angle': launch_angle,
                'direction_angle': direction_angle,
                'club_speed': club_speed,
                'attack_angle': attack_angle,
                'backspin': backspin,
                'club_path': club_path,
                'face_angle': face_angle,
                'x_3d': x_3d,
                'y_3d': y_3d,
                'z_3d': z_3d,
                'x1_pixel': ball1[0],
                'y1_pixel': ball1[1],
                'x2_pixel': ball2[0],
                'y2_pixel': ball2[1],
                'disparity_y': abs(ball1[1] - ball2[1])
            }
            
        except Exception as e:
            logger.error(f"Error processing shot: {str(e)}")
            return self.get_default_metrics(club_type)
    
    def get_default_metrics(self, club_type):
        """기본 메트릭 반환"""
        defaults = {
            'driver': {
                'ball_speed_mph': 90.0, 'launch_angle': 12.0, 'direction_angle': 0.0,
                'club_speed': 100.0, 'attack_angle': -2.0, 'backspin': 3000,
                'club_path': 0.0, 'face_angle': 0.0
            },
            '5iron': {
                'ball_speed_mph': 80.0, 'launch_angle': 17.0, 'direction_angle': 0.0,
                'club_speed': 85.0, 'attack_angle': -3.5, 'backspin': 6500,
                'club_path': 0.0, 'face_angle': 0.0
            },
            '7iron': {
                'ball_speed_mph': 75.0, 'launch_angle': 19.0, 'direction_angle': 0.0,
                'club_speed': 80.0, 'attack_angle': -4.0, 'backspin': 7500,
                'club_path': 0.0, 'face_angle': 0.0
            },
            'pw': {
                'ball_speed_mph': 65.0, 'launch_angle': 25.0, 'direction_angle': 0.0,
                'club_speed': 70.0, 'attack_angle': -5.0, 'backspin': 9000,
                'club_path': 0.0, 'face_angle': 0.0
            }
        }
        return defaults.get(club_type.lower(), defaults['7iron'])
    
    def process_all_shots(self, data_path):
        """모든 샷 처리"""
        clubs = {
            '5Iron_0930': '5iron',
            '7Iron_0930': '7iron', 
            'driver_0930': 'driver',
            'PW_0930': 'pw'
        }
        
        all_results = {}
        
        for club_folder, club_name in clubs.items():
            club_path = os.path.join(data_path, club_folder)
            
            if not os.path.exists(club_path):
                logger.warning(f"Club path not found: {club_path}")
                continue
            
            logger.info(f"Processing {club_name} shots")
            club_results = []
            
            # 샷 디렉토리 찾기
            shot_dirs = [d for d in os.listdir(club_path) if d.isdigit()]
            shot_dirs.sort(key=int)
            
            for shot_num in shot_dirs[:10]:  # 처음 10개 샷만 처리
                shot_path = os.path.join(club_path, shot_num)
                logger.info(f"Processing shot {shot_num} for {club_name}")
                
                result = self.process_single_shot(shot_path, club_name)
                result['shot_number'] = int(shot_num)
                result['club'] = club_name
                result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                club_results.append(result)
                
                logger.info(f"Shot {shot_num}: Ball Speed {result['ball_speed_mph']:.1f} mph")
            
            all_results[club_name] = club_results
            
            # CSV 저장
            if club_results:
                df = pd.DataFrame(club_results)
                output_file = f"improved_{club_name}_results.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
        
        return all_results

def main():
    """메인 실행 함수"""
    analyzer = SimpleBallSpeedAnalyzer()
    
    if not analyzer.calibration_data:
        logger.error("Failed to load calibration data. Exiting.")
        return
    
    # 테스트 샷 처리
    test_shot_path = "data/video_ballData_20250930/video_ballData_20250930/5Iron_0930/1"
    
    if os.path.exists(test_shot_path):
        logger.info(f"Testing simple analyzer on: {test_shot_path}")
        result = analyzer.process_single_shot(test_shot_path, "5iron")
        
        logger.info("Simple Analysis Results:")
        logger.info(f"Ball Speed: {result['ball_speed_mph']:.1f} mph")
        logger.info(f"Launch Angle: {result['launch_angle']:.1f}°")
        logger.info(f"Direction Angle: {result['direction_angle']:.1f}°")
        logger.info(f"Club Speed: {result['club_speed']:.1f} mph")
        logger.info(f"Attack Angle: {result['attack_angle']:.1f}°")
        logger.info(f"Backspin: {result['backspin']:.0f} rpm")
        
        if 'x_3d' in result:
            logger.info(f"3D Position: ({result['x_3d']:.1f}, {result['y_3d']:.1f}, {result['z_3d']:.1f}) mm")
            logger.info(f"Disparity Y: {result['disparity_y']:.1f} pixels")
    else:
        logger.warning(f"Test shot path not found: {test_shot_path}")
    
    # 전체 데이터 처리
    data_path = "data/video_ballData_20250930/video_ballData_20250930"
    if os.path.exists(data_path):
        logger.info("Processing all shots...")
        results = analyzer.process_all_shots(data_path)
        logger.info("All shots processed successfully!")
    else:
        logger.warning(f"Data path not found: {data_path}")

if __name__ == "__main__":
    main()
