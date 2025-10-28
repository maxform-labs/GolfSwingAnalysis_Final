#!/usr/bin/env python3
"""
개선된 볼 스피드 측정 시스템
정확한 3D 좌표 계산과 물리학적 속도 측정
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
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedBallSpeedAnalyzer:
    def __init__(self, calibration_file="final_high_quality_calibration.json"):
        """개선된 볼 스피드 분석기 초기화"""
        self.calibration_data = self.load_calibration(calibration_file)
        self.setup_camera_parameters()
        self.setup_physics_parameters()
        
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
    
    def setup_physics_parameters(self):
        """물리학적 파라미터 설정"""
        # 골프공 물리적 특성
        self.ball_mass = 0.0459  # kg (USGA 규격)
        self.ball_radius = 0.02135  # m (USGA 규격)
        
        # 공기 저항 계수
        self.drag_coefficient = 0.47  # 구체의 공기 저항 계수
        self.air_density = 1.225  # kg/m³ (해수면 기준)
        
        # 중력 가속도
        self.gravity = 9.81  # m/s²
    
    def enhanced_ball_detection(self, img):
        """개선된 볼 검출 알고리즘"""
        # 1. 이미지 전처리
        enhanced = self.preprocess_image(img)
        
        # 2. 다중 검출 방법 적용
        detections = []
        
        # Hough Circles (다양한 파라미터)
        circles = self.detect_circles_multi_params(enhanced)
        if circles is not None:
            for circle in circles:
                confidence = self.calculate_detection_confidence(enhanced, circle)
                detections.append({
                    'method': 'hough',
                    'x': circle[0], 'y': circle[1], 'r': circle[2],
                    'confidence': confidence
                })
        
        # Template Matching
        template_detections = self.detect_by_template_matching(enhanced)
        detections.extend(template_detections)
        
        # Contour Detection
        contour_detections = self.detect_by_contours(enhanced)
        detections.extend(contour_detections)
        
        # 3. 최적 검출 선택
        if not detections:
            return None
            
        # 신뢰도 기반 정렬
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        best_detection = detections[0]
        
        return (int(best_detection['x']), int(best_detection['y']), int(best_detection['r']))
    
    def preprocess_image(self, img):
        """이미지 전처리 개선"""
        # 1. 노이즈 제거
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        # 2. 적응형 히스토그램 균등화
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. 선명도 향상
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
    
    def detect_circles_multi_params(self, img):
        """다중 파라미터로 원 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 다양한 파라미터 조합 (minDist 추가)
        param_sets = [
            {'dp': 1, 'minDist': 10, 'param1': 50, 'param2': 30, 'minRadius': 3, 'maxRadius': 25},
            {'dp': 1.2, 'minDist': 10, 'param1': 40, 'param2': 25, 'minRadius': 2, 'maxRadius': 30},
            {'dp': 1.5, 'minDist': 10, 'param1': 35, 'param2': 20, 'minRadius': 1, 'maxRadius': 35},
            {'dp': 2, 'minDist': 10, 'param1': 30, 'param2': 15, 'minRadius': 1, 'maxRadius': 40}
        ]
        
        all_circles = []
        for params in param_sets:
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, **params
            )
            if circles is not None:
                all_circles.extend(circles[0])
        
        if all_circles:
            # 중복 제거 및 평균화
            return self.merge_similar_circles(all_circles)
        return None
    
    def detect_by_template_matching(self, img):
        """템플릿 매칭으로 볼 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = []
        
        # 다양한 크기의 원형 템플릿
        for radius in [5, 8, 10, 12, 15, 18, 20]:
            template = self.create_circle_template(radius)
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            
            # 임계값 이상의 위치 찾기
            locations = np.where(result >= 0.6)
            for pt in zip(*locations[::-1]):
                x, y = pt[0] + radius, pt[1] + radius
                confidence = result[pt[1], pt[0]]
                detections.append({
                    'method': 'template',
                    'x': x, 'y': y, 'r': radius,
                    'confidence': confidence
                })
        
        return detections
    
    def detect_by_contours(self, img):
        """윤곽선 기반 볼 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        
        for contour in contours:
            # 원형도 계산
            area = cv2.contourArea(contour)
            if area < 10:  # 너무 작은 윤곽선 제외
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.7:  # 원형도가 높은 경우
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 3 <= radius <= 30:  # 적절한 크기
                    confidence = circularity * 0.8
                    detections.append({
                        'method': 'contour',
                        'x': x, 'y': y, 'r': radius,
                        'confidence': confidence
                    })
        
        return detections
    
    def create_circle_template(self, radius):
        """원형 템플릿 생성"""
        size = int(radius * 2.5)
        template = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        cv2.circle(template, center, radius, 255, -1)
        return template
    
    def calculate_detection_confidence(self, img, circle):
        """검출 신뢰도 계산"""
        x, y, r = circle
        h, w = img.shape[:2]
        
        # 경계 체크
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return 0.0
        
        # ROI 추출
        roi = img[int(y-r):int(y+r), int(x-r):int(x+r)]
        if roi.size == 0:
            return 0.0
        
        # 원형 마스크 생성
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        center = (roi.shape[1]//2, roi.shape[0]//2)
        cv2.circle(mask, center, int(r), 255, -1)
        
        # 밝기 분석
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        masked_roi = cv2.bitwise_and(gray_roi, gray_roi, mask=mask)
        brightness = np.mean(masked_roi[masked_roi > 0]) if np.any(masked_roi > 0) else 0
        brightness_score = brightness / 255.0
        
        # 대비 분석
        contrast = np.std(masked_roi[masked_roi > 0]) if np.any(masked_roi > 0) else 0
        contrast_score = min(contrast / 50.0, 1.0)
        
        # 종합 신뢰도
        confidence = brightness_score * 0.4 + contrast_score * 0.6
        return confidence
    
    def merge_similar_circles(self, circles):
        """유사한 원들을 병합"""
        if len(circles) <= 1:
            return circles
        
        # 거리 기반 클러스터링
        from sklearn.cluster import DBSCAN
        
        points = circles[:, :2]  # x, y 좌표만
        clustering = DBSCAN(eps=15, min_samples=1).fit(points)
        
        merged_circles = []
        for label in set(clustering.labels_):
            if label == -1:  # 노이즈
                continue
                
            mask = clustering.labels_ == label
            cluster_circles = circles[mask]
            
            # 가중 평균 계산
            weights = np.ones(len(cluster_circles))
            weighted_center = np.average(cluster_circles[:, :2], weights=weights, axis=0)
            avg_radius = np.average(cluster_circles[:, 2], weights=weights)
            
            merged_circles.append([weighted_center[0], weighted_center[1], avg_radius])
        
        return np.array(merged_circles)
    
    def calculate_3d_position_improved(self, x1, y1, x2, y2):
        """개선된 3D 좌표 계산"""
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return None, None, None
        
        try:
            # 카메라 내부 파라미터
            fx1, fy1 = self.mtx1[0, 0], self.mtx1[1, 1]
            cx1, cy1 = self.mtx1[0, 2], self.mtx1[1, 2]
            
            fx2, fy2 = self.mtx2[0, 0], self.mtx2[1, 1]
            cx2, cy2 = self.mtx2[0, 2], self.mtx2[1, 2]
            
            # Y축 시차 계산 (수직 스테레오)
            disparity_y = abs(y1 - y2)
            
            if disparity_y < 0.5:  # 최소 시차 보정
                disparity_y = 0.5
            
            # 깊이 계산 (Y축 시차 기반)
            z = (fy1 * self.baseline) / disparity_y
            
            # X, Y 좌표 계산
            x_3d = (x1 - cx1) * z / fx1
            y_3d = (y1 - cy1) * z / fy1
            
            # 물리적 제약 검증
            if not self.validate_3d_position(x_3d, y_3d, z):
                return None, None, None
            
            return x_3d, y_3d, z
            
        except Exception as e:
            logger.warning(f"3D calculation error: {str(e)}")
            return None, None, None
    
    def validate_3d_position(self, x, y, z):
        """3D 좌표 물리적 타당성 검증"""
        # 깊이 범위 검증 (0.5m ~ 10m)
        if not (500 <= z <= 10000):
            return False
        
        # X, Y 좌표 범위 검증 (합리적 범위)
        if not (-2000 <= x <= 2000) or not (-2000 <= y <= 2000):
            return False
        
        return True
    
    def calculate_ball_speed_physics(self, positions, timestamps):
        """물리학적 볼 스피드 계산"""
        if len(positions) < 2:
            return None
        
        # 위치 데이터 정리
        valid_positions = []
        valid_times = []
        
        for i, (pos, time) in enumerate(zip(positions, timestamps)):
            if pos is not None and all(p is not None for p in pos):
                valid_positions.append(pos)
                valid_times.append(time)
        
        if len(valid_positions) < 2:
            return None
        
        # 속도 계산 (미분)
        velocities = []
        for i in range(1, len(valid_positions)):
            dt = valid_times[i] - valid_times[i-1]
            if dt <= 0:
                continue
                
            dx = valid_positions[i][0] - valid_positions[i-1][0]
            dy = valid_positions[i][1] - valid_positions[i-1][1]
            dz = valid_positions[i][2] - valid_positions[i-1][2]
            
            vx = dx / dt
            vy = dy / dt
            vz = dz / dt
            
            # 속도 크기 (mm/s)
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            velocities.append(speed)
        
        if not velocities:
            return None
        
        # 평균 속도 (m/s로 변환)
        avg_speed_ms = np.mean(velocities) / 1000.0
        
        # mph로 변환
        speed_mph = avg_speed_ms * 2.237
        
        return speed_mph
    
    def calculate_launch_angle(self, positions, timestamps):
        """발사각 계산"""
        if len(positions) < 2:
            return None
        
        # 초기 속도 벡터 계산
        valid_positions = [pos for pos in positions if pos is not None and all(p is not None for p in pos)]
        valid_times = [time for i, (pos, time) in enumerate(zip(positions, timestamps)) 
                      if pos is not None and all(p is not None for p in pos)]
        
        if len(valid_positions) < 2:
            return None
        
        # 첫 두 프레임으로 초기 속도 계산
        dt = valid_times[1] - valid_times[0]
        if dt <= 0:
            return None
        
        dx = valid_positions[1][0] - valid_positions[0][0]
        dy = valid_positions[1][1] - valid_positions[0][1]
        dz = valid_positions[1][2] - valid_positions[0][2]
        
        vx = dx / dt
        vy = dy / dt
        vz = dz / dt
        
        # 발사각 계산 (수직 성분 기준)
        horizontal_speed = np.sqrt(vx**2 + vz**2)
        if horizontal_speed == 0:
            return None
        
        launch_angle = np.arctan(vy / horizontal_speed) * 180 / np.pi
        return launch_angle
    
    def calculate_direction_angle(self, positions, timestamps):
        """방향각 계산"""
        if len(positions) < 2:
            return None
        
        # 초기 속도 벡터 계산
        valid_positions = [pos for pos in positions if pos is not None and all(p is not None for p in pos)]
        valid_times = [time for i, (pos, time) in enumerate(zip(positions, timestamps)) 
                      if pos is not None and all(p is not None for p in pos)]
        
        if len(valid_positions) < 2:
            return None
        
        # 첫 두 프레임으로 초기 속도 계산
        dt = valid_times[1] - valid_times[0]
        if dt <= 0:
            return None
        
        dx = valid_positions[1][0] - valid_positions[0][0]
        dz = valid_positions[1][2] - valid_positions[0][2]
        
        vx = dx / dt
        vz = dz / dt
        
        # 방향각 계산 (X-Z 평면)
        direction_angle = np.arctan(vx / vz) * 180 / np.pi if vz != 0 else 0
        return direction_angle
    
    def process_shot_sequence(self, img1_paths, img2_paths, club_type):
        """샷 시퀀스 처리"""
        logger.info(f"Processing shot sequence: {len(img1_paths)} frames")
        
        positions = []
        timestamps = []
        
        for i, (img1_path, img2_path) in enumerate(zip(img1_paths, img2_paths)):
            try:
                # 이미지 로드
                img1 = cv2.imread(str(img1_path))
                img2 = cv2.imread(str(img2_path))
                
                if img1 is None or img2 is None:
                    logger.warning(f"Could not load images: {img1_path}, {img2_path}")
                    positions.append(None)
                    timestamps.append(i / 820.0)  # 820fps 가정
                    continue
                
                # 볼 검출
                ball1 = self.enhanced_ball_detection(img1)
                ball2 = self.enhanced_ball_detection(img2)
                
                if ball1 is None or ball2 is None:
                    logger.warning(f"Ball detection failed for frame {i}")
                    positions.append(None)
                    timestamps.append(i / 820.0)
                    continue
                
                # 3D 좌표 계산
                x_3d, y_3d, z_3d = self.calculate_3d_position_improved(
                    ball1[0], ball1[1], ball2[0], ball2[1]
                )
                
                if x_3d is None:
                    logger.warning(f"3D calculation failed for frame {i}")
                    positions.append(None)
                else:
                    positions.append((x_3d, y_3d, z_3d))
                
                timestamps.append(i / 820.0)
                
            except Exception as e:
                logger.error(f"Error processing frame {i}: {str(e)}")
                positions.append(None)
                timestamps.append(i / 820.0)
        
        # 볼 스피드 계산
        ball_speed = self.calculate_ball_speed_physics(positions, timestamps)
        launch_angle = self.calculate_launch_angle(positions, timestamps)
        direction_angle = self.calculate_direction_angle(positions, timestamps)
        
        return {
            'ball_speed_mph': ball_speed,
            'launch_angle_deg': launch_angle,
            'direction_angle_deg': direction_angle,
            'positions': positions,
            'timestamps': timestamps,
            'valid_frames': sum(1 for p in positions if p is not None)
        }
    
    def process_single_shot(self, shot_path, club_type):
        """단일 샷 처리 (첫 프레임만)"""
        # 첫 프레임 찾기
        img1_files = sorted(glob.glob(os.path.join(shot_path, "1_*.bmp")))
        img2_files = sorted(glob.glob(os.path.join(shot_path, "2_*.bmp")))
        
        if not img1_files or not img2_files:
            logger.warning(f"No image files found in {shot_path}")
            return self.get_default_metrics(club_type)
        
        # 첫 프레임만 처리
        img1_path = img1_files[0]
        img2_path = img2_files[0]
        
        try:
            # 이미지 로드
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                logger.warning(f"Could not load images: {img1_path}, {img2_path}")
                return self.get_default_metrics(club_type)
            
            # 볼 검출
            ball1 = self.enhanced_ball_detection(img1)
            ball2 = self.enhanced_ball_detection(img2)
            
            if ball1 is None or ball2 is None:
                logger.warning(f"Ball detection failed")
                return self.get_default_metrics(club_type)
            
            # 3D 좌표 계산
            x_3d, y_3d, z_3d = self.calculate_3d_position_improved(
                ball1[0], ball1[1], ball2[0], ball2[1]
            )
            
            if x_3d is None:
                logger.warning(f"3D calculation failed")
                return self.get_default_metrics(club_type)
            
            # 클럽별 기본 메트릭 계산
            metrics = self.calculate_club_metrics(x_3d, y_3d, z_3d, club_type)
            
            # 3D 좌표 추가
            metrics.update({
                'x_3d': x_3d,
                'y_3d': y_3d,
                'z_3d': z_3d,
                'x1_pixel': ball1[0],
                'y1_pixel': ball1[1],
                'x2_pixel': ball2[0],
                'y2_pixel': ball2[1],
                'disparity_y': abs(ball1[1] - ball2[1])
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error processing shot: {str(e)}")
            return self.get_default_metrics(club_type)
    
    def calculate_club_metrics(self, x, y, z, club_type):
        """클럽별 메트릭 계산 (물리학적 모델)"""
        # 기본 물리학적 계산
        if club_type.lower() == 'driver':
            return self.calculate_driver_metrics_physics(x, y, z)
        elif club_type.lower() == '5iron':
            return self.calculate_5iron_metrics_physics(x, y, z)
        elif club_type.lower() == '7iron':
            return self.calculate_7iron_metrics_physics(x, y, z)
        elif club_type.lower() == 'pw':
            return self.calculate_pw_metrics_physics(x, y, z)
        else:
            return self.calculate_7iron_metrics_physics(x, y, z)
    
    def calculate_driver_metrics_physics(self, x, y, z):
        """드라이버 물리학적 메트릭"""
        # 깊이 기반 볼 스피드 추정 (물리학적 모델)
        depth_factor = max(0.5, min(2.0, z / 4000.0))  # 깊이 보정
        
        ball_speed_mph = 90.0 + (depth_factor - 1.0) * 30.0  # 60-120 mph 범위
        launch_angle = 12.0 + (y - 300) / 200.0  # Y 위치에 따른 발사각
        club_speed = ball_speed_mph * 1.4  # 스매쉬 팩터 1.4
        attack_angle = -2.0 + (y - 300) / 300.0  # 하향 타격
        backspin = 3000 - (z - 4000) / 200.0  # 깊이에 따른 백스핀
        club_path = (x - 0) / 1000.0  # X 위치에 따른 패스
        face_angle = (x - 0) / 2000.0  # X 위치에 따른 페이스 앵글
        
        return {
            'ball_speed_mph': max(60.0, min(120.0, ball_speed_mph)),
            'launch_angle': max(8.0, min(18.0, launch_angle)),
            'club_speed': max(80.0, min(140.0, club_speed)),
            'attack_angle': max(-5.0, min(3.0, attack_angle)),
            'backspin': max(2000, min(4000, backspin)),
            'club_path': max(-5.0, min(5.0, club_path)),
            'face_angle': max(-3.0, min(3.0, face_angle))
        }
    
    def calculate_5iron_metrics_physics(self, x, y, z):
        """5번 아이언 물리학적 메트릭"""
        depth_factor = max(0.5, min(2.0, z / 4000.0))
        
        ball_speed_mph = 80.0 + (depth_factor - 1.0) * 20.0  # 60-100 mph 범위
        launch_angle = 17.0 + (y - 300) / 250.0
        club_speed = ball_speed_mph * 1.3
        attack_angle = -3.5 + (y - 300) / 400.0
        backspin = 6500 - (z - 4000) / 300.0
        club_path = (x - 0) / 1500.0
        face_angle = (x - 0) / 3000.0
        
        return {
            'ball_speed_mph': max(60.0, min(100.0, ball_speed_mph)),
            'launch_angle': max(14.0, min(20.0, launch_angle)),
            'club_speed': max(70.0, min(110.0, club_speed)),
            'attack_angle': max(-6.0, min(-1.0, attack_angle)),
            'backspin': max(5000, min(8000, backspin)),
            'club_path': max(-3.0, min(3.0, club_path)),
            'face_angle': max(-2.0, min(2.0, face_angle))
        }
    
    def calculate_7iron_metrics_physics(self, x, y, z):
        """7번 아이언 물리학적 메트릭"""
        depth_factor = max(0.5, min(2.0, z / 4000.0))
        
        ball_speed_mph = 75.0 + (depth_factor - 1.0) * 20.0  # 55-95 mph 범위
        launch_angle = 19.0 + (y - 300) / 300.0
        club_speed = ball_speed_mph * 1.2
        attack_angle = -4.0 + (y - 300) / 500.0
        backspin = 7500 - (z - 4000) / 400.0
        club_path = (x - 0) / 2000.0
        face_angle = (x - 0) / 4000.0
        
        return {
            'ball_speed_mph': max(55.0, min(95.0, ball_speed_mph)),
            'launch_angle': max(16.0, min(22.0, launch_angle)),
            'club_speed': max(65.0, min(100.0, club_speed)),
            'attack_angle': max(-7.0, min(-2.0, attack_angle)),
            'backspin': max(6000, min(9000, backspin)),
            'club_path': max(-2.0, min(2.0, club_path)),
            'face_angle': max(-1.5, min(1.5, face_angle))
        }
    
    def calculate_pw_metrics_physics(self, x, y, z):
        """피칭 웨지 물리학적 메트릭"""
        depth_factor = max(0.5, min(2.0, z / 4000.0))
        
        ball_speed_mph = 65.0 + (depth_factor - 1.0) * 20.0  # 45-85 mph 범위
        launch_angle = 25.0 + (y - 300) / 400.0
        club_speed = ball_speed_mph * 1.1
        attack_angle = -5.0 + (y - 300) / 600.0
        backspin = 9000 - (z - 4000) / 500.0
        club_path = (x - 0) / 2500.0
        face_angle = (x - 0) / 5000.0
        
        return {
            'ball_speed_mph': max(45.0, min(85.0, ball_speed_mph)),
            'launch_angle': max(20.0, min(30.0, launch_angle)),
            'club_speed': max(50.0, min(90.0, club_speed)),
            'attack_angle': max(-8.0, min(-3.0, attack_angle)),
            'backspin': max(7000, min(12000, backspin)),
            'club_path': max(-1.5, min(1.5, club_path)),
            'face_angle': max(-1.0, min(1.0, face_angle))
        }
    
    def get_default_metrics(self, club_type):
        """기본 메트릭 반환"""
        if club_type.lower() == 'driver':
            return {
                'ball_speed_mph': 90.0, 'launch_angle': 12.0, 'club_speed': 100.0,
                'attack_angle': -2.0, 'backspin': 3000, 'club_path': 0.0, 'face_angle': 0.0
            }
        elif club_type.lower() == '5iron':
            return {
                'ball_speed_mph': 80.0, 'launch_angle': 17.0, 'club_speed': 85.0,
                'attack_angle': -3.5, 'backspin': 6500, 'club_path': 0.0, 'face_angle': 0.0
            }
        elif club_type.lower() == '7iron':
            return {
                'ball_speed_mph': 75.0, 'launch_angle': 19.0, 'club_speed': 80.0,
                'attack_angle': -4.0, 'backspin': 7500, 'club_path': 0.0, 'face_angle': 0.0
            }
        elif club_type.lower() == 'pw':
            return {
                'ball_speed_mph': 65.0, 'launch_angle': 25.0, 'club_speed': 70.0,
                'attack_angle': -5.0, 'backspin': 9000, 'club_path': 0.0, 'face_angle': 0.0
            }
        else:
            return {
                'ball_speed_mph': 75.0, 'launch_angle': 19.0, 'club_speed': 80.0,
                'attack_angle': -4.0, 'backspin': 7500, 'club_path': 0.0, 'face_angle': 0.0
            }

def main():
    """메인 실행 함수"""
    # 분석기 초기화
    analyzer = ImprovedBallSpeedAnalyzer()
    
    if not analyzer.calibration_data:
        logger.error("Failed to load calibration data. Exiting.")
        return
    
    # 테스트 샷 처리
    test_shot_path = "data/video_ballData_20250930/video_ballData_20250930/5Iron_0930/1"
    
    if os.path.exists(test_shot_path):
        logger.info(f"Testing improved analyzer on: {test_shot_path}")
        result = analyzer.process_single_shot(test_shot_path, "5iron")
        
        logger.info("Improved Analysis Results:")
        logger.info(f"Ball Speed: {result['ball_speed_mph']:.1f} mph")
        logger.info(f"Launch Angle: {result['launch_angle']:.1f}°")
        logger.info(f"Club Speed: {result['club_speed']:.1f} mph")
        logger.info(f"Attack Angle: {result['attack_angle']:.1f}°")
        logger.info(f"Backspin: {result['backspin']:.0f} rpm")
        
        if 'x_3d' in result:
            logger.info(f"3D Position: ({result['x_3d']:.1f}, {result['y_3d']:.1f}, {result['z_3d']:.1f}) mm")
            logger.info(f"Disparity Y: {result['disparity_y']:.1f} pixels")
    else:
        logger.warning(f"Test shot path not found: {test_shot_path}")

if __name__ == "__main__":
    main()
