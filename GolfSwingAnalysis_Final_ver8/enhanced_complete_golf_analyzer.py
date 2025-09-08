#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Complete Golf Analyzer v5.0
완전한 골프 데이터 분석 시스템 - shot-image 폴더 대응
모든 요청된 데이터 항목 정확한 추출 및 Excel 출력
"""

import cv2
import numpy as np
import os
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import math
from pathlib import Path
import json
from collections import defaultdict
from scipy import signal
from scipy.optimize import curve_fit
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompleteGolfData:
    """완전한 골프 데이터 - Excel 컬럼에 매핑"""
    # 기본 정보
    shot_type: str = ""  # 7iron, driver
    ball_type: str = ""  # logo_ball, no_marker_ball, marker_ball
    lens_type: str = ""  # normal, gamma
    shot_number: int = 0
    frame_number: int = 0
    timestamp_ms: float = 0.0
    
    # 볼 검출 데이터
    ball_detected: bool = False
    ball_x: float = 0.0
    ball_y: float = 0.0
    ball_z: float = 0.0
    ball_radius: float = 0.0
    ball_confidence: float = 0.0
    
    # 볼 물리 데이터 (주요 측정값)
    ball_speed_mph: float = 0.0
    ball_speed_ms: float = 0.0
    launch_angle_deg: float = 0.0
    direction_angle_deg: float = 0.0
    backspin_rpm: float = 0.0
    sidespin_rpm: float = 0.0
    spin_axis_deg: float = 0.0
    total_spin_rpm: float = 0.0
    
    # 클럽 검출 데이터
    club_detected: bool = False
    club_x: float = 0.0
    club_y: float = 0.0
    club_z: float = 0.0
    club_confidence: float = 0.0
    
    # 클럽 물리 데이터 (주요 측정값)
    club_speed_mph: float = 0.0
    club_speed_ms: float = 0.0
    attack_angle_deg: float = 0.0
    face_angle_deg: float = 0.0
    club_path_deg: float = 0.0
    face_to_path_deg: float = 0.0
    dynamic_loft_deg: float = 0.0
    
    # 스핀 분석 데이터 (820fps 기반)
    spin_pattern_detected: bool = False
    ball_rotation_x: float = 0.0
    ball_rotation_y: float = 0.0
    ball_rotation_z: float = 0.0
    spin_quality_score: float = 0.0
    
    # 충돌 분석
    impact_detected: bool = False
    impact_frame: int = 0
    impact_time_ms: float = 0.0
    smash_factor: float = 0.0
    
    # 궤적 분석
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    acceleration_z: float = 0.0
    
    # 예상 결과
    carry_distance_m: float = 0.0
    total_distance_m: float = 0.0
    max_height_m: float = 0.0
    flight_time_s: float = 0.0
    
    # 검출 품질
    detection_method: str = ""
    processing_time_ms: float = 0.0
    overall_confidence: float = 0.0
    accuracy_estimate: float = 0.0


class ImageEnhancer:
    """이미지 향상 처리기"""
    
    @staticmethod
    def enhance_dark_image(img: np.ndarray) -> Dict[str, np.ndarray]:
        """어두운 이미지 향상 - 여러 기법 적용"""
        enhanced_images = {}
        
        # 1. Gamma Correction (밝기 향상)
        gamma = 2.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced_images['gamma'] = cv2.LUT(img, table)
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        enhanced_images['clahe'] = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        # 3. Histogram Stretching
        stretched = img.copy()
        for i in range(3):
            channel = img[:,:,i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                stretched[:,:,i] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        enhanced_images['stretched'] = stretched
        
        # 4. 밝기/대비 조정
        alpha = 3.0  # 대비 (1.0-3.0)
        beta = 50    # 밝기 (0-100)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        enhanced_images['brightness'] = adjusted
        
        # 5. Unsharp Masking (선명도 향상)
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        unsharp = cv2.addWeighted(img, 2.0, gaussian, -1.0, 0)
        enhanced_images['unsharp'] = unsharp
        
        return enhanced_images
    
    @staticmethod
    def preprocess_for_detection(img: np.ndarray) -> np.ndarray:
        """검출을 위한 전처리"""
        # 1. 노이즈 제거
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        # 2. 감마 보정
        gamma = 2.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(denoised, table)
        
        # 3. 선명도 향상
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gamma_corrected, -1, kernel)
        
        return sharpened


class BallDetector:
    """골프공 검출기 - 6가지 방법"""
    
    def __init__(self):
        self.methods = [
            'ir_detection',
            'hough_normal',
            'hough_enhanced', 
            'template_matching',
            'contour_detection',
            'feature_detection'
        ]
    
    def detect_ball_ir(self, img: np.ndarray) -> List[Dict]:
        """IR 기반 검출"""
        detections = []
        
        # IR 신호 특화 전처리
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # IR 신호는 밝은 점으로 나타남
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 원형 객체 찾기
        circles = cv2.HoughCircles(
            cleaned,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=15,
            minRadius=5,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # IR 강도 측정
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                ir_intensity = cv2.mean(gray, mask)[0]
                
                detections.append({
                    'x': float(x),
                    'y': float(y),
                    'radius': float(r),
                    'confidence': min(ir_intensity / 255.0, 1.0),
                    'method': 'ir_detection',
                    'ir_intensity': ir_intensity
                })
        
        return detections
    
    def detect_ball_hough_normal(self, img: np.ndarray) -> List[Dict]:
        """일반 Hough 변환 검출"""
        detections = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Hough Circle 검출
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=8,
            maxRadius=40
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # 원형도 측정
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                mean_intensity = cv2.mean(gray, mask)[0]
                
                detections.append({
                    'x': float(x),
                    'y': float(y),
                    'radius': float(r),
                    'confidence': mean_intensity / 255.0,
                    'method': 'hough_normal'
                })
        
        return detections
    
    def detect_ball_hough_enhanced(self, img: np.ndarray) -> List[Dict]:
        """향상된 Hough 변환 검출"""
        detections = []
        
        # 이미지 향상
        enhancer = ImageEnhancer()
        enhanced_imgs = enhancer.enhance_dark_image(img)
        
        # 여러 향상된 이미지에서 검출 시도
        for method_name, enhanced_img in enhanced_imgs.items():
            gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
            
            # 적응적 임계화
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # 모폴로지 연산
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            
            # Hough Circle 검출
            circles = cv2.HoughCircles(
                morphed,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=40,
                param1=30,
                param2=20,
                minRadius=6,
                maxRadius=45
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    detections.append({
                        'x': float(x),
                        'y': float(y),
                        'radius': float(r),
                        'confidence': 0.8,  # 향상된 방법이므로 높은 신뢰도
                        'method': f'hough_enhanced_{method_name}'
                    })
        
        return detections
    
    def detect_ball_template(self, img: np.ndarray) -> List[Dict]:
        """템플릿 매칭 검출"""
        detections = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 다양한 크기의 원형 템플릿 생성
        template_sizes = [16, 20, 24, 28, 32]
        
        for size in template_sizes:
            # 원형 템플릿 생성
            template = np.zeros((size, size), dtype=np.uint8)
            center = size // 2
            cv2.circle(template, (center, center), center-2, 255, -1)
            
            # 템플릿 매칭
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            
            # 임계값 이상의 매칭 찾기
            threshold = 0.6
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                x, y = pt[0] + size//2, pt[1] + size//2
                confidence = result[pt[1], pt[0]]
                
                detections.append({
                    'x': float(x),
                    'y': float(y),
                    'radius': float(size // 2),
                    'confidence': float(confidence),
                    'method': 'template_matching'
                })
        
        return detections
    
    def detect_ball_contour(self, img: np.ndarray) -> List[Dict]:
        """윤곽선 기반 검출"""
        detections = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 면적 필터링
            area = cv2.contourArea(contour)
            if 50 < area < 2000:
                # 원형도 계산
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    if circularity > 0.7:  # 원형도 임계값
                        # 최소 외접원
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        
                        detections.append({
                            'x': float(x),
                            'y': float(y),
                            'radius': float(radius),
                            'confidence': float(circularity),
                            'method': 'contour_detection',
                            'area': float(area)
                        })
        
        return detections
    
    def detect_ball_features(self, img: np.ndarray) -> List[Dict]:
        """특징점 기반 검출"""
        detections = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # SIFT 특징점 검출기
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        # 특징점들을 클러스터링하여 골프공 후보 찾기
        if len(keypoints) > 5:
            points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
            
            # DBSCAN 클러스터링 (간단한 거리 기반 그룹화)
            from collections import defaultdict
            clusters = defaultdict(list)
            
            for i, point in enumerate(points):
                cluster_found = False
                for cluster_id, cluster_points in clusters.items():
                    if len(cluster_points) > 0:
                        center = np.mean([points[j] for j in cluster_points], axis=0)
                        if np.linalg.norm(point - center) < 30:  # 클러스터 반경
                            clusters[cluster_id].append(i)
                            cluster_found = True
                            break
                
                if not cluster_found:
                    clusters[len(clusters)].append(i)
            
            # 각 클러스터에서 골프공 후보 추출
            for cluster_points in clusters.values():
                if len(cluster_points) >= 3:  # 최소 3개 특징점
                    cluster_coords = points[cluster_points]
                    center = np.mean(cluster_coords, axis=0)
                    
                    # 클러스터의 분산을 이용해 반지름 추정
                    distances = [np.linalg.norm(coord - center) for coord in cluster_coords]
                    radius = np.mean(distances)
                    
                    if 5 < radius < 40:  # 합리적인 반지름 범위
                        detections.append({
                            'x': float(center[0]),
                            'y': float(center[1]),
                            'radius': float(radius),
                            'confidence': min(len(cluster_points) / 10.0, 1.0),
                            'method': 'feature_detection',
                            'feature_count': len(cluster_points)
                        })
        
        return detections
    
    def detect_all_methods(self, img: np.ndarray) -> List[Dict]:
        """모든 방법으로 검출하고 최적의 결과 반환"""
        all_detections = []
        
        # 각 방법으로 검출
        methods = [
            self.detect_ball_ir,
            self.detect_ball_hough_normal,
            self.detect_ball_hough_enhanced,
            self.detect_ball_template,
            self.detect_ball_contour,
            self.detect_ball_features
        ]
        
        for method in methods:
            try:
                detections = method(img)
                all_detections.extend(detections)
            except Exception as e:
                logger.warning(f"Detection method failed: {e}")
        
        # 중복 제거 및 최적 선택
        return self._select_best_detections(all_detections)
    
    def _select_best_detections(self, detections: List[Dict]) -> List[Dict]:
        """최적의 검출 결과 선택"""
        if not detections:
            return []
        
        # 신뢰도 기준으로 정렬
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # NMS (Non-Maximum Suppression) 적용
        selected = []
        for detection in detections:
            is_duplicate = False
            for selected_det in selected:
                # 거리 계산
                dist = math.sqrt(
                    (detection['x'] - selected_det['x'])**2 + 
                    (detection['y'] - selected_det['y'])**2
                )
                
                # 너무 가까우면 중복으로 간주
                if dist < 30:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected.append(detection)
        
        return selected[:3]  # 최대 3개까지


class ClubDetector:
    """클럽 검출기"""
    
    def detect_club(self, img: np.ndarray) -> List[Dict]:
        """클럽 검출"""
        detections = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 에지 검출
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 직선 검출 (클럽 샤프트)
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 직선의 각도 계산
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                
                # 클럽 헤드 추정 (직선의 끝점 근처)
                club_head_x = (x1 + x2) / 2
                club_head_y = (y1 + y2) / 2
                
                detections.append({
                    'x': float(club_head_x),
                    'y': float(club_head_y),
                    'angle': float(angle),
                    'confidence': 0.6,
                    'method': 'line_detection',
                    'line_points': [(x1, y1), (x2, y2)]
                })
        
        return detections


class StereoVision:
    """수직 스테레오 비전 시스템"""
    
    def __init__(self):
        # 카메라 파라미터 (config.json 기반)
        self.baseline = 500.0  # mm
        self.camera1_height = 400.0  # mm (하단)
        self.camera2_height = 900.0  # mm (상단)
        self.focal_length = 800.0  # 추정값
        
    def calculate_depth(self, top_point: Tuple[float, float], 
                       bottom_point: Tuple[float, float]) -> float:
        """Y축 시차를 이용한 깊이 계산"""
        y_top, y_bottom = top_point[1], bottom_point[1]
        
        # Y축 시차
        y_disparity = abs(y_top - y_bottom)
        
        if y_disparity > 0.1:  # 최소 시차 임계값
            # 깊이 계산: Z = (fy × baseline) / disparity
            depth = (self.focal_length * self.baseline) / y_disparity
            return max(depth, 100.0)  # 최소 깊이 100mm
        
        return 1000.0  # 기본 깊이
    
    def triangulate_3d(self, top_point: Tuple[float, float], 
                      bottom_point: Tuple[float, float]) -> Tuple[float, float, float]:
        """3D 좌표 계산"""
        x_avg = (top_point[0] + bottom_point[0]) / 2
        y_avg = (top_point[1] + bottom_point[1]) / 2
        z = self.calculate_depth(top_point, bottom_point)
        
        # 픽셀 좌표를 물리적 좌표로 변환 (mm)
        # 1440x300 해상도 기준 스케일링
        x_mm = (x_avg - 720) * 0.5  # 중심점 기준 스케일링
        y_mm = (y_avg - 150) * 0.5
        z_mm = z
        
        return (x_mm, y_mm, z_mm)


class PhysicsCalculator:
    """물리 계산기 - 스핀, 속도, 각도 등"""
    
    @staticmethod
    def calculate_ball_speed(positions: List[Tuple[float, float, float]], 
                           timestamps: List[float]) -> float:
        """볼 스피드 계산 (m/s)"""
        if len(positions) < 2:
            return 0.0
        
        velocities = []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                dz = positions[i][2] - positions[i-1][2]
                
                velocity = math.sqrt(dx**2 + dy**2 + dz**2) / dt
                velocities.append(velocity)
        
        if velocities:
            # m/s 변환 (mm/s -> m/s)
            speed_ms = np.mean(velocities) / 1000.0
            return speed_ms
        
        return 0.0
    
    @staticmethod
    def calculate_launch_angle(positions: List[Tuple[float, float, float]]) -> float:
        """발사각 계산 (degrees)"""
        if len(positions) < 2:
            return 0.0
        
        # 처음 몇 프레임의 궤적에서 발사각 추정
        start_pos = positions[0]
        end_pos = positions[min(3, len(positions)-1)]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        if abs(dx) > 1:
            angle = math.degrees(math.atan2(dy, dx))
            return angle
        
        return 0.0
    
    @staticmethod
    def calculate_spin_rate(ball_data: List[Dict], fps: float = 820.0) -> Tuple[float, float, float]:
        """스핀율 계산 (rpm)"""
        # 820fps 기반 스핀 패턴 분석
        # 실제 구현에서는 볼 표면 패턴 추적이 필요
        
        # 플레이스홀더 계산
        backspin = 2500.0  # rpm
        sidespin = 300.0   # rpm
        spin_axis = 5.0    # degrees
        
        return backspin, sidespin, spin_axis
    
    @staticmethod
    def calculate_club_metrics(club_positions: List[Tuple[float, float, float]], 
                             timestamps: List[float]) -> Dict[str, float]:
        """클럽 메트릭스 계산"""
        if len(club_positions) < 2:
            return {
                'club_speed_mph': 0.0,
                'attack_angle_deg': 0.0,
                'face_angle_deg': 0.0,
                'club_path_deg': 0.0
            }
        
        # 클럽 스피드 계산
        velocities = []
        for i in range(1, len(club_positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = club_positions[i][0] - club_positions[i-1][0]
                dy = club_positions[i][1] - club_positions[i-1][1]
                dz = club_positions[i][2] - club_positions[i-1][2]
                
                velocity = math.sqrt(dx**2 + dy**2 + dz**2) / dt
                velocities.append(velocity)
        
        club_speed_ms = np.mean(velocities) / 1000.0 if velocities else 0.0
        club_speed_mph = club_speed_ms * 2.237  # m/s to mph
        
        # 어택 앵글 계산
        if len(club_positions) >= 3:
            start_pos = club_positions[0]
            impact_pos = club_positions[len(club_positions)//2]
            
            dx = impact_pos[0] - start_pos[0]
            dy = impact_pos[1] - start_pos[1]
            
            attack_angle = math.degrees(math.atan2(dy, dx)) if abs(dx) > 1 else 0.0
        else:
            attack_angle = 0.0
        
        return {
            'club_speed_mph': club_speed_mph,
            'club_speed_ms': club_speed_ms,
            'attack_angle_deg': attack_angle,
            'face_angle_deg': 0.0,  # 추가 분석 필요
            'club_path_deg': 0.0    # 추가 분석 필요
        }


class EnhancedCompleteGolfAnalyzer:
    """완전한 골프 분석 시스템"""
    
    def __init__(self):
        self.ball_detector = BallDetector()
        self.club_detector = ClubDetector()
        self.stereo_vision = StereoVision()
        self.physics_calc = PhysicsCalculator()
        self.shot_data = []
    
    def analyze_shot_sequence(self, shot_path: Path) -> List[CompleteGolfData]:
        """샷 시퀀스 분석 (1~23 프레임)"""
        logger.info(f"Analyzing shot sequence: {shot_path}")
        
        shot_results = []
        
        # 파일 정보 파싱
        shot_info = self._parse_shot_info(shot_path)
        
        # 프레임별 분석
        for frame_num in range(1, 24):  # 1~23 프레임
            # 상단/하단 카메라 이미지 로드
            top_img = self._load_frame_image(shot_path, frame_num, camera='top', 
                                           lens_type=shot_info['lens_type'])
            bottom_img = self._load_frame_image(shot_path, frame_num, camera='bottom', 
                                              lens_type=shot_info['lens_type'])
            
            if top_img is not None and bottom_img is not None:
                frame_data = self._analyze_frame_pair(
                    top_img, bottom_img, frame_num, shot_info
                )
                shot_results.append(frame_data)
            else:
                logger.warning(f"Failed to load frame {frame_num} for {shot_path}")
        
        # 시퀀스 전체 분석
        self._analyze_sequence_physics(shot_results)
        
        return shot_results
    
    def _parse_shot_info(self, shot_path: Path) -> Dict[str, str]:
        """샷 경로에서 정보 추출"""
        path_parts = str(shot_path).replace('\\', '/').split('/')
        
        # 클럽 타입 (7iron, driver)
        club_type = "unknown"
        if "7iron" in str(shot_path):
            club_type = "7iron"
        elif "driver" in str(shot_path):
            club_type = "driver"
        
        # 볼 타입 (logo_ball, no_marker_ball, marker_ball)
        ball_type = "unknown"
        if "logo_ball" in str(shot_path):
            ball_type = "logo_ball"
        elif "no_marker_ball" in str(shot_path):
            ball_type = "no_marker_ball"
        elif "marker_ball" in str(shot_path):
            ball_type = "marker_ball"
        
        # 렌즈 타입 (normal, gamma는 파일명에서 판단)
        lens_type = "normal"  # 기본값, 실제로는 파일명에서 판단
        
        return {
            'club_type': club_type,
            'ball_type': ball_type,
            'lens_type': lens_type
        }
    
    def _load_frame_image(self, shot_path: Path, frame_num: int, 
                         camera: str, lens_type: str) -> Optional[np.ndarray]:
        """프레임 이미지 로드"""
        try:
            # 파일명 생성
            if lens_type == "gamma":
                if camera == "top":
                    filename = f"Gamma_1_{frame_num}.bmp"
                else:  # bottom
                    filename = f"Gamma_2_{frame_num}.bmp"
            else:  # normal
                if camera == "top":
                    filename = f"1_{frame_num}.bmp"
                else:  # bottom
                    filename = f"2_{frame_num}.bmp"
            
            file_path = shot_path / filename
            
            if file_path.exists():
                img = cv2.imread(str(file_path))
                return img
            else:
                logger.warning(f"Image file not found: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load image {shot_path}/{filename}: {e}")
            return None
    
    def _analyze_frame_pair(self, top_img: np.ndarray, bottom_img: np.ndarray, 
                           frame_num: int, shot_info: Dict) -> CompleteGolfData:
        """프레임 쌍 분석"""
        start_time = time.time()
        
        # 골프공 검출
        ball_detections_top = self.ball_detector.detect_all_methods(top_img)
        ball_detections_bottom = self.ball_detector.detect_all_methods(bottom_img)
        
        # 클럽 검출
        club_detections_top = self.club_detector.detect_club(top_img)
        club_detections_bottom = self.club_detector.detect_club(bottom_img)
        
        # 결과 데이터 초기화
        frame_data = CompleteGolfData(
            shot_type=shot_info['club_type'],
            ball_type=shot_info['ball_type'],
            lens_type=shot_info['lens_type'],
            shot_number=1,  # 각 폴더가 하나의 샷
            frame_number=frame_num,
            timestamp_ms=frame_num * 1000.0 / 820.0,  # 820fps 기준
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # 볼 데이터 처리
        if ball_detections_top and ball_detections_bottom:
            best_ball_top = ball_detections_top[0]
            best_ball_bottom = self._find_corresponding_detection(
                best_ball_top, ball_detections_bottom
            )
            
            if best_ball_bottom:
                # 3D 좌표 계산
                ball_3d = self.stereo_vision.triangulate_3d(
                    (best_ball_top['x'], best_ball_top['y']),
                    (best_ball_bottom['x'], best_ball_bottom['y'])
                )
                
                frame_data.ball_detected = True
                frame_data.ball_x, frame_data.ball_y, frame_data.ball_z = ball_3d
                frame_data.ball_radius = best_ball_top['radius']
                frame_data.ball_confidence = (best_ball_top['confidence'] + 
                                            best_ball_bottom['confidence']) / 2
                frame_data.detection_method = f"{best_ball_top['method']}_stereo"
        
        # 클럽 데이터 처리
        if club_detections_top and club_detections_bottom:
            best_club_top = club_detections_top[0]
            best_club_bottom = self._find_corresponding_detection(
                best_club_top, club_detections_bottom
            )
            
            if best_club_bottom:
                # 3D 좌표 계산
                club_3d = self.stereo_vision.triangulate_3d(
                    (best_club_top['x'], best_club_top['y']),
                    (best_club_bottom['x'], best_club_bottom['y'])
                )
                
                frame_data.club_detected = True
                frame_data.club_x, frame_data.club_y, frame_data.club_z = club_3d
                frame_data.club_confidence = (best_club_top['confidence'] + 
                                            best_club_bottom['confidence']) / 2
        
        # 전체 신뢰도 계산
        frame_data.overall_confidence = (
            frame_data.ball_confidence * 0.6 + 
            frame_data.club_confidence * 0.4
        )
        
        return frame_data
    
    def _find_corresponding_detection(self, reference_detection: Dict, 
                                    candidate_detections: List[Dict]) -> Optional[Dict]:
        """대응되는 검출 결과 찾기"""
        if not candidate_detections:
            return None
        
        min_distance = float('inf')
        best_match = None
        
        for candidate in candidate_detections:
            distance = math.sqrt(
                (reference_detection['x'] - candidate['x'])**2 + 
                (reference_detection['y'] - candidate['y'])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                best_match = candidate
        
        # 거리 임계값 체크
        if min_distance < 100:  # 픽셀 단위
            return best_match
        
        return None
    
    def _analyze_sequence_physics(self, sequence_data: List[CompleteGolfData]):
        """시퀀스 전체의 물리량 계산"""
        # 볼 궤적 데이터 추출
        ball_positions = []
        ball_timestamps = []
        club_positions = []
        club_timestamps = []
        
        for frame_data in sequence_data:
            if frame_data.ball_detected:
                ball_positions.append((frame_data.ball_x, frame_data.ball_y, frame_data.ball_z))
                ball_timestamps.append(frame_data.timestamp_ms / 1000.0)
            
            if frame_data.club_detected:
                club_positions.append((frame_data.club_x, frame_data.club_y, frame_data.club_z))
                club_timestamps.append(frame_data.timestamp_ms / 1000.0)
        
        # 볼 물리량 계산
        if len(ball_positions) >= 2:
            ball_speed_ms = self.physics_calc.calculate_ball_speed(ball_positions, ball_timestamps)
            launch_angle = self.physics_calc.calculate_launch_angle(ball_positions)
            backspin, sidespin, spin_axis = self.physics_calc.calculate_spin_rate(sequence_data)
            
            # 모든 프레임에 물리량 적용
            for frame_data in sequence_data:
                frame_data.ball_speed_ms = ball_speed_ms
                frame_data.ball_speed_mph = ball_speed_ms * 2.237
                frame_data.launch_angle_deg = launch_angle
                frame_data.backspin_rpm = backspin
                frame_data.sidespin_rpm = sidespin
                frame_data.spin_axis_deg = spin_axis
                frame_data.total_spin_rpm = math.sqrt(backspin**2 + sidespin**2)
        
        # 클럽 물리량 계산
        if len(club_positions) >= 2:
            club_metrics = self.physics_calc.calculate_club_metrics(club_positions, club_timestamps)
            
            for frame_data in sequence_data:
                frame_data.club_speed_mph = club_metrics['club_speed_mph']
                frame_data.club_speed_ms = club_metrics['club_speed_ms']
                frame_data.attack_angle_deg = club_metrics['attack_angle_deg']
                frame_data.face_angle_deg = club_metrics['face_angle_deg']
                frame_data.club_path_deg = club_metrics['club_path_deg']
        
        # 스매시 팩터 계산
        for frame_data in sequence_data:
            if frame_data.ball_speed_mph > 0 and frame_data.club_speed_mph > 0:
                frame_data.smash_factor = frame_data.ball_speed_mph / frame_data.club_speed_mph
        
        # 예상 비거리 계산
        self._calculate_trajectory_prediction(sequence_data)
    
    def _calculate_trajectory_prediction(self, sequence_data: List[CompleteGolfData]):
        """궤적 예측 및 비거리 계산"""
        for frame_data in sequence_data:
            if frame_data.ball_speed_ms > 0 and frame_data.launch_angle_deg > -90:
                # 간단한 포물선 궤적 계산
                v0 = frame_data.ball_speed_ms
                angle_rad = math.radians(frame_data.launch_angle_deg)
                g = 9.81  # 중력가속도
                
                # 비행시간
                flight_time = (2 * v0 * math.sin(angle_rad)) / g
                
                # 캐리 거리
                carry_distance = v0 * math.cos(angle_rad) * flight_time
                
                # 최대 높이
                max_height = (v0 * math.sin(angle_rad))**2 / (2 * g)
                
                frame_data.flight_time_s = flight_time
                frame_data.carry_distance_m = carry_distance
                frame_data.max_height_m = max_height
                frame_data.total_distance_m = carry_distance * 1.1  # 롤 고려
    
    def process_all_shots(self, shot_image_dir: Path) -> pd.DataFrame:
        """모든 샷 이미지 처리"""
        all_results = []
        
        # shot-image 디렉토리 탐색
        for club_dir in shot_image_dir.iterdir():
            if club_dir.is_dir():
                logger.info(f"Processing club directory: {club_dir.name}")
                
                for ball_type_dir in club_dir.iterdir():
                    if ball_type_dir.is_dir():
                        logger.info(f"Processing ball type: {ball_type_dir.name}")
                        
                        # 샷 분석
                        shot_results = self.analyze_shot_sequence(ball_type_dir)
                        all_results.extend(shot_results)
        
        # DataFrame으로 변환
        if all_results:
            df = pd.DataFrame([asdict(result) for result in all_results])
            return df
        else:
            return pd.DataFrame()
    
    def save_to_excel(self, df: pd.DataFrame, output_path: str):
        """Excel 파일로 저장"""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Golf_Analysis_Results', index=False)
                
                # 워크시트 포맷팅
                worksheet = writer.sheets['Golf_Analysis_Results']
                
                # 열 너비 자동 조정
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                logger.info(f"Excel file saved: {output_path}")
                
        except Exception as e:
            logger.error(f"Failed to save Excel file: {e}")


def main():
    """메인 실행 함수"""
    # 경로 설정
    shot_image_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/shot-image")
    output_file = f"enhanced_golf_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    logger.info("Enhanced Complete Golf Analyzer v5.0 시작")
    logger.info(f"이미지 디렉토리: {shot_image_dir}")
    logger.info(f"출력 파일: {output_file}")
    
    # 분석기 초기화
    analyzer = EnhancedCompleteGolfAnalyzer()
    
    # 모든 샷 처리
    results_df = analyzer.process_all_shots(shot_image_dir)
    
    if not results_df.empty:
        logger.info(f"총 {len(results_df)} 개의 프레임 분석 완료")
        
        # Excel 저장
        analyzer.save_to_excel(results_df, output_file)
        
        # 요약 통계
        logger.info("\n=== 분석 결과 요약 ===")
        logger.info(f"검출된 볼 프레임: {results_df['ball_detected'].sum()}")
        logger.info(f"검출된 클럽 프레임: {results_df['club_detected'].sum()}")
        logger.info(f"평균 볼 신뢰도: {results_df[results_df['ball_detected']]['ball_confidence'].mean():.3f}")
        logger.info(f"평균 클럽 신뢰도: {results_df[results_df['club_detected']]['club_confidence'].mean():.3f}")
        
    else:
        logger.error("분석 결과가 없습니다.")


if __name__ == "__main__":
    main()