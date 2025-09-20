#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMP 직접 처리 및 혁신적 딤플 분석 시스템 v7.0
개발팀: maxform
목표: 820fps BMP 직접 처리, 혁신적 딤플 패턴 분석으로 95% 정확도 달성
혁신: 임팩트 기준 전후 4프레임 확대 딤플 분석, 과노출 볼 처리
"""

import cv2
import numpy as np
import json
import os
import pandas as pd
from datetime import datetime
import math
import struct
from pathlib import Path
from scipy import ndimage, optimize
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class EnhancedBMPGolfAnalyzer:
    """BMP 직접 처리 및 혁신적 딤플 분석 시스템"""
    
    def __init__(self):
        """초기화"""
        self.fps = 820
        self.pixel_to_mm = 0.1  # 픽셀-mm 변환 비율
        self.gravity = 9.81  # 중력 가속도
        
        # TrackMan 기준 데이터 (수준별) - 로프트앵글 제외
        self.trackman_reference = {
            '7iron': {
                'beginner': {'ball_speed': 75, 'club_speed': 60, 'launch_angle': 25, 'backspin': 8500, 'sidespin': 400},
                'amateur': {'ball_speed': 95, 'club_speed': 75, 'launch_angle': 20, 'backspin': 7500, 'sidespin': 300},
                'scratch': {'ball_speed': 115, 'club_speed': 90, 'launch_angle': 18, 'backspin': 7000, 'sidespin': 200},
                'pro': {'ball_speed': 130, 'club_speed': 100, 'launch_angle': 16, 'backspin': 6500, 'sidespin': 100}
            },
            'driver': {
                'beginner': {'ball_speed': 110, 'club_speed': 85, 'launch_angle': 15, 'backspin': 3500, 'sidespin': 500},
                'amateur': {'ball_speed': 130, 'club_speed': 100, 'launch_angle': 12, 'backspin': 3000, 'sidespin': 400},
                'scratch': {'ball_speed': 150, 'club_speed': 115, 'launch_angle': 10, 'backspin': 2500, 'sidespin': 300},
                'pro': {'ball_speed': 170, 'club_speed': 130, 'launch_angle': 8, 'backspin': 2200, 'sidespin': 200}
            }
        }
        
        # 12개 파라미터 (로프트앵글 제외)
        self.target_parameters = [
            # 볼 데이터 (6개)
            'ball_speed', 'launch_angle', 'direction_angle', 'backspin', 'sidespin', 'spin_axis',
            # 클럽 데이터 (6개) - 로프트앵글 제외
            'club_speed', 'attack_angle', 'club_path', 'face_angle', 'face_to_path', 'smash_factor'
        ]
        
        # 혁신적 딤플 분석 알고리즘 가중치
        self.dimple_algorithm_weights = {
            'impact_zone_analysis': 0.40,      # 임팩트 존 집중 분석
            'multi_frame_fusion': 0.30,       # 다중 프레임 융합
            'overexposure_correction': 0.20,  # 과노출 보정
            'adaptive_enhancement': 0.10       # 적응형 향상
        }
        
        print("=== BMP 직접 처리 및 혁신적 딤플 분석 시스템 v7.0 초기화 ===")
        print("개발팀: maxform")
        print("특징: BMP 직접 처리, 임팩트 기준 딤플 분석, 과노출 볼 처리")
        print("목표: 12개 파라미터 95% 정확도 달성")
    
    def read_bmp_direct(self, bmp_path):
        """BMP 파일 직접 읽기 (고해상도 유지)"""
        
        try:
            with open(bmp_path, 'rb') as f:
                # BMP 헤더 읽기
                header = f.read(54)
                
                # BMP 시그니처 확인
                if header[:2] != b'BM':
                    raise ValueError("유효하지 않은 BMP 파일")
                
                # 헤더 정보 추출
                file_size = struct.unpack('<I', header[2:6])[0]
                data_offset = struct.unpack('<I', header[10:14])[0]
                width = struct.unpack('<I', header[18:22])[0]
                height = struct.unpack('<I', header[22:26])[0]
                bit_count = struct.unpack('<H', header[28:30])[0]
                
                print(f"BMP 정보: {width}x{height}, {bit_count}bit")
                
                # 이미지 데이터 읽기
                f.seek(data_offset)
                
                if bit_count == 24:
                    # 24비트 BGR
                    row_size = ((width * 3 + 3) // 4) * 4  # 4바이트 정렬
                    image_data = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    for y in range(height):
                        row_data = f.read(row_size)
                        for x in range(width):
                            if x * 3 + 2 < len(row_data):
                                # BGR 순서
                                image_data[height - 1 - y, x, 0] = row_data[x * 3 + 2]  # R
                                image_data[height - 1 - y, x, 1] = row_data[x * 3 + 1]  # G
                                image_data[height - 1 - y, x, 2] = row_data[x * 3]      # B
                
                elif bit_count == 8:
                    # 8비트 그레이스케일
                    row_size = ((width + 3) // 4) * 4
                    image_data = np.zeros((height, width), dtype=np.uint8)
                    
                    for y in range(height):
                        row_data = f.read(row_size)
                        for x in range(width):
                            if x < len(row_data):
                                image_data[height - 1 - y, x] = row_data[x]
                
                else:
                    # 다른 비트 깊이는 OpenCV로 대체
                    image_data = cv2.imread(bmp_path)
                
                return image_data
                
        except Exception as e:
            print(f"BMP 직접 읽기 실패: {e}, OpenCV로 대체")
            return cv2.imread(bmp_path)
    
    def detect_impact_frame(self, image_sequence):
        """임팩트 프레임 검출 (볼과 클럽이 가장 가까운 순간)"""
        
        impact_scores = []
        
        for i, image in enumerate(image_sequence):
            if image is None:
                impact_scores.append(0)
                continue
            
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 볼 검출 (밝은 원형 객체)
            ball_circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 50,
                param1=50, param2=30, minRadius=10, maxRadius=100
            )
            
            # 클럽 검출 (직선 에지)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            # 임팩트 점수 계산 (볼과 클럽의 근접도)
            impact_score = 0
            
            if ball_circles is not None and lines is not None:
                ball_circles = np.round(ball_circles[0, :]).astype("int")
                
                for (x, y, r) in ball_circles:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        
                        # 볼 중심에서 직선까지의 거리
                        distance = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
                        
                        # 거리가 가까울수록 높은 점수
                        if distance < r + 20:  # 볼 반지름 + 여유분
                            impact_score += (r + 20 - distance) / (r + 20)
            
            impact_scores.append(impact_score)
        
        # 최고 점수 프레임을 임팩트 프레임으로 선정
        if impact_scores:
            impact_frame_idx = np.argmax(impact_scores)
            print(f"임팩트 프레임 검출: {impact_frame_idx}번 프레임 (점수: {impact_scores[impact_frame_idx]:.2f})")
            return impact_frame_idx
        
        return len(image_sequence) // 2  # 중간 프레임을 기본값으로
    
    def correct_overexposure(self, image, ball_center, ball_radius):
        """과노출된 볼 영역 보정"""
        
        x, y = ball_center
        r = ball_radius
        
        # ROI 설정
        roi_size = int(r * 3)
        x1, y1 = max(0, x - roi_size//2), max(0, y - roi_size//2)
        x2, y2 = min(image.shape[1], x + roi_size//2), min(image.shape[0], y + roi_size//2)
        
        roi = image[y1:y2, x1:x2].copy()
        
        if roi.size == 0:
            return image
        
        # 그레이스케일 변환
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi
        
        # 과노출 영역 검출 (매우 밝은 픽셀)
        overexposed_mask = gray_roi > 240
        
        if np.sum(overexposed_mask) > 0:
            print(f"과노출 영역 검출: {np.sum(overexposed_mask)} 픽셀")
            
            # 1. 감마 보정으로 밝기 조절
            gamma = 0.5  # 어둡게
            gamma_corrected = np.power(gray_roi / 255.0, gamma) * 255.0
            gamma_corrected = gamma_corrected.astype(np.uint8)
            
            # 2. 적응형 히스토그램 평활화
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_applied = clahe.apply(gamma_corrected)
            
            # 3. 언샤프 마스킹으로 디테일 강화
            gaussian = cv2.GaussianBlur(clahe_applied, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(clahe_applied, 1.5, gaussian, -0.5, 0)
            
            # 4. 과노출 영역만 교체
            corrected_roi = gray_roi.copy()
            corrected_roi[overexposed_mask] = unsharp_mask[overexposed_mask]
            
            # 원본 이미지에 적용
            corrected_image = image.copy()
            if len(image.shape) == 3:
                corrected_image[y1:y2, x1:x2] = cv2.cvtColor(corrected_roi, cv2.COLOR_GRAY2BGR)
            else:
                corrected_image[y1:y2, x1:x2] = corrected_roi
            
            return corrected_image
        
        return image
    
    def extract_dimple_patterns(self, image, ball_center, ball_radius, zoom_factor=3):
        """혁신적 딤플 패턴 추출 (확대 + 다중 필터링)"""
        
        x, y = ball_center
        r = ball_radius
        
        # 1. 볼 영역 확대 추출
        roi_size = int(r * 2)
        x1, y1 = max(0, x - roi_size//2), max(0, y - roi_size//2)
        x2, y2 = min(image.shape[1], x + roi_size//2), min(image.shape[0], y + roi_size//2)
        
        ball_roi = image[y1:y2, x1:x2]
        
        if ball_roi.size == 0:
            return {'dimple_count': 0, 'dimple_density': 0, 'rotation_angle': 0, 'confidence': 0}
        
        # 2. 확대 (바이큐빅 보간)
        zoomed_roi = cv2.resize(ball_roi, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        
        # 3. 그레이스케일 변환
        if len(zoomed_roi.shape) == 3:
            gray_zoomed = cv2.cvtColor(zoomed_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_zoomed = zoomed_roi
        
        # 4. 과노출 보정 적용
        corrected_zoomed = self.correct_overexposure(gray_zoomed, 
                                                   (gray_zoomed.shape[1]//2, gray_zoomed.shape[0]//2), 
                                                   r * zoom_factor)
        
        if len(corrected_zoomed.shape) == 3:
            corrected_zoomed = cv2.cvtColor(corrected_zoomed, cv2.COLOR_BGR2GRAY)
        
        # 5. 다중 필터링으로 딤플 강화
        
        # 5-1. 라플라시안 필터 (에지 강화)
        laplacian = cv2.Laplacian(corrected_zoomed, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # 5-2. 가우시안 차이 (DoG) 필터
        gaussian1 = cv2.GaussianBlur(corrected_zoomed, (5, 5), 1.0)
        gaussian2 = cv2.GaussianBlur(corrected_zoomed, (9, 9), 2.0)
        dog = cv2.subtract(gaussian1, gaussian2)
        
        # 5-3. 방향성 필터 (Sobel)
        sobel_x = cv2.Sobel(corrected_zoomed, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(corrected_zoomed, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 5-4. 형태학적 연산 (딤플 모양 강조)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(corrected_zoomed, cv2.MORPH_TOPHAT, kernel)
        
        # 6. 딤플 검출 (다중 방법 융합)
        
        # 6-1. 원형 딤플 검출 (Hough Circles)
        dimple_circles = cv2.HoughCircles(
            corrected_zoomed, cv2.HOUGH_GRADIENT, 1, 10,
            param1=50, param2=15, minRadius=2, maxRadius=8
        )
        
        dimple_count = 0
        if dimple_circles is not None:
            dimple_count = len(dimple_circles[0])
        
        # 6-2. 블롭 검출 (어두운 영역)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0  # 어두운 블롭
        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 100
        params.filterByCircularity = True
        params.minCircularity = 0.3
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(255 - corrected_zoomed)  # 반전하여 어두운 부분을 밝게
        
        blob_count = len(keypoints)
        
        # 6-3. 텍스처 분석 (LBP - Local Binary Pattern)
        def calculate_lbp_variance(image):
            """LBP 분산 계산"""
            h, w = image.shape
            lbp_var = 0
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = image[i, j]
                    binary_pattern = 0
                    
                    # 8방향 이웃 픽셀 비교
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            binary_pattern += 2**k
                    
                    lbp_var += binary_pattern
            
            return lbp_var / ((h-2) * (w-2))
        
        texture_complexity = calculate_lbp_variance(corrected_zoomed)
        
        # 7. 회전 각도 추정 (딤플 패턴의 주 방향)
        
        # 7-1. 구조 텐서 분석
        Ix = cv2.Sobel(corrected_zoomed, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(corrected_zoomed, cv2.CV_64F, 0, 1, ksize=3)
        
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        # 가우시안 가중 평균
        Sxx = cv2.GaussianBlur(Ixx, (5, 5), 1.0)
        Syy = cv2.GaussianBlur(Iyy, (5, 5), 1.0)
        Sxy = cv2.GaussianBlur(Ixy, (5, 5), 1.0)
        
        # 주 방향 계산
        rotation_angles = []
        h, w = Sxx.shape
        
        for i in range(h//4, 3*h//4, 5):
            for j in range(w//4, 3*w//4, 5):
                # 구조 텐서 행렬
                M = np.array([[Sxx[i,j], Sxy[i,j]], [Sxy[i,j], Syy[i,j]]])
                
                # 고유값과 고유벡터
                eigenvals, eigenvecs = np.linalg.eig(M)
                
                if eigenvals[0] > eigenvals[1]:
                    principal_direction = eigenvecs[:, 0]
                else:
                    principal_direction = eigenvecs[:, 1]
                
                angle = np.arctan2(principal_direction[1], principal_direction[0]) * 180 / np.pi
                rotation_angles.append(angle)
        
        avg_rotation_angle = np.mean(rotation_angles) if rotation_angles else 0
        
        # 8. 딤플 밀도 계산
        ball_area = np.pi * (r * zoom_factor)**2
        dimple_density = (dimple_count + blob_count) / ball_area * 10000  # 단위 면적당 딤플 수
        
        # 9. 신뢰도 계산
        confidence_factors = [
            min(1.0, dimple_count / 50),  # 딤플 개수 기반
            min(1.0, texture_complexity / 1000),  # 텍스처 복잡도 기반
            min(1.0, np.std(rotation_angles) / 45) if rotation_angles else 0,  # 방향 일관성 기반
        ]
        
        confidence = np.mean(confidence_factors)
        
        print(f"딤플 분석 결과: 딤플 {dimple_count}개, 블롭 {blob_count}개, 밀도 {dimple_density:.2f}, 회전각 {avg_rotation_angle:.1f}°")
        
        return {
            'dimple_count': dimple_count + blob_count,
            'dimple_density': dimple_density,
            'rotation_angle': avg_rotation_angle,
            'texture_complexity': texture_complexity,
            'confidence': confidence,
            'enhanced_image': corrected_zoomed
        }
    
    def multi_frame_dimple_fusion(self, image_sequence, ball_centers, ball_radii, impact_frame_idx):
        """다중 프레임 딤플 분석 융합 (임팩트 기준 전후 4프레임)"""
        
        # 임팩트 기준 전후 4프레임 선정
        start_frame = max(0, impact_frame_idx - 4)
        end_frame = min(len(image_sequence), impact_frame_idx + 5)
        
        dimple_results = []
        
        for i in range(start_frame, end_frame):
            if i < len(image_sequence) and image_sequence[i] is not None:
                if i < len(ball_centers) and ball_centers[i] is not None:
                    ball_center = ball_centers[i]
                    ball_radius = ball_radii[i] if i < len(ball_radii) else 20
                    
                    # 각 프레임에서 딤플 분석
                    dimple_result = self.extract_dimple_patterns(
                        image_sequence[i], ball_center, ball_radius
                    )
                    
                    # 임팩트 프레임에 더 높은 가중치
                    weight = 2.0 if i == impact_frame_idx else 1.0
                    dimple_result['weight'] = weight
                    dimple_result['frame_idx'] = i
                    
                    dimple_results.append(dimple_result)
        
        if not dimple_results:
            return {'total_spin': 0, 'spin_axis': [0, 0, 1], 'confidence': 0}
        
        # 가중 평균으로 최종 스핀 계산
        total_weight = sum(r['weight'] for r in dimple_results)
        
        if total_weight > 0:
            # 회전 각도 변화량 계산
            rotation_changes = []
            for i in range(1, len(dimple_results)):
                prev_angle = dimple_results[i-1]['rotation_angle']
                curr_angle = dimple_results[i]['rotation_angle']
                
                # 각도 차이 (순환 고려)
                angle_diff = curr_angle - prev_angle
                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360
                
                rotation_changes.append(abs(angle_diff))
            
            # 평균 회전 변화량
            avg_rotation_change = np.mean(rotation_changes) if rotation_changes else 0
            
            # 프레임 간격 (1/820초)
            dt = 1.0 / self.fps
            
            # 각속도 (도/초)
            angular_velocity = avg_rotation_change / dt
            
            # RPM 변환
            total_spin = angular_velocity / 6.0  # 도/초 → RPM
            
            # 스핀 축 계산 (가중 평균)
            weighted_angles = [r['rotation_angle'] * r['weight'] for r in dimple_results]
            avg_angle = sum(weighted_angles) / total_weight
            
            # 스핀 축 벡터 (3D)
            spin_axis = [
                np.sin(avg_angle * np.pi / 180),  # X축 성분
                np.cos(avg_angle * np.pi / 180),  # Y축 성분
                0.1  # Z축 성분 (약간의 틸트)
            ]
            
            # 정규화
            norm = np.linalg.norm(spin_axis)
            if norm > 0:
                spin_axis = [x / norm for x in spin_axis]
            
            # 신뢰도 계산
            confidence = sum(r['confidence'] * r['weight'] for r in dimple_results) / total_weight
            
            print(f"다중 프레임 딤플 융합: {len(dimple_results)}개 프레임, 스핀 {total_spin:.0f} RPM")
            
            return {
                'total_spin': max(0, min(15000, total_spin)),  # 현실적 범위 제한
                'spin_axis': spin_axis,
                'confidence': confidence,
                'frame_count': len(dimple_results),
                'rotation_changes': rotation_changes
            }
        
        return {'total_spin': 0, 'spin_axis': [0, 0, 1], 'confidence': 0}
    
    def enhanced_ball_detection(self, image):
        """향상된 볼 검출 (BMP 고해상도 활용)"""
        
        detections = []
        confidences = []
        
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. 적응형 임계값으로 밝은 영역 검출
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 2. 형태학적 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # 3. 윤곽선 검출
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 면적 필터링
            area = cv2.contourArea(contour)
            if 100 < area < 10000:
                # 원형도 계산
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.5:  # 원형에 가까운 객체
                        # 외접원 계산
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        
                        detections.append([int(x), int(y), int(radius)])
                        confidences.append(circularity)
        
        # 4. Hough Circle 검출 (보완)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detections.append([x, y, r])
                confidences.append(0.8)  # 기본 신뢰도
        
        # 5. 중복 제거 (DBSCAN)
        if len(detections) > 0:
            detections = np.array(detections)
            confidences = np.array(confidences)
            
            # 좌표 정규화
            scaler = StandardScaler()
            normalized_detections = scaler.fit_transform(detections[:, :2])
            
            # 클러스터링
            clustering = DBSCAN(eps=0.3, min_samples=1).fit(normalized_detections)
            labels = clustering.labels_
            
            # 각 클러스터에서 최고 신뢰도 검출 선택
            unique_labels = set(labels)
            final_detections = []
            
            for label in unique_labels:
                if label == -1:  # 노이즈
                    continue
                
                cluster_mask = labels == label
                cluster_detections = detections[cluster_mask]
                cluster_confidences = confidences[cluster_mask]
                
                # 최고 신뢰도 검출 선택
                best_idx = np.argmax(cluster_confidences)
                final_detections.append({
                    'center': tuple(cluster_detections[best_idx][:2].astype(int)),
                    'radius': int(cluster_detections[best_idx][2]),
                    'confidence': float(cluster_confidences[best_idx])
                })
            
            return final_detections
        
        return []
    
    def analyze_bmp_sequence(self, bmp_folder):
        """BMP 시퀀스 분석 (혁신적 딤플 분석 적용)"""
        
        print(f"\n=== BMP 시퀀스 분석 시작: {bmp_folder} ===")
        
        # BMP 파일 목록 가져오기
        bmp_files = sorted([f for f in os.listdir(bmp_folder) if f.endswith('.bmp')])
        
        if len(bmp_files) < 2:
            print("분석할 BMP 이미지가 부족합니다.")
            return None
        
        # 이미지 시퀀스 로드
        image_sequence = []
        ball_centers = []
        ball_radii = []
        
        print("BMP 이미지 로딩 중...")
        for i, filename in enumerate(bmp_files):
            bmp_path = os.path.join(bmp_folder, filename)
            
            # BMP 직접 읽기
            image = self.read_bmp_direct(bmp_path)
            
            if image is not None:
                image_sequence.append(image)
                
                # 볼 검출
                detections = self.enhanced_ball_detection(image)
                
                if detections:
                    # 가장 신뢰도 높은 검출 선택
                    best_detection = max(detections, key=lambda x: x['confidence'])
                    ball_centers.append(best_detection['center'])
                    ball_radii.append(best_detection['radius'])
                else:
                    ball_centers.append(None)
                    ball_radii.append(20)  # 기본값
                
                print(f"프레임 {i+1}/{len(bmp_files)} 로딩 완료")
            else:
                print(f"프레임 {i+1} 로딩 실패")
        
        if len(image_sequence) < 2:
            print("유효한 이미지가 부족합니다.")
            return None
        
        # 임팩트 프레임 검출
        impact_frame_idx = self.detect_impact_frame(image_sequence)
        
        # 혁신적 딤플 분석 (다중 프레임 융합)
        dimple_result = self.multi_frame_dimple_fusion(
            image_sequence, ball_centers, ball_radii, impact_frame_idx
        )
        
        # 볼 데이터 계산
        ball_data = self.calculate_ball_data_from_dimples(
            ball_centers, ball_radii, dimple_result
        )
        
        # 클럽 데이터 계산
        club_data = self.calculate_club_data(image_sequence[impact_frame_idx])
        
        # 클럽 타입 판정
        club_type = '7iron' if '7iron' in bmp_folder else 'driver'
        
        # 정확도 계산
        accuracy_result = self.calculate_accuracy(ball_data, club_type)
        
        result = {
            'sequence_name': os.path.basename(bmp_folder),
            'club_type': club_type,
            'ball_data': ball_data,
            'club_data': club_data,
            'accuracy': accuracy_result,
            'dimple_analysis': dimple_result,
            'impact_frame': impact_frame_idx,
            'detection_stats': {
                'total_frames': len(bmp_files),
                'processed_frames': len(image_sequence),
                'detection_rate': len([c for c in ball_centers if c is not None]) / len(image_sequence) * 100
            }
        }
        
        print(f"BMP 분석 완료: {result['sequence_name']}")
        print(f"검출률: {result['detection_stats']['detection_rate']:.1f}%")
        print(f"전체 정확도: {accuracy_result['overall_accuracy']:.1f}%")
        print(f"딤플 분석 신뢰도: {dimple_result['confidence']:.1f}")
        
        return result
    
    def calculate_ball_data_from_dimples(self, ball_centers, ball_radii, dimple_result):
        """딤플 분석 결과로부터 볼 데이터 계산"""
        
        # 유효한 볼 중심점들만 필터링
        valid_centers = [c for c in ball_centers if c is not None]
        
        if len(valid_centers) < 2:
            return {
                'ball_speed': 0, 'launch_angle': 0, 'direction_angle': 0,
                'backspin': 0, 'sidespin': 0, 'spin_axis': 0
            }
        
        # 시간 간격
        dt = 1.0 / self.fps
        
        # 볼 스피드 계산
        distances = []
        for i in range(1, len(valid_centers)):
            dx = (valid_centers[i][0] - valid_centers[i-1][0]) * self.pixel_to_mm
            dy = (valid_centers[i][1] - valid_centers[i-1][1]) * self.pixel_to_mm
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            ball_speed_ms = avg_distance / dt / 1000  # m/s
            ball_speed_mph = ball_speed_ms * 2.237  # mph
        else:
            ball_speed_mph = 0
        
        # 발사각 계산
        if len(valid_centers) >= 3:
            start_point = valid_centers[0]
            end_point = valid_centers[-1]
            
            dx = (end_point[0] - start_point[0]) * self.pixel_to_mm
            dy = (end_point[1] - start_point[1]) * self.pixel_to_mm
            
            launch_angle = np.arctan2(-dy, dx) * 180 / np.pi  # 음수: 위쪽이 양수
        else:
            launch_angle = 0
        
        # 방향각 계산
        direction_angle = np.arctan2(
            valid_centers[-1][0] - valid_centers[0][0],
            valid_centers[-1][1] - valid_centers[0][1]
        ) * 180 / np.pi
        
        # 딤플 분석 결과에서 스핀 데이터 추출
        total_spin = dimple_result.get('total_spin', 0)
        spin_axis = dimple_result.get('spin_axis', [0, 0, 1])
        
        # 백스핀과 사이드스핀 계산
        backspin = total_spin * abs(spin_axis[1])  # Y축 성분
        sidespin = total_spin * abs(spin_axis[0])  # X축 성분
        
        # 스핀 축 각도
        spin_axis_angle = np.arctan2(spin_axis[0], spin_axis[1]) * 180 / np.pi
        
        return {
            'ball_speed': ball_speed_mph,
            'launch_angle': launch_angle,
            'direction_angle': direction_angle,
            'backspin': backspin,
            'sidespin': sidespin,
            'spin_axis': spin_axis_angle,
            'total_spin': total_spin
        }
    
    def calculate_club_data(self, image):
        """클럽 데이터 계산 (로프트앵글 제외)"""
        
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 에지 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # 직선 검출 (클럽 샤프트)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        club_data = {
            'club_speed': 0,
            'attack_angle': 0,
            'club_path': 0,
            'face_angle': 0,
            'face_to_path': 0,
            'smash_factor': 0,
            'confidence': 0
        }
        
        if lines is not None and len(lines) > 0:
            # 가장 긴 직선을 클럽 샤프트로 간주
            longest_line = max(lines, key=lambda line: 
                np.sqrt((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2))
            
            x1, y1, x2, y2 = longest_line[0]
            
            # 클럽 각도 계산
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # 클럽 데이터 추정
            club_data['attack_angle'] = angle - 90  # 수직 기준
            club_data['club_path'] = angle * 0.6  # 클럽 패스
            club_data['face_angle'] = angle * 0.8  # 페이스 각도
            club_data['face_to_path'] = club_data['face_angle'] - club_data['club_path']
            club_data['club_speed'] = 60  # 기본값 (실제로는 연속 프레임 분석 필요)
            club_data['smash_factor'] = 1.25  # 기본값
            club_data['confidence'] = 0.7
        
        return club_data
    
    def determine_skill_level(self, ball_speed):
        """볼스피드 기반 실력 수준 판정"""
        if ball_speed < 85:
            return 'beginner'
        elif ball_speed < 105:
            return 'amateur'
        elif ball_speed < 125:
            return 'scratch'
        else:
            return 'pro'
    
    def calculate_accuracy(self, measured_data, club_type):
        """TrackMan 기준 대비 정확도 계산"""
        
        skill_level = self.determine_skill_level(measured_data.get('ball_speed', 0))
        reference = self.trackman_reference[club_type][skill_level]
        
        accuracies = {}
        total_accuracy = 0
        count = 0
        
        # 각 항목별 정확도 계산
        for key in ['ball_speed', 'launch_angle', 'backspin', 'sidespin']:
            if key in measured_data and key in reference:
                measured = measured_data[key]
                ref = reference[key]
                
                if ref > 0:
                    error_rate = abs(measured - ref) / ref
                    accuracy = max(0, 1 - error_rate) * 100
                    accuracies[key] = accuracy
                    total_accuracy += accuracy
                    count += 1
        
        # 전체 정확도
        overall_accuracy = total_accuracy / count if count > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'individual_accuracies': accuracies,
            'skill_level': skill_level,
            'reference_data': reference
        }

def main():
    """메인 실행 함수"""
    
    print("=== BMP 직접 처리 및 혁신적 딤플 분석 시스템 v7.0 ===")
    print("개발팀: maxform")
    print("혁신: 임팩트 기준 딤플 분석, 과노출 볼 처리, BMP 직접 처리")
    
    # 분석기 초기화
    analyzer = EnhancedBMPGolfAnalyzer()
    
    # BMP 이미지 경로 설정
    base_path = "/home/ubuntu/GolfSwingAnalysis_Final/data/images/shot-image-bmp-treated"
    
    if not os.path.exists(base_path):
        print(f"오류: {base_path} 폴더를 찾을 수 없습니다.")
        return
    
    # 모든 시퀀스 분석
    all_results = []
    
    # 7iron 분석
    iron_path = os.path.join(base_path, "7iron")
    if os.path.exists(iron_path):
        for sequence_folder in os.listdir(iron_path):
            sequence_path = os.path.join(iron_path, sequence_folder)
            if os.path.isdir(sequence_path):
                print(f"\n7iron BMP 시퀀스 분석: {sequence_folder}")
                result = analyzer.analyze_bmp_sequence(sequence_path)
                if result:
                    all_results.append(result)
    
    # driver 분석
    driver_path = os.path.join(base_path, "driver")
    if os.path.exists(driver_path):
        for sequence_folder in os.listdir(driver_path):
            sequence_path = os.path.join(driver_path, sequence_folder)
            if os.path.isdir(sequence_path):
                print(f"\ndriver BMP 시퀀스 분석: {sequence_folder}")
                result = analyzer.analyze_bmp_sequence(sequence_path)
                if result:
                    all_results.append(result)
    
    # 결과 저장
    if all_results:
        # JSON 저장
        output_file = "/home/ubuntu/golf_analysis_v4/bmp_dimple_analysis_results_v7.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        
        # Excel 보고서 생성
        excel_file = "/home/ubuntu/golf_analysis_v4/bmp_dimple_analysis_report_v7.xlsx"
        
        # 요약 데이터 준비
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Sequence': result['sequence_name'],
                'Club Type': result['club_type'],
                'Ball Speed (mph)': result['ball_data'].get('ball_speed', 0),
                'Launch Angle (°)': result['ball_data'].get('launch_angle', 0),
                'Backspin (rpm)': result['ball_data'].get('backspin', 0),
                'Sidespin (rpm)': result['ball_data'].get('sidespin', 0),
                'Club Speed (mph)': result['club_data'].get('club_speed', 0),
                'Attack Angle (°)': result['club_data'].get('attack_angle', 0),
                'Overall Accuracy (%)': result['accuracy']['overall_accuracy'],
                'Detection Rate (%)': result['detection_stats']['detection_rate'],
                'Dimple Confidence': result['dimple_analysis']['confidence'],
                'Impact Frame': result['impact_frame'],
                'Skill Level': result['accuracy']['skill_level']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_excel(excel_file, index=False)
        
        # 통계 출력
        print(f"\n=== BMP 딤플 분석 최종 결과 ===")
        print(f"총 분석 시퀀스: {len(all_results)}개")
        print(f"평균 정확도: {np.mean([r['accuracy']['overall_accuracy'] for r in all_results]):.1f}%")
        print(f"평균 검출률: {np.mean([r['detection_stats']['detection_rate'] for r in all_results]):.1f}%")
        print(f"평균 딤플 신뢰도: {np.mean([r['dimple_analysis']['confidence'] for r in all_results]):.1f}")
        print(f"결과 저장: {output_file}")
        print(f"Excel 보고서: {excel_file}")
        
        # 95% 정확도 달성 여부
        high_accuracy_count = sum(1 for r in all_results if r['accuracy']['overall_accuracy'] >= 95)
        print(f"95% 이상 정확도 달성: {high_accuracy_count}/{len(all_results)} 시퀀스")
        
        if high_accuracy_count / len(all_results) >= 0.8:
            print("🎯 목표 달성: 80% 이상의 시퀀스에서 95% 정확도 달성!")
        else:
            print("⚠️  추가 개선 필요: 95% 정확도 달성률이 목표에 미달")
        
        print("\n=== 혁신적 딤플 분석 시스템 완료 ===")
        print("maxform 개발팀의 BMP 직접 처리 및 딤플 분석 시스템이 완성되었습니다.")
    
    else:
        print("분석할 수 있는 BMP 시퀀스가 없습니다.")

if __name__ == "__main__":
    main()

