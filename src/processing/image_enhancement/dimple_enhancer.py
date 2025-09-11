#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimple Enhancement for Golf Ball Analysis
골프공 딤플 검출을 위한 이미지 전처리 시스템
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

class DimpleEnhancer:
    """딤플 검출을 위한 이미지 전처리 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 딤플 검출을 위한 고급 파라미터
        self.dimple_params = {
            'min_radius': 2,      # 최소 딤플 반지름 (픽셀)
            'max_radius': 8,      # 최대 딤플 반지름 (픽셀)
            'min_distance': 10,   # 딤플 간 최소 거리
            'threshold': 0.3,     # 딤플 검출 임계값
            'quality': 0.7,       # 딤플 품질 임계값
            'brightness_threshold': 150,  # 밝기 임계값
            'contrast_alpha': 1.5,  # 대비 증가 계수
            'gamma_correction': 0.6  # 감마 보정 값
        }
        
        # BMP 전용 딤플 검출 파라미터
        self.bmp_dimple_params = {
            'min_radius': 1,        # 더 작은 딤플도 검출
            'max_radius': 12,       # 더 큰 딤플도 검출
            'min_distance': 8,      # 딤플 간 거리 조정
            'threshold': 0.2,       # 더 낮은 임계값
            'quality': 0.5,         # 더 낮은 품질 기준
            'brightness_threshold': 150,  # 밝기 임계값
            'contrast_alpha': 1.5,  # 대비 증가 계수
            'gamma_correction': 0.6  # 감마 보정 값
        }
        
        # 고주파 강조 필터 (딤플 검출용)
        self.dimple_kernel = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  2,  2,  2, -1],
            [-1,  2,  8,  2, -1],
            [-1,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1]
        ])
    
    def enhance_dimple_visibility(self, image: np.ndarray) -> np.ndarray:
        """
        딤플 가시성 향상 (BMP 선명도 유지)
        
        Args:
            image: 입력 이미지
            
        Returns:
            딤플이 강조된 이미지
        """
        # 1. 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 2. 노이즈 제거 (가벼운 가우시안)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 3. 딤플 강조 필터 적용
        enhanced = cv2.filter2D(denoised, -1, self.dimple_kernel)
        
        # 4. 히스토그램 균등화 (딤플 대비 향상)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
        
        # 5. 언샤프 마스킹 (선명도 향상)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return enhanced
    
    def enhance_dimple_visibility_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        BMP 딤플 가시성 고급 향상 (기존 JPG 처리 방식 적용)
        
        Args:
            image: 입력 이미지
            
        Returns:
            딤플이 강조된 이미지
        """
        # 1. 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 2. 히스토그램 분석으로 밝기 조정
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # 3. 밝은 이미지 감지 및 감마 보정
        mean_brightness = np.mean(gray)
        if mean_brightness > 150:  # 너무 밝은 이미지
            # 감마 보정 (어둡게)
            gamma = 0.7
            gray = np.power(gray / 255.0, gamma) * 255.0
            gray = gray.astype(np.uint8)
        
        # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 5. 언샤프 마스킹 (선명도 향상)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # 6. 딤플 특화 필터 적용
        enhanced = cv2.filter2D(enhanced, -1, self.dimple_kernel)
        
        # 7. 이진화를 통한 딤플 강조
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 8. 모폴로지 연산으로 딤플 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def apply_jpg_processing_methods(self, image: np.ndarray) -> np.ndarray:
        """
        기존 JPG 처리 방식을 BMP에 적용
        
        Args:
            image: 입력 이미지
            
        Returns:
            JPG 처리 방식이 적용된 이미지
        """
        # 1. 감마 보정 (기존 JPG 처리의 핵심)
        gamma = 1.7  # JPG 처리에서 사용된 감마 값
        gamma_corrected = np.power(image / 255.0, 1/gamma) * 255.0
        gamma_corrected = gamma_corrected.astype(np.uint8)
        
        # 2. 대비 향상
        alpha = 1.2  # 대비 증가
        beta = 30    # 밝기 조정
        enhanced = cv2.convertScaleAbs(gamma_corrected, alpha=alpha, beta=beta)
        
        # 3. 노이즈 제거
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 4. 딤플 검출용 엣지 강화
        sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.uint8(sobel / sobel.max() * 255)
        
        # 5. 원본과 엣지 강화 이미지 결합
        final = cv2.addWeighted(enhanced, 0.7, sobel, 0.3, 0)
        
        return final
    
    def detect_dimple_patterns(self, image: np.ndarray, ball_center: Tuple[int, int], 
                             ball_radius: int) -> list:
        """
        딤플 패턴 검출
        
        Args:
            image: 입력 이미지
            ball_center: 볼 중심 좌표
            ball_radius: 볼 반지름
            
        Returns:
            딤플 정보 리스트
        """
        # 볼 영역 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, ball_center, ball_radius, 255, -1)
        
        # 딤플 가시성 향상
        enhanced = self.enhance_dimple_visibility(image)
        
        # 딤플 검출 (다중 방법)
        dimples = []
        
        # 1. HoughCircles로 딤플 검출
        circles = cv2.HoughCircles(
            enhanced, cv2.HOUGH_GRADIENT,
            dp=1, minDist=self.dimple_params['min_distance'],
            param1=50, param2=self.dimple_params['threshold'] * 100,
            minRadius=self.dimple_params['min_radius'],
            maxRadius=self.dimple_params['max_radius']
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # 볼 영역 내부인지 확인
                if cv2.pointPolygonTest(
                    np.array([[ball_center[0], ball_center[1] - ball_radius],
                             [ball_center[0] + ball_radius, ball_center[1]],
                             [ball_center[0], ball_center[1] + ball_radius],
                             [ball_center[0] - ball_radius, ball_center[1]]]), 
                    (x, y), False) >= 0:
                    
                    # 딤플 품질 평가
                    quality = self.evaluate_dimple_quality(enhanced, (x, y), r)
                    if quality > self.dimple_params['quality']:
                        dimples.append({
                            'center': (x, y),
                            'radius': r,
                            'quality': quality,
                            'method': 'hough'
                        })
        
        # 2. 특징점 기반 딤플 검출
        keypoints = self.detect_dimple_keypoints(enhanced, ball_center, ball_radius)
        for kp in keypoints:
            dimples.append({
                'center': (int(kp.pt[0]), int(kp.pt[1])),
                'radius': int(kp.size / 2),
                'quality': kp.response,
                'method': 'keypoint'
            })
        
        return dimples
    
    def detect_dimple_keypoints(self, image: np.ndarray, ball_center: Tuple[int, int], 
                              ball_radius: int) -> list:
        """
        특징점 기반 딤플 검출
        
        Args:
            image: 입력 이미지
            ball_center: 볼 중심 좌표
            ball_radius: 볼 반지름
            
        Returns:
            딤플 키포인트 리스트
        """
        # 볼 영역 마스크
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, ball_center, ball_radius, 255, -1)
        
        # SIFT 특징점 검출기 (딤플에 특화)
        sift = cv2.SIFT_create(
            nfeatures=100,
            contrastThreshold=0.03,  # 낮은 임계값으로 미세한 딤플도 검출
            edgeThreshold=10
        )
        
        keypoints = sift.detect(image, mask)
        
        # 딤플 특성에 맞는 키포인트만 필터링
        dimple_keypoints = []
        for kp in keypoints:
            # 크기가 딤플 범위 내인지 확인
            if (self.dimple_params['min_radius'] <= kp.size/2 <= 
                self.dimple_params['max_radius']):
                dimple_keypoints.append(kp)
        
        return dimple_keypoints
    
    def evaluate_dimple_quality(self, image: np.ndarray, center: Tuple[int, int], 
                               radius: int) -> float:
        """
        딤플 품질 평가
        
        Args:
            image: 입력 이미지
            center: 딤플 중심 좌표
            radius: 딤플 반지름
            
        Returns:
            딤플 품질 점수 (0-1)
        """
        x, y = center
        r = radius
        
        # 딤플 영역 추출
        roi = image[max(0, y-r):min(image.shape[0], y+r), 
                   max(0, x-r):min(image.shape[1], x+r)]
        
        if roi.size == 0:
            return 0.0
        
        # 1. 대비 평가
        contrast = np.std(roi) / 255.0
        
        # 2. 원형도 평가
        mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        masked_roi = cv2.bitwise_and(roi, mask)
        
        # 3. 중심에서 가장자리까지의 그라데이션
        center_intensity = masked_roi[r, r] if r < masked_roi.shape[0] and r < masked_roi.shape[1] else 0
        edge_intensity = np.mean(masked_roi[mask > 0])
        
        gradient = abs(center_intensity - edge_intensity) / 255.0
        
        # 종합 품질 점수
        quality = (contrast * 0.4 + gradient * 0.6)
        
        return min(1.0, quality)
    
    def update_dimple_params(self, **kwargs):
        """
        딤플 검출 파라미터 업데이트
        
        Args:
            **kwargs: 업데이트할 파라미터들
        """
        for key, value in kwargs.items():
            if key in self.dimple_params:
                self.dimple_params[key] = value
                self.logger.info(f"딤플 파라미터 업데이트: {key} = {value}")
            else:
                self.logger.warning(f"알 수 없는 파라미터: {key}")
    
    def detect_dimples_improved(self, image: np.ndarray, ball_center: Tuple[int, int], 
                               ball_radius: int) -> List[Dict]:
        """
        개선된 딤플 검출 (BMP 전용)
        
        Args:
            image: 입력 이미지
            ball_center: 볼 중심 좌표
            ball_radius: 볼 반지름
            
        Returns:
            딤플 정보 리스트
        """
        # 볼 영역 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, ball_center, ball_radius, 255, -1)
        
        # 1. HoughCircles로 원형 딤플 검출
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT,
            dp=1, minDist=self.bmp_dimple_params['min_distance'],
            param1=50, param2=self.bmp_dimple_params['threshold'] * 100,
            minRadius=self.bmp_dimple_params['min_radius'],
            maxRadius=self.bmp_dimple_params['max_radius']
        )
        
        dimples = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # 볼 영역 내부인지 확인
                if cv2.pointPolygonTest(
                    np.array([[ball_center[0], ball_center[1] - ball_radius],
                             [ball_center[0] + ball_radius, ball_center[1]],
                             [ball_center[0], ball_center[1] + ball_radius],
                             [ball_center[0] - ball_radius, ball_center[1]]]), 
                    (x, y), False) >= 0:
                    
                    # 딤플 품질 평가
                    quality = self.evaluate_dimple_quality_improved(image, (x, y), r)
                    if quality > self.bmp_dimple_params['quality']:
                        dimples.append({
                            'center': (x, y),
                            'radius': r,
                            'quality': quality,
                            'method': 'hough_improved'
                        })
        
        # 2. 특징점 기반 딤플 검출 (SIFT)
        keypoints = self.detect_dimple_keypoints_improved(image, ball_center, ball_radius)
        for kp in keypoints:
            dimples.append({
                'center': (int(kp.pt[0]), int(kp.pt[1])),
                'radius': int(kp.size / 2),
                'quality': kp.response,
                'method': 'sift_improved'
            })
        
        return dimples
    
    def detect_dimple_keypoints_improved(self, image: np.ndarray, ball_center: Tuple[int, int], 
                                       ball_radius: int) -> list:
        """
        개선된 특징점 기반 딤플 검출
        
        Args:
            image: 입력 이미지
            ball_center: 볼 중심 좌표
            ball_radius: 볼 반지름
            
        Returns:
            딤플 키포인트 리스트
        """
        # 볼 영역 마스크
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, ball_center, ball_radius, 255, -1)
        
        # SIFT 특징점 검출기 (딤플에 특화)
        sift = cv2.SIFT_create(
            nfeatures=200,  # 더 많은 특징점 검출
            contrastThreshold=0.02,  # 더 낮은 임계값
            edgeThreshold=15  # 더 높은 엣지 임계값
        )
        
        keypoints = sift.detect(image, mask)
        
        # 딤플 특성에 맞는 키포인트만 필터링
        dimple_keypoints = []
        for kp in keypoints:
            # 크기가 딤플 범위 내인지 확인
            if (self.bmp_dimple_params['min_radius'] <= kp.size/2 <= 
                self.bmp_dimple_params['max_radius']):
                dimple_keypoints.append(kp)
        
        return dimple_keypoints
    
    def evaluate_dimple_quality_improved(self, image: np.ndarray, center: Tuple[int, int], 
                                       radius: int) -> float:
        """
        개선된 딤플 품질 평가 (BMP 전용)
        
        Args:
            image: 입력 이미지
            center: 딤플 중심 좌표
            radius: 딤플 반지름
            
        Returns:
            딤플 품질 점수 (0-1)
        """
        x, y = center
        r = radius
        
        # 딤플 영역 추출
        roi = image[max(0, y-r):min(image.shape[0], y+r), 
                   max(0, x-r):min(image.shape[1], x+r)]
        
        if roi.size == 0:
            return 0.0
        
        # 1. 대비 평가 (개선된 방식)
        contrast = np.std(roi) / 255.0
        
        # 2. 원형도 평가
        mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        masked_roi = cv2.bitwise_and(roi, mask)
        
        # 3. 중심에서 가장자리까지의 그라데이션 (개선된 방식)
        center_intensity = masked_roi[r, r] if r < masked_roi.shape[0] and r < masked_roi.shape[1] else 0
        edge_intensity = np.mean(masked_roi[mask > 0])
        
        gradient = abs(center_intensity - edge_intensity) / 255.0
        
        # 4. 텍스처 분석 (새로 추가)
        texture_score = self.analyze_texture_pattern(masked_roi)
        
        # 5. 종합 품질 점수 (개선된 가중치)
        quality = (contrast * 0.3 + gradient * 0.4 + texture_score * 0.3)
        
        return min(1.0, quality)
    
    def analyze_texture_pattern(self, roi: np.ndarray) -> float:
        """
        텍스처 패턴 분석 (딤플 특성 감지)
        
        Args:
            roi: 딤플 영역
            
        Returns:
            텍스처 점수 (0-1)
        """
        if roi.size == 0:
            return 0.0
        
        # 1. 라플라시안 필터로 텍스처 강조
        laplacian = cv2.Laplacian(roi, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # 2. 정규화
        texture_score = min(1.0, texture_variance / 1000.0)
        
        return texture_score
