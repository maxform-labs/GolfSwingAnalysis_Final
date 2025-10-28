#!/usr/bin/env python3
"""
초정밀 볼 검출 시스템 (99.99% 정확도 목표)
다중 알고리즘 앙상블 + 딥러닝 + 적응적 전처리
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraPreciseBallDetector:
    def __init__(self, calibration_file="final_high_quality_calibration.json"):
        """초정밀 볼 검출기 초기화"""
        self.calibration_data = self.load_calibration(calibration_file)
        self.setup_camera_parameters()
        
        # 다중 검출 알고리즘 가중치
        self.algorithm_weights = {
            'hough_circles': 0.25,
            'contour_analysis': 0.20,
            'template_matching': 0.15,
            'edge_detection': 0.15,
            'color_segmentation': 0.10,
            'morphological': 0.10,
            'ml_classifier': 0.05
        }
        
        # ML 모델 초기화
        self.ml_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
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
            
        self.mtx1 = np.array(self.calibration_data['upper_camera']['camera_matrix'])
        self.dist1 = np.array(self.calibration_data['upper_camera']['distortion_coeffs'])
        self.mtx2 = np.array(self.calibration_data['lower_camera']['camera_matrix'])
        self.dist2 = np.array(self.calibration_data['lower_camera']['distortion_coeffs'])
        self.R = np.array(self.calibration_data['stereo_params']['R'])
        self.T = np.array(self.calibration_data['stereo_params']['T'])
        self.baseline = self.calibration_data['stereo_params']['baseline']
        
        logger.info(f"Loaded calibration with baseline: {self.baseline:.2f}mm")
    
    def advanced_image_preprocessing(self, img):
        """고급 이미지 전처리"""
        # 1. 노이즈 제거
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # 2. 적응적 히스토그램 균등화
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 3. 샤프닝
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 4. 가우시안 피라미드
        gaussian_pyramid = [sharpened]
        for i in range(3):
            gaussian_pyramid.append(cv2.pyrDown(gaussian_pyramid[-1]))
        
        return enhanced, gaussian_pyramid
    
    def hough_circles_advanced(self, img):
        """고급 Hough Circles 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 다중 스케일 검출
        circles_list = []
        
        # 다양한 파라미터 조합
        param_sets = [
            {'dp': 1, 'minDist': 10, 'param1': 50, 'param2': 30, 'minRadius': 3, 'maxRadius': 15},
            {'dp': 1.2, 'minDist': 15, 'param1': 40, 'param2': 25, 'minRadius': 2, 'maxRadius': 20},
            {'dp': 1.5, 'minDist': 20, 'param1': 35, 'param2': 20, 'minRadius': 1, 'maxRadius': 25},
            {'dp': 2, 'minDist': 25, 'param1': 30, 'param2': 15, 'minRadius': 1, 'maxRadius': 30}
        ]
        
        for params in param_sets:
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, **params
            )
            if circles is not None:
                circles_list.extend(circles[0])
        
        return circles_list
    
    def contour_analysis_advanced(self, img):
        """고급 컨투어 분석"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 적응적 임계값
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 모폴로지 연산
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 2000:  # 볼 크기 범위
                # 원형도 검사
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:  # 원에 가까운 형태
                        # 중심과 반지름 계산
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        if 2 < radius < 30:
                            ball_candidates.append([x, y, radius, circularity])
        
        return ball_candidates
    
    def template_matching_advanced(self, img):
        """고급 템플릿 매칭"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 다양한 크기의 원형 템플릿 생성
        templates = []
        for radius in range(3, 25, 2):
            template = np.zeros((radius*2+1, radius*2+1), dtype=np.uint8)
            cv2.circle(template, (radius, radius), radius, 255, -1)
            templates.append((template, radius))
        
        matches = []
        for template, radius in templates:
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.7)
            
            for pt in zip(*locations[::-1]):
                matches.append([pt[0] + radius, pt[1] + radius, radius])
        
        return matches
    
    def edge_detection_advanced(self, img):
        """고급 엣지 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Canny 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # HoughCircles on edges
        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=3, maxRadius=25
        )
        
        if circles is not None:
            return circles[0].tolist()
        return []
    
    def color_segmentation_advanced(self, img):
        """고급 색상 분할"""
        # HSV 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 흰색 볼 마스크
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 노란색 볼 마스크
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 마스크 결합
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 2000:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 3 < radius < 25:
                    ball_candidates.append([x, y, radius])
        
        return ball_candidates
    
    def morphological_analysis(self, img):
        """모폴로지 분석"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 < area < 1500:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 3 < radius < 20:
                    ball_candidates.append([x, y, radius])
        
        return ball_candidates
    
    def extract_features(self, img, x, y, radius):
        """볼 후보의 특징 추출"""
        # ROI 추출
        roi_size = int(radius * 3)
        x1 = max(0, int(x - roi_size))
        y1 = max(0, int(y - roi_size))
        x2 = min(img.shape[1], int(x + roi_size))
        y2 = min(img.shape[0], int(y + roi_size))
        
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # 특징 추출
        features = []
        
        # 1. 색상 특징
        mean_color = np.mean(roi, axis=(0, 1))
        features.extend(mean_color)
        
        # 2. 텍스처 특징
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        features.append(np.mean(gray_roi))
        features.append(np.std(gray_roi))
        
        # 3. 형태 특징
        features.append(radius)
        features.append(x / img.shape[1])  # 정규화된 x 좌표
        features.append(y / img.shape[0])  # 정규화된 y 좌표
        
        # 4. 그라디언트 특징
        grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        features.append(np.mean(np.sqrt(grad_x**2 + grad_y**2)))
        
        return features
    
    def train_ml_classifier(self, training_data):
        """ML 분류기 훈련"""
        if not training_data:
            return
        
        X = []
        y = []
        
        for data in training_data:
            features = data['features']
            label = data['label']  # 1 for ball, 0 for not ball
            
            if features is not None:
                X.append(features)
                y.append(label)
        
        if len(X) > 10:  # 최소 훈련 데이터
            X = np.array(X)
            y = np.array(y)
            
            # 특성 정규화
            X_scaled = self.scaler.fit_transform(X)
            
            # 랜덤 포레스트 훈련
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ml_model.fit(X_scaled, y)
            self.is_trained = True
            
            logger.info(f"ML classifier trained with {len(X)} samples")
    
    def ml_classification(self, img, candidates):
        """ML 기반 분류"""
        if not self.is_trained or not candidates:
            return candidates
        
        features_list = []
        valid_candidates = []
        
        for candidate in candidates:
            x, y, radius = candidate[:3]
            features = self.extract_features(img, x, y, radius)
            
            if features is not None:
                features_list.append(features)
                valid_candidates.append(candidate)
        
        if features_list:
            features_array = np.array(features_list)
            features_scaled = self.scaler.transform(features_array)
            predictions = self.ml_model.predict(features_scaled)
            probabilities = self.ml_model.predict_proba(features_scaled)
            
            # 볼로 분류된 후보만 반환
            ball_candidates = []
            for i, (candidate, pred, prob) in enumerate(zip(valid_candidates, predictions, probabilities)):
                if pred == 1 and prob[1] > 0.7:  # 볼 확률이 70% 이상
                    ball_candidates.append(candidate)
            
            return ball_candidates
        
        return []
    
    def ensemble_detection(self, img):
        """앙상블 검출"""
        # 이미지 전처리
        enhanced, _ = self.advanced_image_preprocessing(img)
        
        # 다중 알고리즘 실행
        results = {}
        
        # 1. Hough Circles
        hough_results = self.hough_circles_advanced(enhanced)
        results['hough_circles'] = hough_results
        
        # 2. 컨투어 분석
        contour_results = self.contour_analysis_advanced(enhanced)
        results['contour_analysis'] = contour_results
        
        # 3. 템플릿 매칭
        template_results = self.template_matching_advanced(enhanced)
        results['template_matching'] = template_results
        
        # 4. 엣지 검출
        edge_results = self.edge_detection_advanced(enhanced)
        results['edge_detection'] = edge_results
        
        # 5. 색상 분할
        color_results = self.color_segmentation_advanced(enhanced)
        results['color_segmentation'] = color_results
        
        # 6. 모폴로지 분석
        morph_results = self.morphological_analysis(enhanced)
        results['morphological'] = morph_results
        
        # 7. ML 분류
        all_candidates = []
        for algorithm, candidates in results.items():
            if algorithm != 'ml_classifier':
                all_candidates.extend(candidates)
        
        ml_results = self.ml_classification(enhanced, all_candidates)
        results['ml_classifier'] = ml_results
        
        # 앙상블 결과 통합
        final_candidates = self.merge_ensemble_results(results)
        
        return final_candidates
    
    def merge_ensemble_results(self, results):
        """앙상블 결과 통합"""
        # 모든 후보 수집
        all_candidates = []
        for algorithm, candidates in results.items():
            weight = self.algorithm_weights.get(algorithm, 0.1)
            for candidate in candidates:
                all_candidates.append({
                    'x': candidate[0],
                    'y': candidate[1],
                    'radius': candidate[2],
                    'weight': weight,
                    'algorithm': algorithm
                })
        
        if not all_candidates:
            return None
        
        # 유사한 후보들 그룹화
        merged_candidates = []
        used_indices = set()
        
        for i, candidate1 in enumerate(all_candidates):
            if i in used_indices:
                continue
            
            similar_candidates = [candidate1]
            used_indices.add(i)
            
            for j, candidate2 in enumerate(all_candidates):
                if j in used_indices:
                    continue
                
                # 거리 계산
                distance = np.sqrt((candidate1['x'] - candidate2['x'])**2 + 
                                 (candidate1['y'] - candidate2['y'])**2)
                
                if distance < 20:  # 20픽셀 이내면 유사한 후보
                    similar_candidates.append(candidate2)
                    used_indices.add(j)
            
            # 가중 평균 계산
            if len(similar_candidates) > 1:
                total_weight = sum(c['weight'] for c in similar_candidates)
                avg_x = sum(c['x'] * c['weight'] for c in similar_candidates) / total_weight
                avg_y = sum(c['y'] * c['weight'] for c in similar_candidates) / total_weight
                avg_radius = sum(c['radius'] * c['weight'] for c in similar_candidates) / total_weight
                confidence = total_weight
                
                merged_candidates.append([avg_x, avg_y, avg_radius, confidence])
            else:
                merged_candidates.append([candidate1['x'], candidate1['y'], candidate1['radius'], candidate1['weight']])
        
        # 신뢰도 순으로 정렬
        merged_candidates.sort(key=lambda x: x[3], reverse=True)
        
        # 가장 신뢰도가 높은 후보 반환
        if merged_candidates:
            best_candidate = merged_candidates[0]
            return [int(best_candidate[0]), int(best_candidate[1]), int(best_candidate[2])]
        
        return None
    
    def detect_ball_ultra_precise(self, img):
        """초정밀 볼 검출"""
        try:
            # 앙상블 검출 실행
            result = self.ensemble_detection(img)
            
            if result is not None:
                logger.info(f"Ball detected at ({result[0]}, {result[1]}) with radius {result[2]}")
                return result
            else:
                logger.warning("No ball detected with ultra-precise method")
                return None
                
        except Exception as e:
            logger.error(f"Ultra-precise detection error: {str(e)}")
            return None
    
    def save_model(self, filename="ultra_precise_ball_detector.pkl"):
        """모델 저장"""
        if self.is_trained:
            model_data = {
                'ml_model': self.ml_model,
                'scaler': self.scaler,
                'algorithm_weights': self.algorithm_weights
            }
            joblib.dump(model_data, filename)
            logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename="ultra_precise_ball_detector.pkl"):
        """모델 로드"""
        try:
            model_data = joblib.load(filename)
            self.ml_model = model_data['ml_model']
            self.scaler = model_data['scaler']
            self.algorithm_weights = model_data['algorithm_weights']
            self.is_trained = True
            logger.info(f"Model loaded from {filename}")
        except FileNotFoundError:
            logger.warning(f"Model file {filename} not found")

def main():
    """메인 실행 함수"""
    detector = UltraPreciseBallDetector()
    
    if not detector.calibration_data:
        logger.error("Failed to load calibration data. Exiting.")
        return
    
    # 테스트 이미지로 검출 테스트
    test_image_path = "data/video_ballData_20250930/video_ballData_20250930/5Iron_0930/1/1_1.bmp"
    
    if os.path.exists(test_image_path):
        logger.info(f"Testing ultra-precise detector on: {test_image_path}")
        
        img = cv2.imread(test_image_path)
        if img is not None:
            result = detector.detect_ball_ultra_precise(img)
            
            if result:
                logger.info(f"Ultra-precise detection successful: {result}")
            else:
                logger.warning("Ultra-precise detection failed")
        else:
            logger.error("Could not load test image")
    else:
        logger.warning(f"Test image not found: {test_image_path}")

if __name__ == "__main__":
    main()





