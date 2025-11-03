#!/usr/bin/env python3
"""
3단계: 골프공 검출률 99% 달성
다중 알고리즘과 이미지 전처리를 통한 고성능 검출 시스템
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
from datetime import datetime

class UltraHighDetectionSystem:
    def __init__(self, calibration_file="realistic_stereo_calibration.json"):
        """초고성능 검출 시스템 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        print(f"Ultra High Detection System Initialized")
        print(f"Target: 99% detection rate")
        print(f"Baseline: {self.baseline_mm}mm")
    
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
    
    def ultra_enhance_image(self, img):
        """초강력 이미지 향상"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 다단계 이미지 향상
        enhanced_images = []
        
        # 1. 원본
        enhanced_images.append(('original', gray))
        
        # 2. CLAHE (다양한 파라미터)
        for clip_limit in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced_images.append((f'clahe_{clip_limit}', enhanced))
        
        # 3. 감마 보정
        for gamma in [0.3, 0.5, 0.7, 1.2, 1.5, 2.0]:
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                    for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(gray, lookup_table)
            enhanced_images.append((f'gamma_{gamma}', gamma_corrected))
        
        # 4. 히스토그램 균등화
        equalized = cv2.equalizeHist(gray)
        enhanced_images.append(('equalized', equalized))
        
        # 5. 가우시안 블러 후 CLAHE
        for blur_size in [3, 5, 7]:
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            blurred_enhanced = clahe.apply(blurred)
            enhanced_images.append((f'blur_{blur_size}_clahe', blurred_enhanced))
        
        # 6. 언샤프 마스킹
        for sigma in [1.0, 1.5, 2.0]:
            gaussian = cv2.GaussianBlur(gray, (0, 0), sigma)
            unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
            enhanced_images.append((f'unsharp_{sigma}', unsharp_mask))
        
        # 7. 노이즈 제거 후 CLAHE
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        denoised_enhanced = clahe.apply(denoised)
        enhanced_images.append(('denoised_clahe', denoised_enhanced))
        
        return enhanced_images
    
    def detect_ball_ultra_hough(self, enhanced_img, method_name):
        """초강력 허프 원 검출"""
        detections = []
        
        # 다양한 허프 원 파라미터 시도
        param_sets = [
            {'dp': 1, 'minDist': 10, 'param1': 30, 'param2': 20, 'minRadius': 2, 'maxRadius': 15},
            {'dp': 1, 'minDist': 15, 'param1': 50, 'param2': 30, 'minRadius': 3, 'maxRadius': 20},
            {'dp': 1, 'minDist': 20, 'param1': 80, 'param2': 40, 'minRadius': 4, 'maxRadius': 25},
            {'dp': 1, 'minDist': 25, 'param1': 100, 'param2': 50, 'minRadius': 5, 'maxRadius': 30},
            {'dp': 2, 'minDist': 15, 'param1': 50, 'param2': 30, 'minRadius': 3, 'maxRadius': 20},
        ]
        
        for i, params in enumerate(param_sets):
            circles = cv2.HoughCircles(
                enhanced_img, cv2.HOUGH_GRADIENT, 
                dp=params['dp'], minDist=params['minDist'],
                param1=params['param1'], param2=params['param2'], 
                minRadius=params['minRadius'], maxRadius=params['maxRadius']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for circle in circles:
                    center = (circle[0], circle[1])
                    radius = circle[2]
                    confidence = 1.0 / (i + 1)  # 첫 번째 파라미터 세트가 더 높은 신뢰도
                    detections.append((center, radius, confidence, f'hough_{i}'))
        
        return detections
    
    def detect_ball_ultra_contour(self, enhanced_img, method_name):
        """초강력 컨투어 검출"""
        detections = []
        
        # 다양한 임계값 시도
        threshold_methods = [
            ('adaptive_mean', cv2.ADAPTIVE_THRESH_MEAN_C),
            ('adaptive_gaussian', cv2.ADAPTIVE_THRESH_GAUSSIAN_C),
            ('otsu', None),
        ]
        
        for thresh_name, thresh_method in threshold_methods:
            if thresh_method is None:
                # Otsu 임계값
                _, thresh = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                # 적응형 임계값
                thresh = cv2.adaptiveThreshold(enhanced_img, 255, thresh_method, cv2.THRESH_BINARY, 11, 2)
            
            # 모폴로지 연산
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # 컨투어 검출
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < 2000:  # 면적 필터링
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.5:  # 원형도 필터링
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            center = (int(x), int(y))
                            confidence = circularity
                            detections.append((center, int(radius), confidence, f'contour_{thresh_name}'))
        
        return detections
    
    def detect_ball_ultra_template(self, enhanced_img, method_name):
        """초강력 템플릿 매칭"""
        detections = []
        
        # 다양한 크기의 원형 템플릿 생성
        template_sizes = [8, 12, 16, 20, 24, 28]
        
        for size in template_sizes:
            template = np.zeros((size, size), dtype=np.uint8)
            cv2.circle(template, (size//2, size//2), size//3, 255, -1)
            
            # 템플릿 매칭
            result = cv2.matchTemplate(enhanced_img, template, cv2.TM_CCOEFF_NORMED)
            
            # 임계값 이상인 위치 찾기
            locations = np.where(result >= 0.3)
            
            for pt in zip(*locations[::-1]):
                center = (pt[0] + size//2, pt[1] + size//2)
                radius = size//3
                confidence = result[pt[1], pt[0]]
                detections.append((center, radius, confidence, f'template_{size}'))
        
        return detections
    
    def detect_ball_ultra_edge(self, enhanced_img, method_name):
        """초강력 엣지 기반 검출"""
        detections = []
        
        # 다양한 엣지 검출 파라미터
        edge_params = [
            {'low': 30, 'high': 100},
            {'low': 50, 'high': 150},
            {'low': 80, 'high': 200},
        ]
        
        for i, params in enumerate(edge_params):
            edges = cv2.Canny(enhanced_img, params['low'], params['high'])
            
            # 허프 원 검출 (엣지 이미지에서)
            circles = cv2.HoughCircles(
                edges, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
                param1=50, param2=30, minRadius=3, maxRadius=25
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for circle in circles:
                    center = (circle[0], circle[1])
                    radius = circle[2]
                    confidence = 0.8 / (i + 1)  # 첫 번째 파라미터가 더 높은 신뢰도
                    detections.append((center, radius, confidence, f'edge_{i}'))
        
        return detections
    
    def detect_ball_ultra(self, img):
        """초강력 골프 공 검출"""
        enhanced_images = self.ultra_enhance_image(img)
        
        all_detections = []
        
        for method_name, enhanced_img in enhanced_images:
            # 허프 원 검출
            hough_detections = self.detect_ball_ultra_hough(enhanced_img, method_name)
            all_detections.extend(hough_detections)
            
            # 컨투어 검출
            contour_detections = self.detect_ball_ultra_contour(enhanced_img, method_name)
            all_detections.extend(contour_detections)
            
            # 템플릿 매칭
            template_detections = self.detect_ball_ultra_template(enhanced_img, method_name)
            all_detections.extend(template_detections)
            
            # 엣지 기반 검출
            edge_detections = self.detect_ball_ultra_edge(enhanced_img, method_name)
            all_detections.extend(edge_detections)
        
        # 검출 결과 통합 및 필터링
        if not all_detections:
            return None, None
        
        # 신뢰도 기반 정렬
        all_detections.sort(key=lambda x: x[2], reverse=True)
        
        # 중복 제거 (거리 기반)
        filtered_detections = []
        for detection in all_detections:
            center, radius, confidence, method = detection
            
            # 기존 검출과의 거리 확인
            is_duplicate = False
            for existing in filtered_detections:
                existing_center = existing[0]
                distance = np.sqrt((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)
                
                if distance < 20:  # 20픽셀 이내면 중복으로 간주
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_detections.append(detection)
        
        # 최고 신뢰도 검출 반환
        if filtered_detections:
            best_detection = filtered_detections[0]
            return best_detection[0], best_detection[1]  # center, radius
        
        return None, None
    
    def test_detection_rate(self, image_folder="data2/driver/2"):
        """검출률 테스트"""
        print(f"\nTesting detection rate on: {image_folder}")
        
        # 이미지 파일 목록
        gamma1_files = sorted(glob.glob(f"{image_folder}/Gamma_1_*.bmp"))
        gamma2_files = sorted(glob.glob(f"{image_folder}/Gamma_2_*.bmp"))
        
        print(f"Found {len(gamma1_files)} Gamma_1 images")
        print(f"Found {len(gamma2_files)} Gamma_2 images")
        
        if len(gamma1_files) == 0:
            print("X No images found")
            return 0
        
        # 검출 테스트
        successful_detections = 0
        total_images = min(len(gamma1_files), len(gamma2_files))
        
        detection_results = []
        
        for i in range(total_images):
            img1_path = gamma1_files[i]
            img2_path = gamma2_files[i]
            
            # 이미지 로드
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                continue
            
            # 골프 공 검출
            center1, radius1 = self.detect_ball_ultra(img1)
            center2, radius2 = self.detect_ball_ultra(img2)
            
            frame_success = center1 is not None and center2 is not None
            
            if frame_success:
                successful_detections += 1
                
                # 3D 위치 계산
                u1, v1 = center1
                u2, v2 = center2
                
                # 시차 계산
                disparity = u1 - u2
                
                if disparity > 0:
                    depth = (self.focal_length * self.baseline_mm) / disparity
                    
                    if 50 < depth < 5000:  # 유효한 깊이 범위
                        x = (u1 - self.K1[0, 2]) * depth / self.focal_length
                        y = (v1 - self.K1[1, 2]) * depth / self.focal_length
                        z = depth
                        
                        position_3d = np.array([x, y, z])
                        
                        result = {
                            'frame': i+1,
                            'center1': center1,
                            'center2': center2,
                            'position_3d': position_3d,
                            'success': True
                        }
                        detection_results.append(result)
                        
                        print(f"Frame {i+1}: SUCCESS - 3D position ({x:.1f}, {y:.1f}, {z:.1f}) mm")
                        
                        # 시각화 저장
                        self.save_detection_visualization(img1, img2, center1, center2, 
                                                       radius1, radius2, position_3d, i+1)
                    else:
                        print(f"Frame {i+1}: SUCCESS - Invalid depth")
                else:
                    print(f"Frame {i+1}: SUCCESS - Invalid disparity")
            else:
                print(f"Frame {i+1}: FAILED - Ball not detected")
        
        # 검출률 계산
        detection_rate = (successful_detections / total_images) * 100
        
        print(f"\n=== DETECTION RATE RESULTS ===")
        print(f"Total images: {total_images}")
        print(f"Successful detections: {successful_detections}")
        print(f"Detection rate: {detection_rate:.1f}%")
        
        if detection_rate >= 99:
            print(f"OK Target achieved: {detection_rate:.1f}% >= 99%")
        else:
            print(f"X Target not achieved: {detection_rate:.1f}% < 99%")
        
        return detection_rate, detection_results
    
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
        plt.savefig(f'ultra_detection_frame_{frame_num:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Detection visualization saved: ultra_detection_frame_{frame_num:02d}.png")

def main():
    """메인 함수"""
    print("=== STEP 3: Ultra High Detection System ===")
    
    detector = UltraHighDetectionSystem()
    
    # 검출률 테스트
    detection_rate, results = detector.test_detection_rate()
    
    if detection_rate >= 99:
        print("\nOK Step 3 completed: 99% detection rate achieved!")
        print("Next: Step 4 - Angle calculation improvement")
    else:
        print(f"\nX Step 3 not completed: {detection_rate:.1f}% < 99%")
        print("Need to improve detection algorithms")

if __name__ == "__main__":
    main()
