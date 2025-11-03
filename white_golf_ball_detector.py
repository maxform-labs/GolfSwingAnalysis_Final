#!/usr/bin/env python3
"""
흰색 골프공 전용 검출 시스템
색상 기반 검출으로 99% 검출률 달성
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
from datetime import datetime

class WhiteGolfBallDetector:
    def __init__(self, calibration_file="realistic_stereo_calibration.json"):
        """흰색 골프공 검출기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        print(f"White Golf Ball Detector Initialized")
        print(f"Target: 99% detection rate for white golf balls")
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
    
    def detect_white_ball_color_based(self, img):
        """색상 기반 흰색 골프공 검출"""
        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 흰색 범위 정의 (HSV)
        # 흰색은 채도(S)가 낮고 명도(V)가 높음
        lower_white = np.array([0, 0, 200])      # 하한선
        upper_white = np.array([180, 30, 255])   # 상한선
        
        # 흰색 마스크 생성
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, white_mask
        
        # 가장 원형에 가까운 컨투어 선택
        best_contour = None
        best_circularity = 0
        best_center = None
        best_radius = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 2000:  # 면적 필터링
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.6:  # 원형도 필터링
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        center = (int(x), int(y))
                        
                        if circularity > best_circularity:
                            best_circularity = circularity
                            best_contour = contour
                            best_center = center
                            best_radius = int(radius)
        
        return best_center, best_radius, white_mask
    
    def detect_white_ball_enhanced(self, img):
        """향상된 흰색 골프공 검출"""
        # 1. 색상 기반 검출
        center1, radius1, mask1 = self.detect_white_ball_color_based(img)
        
        if center1 is not None:
            return center1, radius1, mask1
        
        # 2. 이미지 향상 후 재시도
        # CLAHE 적용
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # RGB로 다시 변환
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        center2, radius2, mask2 = self.detect_white_ball_color_based(enhanced_rgb)
        
        if center2 is not None:
            return center2, radius2, mask2
        
        # 3. 감마 보정 후 재시도
        gamma = 0.7
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(img, lookup_table)
        
        center3, radius3, mask3 = self.detect_white_ball_color_based(gamma_corrected)
        
        if center3 is not None:
            return center3, radius3, mask3
        
        # 4. 허프 원 검출 (최후 수단)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 허프 원 검출
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=3, maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # 가장 큰 원 선택
            largest_circle = max(circles, key=lambda x: x[2])
            center = (largest_circle[0], largest_circle[1])
            radius = largest_circle[2]
            
            # 흰색 영역인지 확인
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            
            # 마스크 영역의 평균 밝기 확인
            mean_brightness = cv2.mean(gray, mask)[0]
            
            if mean_brightness > 100:  # 충분히 밝은 영역
                return center, radius, mask
        
        return None, None, None
    
    def detect_white_ball_ultra(self, img):
        """초강력 흰색 골프공 검출"""
        # 다중 방법 시도
        methods = [
            self.detect_white_ball_enhanced,
            self.detect_white_ball_color_based,
        ]
        
        for method in methods:
            center, radius, mask = method(img)
            if center is not None:
                return center, radius, mask
        
        return None, None, None
    
    def test_white_ball_detection(self, image_folder="data2/driver/2"):
        """흰색 골프공 검출 테스트"""
        print(f"\nTesting white golf ball detection on: {image_folder}")
        
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
            
            # 흰색 골프공 검출
            center1, radius1, mask1 = self.detect_white_ball_ultra(img1)
            center2, radius2, mask2 = self.detect_white_ball_ultra(img2)
            
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
                                                       radius1, radius2, position_3d, i+1, mask1, mask2)
                    else:
                        print(f"Frame {i+1}: SUCCESS - Invalid depth")
                else:
                    print(f"Frame {i+1}: SUCCESS - Invalid disparity")
            else:
                print(f"Frame {i+1}: FAILED - White ball not detected")
        
        # 검출률 계산
        detection_rate = (successful_detections / total_images) * 100
        
        print(f"\n=== WHITE GOLF BALL DETECTION RESULTS ===")
        print(f"Total images: {total_images}")
        print(f"Successful detections: {successful_detections}")
        print(f"Detection rate: {detection_rate:.1f}%")
        
        if detection_rate >= 99:
            print(f"OK Target achieved: {detection_rate:.1f}% >= 99%")
        else:
            print(f"X Target not achieved: {detection_rate:.1f}% < 99%")
        
        return detection_rate, detection_results
    
    def save_detection_visualization(self, img1, img2, center1, center2, radius1, radius2, 
                                   position_3d, frame_num, mask1=None, mask2=None):
        """검출 결과 시각화 저장"""
        # 이미지에 원 그리기
        img1_vis = img1.copy()
        img2_vis = img2.copy()
        
        cv2.circle(img1_vis, center1, radius1, (0, 255, 0), 2)
        cv2.circle(img1_vis, center1, 2, (0, 0, 255), -1)
        
        cv2.circle(img2_vis, center2, radius2, (0, 255, 0), 2)
        cv2.circle(img2_vis, center2, 2, (0, 0, 255), -1)
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 원본 이미지
        axes[0, 0].imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Camera 1 - Frame {frame_num}\nBall at ({center1[0]}, {center1[1]})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'Camera 2 - Frame {frame_num}\nBall at ({center2[0]}, {center2[1]})')
        axes[0, 1].axis('off')
        
        # 마스크 이미지
        if mask1 is not None:
            axes[1, 0].imshow(mask1, cmap='gray')
            axes[1, 0].set_title(f'Camera 1 - White Mask')
            axes[1, 0].axis('off')
        
        if mask2 is not None:
            axes[1, 1].imshow(mask2, cmap='gray')
            axes[1, 1].set_title(f'Camera 2 - White Mask')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'white_ball_detection_frame_{frame_num:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Detection visualization saved: white_ball_detection_frame_{frame_num:02d}.png")
    
    def analyze_white_ball_characteristics(self, image_folder="data2/driver/2"):
        """흰색 골프공 특성 분석"""
        print(f"\n=== WHITE GOLF BALL CHARACTERISTICS ANALYSIS ===")
        
        # 샘플 이미지 분석
        gamma1_files = sorted(glob.glob(f"{image_folder}/Gamma_1_*.bmp"))
        
        if len(gamma1_files) == 0:
            print("X No images found for analysis")
            return
        
        # 첫 번째 이미지 분석
        img = cv2.imread(gamma1_files[0])
        if img is None:
            print("X Failed to load sample image")
            return
        
        # HSV 색공간 분석
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 흰색 영역 분석
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 통계 분석
        white_pixels = np.sum(white_mask > 0)
        total_pixels = white_mask.size
        white_ratio = white_pixels / total_pixels * 100
        
        print(f"Image analysis:")
        print(f"  Image size: {img.shape}")
        print(f"  White pixels: {white_pixels} ({white_ratio:.2f}%)")
        print(f"  Total pixels: {total_pixels}")
        
        # 히스토그램 분석
        h, s, v = cv2.split(hsv)
        
        print(f"\nHSV histogram analysis:")
        print(f"  H (Hue) range: {h.min()} - {h.max()}")
        print(f"  S (Saturation) range: {s.min()} - {s.max()}")
        print(f"  V (Value) range: {v.min()} - {v.max()}")
        
        # 흰색 영역의 평균값
        white_h = h[white_mask > 0]
        white_s = s[white_mask > 0]
        white_v = v[white_mask > 0]
        
        if len(white_h) > 0:
            print(f"\nWhite region statistics:")
            print(f"  Average H: {white_h.mean():.2f}")
            print(f"  Average S: {white_s.mean():.2f}")
            print(f"  Average V: {white_v.mean():.2f}")
        
        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 원본 이미지
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # HSV 채널
        axes[0, 1].imshow(h, cmap='hsv')
        axes[0, 1].set_title('Hue Channel')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(s, cmap='gray')
        axes[0, 2].set_title('Saturation Channel')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(v, cmap='gray')
        axes[1, 0].set_title('Value Channel')
        axes[1, 0].axis('off')
        
        # 흰색 마스크
        axes[1, 1].imshow(white_mask, cmap='gray')
        axes[1, 1].set_title('White Mask')
        axes[1, 1].axis('off')
        
        # 히스토그램
        axes[1, 2].hist(v.ravel(), bins=256, range=[0, 256], alpha=0.7, color='blue')
        axes[1, 2].set_title('Value Histogram')
        axes[1, 2].set_xlabel('Value')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('white_ball_characteristics_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("OK White ball characteristics analysis saved: white_ball_characteristics_analysis.png")

def main():
    """메인 함수"""
    print("=== WHITE GOLF BALL DETECTION SYSTEM ===")
    
    detector = WhiteGolfBallDetector()
    
    # 흰색 골프공 특성 분석
    detector.analyze_white_ball_characteristics()
    
    # 흰색 골프공 검출 테스트
    detection_rate, results = detector.test_white_ball_detection()
    
    if detection_rate >= 99:
        print("\nOK White golf ball detection: 99% rate achieved!")
        print("Next: Step 4 - Angle calculation improvement")
    else:
        print(f"\nX White golf ball detection: {detection_rate:.1f}% < 99%")
        print("Need to improve white ball detection algorithms")

if __name__ == "__main__":
    main()
