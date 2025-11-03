#!/usr/bin/env python3
"""
디버그용 흰색 골프공 검출 시스템
이미지 로드 및 검출 과정을 자세히 분석
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
from datetime import datetime
import time

class DebugWhiteBallDetector:
    def __init__(self, calibration_file="realistic_stereo_calibration.json"):
        """디버그용 흰색 골프공 검출기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        print(f"Debug White Golf Ball Detector Initialized")
        print(f"Target: Debug detection process and find issues")
    
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
    
    def debug_image_loading(self, image_folder="data2/driver/2"):
        """이미지 로딩 디버그"""
        print(f"\n=== DEBUGGING IMAGE LOADING ===")
        print(f"Image folder: {image_folder}")
        
        # 이미지 파일 목록
        gamma1_files = sorted(glob.glob(f"{image_folder}/Gamma_1_*.bmp"))
        gamma2_files = sorted(glob.glob(f"{image_folder}/Gamma_2_*.bmp"))
        
        print(f"Found {len(gamma1_files)} Gamma_1 images")
        print(f"Found {len(gamma2_files)} Gamma_2 images")
        
        if len(gamma1_files) == 0:
            print("ERROR: No images found!")
            return False
        
        # 첫 번째 이미지 로드 테스트
        img1_path = gamma1_files[0]
        img2_path = gamma2_files[0]
        
        print(f"\nTesting image loading:")
        print(f"Image 1: {img1_path}")
        print(f"Image 2: {img2_path}")
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None:
            print(f"ERROR: Failed to load {img1_path}")
            return False
        else:
            print(f"OK: Image 1 loaded - Shape: {img1.shape}, Type: {img1.dtype}")
        
        if img2 is None:
            print(f"ERROR: Failed to load {img2_path}")
            return False
        else:
            print(f"OK: Image 2 loaded - Shape: {img2.shape}, Type: {img2.dtype}")
        
        # 이미지 통계
        print(f"\nImage 1 statistics:")
        print(f"  Min: {img1.min()}, Max: {img1.max()}, Mean: {img1.mean():.2f}")
        print(f"  Bright pixels (>200): {np.sum(img1 > 200)}")
        
        print(f"\nImage 2 statistics:")
        print(f"  Min: {img2.min()}, Max: {img2.max()}, Mean: {img2.mean():.2f}")
        print(f"  Bright pixels (>200): {np.sum(img2 > 200)}")
        
        return True
    
    def debug_white_detection(self, img, frame_num):
        """흰색 검출 과정 디버그"""
        print(f"\n=== DEBUGGING WHITE DETECTION - Frame {frame_num} ===")
        
        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print(f"HSV conversion: OK - Shape: {hsv.shape}")
        
        # 다양한 흰색 범위 테스트
        white_ranges = [
            ("Basic white", [0, 0, 200], [180, 30, 255]),
            ("Wide white", [0, 0, 180], [180, 50, 255]),
            ("Very wide white", [0, 0, 150], [180, 80, 255]),
            ("Gray-white", [0, 0, 160], [180, 40, 255]),
            ("Bright gray", [0, 0, 140], [180, 60, 255]),
        ]
        
        for name, lower, upper in white_ranges:
            lower_white = np.array(lower)
            upper_white = np.array(upper)
            
            # 흰색 마스크 생성
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            white_pixels = np.sum(white_mask > 0)
            
            print(f"  {name}: {white_pixels} white pixels")
            
            if white_pixels > 100:  # 충분한 흰색 픽셀이 있는 경우
                # 노이즈 제거
                kernel = np.ones((3, 3), np.uint8)
                white_mask_clean = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
                white_mask_clean = cv2.morphologyEx(white_mask_clean, cv2.MORPH_CLOSE, kernel)
                
                clean_pixels = np.sum(white_mask_clean > 0)
                print(f"    After cleaning: {clean_pixels} pixels")
                
                # 컨투어 검출
                contours, _ = cv2.findContours(white_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"    Contours found: {len(contours)}")
                
                if contours:
                    for i, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        if area > 10:  # 작은 컨투어 제외
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * area / (perimeter * perimeter)
                                print(f"      Contour {i}: area={area:.1f}, circularity={circularity:.3f}")
                                
                                if circularity > 0.4:
                                    (x, y), radius = cv2.minEnclosingCircle(contour)
                                    center = (int(x), int(y))
                                    print(f"        -> Potential ball: center={center}, radius={radius:.1f}")
                                    return center, int(radius), white_mask_clean
        
        print(f"  No white ball detected in frame {frame_num}")
        return None, None, None
    
    def test_debug_detection(self, image_folder="data2/driver/2", max_frames=3):
        """디버그 검출 테스트"""
        print(f"\n=== DEBUGGING WHITE BALL DETECTION ===")
        print(f"Testing first {max_frames} frames with detailed analysis")
        
        # 이미지 로딩 디버그
        if not self.debug_image_loading(image_folder):
            return 0, 0, []
        
        # 이미지 파일 목록
        gamma1_files = sorted(glob.glob(f"{image_folder}/Gamma_1_*.bmp"))
        gamma2_files = sorted(glob.glob(f"{image_folder}/Gamma_2_*.bmp"))
        
        successful_detections = 0
        successful_3d_calculations = 0
        total_images = min(len(gamma1_files), len(gamma2_files), max_frames)
        
        detection_results = []
        
        for i in range(total_images):
            print(f"\n" + "="*60)
            print(f"PROCESSING FRAME {i+1}/{total_images}")
            print(f"="*60)
            
            img1_path = gamma1_files[i]
            img2_path = gamma2_files[i]
            
            # 이미지 로드
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                print(f"ERROR: Failed to load images for frame {i+1}")
                continue
            
            # 카메라 1 검출 디버그
            print(f"\n--- Camera 1 Detection ---")
            center1, radius1, mask1 = self.debug_white_detection(img1, i+1)
            
            # 카메라 2 검출 디버그
            print(f"\n--- Camera 2 Detection ---")
            center2, radius2, mask2 = self.debug_white_detection(img2, i+1)
            
            # 결과 분석
            frame_detection_success = center1 is not None and center2 is not None
            
            if frame_detection_success:
                successful_detections += 1
                print(f"\nSUCCESS: Both cameras detected ball")
                print(f"Camera 1: center={center1}, radius={radius1}")
                print(f"Camera 2: center={center2}, radius={radius2}")
                
                # 3D 계산 시도
                u1, v1 = center1
                u2, v2 = center2
                disparity = abs(v1 - v2)
                
                print(f"Disparity: {disparity} pixels")
                
                if disparity > 2:
                    depth = (self.focal_length * self.baseline_mm) / disparity
                    print(f"Calculated depth: {depth:.1f} mm")
                    
                    if 100 < depth < 1500:
                        x = (u1 - self.K1[0, 2]) * depth / self.focal_length
                        y = ((v1 + v2) / 2 - self.K1[1, 2]) * depth / self.focal_length
                        z = depth
                        
                        position_3d = np.array([x, y, z])
                        successful_3d_calculations += 1
                        
                        print(f"3D Position: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
                        
                        result = {
                            'frame': i+1,
                            'center1': center1,
                            'center2': center2,
                            'position_3d': position_3d,
                            'detection_success': True,
                            'calculation_success': True
                        }
                        detection_results.append(result)
                    else:
                        print(f"Invalid depth: {depth:.1f} mm (out of range 100-1500)")
                else:
                    print(f"Invalid disparity: {disparity} (too small)")
            else:
                print(f"\nFAILED: Ball not detected in both cameras")
                if center1 is None:
                    print(f"  Camera 1: No ball detected")
                if center2 is None:
                    print(f"  Camera 2: No ball detected")
        
        # 최종 결과
        detection_rate = (successful_detections / total_images) * 100
        calculation_rate = (successful_3d_calculations / total_images) * 100
        
        print(f"\n" + "="*60)
        print(f"=== DEBUG RESULTS ===")
        print(f"Total images processed: {total_images}")
        print(f"Successful detections: {successful_detections}")
        print(f"Successful 3D calculations: {successful_3d_calculations}")
        print(f"Detection rate: {detection_rate:.1f}%")
        print(f"3D calculation rate: {calculation_rate:.1f}%")
        print(f"="*60)
        
        return detection_rate, calculation_rate, detection_results

def main():
    """메인 함수"""
    print("=== DEBUG WHITE GOLF BALL DETECTION SYSTEM ===")
    print("Detailed analysis of detection process")
    
    detector = DebugWhiteBallDetector()
    
    # 디버그 검출 테스트
    detection_rate, calculation_rate, results = detector.test_debug_detection(max_frames=3)
    
    return detection_rate, calculation_rate, results

if __name__ == "__main__":
    main()
