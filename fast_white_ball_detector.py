#!/usr/bin/env python3
"""
빠른 흰색 골프공 검출 시스템
진행사항 표시와 최적화된 알고리즘으로 빠른 테스트
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

class FastWhiteBallDetector:
    def __init__(self, calibration_file="realistic_stereo_calibration.json"):
        """빠른 흰색 골프공 검출기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        print(f"Fast White Golf Ball Detector Initialized")
        print(f"Target: 99% detection rate + 90% 3D calculation rate")
        print(f"Optimized for speed with progress tracking")
    
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
    
    def detect_white_ball_fast(self, img):
        """빠른 흰색 골프공 검출 (최적화된 버전)"""
        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 가장 효과적인 흰색 범위만 사용
        white_ranges = [
            ([0, 0, 180], [180, 50, 255]),  # 기본 흰색
            ([0, 0, 150], [180, 80, 255]),  # 넓은 흰색
        ]
        
        best_detection = None
        best_score = 0
        
        for lower, upper in white_ranges:
            lower_white = np.array(lower)
            upper_white = np.array(upper)
            
            # 흰색 마스크 생성
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # 빠른 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            
            # 컨투어 검출
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # 가장 원형에 가까운 컨투어 선택
            for contour in contours:
                area = cv2.contourArea(contour)
                if 20 < area < 2000:  # 면적 필터링 (더 엄격하게)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.5:  # 원형도 필터링 (더 엄격하게)
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            center = (int(x), int(y))
                            
                            # 점수 계산
                            score = circularity * (area / 1000)
                            
                            if score > best_score:
                                best_score = score
                                best_detection = (center, int(radius), white_mask)
        
        if best_detection:
            return best_detection[0], best_detection[1], best_detection[2]
        
        return None, None, None
    
    def calculate_3d_position_fast(self, center1, center2):
        """빠른 3D 위치 계산 (최적화된 버전)"""
        u1, v1 = center1
        u2, v2 = center2
        
        # 가장 가능성 높은 시차 방향만 시도
        disparity = abs(v1 - v2)  # 절댓값 사용
        
        if disparity > 2:  # 최소 시차 임계값
            # 깊이 계산
            depth = (self.focal_length * self.baseline_mm) / disparity
            
            # 현실적인 골프공 거리 범위
            if 100 < depth < 1500:  # 더 엄격한 범위
                # 3D 좌표 계산
                x = (u1 - self.K1[0, 2]) * depth / self.focal_length
                y = ((v1 + v2) / 2 - self.K1[1, 2]) * depth / self.focal_length
                z = depth
                
                position_3d = np.array([x, y, z])
                return position_3d, True
        
        return None, False
    
    def test_fast_white_ball_detection(self, image_folder="data2/driver/2", max_frames=10):
        """빠른 흰색 골프공 검출 테스트 (진행사항 표시)"""
        print(f"\nTesting FAST white golf ball detection on: {image_folder}")
        print(f"Processing first {max_frames} frames for speed...")
        
        # 이미지 파일 목록
        gamma1_files = sorted(glob.glob(f"{image_folder}/Gamma_1_*.bmp"))
        gamma2_files = sorted(glob.glob(f"{image_folder}/Gamma_2_*.bmp"))
        
        print(f"Found {len(gamma1_files)} Gamma_1 images")
        print(f"Found {len(gamma2_files)} Gamma_2 images")
        
        if len(gamma1_files) == 0:
            print("X No images found")
            return 0, 0, []
        
        # 검출 테스트 (처음 몇 개 프레임만)
        successful_detections = 0
        successful_3d_calculations = 0
        total_images = min(len(gamma1_files), len(gamma2_files), max_frames)
        
        detection_results = []
        start_time = time.time()
        
        print(f"\nProcessing {total_images} frames...")
        print("=" * 50)
        
        for i in range(total_images):
            frame_start_time = time.time()
            
            img1_path = gamma1_files[i]
            img2_path = gamma2_files[i]
            
            # 이미지 로드
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                print(f"Frame {i+1:2d}: SKIP - Image load failed")
                continue
            
            # 빠른 흰색 골프공 검출
            center1, radius1, mask1 = self.detect_white_ball_fast(img1)
            center2, radius2, mask2 = self.detect_white_ball_fast(img2)
            
            frame_detection_success = center1 is not None and center2 is not None
            
            if frame_detection_success:
                successful_detections += 1
                
                # 빠른 3D 위치 계산
                position_3d, calc_success = self.calculate_3d_position_fast(center1, center2)
                
                if calc_success:
                    successful_3d_calculations += 1
                    
                    result = {
                        'frame': i+1,
                        'center1': center1,
                        'center2': center2,
                        'position_3d': position_3d,
                        'detection_success': True,
                        'calculation_success': True
                    }
                    detection_results.append(result)
                    
                    print(f"Frame {i+1:2d}: SUCCESS - 3D: ({position_3d[0]:6.1f}, {position_3d[1]:6.1f}, {position_3d[2]:6.1f}) mm")
                else:
                    print(f"Frame {i+1:2d}: DETECT - 3D calculation failed")
                    
                    result = {
                        'frame': i+1,
                        'center1': center1,
                        'center2': center2,
                        'position_3d': None,
                        'detection_success': True,
                        'calculation_success': False
                    }
                    detection_results.append(result)
            else:
                print(f"Frame {i+1:2d}: FAILED - Ball not detected")
            
            # 진행사항 표시
            frame_time = time.time() - frame_start_time
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / (i + 1)
            remaining_time = avg_time * (total_images - i - 1)
            
            print(f"         Time: {frame_time:.2f}s | Avg: {avg_time:.2f}s | Remaining: {remaining_time:.1f}s")
            
            # 5프레임마다 중간 결과 표시
            if (i + 1) % 5 == 0:
                current_detection_rate = (successful_detections / (i + 1)) * 100
                current_calculation_rate = (successful_3d_calculations / (i + 1)) * 100
                print(f"         Progress: Detection {current_detection_rate:.1f}%, 3D Calc {current_calculation_rate:.1f}%")
        
        # 최종 결과
        total_time = time.time() - start_time
        detection_rate = (successful_detections / total_images) * 100
        calculation_rate = (successful_3d_calculations / total_images) * 100
        
        print("=" * 50)
        print(f"=== FAST WHITE GOLF BALL DETECTION RESULTS ===")
        print(f"Total images processed: {total_images}")
        print(f"Successful detections: {successful_detections}")
        print(f"Successful 3D calculations: {successful_3d_calculations}")
        print(f"Detection rate: {detection_rate:.1f}%")
        print(f"3D calculation rate: {calculation_rate:.1f}%")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per frame: {total_time/total_images:.2f} seconds")
        
        if detection_rate >= 99:
            print(f"OK Detection target achieved: {detection_rate:.1f}% >= 99%")
        else:
            print(f"X Detection target not achieved: {detection_rate:.1f}% < 99%")
        
        if calculation_rate >= 90:
            print(f"OK 3D calculation target achieved: {calculation_rate:.1f}% >= 90%")
        else:
            print(f"X 3D calculation target not achieved: {calculation_rate:.1f}% < 90%")
        
        # 성공률에 따른 다음 단계 안내
        if detection_rate >= 99 and calculation_rate >= 90:
            print(f"\n🎉 ALL TARGETS ACHIEVED! 🎉")
            print("Next: Step 4 - Angle calculation improvement")
        elif detection_rate >= 99:
            print(f"\n✅ Detection target achieved, 3D calculation needs improvement")
        else:
            print(f"\n❌ Both targets need improvement")
        
        return detection_rate, calculation_rate, detection_results

def main():
    """메인 함수"""
    print("=== FAST WHITE GOLF BALL DETECTION SYSTEM ===")
    print("Optimized for speed with real-time progress tracking")
    
    detector = FastWhiteBallDetector()
    
    # 빠른 흰색 골프공 검출 테스트
    detection_rate, calculation_rate, results = detector.test_fast_white_ball_detection(max_frames=10)
    
    return detection_rate, calculation_rate, results

if __name__ == "__main__":
    main()
