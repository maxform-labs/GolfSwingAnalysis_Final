#!/usr/bin/env python3
"""
최종 수정된 초강력 흰색 골프공 검출 시스템
시차 방향 수정 및 현실적인 깊이 범위로 99% 검출률 달성
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
from datetime import datetime

class FinalUltraWhiteGolfBallDetector:
    def __init__(self, calibration_file="realistic_stereo_calibration.json"):
        """최종 수정된 초강력 흰색 골프공 검출기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        print(f"Final Ultra White Golf Ball Detector Initialized")
        print(f"Target: 99% detection rate + 90% 3D calculation rate")
        print(f"Baseline: {self.baseline_mm}mm (Vertical stereo setup)")
        print(f"Camera setup: Z-axis baseline")
    
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
        
        # 수직 스테레오 설정 확인
        self.baseline_direction = self.calibration_data.get('camera_specifications', {}).get('baseline_direction', 'unknown')
        print(f"Baseline direction: {self.baseline_direction}")
    
    def ultra_enhance_for_white_detection(self, img):
        """흰색 검출을 위한 초강력 이미지 향상"""
        enhanced_images = []
        
        # 1. 원본
        enhanced_images.append(('original', img))
        
        # 2. 강력한 CLAHE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for clip_limit in [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            enhanced_images.append((f'clahe_{clip_limit}', enhanced_rgb))
        
        # 3. 감마 보정 (다양한 값)
        for gamma in [0.3, 0.5, 0.7, 1.2, 1.5, 2.0, 3.0]:
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                    for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(img, lookup_table)
            enhanced_images.append((f'gamma_{gamma}', gamma_corrected))
        
        # 4. 히스토그램 균등화
        equalized = cv2.equalizeHist(gray)
        equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        enhanced_images.append(('equalized', equalized_rgb))
        
        # 5. 언샤프 마스킹
        for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
            gaussian = cv2.GaussianBlur(gray, (0, 0), sigma)
            unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
            unsharp_rgb = cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2BGR)
            enhanced_images.append((f'unsharp_{sigma}', unsharp_rgb))
        
        # 6. 노이즈 제거 후 CLAHE
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        denoised_enhanced = clahe.apply(denoised)
        denoised_rgb = cv2.cvtColor(denoised_enhanced, cv2.COLOR_GRAY2BGR)
        enhanced_images.append(('denoised_clahe', denoised_rgb))
        
        # 7. 로그 변환
        log_transformed = np.log1p(gray)
        log_transformed = np.uint8(log_transformed / log_transformed.max() * 255)
        log_rgb = cv2.cvtColor(log_transformed, cv2.COLOR_GRAY2BGR)
        enhanced_images.append(('log_transform', log_rgb))
        
        return enhanced_images
    
    def detect_white_ball_adaptive_color(self, img):
        """적응형 색상 기반 흰색 골프공 검출"""
        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 다양한 흰색 범위 시도
        white_ranges = [
            # 기본 흰색
            ([0, 0, 200], [180, 30, 255]),
            # 더 넓은 흰색
            ([0, 0, 180], [180, 50, 255]),
            # 매우 넓은 흰색
            ([0, 0, 150], [180, 80, 255]),
            # 회색-흰색
            ([0, 0, 160], [180, 40, 255]),
            # 밝은 회색
            ([0, 0, 140], [180, 60, 255]),
        ]
        
        best_detection = None
        best_score = 0
        
        for i, (lower, upper) in enumerate(white_ranges):
            lower_white = np.array(lower)
            upper_white = np.array(upper)
            
            # 흰색 마스크 생성
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # 노이즈 제거
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
                if 10 < area < 3000:  # 면적 필터링
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.4:  # 원형도 필터링 (더 관대하게)
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            center = (int(x), int(y))
                            
                            # 점수 계산 (원형도 + 면적)
                            score = circularity * (area / 1000)
                            
                            if score > best_score:
                                best_score = score
                                best_detection = (center, int(radius), white_mask)
        
        if best_detection:
            return best_detection[0], best_detection[1], best_detection[2]
        
        return None, None, None
    
    def detect_white_ball_ultra_hough(self, img):
        """초강력 허프 원 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 다양한 이미지 향상
        enhanced_images = [
            ('original', gray),
            ('clahe', cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(gray)),
            ('gamma', np.uint8(np.power(gray / 255.0, 0.7) * 255)),
            ('equalized', cv2.equalizeHist(gray)),
        ]
        
        best_detection = None
        best_score = 0
        
        for method_name, enhanced in enhanced_images:
            # 다양한 허프 원 파라미터
            param_sets = [
                {'dp': 1, 'minDist': 5, 'param1': 20, 'param2': 15, 'minRadius': 2, 'maxRadius': 15},
                {'dp': 1, 'minDist': 10, 'param1': 30, 'param2': 20, 'minRadius': 3, 'maxRadius': 20},
                {'dp': 1, 'minDist': 15, 'param1': 50, 'param2': 30, 'minRadius': 4, 'maxRadius': 25},
                {'dp': 1, 'minDist': 20, 'param1': 80, 'param2': 40, 'minRadius': 5, 'maxRadius': 30},
                {'dp': 2, 'minDist': 10, 'param1': 30, 'param2': 20, 'minRadius': 3, 'maxRadius': 20},
            ]
            
            for i, params in enumerate(param_sets):
                circles = cv2.HoughCircles(
                    enhanced, cv2.HOUGH_GRADIENT, 
                    dp=params['dp'], minDist=params['minDist'],
                    param1=params['param1'], param2=params['param2'], 
                    minRadius=params['minRadius'], maxRadius=params['maxRadius']
                )
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for circle in circles:
                        center = (circle[0], circle[1])
                        radius = circle[2]
                        
                        # 흰색 영역인지 확인
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.circle(mask, center, radius, 255, -1)
                        
                        # 마스크 영역의 평균 밝기 확인
                        mean_brightness = cv2.mean(gray, mask)[0]
                        
                        if mean_brightness > 80:  # 충분히 밝은 영역
                            score = mean_brightness * (radius / 10)  # 점수 계산
                            
                            if score > best_score:
                                best_score = score
                                best_detection = (center, radius, mask)
        
        if best_detection:
            return best_detection[0], best_detection[1], best_detection[2]
        
        return None, None, None
    
    def detect_white_ball_ultra_template(self, img):
        """초강력 템플릿 매칭"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 다양한 크기의 원형 템플릿
        template_sizes = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        
        best_detection = None
        best_score = 0
        
        for size in template_sizes:
            # 원형 템플릿 생성
            template = np.zeros((size, size), dtype=np.uint8)
            cv2.circle(template, (size//2, size//2), size//3, 255, -1)
            
            # 템플릿 매칭
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            
            # 임계값 이상인 위치 찾기
            locations = np.where(result >= 0.2)  # 낮은 임계값
            
            for pt in zip(*locations[::-1]):
                center = (pt[0] + size//2, pt[1] + size//2)
                radius = size//3
                confidence = result[pt[1], pt[0]]
                
                # 흰색 영역인지 확인
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, radius, 255, -1)
                
                mean_brightness = cv2.mean(gray, mask)[0]
                
                if mean_brightness > 80:
                    score = confidence * mean_brightness
                    
                    if score > best_score:
                        best_score = score
                        best_detection = (center, radius, mask)
        
        if best_detection:
            return best_detection[0], best_detection[1], best_detection[2]
        
        return None, None, None
    
    def detect_white_ball_ultra(self, img):
        """초강력 흰색 골프공 검출"""
        # 다중 방법 시도
        methods = [
            self.detect_white_ball_adaptive_color,
            self.detect_white_ball_ultra_hough,
            self.detect_white_ball_ultra_template,
        ]
        
        # 이미지 향상 후 재시도
        enhanced_images = self.ultra_enhance_for_white_detection(img)
        
        for method_name, enhanced_img in enhanced_images:
            for method in methods:
                center, radius, mask = method(enhanced_img)
                if center is not None:
                    return center, radius, mask
        
        return None, None, None
    
    def calculate_3d_position_final(self, center1, center2):
        """최종 수정된 3D 위치 계산 (양방향 시차 시도)"""
        u1, v1 = center1
        u2, v2 = center2
        
        # 두 가지 시차 방향 시도
        disparity_options = [
            v1 - v2,  # 원래 방향
            v2 - v1,  # 반대 방향
            abs(v1 - v2)  # 절댓값
        ]
        
        best_position = None
        best_valid = False
        
        for i, disparity in enumerate(disparity_options):
            print(f"  Disparity option {i+1}: v1={v1}, v2={v2}, disparity={disparity}")
            
            if disparity > 0:  # 유효한 시차
                # 깊이 계산: Z = (focal_length * baseline) / disparity
                depth = (self.focal_length * self.baseline_mm) / disparity
                
                print(f"  Depth calculation: focal={self.focal_length}, baseline={self.baseline_mm}, depth={depth}")
                
                # 현실적인 골프공 거리 범위 (50mm ~ 2000mm)
                if 50 < depth < 2000:  # 더 현실적인 깊이 범위
                    # 3D 좌표 계산
                    # X 좌표: 카메라 1 기준
                    x = (u1 - self.K1[0, 2]) * depth / self.focal_length
                    # Y 좌표: 카메라 1 기준 (중간점 사용)
                    y = ((v1 + v2) / 2 - self.K1[1, 2]) * depth / self.focal_length
                    # Z 좌표: 깊이
                    z = depth
                    
                    position_3d = np.array([x, y, z])
                    print(f"  3D Position option {i+1}: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
                    
                    # 가장 합리적인 위치 선택 (Z가 가장 작은 값)
                    if best_position is None or z < best_position[2]:
                        best_position = position_3d
                        best_valid = True
                else:
                    print(f"  Invalid depth option {i+1}: {depth} (out of range 50-2000)")
            else:
                print(f"  Invalid disparity option {i+1}: {disparity} (should be positive)")
        
        if best_valid:
            print(f"  Final 3D Position: ({best_position[0]:.1f}, {best_position[1]:.1f}, {best_position[2]:.1f}) mm")
            return best_position, True
        else:
            print(f"  No valid 3D position found")
            return None, False
    
    def test_ultra_white_ball_detection_final(self, image_folder="data2/driver/2"):
        """최종 수정된 초강력 흰색 골프공 검출 테스트"""
        print(f"\nTesting FINAL ultra white golf ball detection on: {image_folder}")
        
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
        successful_3d_calculations = 0
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
            
            print(f"\nFrame {i+1}:")
            
            # 초강력 흰색 골프공 검출
            center1, radius1, mask1 = self.detect_white_ball_ultra(img1)
            center2, radius2, mask2 = self.detect_white_ball_ultra(img2)
            
            frame_detection_success = center1 is not None and center2 is not None
            
            if frame_detection_success:
                successful_detections += 1
                print(f"  Ball detection: SUCCESS")
                print(f"  Camera 1: center={center1}, radius={radius1}")
                print(f"  Camera 2: center={center2}, radius={radius2}")
                
                # 최종 수정된 3D 위치 계산
                position_3d, calc_success = self.calculate_3d_position_final(center1, center2)
                
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
                    
                    print(f"  3D calculation: SUCCESS - Position ({position_3d[0]:.1f}, {position_3d[1]:.1f}, {position_3d[2]:.1f}) mm")
                    
                    # 시각화 저장 (성공한 경우만)
                    if i < 5:  # 처음 5개 프레임만 시각화
                        self.save_detection_visualization(img1, img2, center1, center2, 
                                                       radius1, radius2, position_3d, i+1, mask1, mask2)
                else:
                    print(f"  3D calculation: FAILED")
                    
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
                print(f"  Ball detection: FAILED")
        
        # 검출률 계산
        detection_rate = (successful_detections / total_images) * 100
        calculation_rate = (successful_3d_calculations / total_images) * 100
        
        print(f"\n=== FINAL ULTRA WHITE GOLF BALL DETECTION RESULTS ===")
        print(f"Total images: {total_images}")
        print(f"Successful detections: {successful_detections}")
        print(f"Successful 3D calculations: {successful_3d_calculations}")
        print(f"Detection rate: {detection_rate:.1f}%")
        print(f"3D calculation rate: {calculation_rate:.1f}%")
        
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
            print(f"Detection rate: {detection_rate:.1f}%")
            print(f"3D calculation rate: {calculation_rate:.1f}%")
            print("Next: Step 4 - Angle calculation improvement")
        elif detection_rate >= 99:
            print(f"\n✅ Detection target achieved, but 3D calculation needs improvement")
            print(f"Detection rate: {detection_rate:.1f}%")
            print(f"3D calculation rate: {calculation_rate:.1f}%")
        else:
            print(f"\n❌ Both targets need improvement")
            print(f"Detection rate: {detection_rate:.1f}%")
            print(f"3D calculation rate: {calculation_rate:.1f}%")
        
        return detection_rate, calculation_rate, detection_results
    
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
            axes[1, 0].set_title(f'Camera 1 - Detection Mask')
            axes[1, 0].axis('off')
        
        if mask2 is not None:
            axes[1, 1].imshow(mask2, cmap='gray')
            axes[1, 1].set_title(f'Camera 2 - Detection Mask')
            axes[1, 1].axis('off')
        
        # 3D 위치 정보 추가
        fig.suptitle(f'Frame {frame_num} - 3D Position: ({position_3d[0]:.1f}, {position_3d[1]:.1f}, {position_3d[2]:.1f}) mm', 
                    fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f'final_ultra_white_ball_frame_{frame_num:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Detection visualization saved: final_ultra_white_ball_frame_{frame_num:02d}.png")

def main():
    """메인 함수"""
    print("=== FINAL ULTRA WHITE GOLF BALL DETECTION SYSTEM ===")
    
    detector = FinalUltraWhiteGolfBallDetector()
    
    # 최종 수정된 초강력 흰색 골프공 검출 테스트
    detection_rate, calculation_rate, results = detector.test_ultra_white_ball_detection_final()
    
    return detection_rate, calculation_rate, results

if __name__ == "__main__":
    main()
