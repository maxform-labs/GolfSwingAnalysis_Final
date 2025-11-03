#!/usr/bin/env python3
"""
최적화된 골프공 검출 시스템
디버그 결과를 바탕으로 최적의 파라미터 사용
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

class OptimizedGolfBallDetector:
    def __init__(self):
        """최적화된 골프공 검출기 초기화"""
        print("Optimized Golf Ball Detector Initialized")
        print("Using debug-optimized parameters for accurate detection")
        
        # 실제 골프공이 있는 프레임 범위
        self.valid_frames = list(range(1, 8))  # 1~7번 프레임
        print(f"Valid frames with golf balls: {self.valid_frames}")
    
    def detect_golf_ball_optimized(self, img):
        """최적화된 골프공 검출 (디버그 결과 기반)"""
        # 1. 허프 원 검출 (가장 효과적인 방법)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 최적화된 허프 원 파라미터 (디버그 결과)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 
            dp=1, minDist=5,
            param1=20, param2=15, 
            minRadius=2, maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # 가장 밝은 원 선택
            best_circle = self.find_brightest_circle(circles, gray)
            if best_circle is not None:
                center = (best_circle[0], best_circle[1])
                radius = best_circle[2]
                
                # 밝기 확인 (충분히 밝은 영역인지)
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, radius, 255, -1)
                mean_brightness = cv2.mean(gray, mask)[0]
                
                if mean_brightness > 30:  # 낮은 임계값 (어두운 이미지 고려)
                    return center, radius, mask
        
        # 2. HSV 색상 기반 검출 (보조 방법)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 매우 넓은 흰색 범위 (디버그에서 효과적이었던 범위)
        white_ranges = [
            ([0, 0, 160], [180, 80, 255]),  # Very wide white
            ([0, 0, 140], [180, 100, 255]), # Ultra wide white
        ]
        
        for lower, upper in white_ranges:
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
            best_contour = self.find_best_circular_contour(contours, min_area=5)
            
            if best_contour is not None:
                (x, y), radius = cv2.minEnclosingCircle(best_contour)
                center = (int(x), int(y))
                
                # 밝기 확인
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, int(radius), 255, -1)
                mean_brightness = cv2.mean(gray, mask)[0]
                
                if mean_brightness > 30:
                    return center, int(radius), white_mask
        
        return None, None, None
    
    def find_brightest_circle(self, circles, gray_img):
        """가장 밝은 원 찾기"""
        if len(circles) == 0:
            return None
        
        best_circle = None
        best_brightness = 0
        
        for circle in circles:
            center = (circle[0], circle[1])
            radius = circle[2]
            
            # 원 내부의 평균 밝기 계산
            mask = np.zeros(gray_img.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            mean_brightness = cv2.mean(gray_img, mask)[0]
            
            if mean_brightness > best_brightness:
                best_brightness = mean_brightness
                best_circle = circle
        
        return best_circle
    
    def find_best_circular_contour(self, contours, min_area=5):
        """가장 원형에 가까운 컨투어 찾기"""
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.2:  # 관대한 원형도 임계값
                        score = circularity * area
                        if score > best_score:
                            best_score = score
                            best_contour = contour
        
        return best_contour
    
    def create_optimized_visualization(self, img1, img2, center1, center2, radius1, radius2, 
                                     frame_num, disparity=None, position_3d=None):
        """최적화된 검출 결과 시각화"""
        # 이미지 복사
        img1_vis = img1.copy()
        img2_vis = img2.copy()
        
        # 골프공 위치에 원 그리기
        if center1 is not None and radius1 is not None:
            cv2.circle(img1_vis, center1, radius1, (0, 255, 0), 2)  # 녹색 원
            cv2.circle(img1_vis, center1, 2, (0, 0, 255), -1)       # 빨간색 중심점
            cv2.putText(img1_vis, f"({center1[0]}, {center1[1]})", 
                       (center1[0] + 10, center1[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if center2 is not None and radius2 is not None:
            cv2.circle(img2_vis, center2, radius2, (0, 255, 0), 2)  # 녹색 원
            cv2.circle(img2_vis, center2, 2, (0, 0, 255), -1)       # 빨간색 중심점
            cv2.putText(img2_vis, f"({center2[0]}, {center2[1]})", 
                       (center2[0] + 10, center2[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # matplotlib을 사용한 시각화
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 카메라 1 이미지
        axes[0].imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Camera 1 - Frame {frame_num}\nOptimized Golf Ball Detection', 
                         fontsize=14, fontweight='bold', color='blue')
        if center1 is not None:
            axes[0].text(0.02, 0.98, f'Center: ({center1[0]}, {center1[1]})\nRadius: {radius1}px', 
                        transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[0].axis('off')
        
        # 카메라 2 이미지
        axes[1].imshow(cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Camera 2 - Frame {frame_num}\nOptimized Golf Ball Detection', 
                         fontsize=14, fontweight='bold', color='blue')
        if center2 is not None:
            axes[1].text(0.02, 0.98, f'Center: ({center2[0]}, {center2[1]})\nRadius: {radius2}px', 
                        transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1].axis('off')
        
        # 전체 제목
        title = f'OPTIMIZED Golf Ball Detection - Frame {frame_num}'
        if disparity is not None:
            title += f' (Disparity: {disparity:.1f}px)'
        if position_3d is not None:
            title += f' - 3D: ({position_3d[0]:.1f}, {position_3d[1]:.1f}, {position_3d[2]:.1f})mm'
        
        fig.suptitle(title, fontsize=16, fontweight='bold', color='darkblue')
        
        plt.tight_layout()
        
        # 이미지 저장
        filename = f'optimized_golf_ball_frame_{frame_num:02d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Optimized visualization saved: {filename}")
        return filename
    
    def process_optimized_detection(self, image_folder="data2/driver/2"):
        """최적화된 검출 처리"""
        print(f"\n=== PROCESSING WITH OPTIMIZED PARAMETERS ===")
        print(f"Image folder: {image_folder}")
        print(f"Processing frames: {self.valid_frames}")
        
        # 이미지 파일 목록
        gamma1_files = sorted(glob.glob(f"{image_folder}/Gamma_1_*.bmp"))
        gamma2_files = sorted(glob.glob(f"{image_folder}/Gamma_2_*.bmp"))
        
        print(f"Found {len(gamma1_files)} Gamma_1 images")
        print(f"Found {len(gamma2_files)} Gamma_2 images")
        
        if len(gamma1_files) == 0:
            print("ERROR: No images found!")
            return 0, 0, []
        
        successful_detections = 0
        successful_3d_calculations = 0
        total_frames = len(self.valid_frames)
        
        detection_results = []
        
        print(f"\nProcessing {total_frames} valid frames...")
        print("=" * 60)
        
        for i, frame_num in enumerate(self.valid_frames):
            print(f"\nFrame {frame_num} (Valid frame {i+1}/{total_frames}):")
            
            img1_path = gamma1_files[frame_num - 1]  # 0-based indexing
            img2_path = gamma2_files[frame_num - 1]
            
            # 이미지 로드
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                print(f"  ERROR: Failed to load images")
                continue
            
            # 최적화된 골프공 검출
            center1, radius1, mask1 = self.detect_golf_ball_optimized(img1)
            center2, radius2, mask2 = self.detect_golf_ball_optimized(img2)
            
            if center1 is not None and center2 is not None:
                successful_detections += 1
                
                # 시차 계산
                disparity = abs(center1[1] - center2[1])
                
                # 3D 위치 계산
                if disparity > 2:
                    focal_length = 1800.0
                    baseline_mm = 470.0
                    depth = (focal_length * baseline_mm) / disparity
                    
                    if 100 < depth < 1500:
                        x = (center1[0] - 720) * depth / focal_length
                        y = ((center1[1] + center2[1]) / 2 - 540) * depth / focal_length
                        z = depth
                        position_3d = np.array([x, y, z])
                        successful_3d_calculations += 1
                    else:
                        position_3d = None
                else:
                    position_3d = None
                
                print(f"  SUCCESS: Cam1=({center1[0]}, {center1[1]}, r={radius1}), Cam2=({center2[0]}, {center2[1]}, r={radius2})")
                print(f"  Disparity: {disparity:.1f}px")
                if position_3d is not None:
                    print(f"  3D Position: ({position_3d[0]:.1f}, {position_3d[1]:.1f}, {position_3d[2]:.1f})mm")
                
                result = {
                    'frame': frame_num,
                    'center1': center1,
                    'center2': center2,
                    'radius1': radius1,
                    'radius2': radius2,
                    'disparity': disparity,
                    'position_3d': position_3d,
                    'detection_success': True,
                    'calculation_success': position_3d is not None
                }
                detection_results.append(result)
                
                # 최적화된 시각화 생성
                self.create_optimized_visualization(img1, img2, center1, center2, radius1, radius2, 
                                                  frame_num, disparity, position_3d)
            else:
                print(f"  FAILED: Ball not detected (unexpected!)")
                if center1 is None:
                    print(f"    Camera 1: No ball detected")
                if center2 is None:
                    print(f"    Camera 2: No ball detected")
        
        # 최종 결과
        detection_rate = (successful_detections / total_frames) * 100
        calculation_rate = (successful_3d_calculations / total_frames) * 100
        
        print(f"\n" + "=" * 60)
        print(f"=== OPTIMIZED DETECTION RESULTS ===")
        print(f"Valid frames processed: {total_frames}")
        print(f"Successful detections: {successful_detections}")
        print(f"Successful 3D calculations: {successful_3d_calculations}")
        print(f"Detection rate: {detection_rate:.1f}%")
        print(f"3D calculation rate: {calculation_rate:.1f}%")
        print(f"Visualization images saved in current directory")
        print(f"=" * 60)
        
        return detection_rate, calculation_rate, detection_results

def main():
    """메인 함수"""
    print("=== OPTIMIZED GOLF BALL DETECTION SYSTEM ===")
    print("Using debug-optimized parameters for maximum accuracy")
    
    detector = OptimizedGolfBallDetector()
    
    # 최적화된 검출 처리
    detection_rate, calculation_rate, results = detector.process_optimized_detection()
    
    print(f"\nOptimized golf ball detection completed!")
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"3D calculation rate: {calculation_rate:.1f}%")
    print(f"Check the generated PNG files for optimized results!")

if __name__ == "__main__":
    main()
