#!/usr/bin/env python3
"""
실제 이미지 디버그 시스템
골프공이 있는 프레임의 실제 특성을 분석
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

class ActualImageDebugger:
    def __init__(self):
        """실제 이미지 디버거 초기화"""
        print("Actual Image Debugger Initialized")
        print("Analyzing real golf ball characteristics")
    
    def debug_frame(self, frame_num, img1_path, img2_path):
        """특정 프레임 디버그"""
        print(f"\n=== DEBUGGING FRAME {frame_num} ===")
        
        # 이미지 로드
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"ERROR: Failed to load images")
            return None
        
        print(f"Image 1: {img1.shape}, {img1.dtype}")
        print(f"Image 2: {img2.shape}, {img2.dtype}")
        
        # 이미지 통계
        print(f"\nImage 1 statistics:")
        print(f"  Min: {img1.min()}, Max: {img1.max()}, Mean: {img1.mean():.2f}")
        print(f"  Bright pixels (>200): {np.sum(img1 > 200)}")
        print(f"  Very bright pixels (>240): {np.sum(img1 > 240)}")
        
        print(f"\nImage 2 statistics:")
        print(f"  Min: {img2.min()}, Max: {img2.max()}, Mean: {img2.mean():.2f}")
        print(f"  Bright pixels (>200): {np.sum(img2 > 200)}")
        print(f"  Very bright pixels (>240): {np.sum(img2 > 240)}")
        
        # 다양한 방법으로 검출 시도
        detections = []
        
        # 1. HSV 색상 기반 검출 (다양한 파라미터)
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        print(f"\n--- HSV Color Detection ---")
        white_ranges = [
            ("Very strict", [0, 0, 240], [180, 20, 255]),
            ("Strict", [0, 0, 220], [180, 30, 255]),
            ("Normal", [0, 0, 200], [180, 40, 255]),
            ("Wide", [0, 0, 180], [180, 60, 255]),
            ("Very wide", [0, 0, 160], [180, 80, 255]),
            ("Ultra wide", [0, 0, 140], [180, 100, 255]),
        ]
        
        for name, lower, upper in white_ranges:
            lower_white = np.array(lower)
            upper_white = np.array(upper)
            
            # 카메라 1
            mask1 = cv2.inRange(hsv1, lower_white, upper_white)
            white_pixels1 = np.sum(mask1 > 0)
            
            # 카메라 2
            mask2 = cv2.inRange(hsv2, lower_white, upper_white)
            white_pixels2 = np.sum(mask2 > 0)
            
            print(f"  {name}: Cam1={white_pixels1}, Cam2={white_pixels2}")
            
            if white_pixels1 > 20 and white_pixels2 > 20:
                # 컨투어 검출
                contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                print(f"    Contours: Cam1={len(contours1)}, Cam2={len(contours2)}")
                
                # 가장 큰 원형 컨투어 찾기
                best_contour1 = self.find_best_circular_contour(contours1, min_area=5)
                best_contour2 = self.find_best_circular_contour(contours2, min_area=5)
                
                if best_contour1 is not None and best_contour2 is not None:
                    (x1, y1), r1 = cv2.minEnclosingCircle(best_contour1)
                    (x2, y2), r2 = cv2.minEnclosingCircle(best_contour2)
                    
                    center1 = (int(x1), int(y1))
                    center2 = (int(x2), int(y2))
                    radius1 = int(r1)
                    radius2 = int(r2)
                    
                    disparity = abs(y1 - y2)
                    
                    print(f"    Found balls: Cam1=({center1[0]}, {center1[1]}, r={radius1}), Cam2=({center2[0]}, {center2[1]}, r={radius2})")
                    print(f"    Disparity: {disparity:.1f} pixels")
                    
                    detections.append({
                        'method': f'HSV_{name}',
                        'center1': center1,
                        'center2': center2,
                        'radius1': radius1,
                        'radius2': radius2,
                        'disparity': disparity
                    })
        
        # 2. 밝기 기반 검출
        print(f"\n--- Brightness-based Detection ---")
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        brightness_thresholds = [150, 180, 200, 220, 240]
        
        for threshold in brightness_thresholds:
            # 밝은 픽셀 마스크
            bright_mask1 = gray1 > threshold
            bright_mask2 = gray2 > threshold
            
            bright_pixels1 = np.sum(bright_mask1)
            bright_pixels2 = np.sum(bright_mask2)
            
            print(f"  Threshold {threshold}: Cam1={bright_pixels1}, Cam2={bright_pixels2}")
            
            if bright_pixels1 > 20 and bright_pixels2 > 20:
                # 마스크를 uint8로 변환
                mask1 = (bright_mask1 * 255).astype(np.uint8)
                mask2 = (bright_mask2 * 255).astype(np.uint8)
                
                # 노이즈 제거
                kernel = np.ones((3, 3), np.uint8)
                mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
                mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
                
                # 컨투어 검출
                contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                best_contour1 = self.find_best_circular_contour(contours1, min_area=5)
                best_contour2 = self.find_best_circular_contour(contours2, min_area=5)
                
                if best_contour1 is not None and best_contour2 is not None:
                    (x1, y1), r1 = cv2.minEnclosingCircle(best_contour1)
                    (x2, y2), r2 = cv2.minEnclosingCircle(best_contour2)
                    
                    center1 = (int(x1), int(y1))
                    center2 = (int(x2), int(y2))
                    radius1 = int(r1)
                    radius2 = int(r2)
                    
                    disparity = abs(y1 - y2)
                    
                    print(f"    Found balls: Cam1=({center1[0]}, {center1[1]}, r={radius1}), Cam2=({center2[0]}, {center2[1]}, r={radius2})")
                    print(f"    Disparity: {disparity:.1f} pixels")
                    
                    detections.append({
                        'method': f'Brightness_{threshold}',
                        'center1': center1,
                        'center2': center2,
                        'radius1': radius1,
                        'radius2': radius2,
                        'disparity': disparity
                    })
        
        # 3. 허프 원 검출 (다양한 파라미터)
        print(f"\n--- Hough Circle Detection ---")
        for dp in [1, 2]:
            for minDist in [5, 10, 15]:
                for param1 in [20, 30, 50]:
                    for param2 in [15, 20, 30]:
                        circles1 = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                                   param1=param1, param2=param2, minRadius=2, maxRadius=25)
                        circles2 = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                                   param1=param1, param2=param2, minRadius=2, maxRadius=25)
                        
                        if circles1 is not None and circles2 is not None:
                            circles1 = np.round(circles1[0, :]).astype("int")
                            circles2 = np.round(circles2[0, :]).astype("int")
                            
                            # 가장 밝은 원 선택
                            best_circle1 = self.find_brightest_circle(circles1, gray1)
                            best_circle2 = self.find_brightest_circle(circles2, gray2)
                            
                            if best_circle1 is not None and best_circle2 is not None:
                                center1 = (best_circle1[0], best_circle1[1])
                                center2 = (best_circle2[0], best_circle2[1])
                                radius1 = best_circle1[2]
                                radius2 = best_circle2[2]
                                
                                disparity = abs(center1[1] - center2[1])
                                
                                print(f"    Found circles: Cam1=({center1[0]}, {center1[1]}, r={radius1}), Cam2=({center2[0]}, {center2[1]}, r={radius2})")
                                print(f"    Disparity: {disparity:.1f} pixels")
                                
                                detections.append({
                                    'method': f'Hough_dp{dp}_md{minDist}_p1{param1}_p2{param2}',
                                    'center1': center1,
                                    'center2': center2,
                                    'radius1': radius1,
                                    'radius2': radius2,
                                    'disparity': disparity
                                })
                                break
                    if detections and detections[-1]['method'].startswith('Hough'):
                        break
                if detections and detections[-1]['method'].startswith('Hough'):
                    break
            if detections and detections[-1]['method'].startswith('Hough'):
                break
        
        # 최고 검출 결과 선택
        if detections:
            # 시차가 가장 큰 검출 결과 선택
            best_detection = max(detections, key=lambda x: x['disparity'])
            print(f"\nBest detection: {best_detection['method']}")
            print(f"  Disparity: {best_detection['disparity']:.1f} pixels")
            
            return best_detection
        else:
            print(f"\nNo golf ball detected in frame {frame_num}")
            return None
    
    def find_best_circular_contour(self, contours, min_area=10):
        """가장 원형에 가까운 컨투어 찾기"""
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:  # 최소 면적
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.2:  # 더 관대한 원형도 임계값
                        score = circularity * area
                        if score > best_score:
                            best_score = score
                            best_contour = contour
        
        return best_contour
    
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
    
    def debug_multiple_frames(self, image_folder="data2/driver/2", frames=[1, 2, 3]):
        """여러 프레임 디버그"""
        print(f"=== DEBUGGING MULTIPLE FRAMES ===")
        print(f"Image folder: {image_folder}")
        print(f"Frames to debug: {frames}")
        
        # 이미지 파일 목록
        gamma1_files = sorted(glob.glob(f"{image_folder}/Gamma_1_*.bmp"))
        gamma2_files = sorted(glob.glob(f"{image_folder}/Gamma_2_*.bmp"))
        
        print(f"Found {len(gamma1_files)} Gamma_1 images")
        print(f"Found {len(gamma2_files)} Gamma_2 images")
        
        if len(gamma1_files) == 0:
            print("ERROR: No images found!")
            return
        
        successful_detections = 0
        
        for frame_num in frames:
            img1_path = gamma1_files[frame_num - 1]
            img2_path = gamma2_files[frame_num - 1]
            
            detection = self.debug_frame(frame_num, img1_path, img2_path)
            
            if detection:
                successful_detections += 1
                print(f"Frame {frame_num}: SUCCESS - Disparity: {detection['disparity']:.1f} pixels")
            else:
                print(f"Frame {frame_num}: FAILED")
        
        detection_rate = (successful_detections / len(frames)) * 100
        print(f"\n=== DEBUG RESULTS ===")
        print(f"Frames analyzed: {len(frames)}")
        print(f"Successful detections: {successful_detections}")
        print(f"Detection rate: {detection_rate:.1f}%")

def main():
    """메인 함수"""
    print("=== ACTUAL IMAGE DEBUGGER ===")
    print("Debugging real golf ball images to find optimal parameters")
    
    debugger = ActualImageDebugger()
    
    # 골프공이 있는 프레임들 디버그
    debugger.debug_multiple_frames(frames=[1, 2, 3, 4, 5, 6, 7])

if __name__ == "__main__":
    main()
