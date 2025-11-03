#!/usr/bin/env python3
"""
Cam2 검출 디버깅
"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def detect_golf_ball_debug(frame, camera_name):
    """디버그용 골프공 검출"""
    if frame is None or frame.size == 0:
        return None, None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 이미지 통계
    mean_brightness = np.mean(gray)
    max_brightness = np.max(gray)
    min_brightness = np.min(gray)
    
    print(f"  {camera_name} brightness: mean={mean_brightness:.1f}, max={max_brightness}, min={min_brightness}")
    
    # 다양한 임계값으로 시도
    thresholds = [
        (120, 255),
        (150, 255),
        (180, 255),
        (200, 255),
        (mean_brightness + 20, 255),
        (mean_brightness - 30, 255)
    ]
    
    for i, (thresh_low, thresh_high) in enumerate(thresholds):
        if thresh_low < 0 or thresh_low >= thresh_high:
            continue
            
        binary = cv2.inRange(gray, thresh_low, thresh_high)
        bright_pixels = np.sum(binary > 0)
        print(f"    Threshold {i+1} ({thresh_low}-{thresh_high}): {bright_pixels} bright pixels")
        
        if bright_pixels < 10:
            continue
            
        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"    Found {len(contours)} contours")
        
        best_contour = None
        best_score = 0
        
        for j, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            print(f"      Contour {j}: area={area:.0f}")
            
            if 10 < area < 5000:  # 더 관대한 면적 조건
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    print(f"        circularity={circularity:.2f}")
                    
                    if circularity > 0.3:  # 더 관대한 원형도 조건
                        x, y, w, h = cv2.boundingRect(contour)
                        roi = gray[y:y+h, x:x+w]
                        mean_brightness_roi = np.mean(roi)
                        print(f"        brightness={mean_brightness_roi:.1f}")
                        
                        if mean_brightness_roi > thresh_low * 0.5:  # 더 관대한 밝기 조건
                            score = circularity * (area / 100) * (mean_brightness_roi / 255)
                            print(f"        score={score:.2f}")
                            
                            if score > best_score:
                                best_score = score
                                best_contour = contour
        
        if best_contour is not None and best_score > 0.05:  # 더 낮은 점수 임계값
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center = (int(x), int(y))
            print(f"    BEST DETECTION: center=({center[0]}, {center[1]}), radius={int(radius)}, score={best_score:.2f}")
            return center, int(radius)
    
    print(f"  {camera_name}: No detection")
    return None, None

def debug_cam2_detection():
    """Cam2 검출 디버깅"""
    driver_dir = "data2/driver/2"
    
    # Cam2 이미지들
    pattern_cam2 = os.path.join(driver_dir, "Cam2_*.bmp")
    files_cam2 = sorted(glob.glob(pattern_cam2), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    print(f"Found {len(files_cam2)} Cam2 images")
    
    # 처음 5개 프레임 디버깅
    for i in range(min(5, len(files_cam2))):
        filename = os.path.basename(files_cam2[i])
        frame_num = int(filename.split('_')[1].split('.')[0])
        
        print(f"\n=== Frame {frame_num} ===")
        
        # 이미지 로드
        img = cv2.imread(files_cam2[i])
        
        if img is None:
            print(f"Failed to load image")
            continue
        
        print(f"Image size: {img.shape}")
        
        # 골프공 검출
        center, radius = detect_golf_ball_debug(img, "Cam2")
        
        if center is not None:
            print(f"SUCCESS: Detected at {center}, radius {radius}")
        else:
            print(f"FAILED: No detection")

if __name__ == "__main__":
    debug_cam2_detection()
