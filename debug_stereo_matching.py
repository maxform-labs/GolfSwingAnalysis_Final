#!/usr/bin/env python3
"""
스테레오 매칭 디버깅
"""

import cv2
import numpy as np
import os
import glob

def detect_golf_ball_adaptive(frame):
    """적응형 골프공 검출"""
    if frame is None or frame.size == 0:
        return None, None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness > 150:
        threshold_low = max(180, mean_brightness - 30)
    else:
        threshold_low = max(120, mean_brightness + 20)
    threshold_high = 255
    
    binary = cv2.inRange(gray, threshold_low, threshold_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_contour = None
    best_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 3000:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.5:
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = gray[y:y+h, x:x+w]
                    mean_brightness_roi = np.mean(roi)
                    if mean_brightness_roi > threshold_low * 0.8:
                        score = circularity * (area / 100) * (mean_brightness_roi / 255)
                        if score > best_score:
                            best_score = score
                            best_contour = contour
    
    if best_contour is not None and best_score > 0.1:
        (x, y), radius = cv2.minEnclosingCircle(best_contour)
        center = (int(x), int(y))
        return center, int(radius)
    
    return None, None

def debug_stereo_matching():
    """스테레오 매칭 디버깅"""
    driver_dir = "data2/driver/2"
    
    # 이미지 파일 목록
    pattern_cam1 = os.path.join(driver_dir, "Cam1_*.bmp")
    pattern_cam2 = os.path.join(driver_dir, "Cam2_*.bmp")
    files_cam1 = sorted(glob.glob(pattern_cam1), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    files_cam2 = sorted(glob.glob(pattern_cam2), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    print(f"Found {len(files_cam1)} Cam1 images")
    print(f"Found {len(files_cam2)} Cam2 images")
    
    # 처음 몇 프레임만 디버깅
    for i in range(min(5, len(files_cam1), len(files_cam2))):
        filename1 = os.path.basename(files_cam1[i])
        filename2 = os.path.basename(files_cam2[i])
        frame_num1 = int(filename1.split('_')[1].split('.')[0])
        frame_num2 = int(filename2.split('_')[1].split('.')[0])
        
        print(f"\nFrame {frame_num1}:")
        print(f"  Cam1 file: {filename1}")
        print(f"  Cam2 file: {filename2}")
        
        # 이미지 로드
        img1 = cv2.imread(files_cam1[i])
        img2 = cv2.imread(files_cam2[i])
        
        if img1 is None or img2 is None:
            print(f"  Failed to load images")
            continue
        
        print(f"  Cam1 size: {img1.shape}")
        print(f"  Cam2 size: {img2.shape}")
        
        # 골프공 검출
        center1, radius1 = detect_golf_ball_adaptive(img1)
        center2, radius2 = detect_golf_ball_adaptive(img2)
        
        print(f"  Cam1 detection: {center1}, radius: {radius1}")
        print(f"  Cam2 detection: {center2}, radius: {radius2}")
        
        if center1 is not None and center2 is not None:
            u1, v1 = center1
            u2, v2 = center2
            
            print(f"  Cam1 center: ({u1}, {v1})")
            print(f"  Cam2 center: ({u2}, {v2})")
            print(f"  X disparity: {u1 - u2}")
            print(f"  Y disparity: {v1 - v2}")
            print(f"  Y disparity abs: {abs(v1 - v2)}")
            
            # 깊이 계산 시뮬레이션
            focal_length = 1200
            baseline = 470
            
            for disparity_type in ['X', 'Y']:
                if disparity_type == 'X':
                    disparity = u1 - u2
                else:
                    disparity = abs(v1 - v2)
                
                if disparity > 0:
                    depth = (focal_length * baseline) / disparity
                    print(f"  {disparity_type} disparity depth: {depth:.2f}mm")
                else:
                    print(f"  {disparity_type} disparity: {disparity} (invalid)")

if __name__ == "__main__":
    debug_stereo_matching()
