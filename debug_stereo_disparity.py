#!/usr/bin/env python3
"""
스테레오 시차 디버깅
"""

import cv2
import numpy as np
import os
import glob

def detect_golf_ball_adaptive(frame):
    """완화된 조건의 골프공 검출"""
    if frame is None or frame.size == 0:
        return None, None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness > 150:
        threshold_low = max(120, mean_brightness - 30)
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
        if 10 < area < 5000:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3:
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = gray[y:y+h, x:x+w]
                    mean_brightness_roi = np.mean(roi)
                    if mean_brightness_roi > threshold_low * 0.5:
                        score = circularity * (area / 100) * (mean_brightness_roi / 255)
                        if score > best_score:
                            best_score = score
                            best_contour = contour
    
    if best_contour is not None and best_score > 0.05:
        (x, y), radius = cv2.minEnclosingCircle(best_contour)
        center = (int(x), int(y))
        return center, int(radius)
    
    return None, None

def debug_stereo_disparity():
    """스테레오 시차 디버깅"""
    driver_dir = "data2/driver/2"
    
    # 이미지 파일 목록
    pattern_cam1 = os.path.join(driver_dir, "Cam1_*.bmp")
    pattern_cam2 = os.path.join(driver_dir, "Cam2_*.bmp")
    files_cam1 = sorted(glob.glob(pattern_cam1), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    files_cam2 = sorted(glob.glob(pattern_cam2), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    print(f"Found {len(files_cam1)} Cam1 images")
    print(f"Found {len(files_cam2)} Cam2 images")
    
    # 처음 5개 프레임 디버깅
    for i in range(min(5, len(files_cam1), len(files_cam2))):
        filename1 = os.path.basename(files_cam1[i])
        filename2 = os.path.basename(files_cam2[i])
        frame_num1 = int(filename1.split('_')[1].split('.')[0])
        frame_num2 = int(filename2.split('_')[1].split('.')[0])
        
        print(f"\n=== Frame {frame_num1} ===")
        
        # 이미지 로드
        img1 = cv2.imread(files_cam1[i])
        img2 = cv2.imread(files_cam2[i])
        
        if img1 is None or img2 is None:
            print(f"Failed to load images")
            continue
        
        # 골프공 검출
        center1, radius1 = detect_golf_ball_adaptive(img1)
        center2, radius2 = detect_golf_ball_adaptive(img2)
        
        print(f"Cam1 detection: {center1}, radius: {radius1}")
        print(f"Cam2 detection: {center2}, radius: {radius2}")
        
        if center1 is not None and center2 is not None:
            u1, v1 = center1
            u2, v2 = center2
            
            print(f"Cam1 center: ({u1}, {v1})")
            print(f"Cam2 center: ({u2}, {v2})")
            print(f"X disparity (u1-u2): {u1 - u2}")
            print(f"Y disparity (v1-v2): {v1 - v2}")
            print(f"Y disparity abs: {abs(v1 - v2)}")
            
            # 깊이 계산 시뮬레이션
            focal_length = 1200
            baseline = 470
            
            # X 시차로 깊이 계산
            x_disparity = u1 - u2
            if x_disparity > 0:
                x_depth = (focal_length * baseline) / x_disparity
                print(f"X disparity depth: {x_depth:.2f}mm")
            else:
                print(f"X disparity: {x_disparity} (invalid)")
            
            # Y 시차로 깊이 계산
            y_disparity = abs(v1 - v2)
            if y_disparity > 0:
                y_depth = (focal_length * baseline) / y_disparity
                print(f"Y disparity depth: {y_depth:.2f}mm")
            else:
                print(f"Y disparity: {y_disparity} (invalid)")
            
            # 3D 좌표 계산 (X 시차 사용)
            if x_disparity > 0:
                depth = x_depth
                x_3d = (u1 - 720) * depth / focal_length  # 720은 주점
                y_3d = (v1 - 540) * depth / focal_length  # 540은 주점
                z_3d = depth
                print(f"3D position (X disparity): ({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f})")
            
            # 3D 좌표 계산 (Y 시차 사용)
            if y_disparity > 0:
                depth = y_depth
                x_3d = (u1 - 720) * depth / focal_length
                y_3d = (v1 - 540) * depth / focal_length
                z_3d = depth
                print(f"3D position (Y disparity): ({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f})")
        else:
            print(f"One or both cameras failed to detect")

if __name__ == "__main__":
    debug_stereo_disparity()
