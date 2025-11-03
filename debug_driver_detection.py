#!/usr/bin/env python3
"""
드라이버 샷 검출 디버깅
왜 3D 위치를 찾지 못하는지 진단
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

def debug_driver_detection():
    """드라이버 샷 검출 디버깅"""
    print("=== DEBUGGING DRIVER DETECTION ===")
    
    # 샷 1의 첫 번째 프레임들 확인
    shot_dir = "data2/driver/1"
    
    # 이미지 파일 목록
    pattern_cam1 = os.path.join(shot_dir, "Cam1_*.bmp")
    pattern_cam2 = os.path.join(shot_dir, "Cam2_*.bmp")
    files_cam1 = sorted(glob.glob(pattern_cam1), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    files_cam2 = sorted(glob.glob(pattern_cam2), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    print(f"Found {len(files_cam1)} Cam1 images, {len(files_cam2)} Cam2 images")
    
    # 첫 5개 프레임 확인
    for i in range(min(5, len(files_cam1), len(files_cam2))):
        print(f"\n--- Frame {i+1} ---")
        
        img1 = cv2.imread(files_cam1[i])
        img2 = cv2.imread(files_cam2[i])
        
        if img1 is not None and img2 is not None:
            print(f"  Cam1 shape: {img1.shape}")
            print(f"  Cam2 shape: {img2.shape}")
            print(f"  Cam1 mean brightness: {np.mean(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)):.1f}")
            print(f"  Cam2 mean brightness: {np.mean(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)):.1f}")
            
            # 골프공 검출 시도
            center1, radius1 = detect_golf_ball_adaptive(img1)
            center2, radius2 = detect_golf_ball_adaptive(img2)
            
            print(f"  Cam1 detection: {center1}, radius: {radius1}")
            print(f"  Cam2 detection: {center2}, radius: {radius2}")
            
            if center1 is not None and center2 is not None:
                # 시차 계산
                y_disparity = abs(center1[1] - center2[1])  # Y 시차
                x_disparity = abs(center1[0] - center2[0])  # X 시차
                print(f"  Y disparity: {y_disparity}")
                print(f"  X disparity: {x_disparity}")
                
                if y_disparity > 1:
                    print(f"  Valid Y disparity for 3D calculation")
                else:
                    print(f"  Y disparity too small for 3D calculation")
                
                if x_disparity > 1:
                    print(f"  Valid X disparity for 3D calculation")
                else:
                    print(f"  X disparity too small for 3D calculation")
            else:
                print(f"  No detection in one or both cameras")
        else:
            print(f"  Could not load images")

def detect_golf_ball_adaptive(frame: np.ndarray) -> tuple:
    """적응형 골프공 검출"""
    if frame is None or frame.size == 0:
        return None, None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    print(f"    Mean brightness: {mean_brightness:.1f}")
    
    if mean_brightness > 150:
        threshold_low = max(120, mean_brightness - 30)
    else:
        threshold_low = max(120, mean_brightness + 20)
    threshold_high = 255
    
    print(f"    Threshold range: {threshold_low}-{threshold_high}")
    
    binary = cv2.inRange(gray, threshold_low, threshold_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"    Found {len(contours)} contours")
    
    best_contour = None
    best_score = 0
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if 50 < area < 10000:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3:
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = gray[y:y+h, x:x+w]
                    mean_brightness_roi = np.mean(roi)
                    if mean_brightness_roi > threshold_low * 0.5:
                        score = circularity * (area / 100) * (mean_brightness_roi / 255)
                        print(f"      Contour {i}: area={area:.0f}, circularity={circularity:.2f}, score={score:.3f}")
                        if score > best_score:
                            best_score = score
                            best_contour = contour
    
    if best_contour is not None and best_score > 0.05:
        (x, y), radius = cv2.minEnclosingCircle(best_contour)
        center = (int(x), int(y))
        print(f"    Best detection: center={center}, radius={int(radius)}, score={best_score:.3f}")
        return center, int(radius)
    
    print(f"    No valid detection (best score: {best_score:.3f})")
    return None, None

if __name__ == "__main__":
    debug_driver_detection()
