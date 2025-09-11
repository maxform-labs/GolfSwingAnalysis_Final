#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Image Test for Ball Detection
"""

import cv2
import numpy as np
from simple_ball_detector import SimpleBallDetector

def test_single_image():
    """단일 이미지 테스트"""
    
    # 이미지 경로
    img_path = "C:/src/GolfSwingAnalysis_Final_ver8/shot-image/7iron/logo_ball-1/1_10.bmp"
    
    print(f"Testing image: {img_path}")
    
    # 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        print("ERROR: Could not load image")
        return
    
    print(f"Image loaded successfully: {img.shape}")
    
    # 볼 검출기 초기화
    detector = SimpleBallDetector()
    
    # 볼 검출 시도
    result = detector.detect_ir_ball(img, 10)
    
    if result:
        print(f"SUCCESS: Ball detected!")
        print(f"  Position: ({result.center_x:.1f}, {result.center_y:.1f})")
        print(f"  Radius: {result.radius:.1f}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Motion State: {result.motion_state}")
        
        # 결과 시각화 이미지 생성
        vis_img = img.copy()
        cv2.circle(vis_img, 
                   (int(result.center_x), int(result.center_y)), 
                   int(result.radius), 
                   (0, 255, 0), 2)
        cv2.circle(vis_img, 
                   (int(result.center_x), int(result.center_y)), 
                   2, 
                   (0, 0, 255), -1)
        
        # 결과 저장
        cv2.imwrite("test_ball_detection_result.jpg", vis_img)
        print("Result image saved: test_ball_detection_result.jpg")
        
    else:
        print("NO BALL DETECTED")
        
        # 이미지 정보 출력
        print(f"Image stats:")
        print(f"  Shape: {img.shape}")
        print(f"  Min/Max pixel values: {img.min()}/{img.max()}")
        print(f"  Mean pixel value: {img.mean():.1f}")
        
        # 그레이스케일 변환하여 히스토그램 확인
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # 밝은 픽셀 비율 확인
        bright_pixels = np.sum(gray > 200)
        total_pixels = gray.shape[0] * gray.shape[1]
        bright_ratio = bright_pixels / total_pixels
        
        print(f"  Bright pixels (>200): {bright_pixels} ({bright_ratio*100:.2f}%)")
        
        # 다른 임계값들도 테스트
        for threshold in [150, 160, 170, 180, 190]:
            _, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            thresh_pixels = np.sum(thresh_img > 0)
            ratio = thresh_pixels / total_pixels
            print(f"  Threshold {threshold}: {thresh_pixels} pixels ({ratio*100:.2f}%)")
        
        # 테스트 이미지 저장
        cv2.imwrite("test_input_image.jpg", img)
        cv2.imwrite("test_gray_image.jpg", gray)
        print("Debug images saved: test_input_image.jpg, test_gray_image.jpg")

if __name__ == "__main__":
    test_single_image()