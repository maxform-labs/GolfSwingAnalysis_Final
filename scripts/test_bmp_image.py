#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMP 이미지 테스트 및 디버그 스크립트
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_and_analyze_bmp(file_path):
    """BMP 파일 로드 및 분석"""
    print(f"\n=== BMP 파일 분석: {file_path} ===")
    
    # PIL로 로드
    try:
        with Image.open(file_path) as pil_img:
            print(f"PIL 모드: {pil_img.mode}")
            print(f"PIL 크기: {pil_img.size}")
            
            # numpy 배열로 변환
            if pil_img.mode == 'L':  # 그레이스케일
                img_array = np.array(pil_img, dtype=np.uint8)
                cv_img = img_array
            elif pil_img.mode == 'RGB':
                img_array = np.array(pil_img)
                cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                pil_img = pil_img.convert('RGB')
                img_array = np.array(pil_img)
                cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
    except Exception as e:
        print(f"PIL 로드 실패: {e}")
        return None
        
    # OpenCV로도 로드해서 비교
    try:
        cv_direct = cv2.imread(file_path)
        if cv_direct is not None:
            print(f"OpenCV 직접 로드: {cv_direct.shape}")
        else:
            print("OpenCV 직접 로드 실패")
    except:
        print("OpenCV 직접 로드 실패")
    
    print(f"최종 이미지 형태: {cv_img.shape}")
    print(f"데이터 타입: {cv_img.dtype}")
    print(f"최소/최대 값: {cv_img.min()}/{cv_img.max()}")
    
    # 그레이스케일 변환
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img.copy()
    
    print(f"그레이스케일 형태: {gray.shape}")
    print(f"픽셀 값 분포: 평균={gray.mean():.1f}, 표준편차={gray.std():.1f}")
    
    # 히스토그램 분석
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    print(f"히스토그램 피크: {np.argmax(hist)}")
    
    # 골프공 검출 시도 (다양한 파라미터)
    print("\n--- 골프공 검출 시도 ---")
    
    # 파라미터 세트들
    param_sets = [
        # (dp, minDist, param1, param2, minR, maxR)
        (1, 50, 50, 30, 20, 100),      # 기본
        (1, 30, 50, 25, 15, 150),      # 더 민감하게
        (2, 20, 40, 20, 10, 200),      # 매우 민감하게
        (1, 40, 60, 35, 25, 80),       # 더 엄격하게
    ]
    
    for i, (dp, minDist, param1, param2, minR, maxR) in enumerate(param_sets):
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
            param1=param1, param2=param2, minRadius=minR, maxRadius=maxR
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"파라미터 세트 {i+1}: {len(circles)}개 원 검출")
            for j, (x, y, r) in enumerate(circles[:3]):  # 상위 3개만
                print(f"  원 {j+1}: 중심=({x},{y}), 반지름={r}")
        else:
            print(f"파라미터 세트 {i+1}: 원 검출 안됨")
    
    return cv_img, gray

def main():
    # 테스트할 파일들
    test_files = [
        "C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/7iron/logo_ball-1/1_1.bmp",
        "C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/7iron/logo_ball-1/1_5.bmp",
        "C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1/1_5.bmp"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            result = load_and_analyze_bmp(test_file)
            if result:
                print("로드 성공!")
            else:
                print("로드 실패!")
        else:
            print(f"파일 없음: {test_file}")

if __name__ == "__main__":
    main()