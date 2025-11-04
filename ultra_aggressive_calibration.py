"""
초공격적 체스보드 검출 - 1개만 더 찾으면 됨!
실패한 이미지들에 집중: Cam1_1, Cam1_2, Cam1_3, Cam1_5, Cam1_6, Cam1_7, Cam1_9
                      Cam2_1, Cam2_2, Cam2_5, Cam2_7, Cam2_12
"""

import cv2
import numpy as np
import glob
import os
import json

def extreme_enhance(img):
    """극단적인 전처리 방법들"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    methods = {}
    
    # 기본
    methods['original'] = gray
    
    # CLAHE 다양한 파라미터
    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    methods['clahe_2.0'] = clahe1.apply(gray)
    
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    methods['clahe_4.0'] = clahe2.apply(gray)
    
    clahe3 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    methods['clahe_16x16'] = clahe3.apply(gray)
    
    # 히스토그램
    methods['hist_eq'] = cv2.equalizeHist(gray)
    
    # 밝기 조절 다양한 레벨
    methods['bright_1.5_30'] = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    methods['bright_2.0_50'] = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
    methods['bright_2.5_70'] = cv2.convertScaleAbs(gray, alpha=2.5, beta=70)
    methods['bright_3.0_100'] = cv2.convertScaleAbs(gray, alpha=3.0, beta=100)
    methods['bright_4.0_120'] = cv2.convertScaleAbs(gray, alpha=4.0, beta=120)
    
    # 감마 보정 다양한 값
    for gamma_val in [1.5, 2.0, 2.5, 3.0]:
        inv_gamma = 1.0 / gamma_val
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        methods[f'gamma_{gamma_val}'] = cv2.LUT(gray, table)
    
    # Bilateral 필터
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    methods['bilateral'] = bilateral
    methods['bilateral_clahe'] = clahe1.apply(bilateral)
    methods['bilateral_hist'] = cv2.equalizeHist(bilateral)
    
    # Gaussian blur + CLAHE
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    methods['blur_clahe'] = clahe1.apply(blur)
    
    # 극단적 조합
    extreme1 = cv2.convertScaleAbs(clahe2.apply(gray), alpha=1.5, beta=30)
    methods['extreme_combo1'] = extreme1
    
    extreme2 = cv2.equalizeHist(cv2.convertScaleAbs(gray, alpha=3.0, beta=80))
    methods['extreme_combo2'] = extreme2
    
    return methods

def try_detection(gray_img):
    """체스보드 검출"""
    patterns = [
        (11, 8), (10, 8), (9, 8), (8, 8),
        (11, 7), (10, 7), (9, 7), (8, 7),
        (10, 6), (9, 6), (8, 6), (7, 6), (6, 6),
        (9, 5), (8, 5), (7, 5), (6, 5), (5, 5),
        (9, 4), (8, 4), (7, 4), (6, 4), (5, 4), (4, 4),
        (7, 9), (6, 9), (6, 8), (5, 8), (5, 7), (5, 6)
    ]
    
    flags_list = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        None,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
    ]
    
    for pattern in patterns:
        for flags in flags_list:
            try:
                if flags is None:
                    ret, corners = cv2.findChessboardCorners(gray_img, pattern, None)
                else:
                    ret, corners = cv2.findChessboardCorners(gray_img, pattern, flags=flags)
                
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
                    return pattern, corners_refined, True
            except:
                continue
    
    return None, None, False

def process_failed_images():
    """실패한 이미지들만 집중 공략"""
    
    # 이전에 실패한 이미지들
    cam1_failed = ['Cam1_1.bmp', 'Cam1_2.bmp', 'Cam1_3.bmp', 'Cam1_5.bmp', 
                   'Cam1_6.bmp', 'Cam1_7.bmp', 'Cam1_9.bmp']
    cam2_failed = ['Cam2_1.bmp', 'Cam2_2.bmp', 'Cam2_5.bmp', 'Cam2_7.bmp', 'Cam2_12.bmp']
    
    print("="*70)
    print("초공격적 체스보드 검출 - 실패 이미지 재도전")
    print("="*70)
    print(f"목표: 공통 이미지 1개만 더 찾으면 스테레오 캘리브레이션 가능!")
    print()
    
    cam1_new = {}
    cam2_new = {}
    
    # Camera 1
    print("Camera 1 실패 이미지 재시도:")
    for basename in cam1_failed:
        img_path = f"chessboard_images/{basename}"
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        print(f"  {basename}...", end=" ")
        
        methods = extreme_enhance(img)
        found = False
        
        for method_name, gray in methods.items():
            pattern, corners, success = try_detection(gray)
            if success:
                img_number = basename.split('_')[1].split('.')[0]
                cam1_new[img_number] = {
                    'basename': basename,
                    'pattern': pattern,
                    'corners': corners,
                    'gray': gray,
                    'method': method_name
                }
                print(f"✓ 성공! {pattern} ({method_name})")
                found = True
                break
        
        if not found:
            print("✗ 여전히 실패")
    
    # Camera 2
    print("\nCamera 2 실패 이미지 재시도:")
    for basename in cam2_failed:
        img_path = f"chessboard_images/{basename}"
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        print(f"  {basename}...", end=" ")
        
        methods = extreme_enhance(img)
        found = False
        
        for method_name, gray in methods.items():
            pattern, corners, success = try_detection(gray)
            if success:
                img_number = basename.split('_')[1].split('.')[0]
                cam2_new[img_number] = {
                    'basename': basename,
                    'pattern': pattern,
                    'corners': corners,
                    'gray': gray,
                    'method': method_name
                }
                print(f"✓ 성공! {pattern} ({method_name})")
                found = True
                break
        
        if not found:
            print("✗ 여전히 실패")
    
    # 새로 찾은 공통 이미지 확인
    print("\n" + "="*70)
    print("결과")
    print("="*70)
    print(f"Camera 1 추가 검출: {len(cam1_new)}개")
    print(f"Camera 2 추가 검출: {len(cam2_new)}개")
    
    common_new = set(cam1_new.keys()) & set(cam2_new.keys())
    print(f"\n새로운 공통 이미지: {len(common_new)}개")
    
    if len(common_new) > 0:
        print("✓✓✓ 공통 이미지 발견!")
        for num in sorted(common_new):
            print(f"  [{num}] Cam1: {cam1_new[num]['pattern']}, Cam2: {cam2_new[num]['pattern']}")
        print(f"\n기존 9개 + 새로운 {len(common_new)}개 = 총 {9 + len(common_new)}개")
        if 9 + len(common_new) >= 10:
            print("✓✓✓ 스테레오 캘리브레이션 가능!")
        else:
            print(f"⚠️ 아직 {10 - (9 + len(common_new))}개 더 필요")
    else:
        print("✗ 새로운 공통 이미지를 찾지 못했습니다")
        
        # 개별 분석
        if len(cam1_new) > 0:
            print(f"\nCamera 1만 추가로 검출된 이미지: {sorted(cam1_new.keys())}")
        if len(cam2_new) > 0:
            print(f"Camera 2만 추가로 검출된 이미지: {sorted(cam2_new.keys())}")
    
    return cam1_new, cam2_new

if __name__ == "__main__":
    cam1_new, cam2_new = process_failed_images()
