#!/usr/bin/env python3
"""
체스보드 검출 테스트 및 디버깅
"""

import cv2
import numpy as np
import glob
import os

def test_chessboard_detection():
    """체스보드 검출 테스트"""
    calibration_path = "data2/Calibration_image_1025"
    
    # 다양한 체스보드 패턴 크기
    patterns = [
        (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4),
        (5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5),
        (6, 6), (7, 6), (8, 6), (9, 6), (10, 6),
        (7, 7), (8, 7), (9, 7), (10, 7),
        (8, 8), (9, 8), (10, 8), (11, 8)
    ]
    
    # 이미지 로드
    cam1_files = sorted(glob.glob(os.path.join(calibration_path, "Cam1_*.bmp")))
    cam2_files = sorted(glob.glob(os.path.join(calibration_path, "Cam2_*.bmp")))
    
    print(f"Cam1 이미지: {len(cam1_files)}개")
    print(f"Cam2 이미지: {len(cam2_files)}개")
    
    if len(cam1_files) == 0:
        print("이미지를 찾을 수 없습니다.")
        return
    
    # 첫 번째 이미지로 테스트
    img1 = cv2.imread(cam1_files[0])
    img2 = cv2.imread(cam2_files[0])
    
    if img1 is None or img2 is None:
        print("이미지 로드 실패")
        return
    
    print(f"이미지 크기: {img1.shape}")
    
    # 다양한 이미지 전처리 방법
    def enhance_image(img, method):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if method == 0:
            return gray
        elif method == 1:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        elif method == 2:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        elif method == 3:
            clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        elif method == 4:
            return cv2.equalizeHist(gray)
        elif method == 5:
            gamma = 0.5
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                    for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(gray, lookup_table)
        elif method == 6:
            gamma = 0.7
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                    for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(gray, lookup_table)
        elif method == 7:
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            return clahe.apply(blurred)
        elif method == 8:
            gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
            return cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        else:
            return gray
    
    # 각 카메라별로 테스트
    for cam_name, img in [("Cam1", img1), ("Cam2", img2)]:
        print(f"\n{cam_name} 체스보드 검출 테스트:")
        
        best_pattern = None
        best_method = 0
        best_count = 0
        
        for pattern in patterns:
            print(f"  패턴 {pattern}:")
            
            for method in range(9):
                enhanced = enhance_image(img, method)
                
                # 다양한 플래그로 시도
                flags_list = [
                    None,
                    cv2.CALIB_CB_ADAPTIVE_THRESH,
                    cv2.CALIB_CB_NORMALIZE_IMAGE,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                    cv2.CALIB_CB_FILTER_QUADS
                ]
                
                for flag_idx, flag in enumerate(flags_list):
                    if flag is None:
                        ret, corners = cv2.findChessboardCorners(enhanced, pattern, None)
                    else:
                        ret, corners = cv2.findChessboardCorners(enhanced, pattern, flag)
                    
                    if ret:
                        print(f"    OK 방법 {method}, 플래그 {flag_idx}")
                        
                        # 시각화
                        img_with_corners = img.copy()
                        cv2.drawChessboardCorners(img_with_corners, pattern, corners, ret)
                        
                        output_path = f"debug_chessboard_{cam_name}_{pattern[0]}x{pattern[1]}_method{method}_flag{flag_idx}.jpg"
                        cv2.imwrite(output_path, img_with_corners)
                        
                        if best_count == 0:
                            best_pattern = pattern
                            best_method = method
                            best_count = 1
                        
                        break
                
                if ret:
                    break
            
            if not ret:
                print(f"    X 모든 방법 실패")
        
        if best_pattern:
            print(f"  최적 패턴: {best_pattern}, 방법: {best_method}")
        else:
            print(f"  X 체스보드 검출 실패")

if __name__ == "__main__":
    test_chessboard_detection()
