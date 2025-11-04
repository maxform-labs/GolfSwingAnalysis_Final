"""
부분 체스보드 허용 캘리브레이션
각 이미지에서 검출 가능한 최대 패턴 사용
스테레오 페어가 다른 패턴 크기여도 OK
"""

import cv2
import numpy as np
import glob
import os
import json
from collections import Counter

def enhance_for_detection(img):
    """체스보드 검출을 위한 전처리"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    # 감마 보정
    gamma = 2.0
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)
    
    # Bilateral 필터
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    methods = [
        ('original', gray),
        ('clahe', clahe.apply(gray)),
        ('hist_eq', cv2.equalizeHist(gray)),
        ('enhanced', cv2.convertScaleAbs(gray, alpha=2.0, beta=50)),
        ('gamma', gamma_corrected),
        ('bilateral_clahe', clahe.apply(bilateral)),
        ('ultra_bright', cv2.convertScaleAbs(gray, alpha=3.0, beta=100)),
    ]
    
    return methods

def try_chessboard_detection(gray_img):
    """큰 패턴부터 시도"""
    patterns = [
        (11, 8), (10, 8), (11, 7), (10, 7),
        (9, 8), (9, 7), (8, 8), (8, 7),
        (9, 6), (8, 6), (7, 6), (6, 6),
        (8, 5), (7, 5), (6, 5), (5, 5),
        (8, 4), (7, 4), (6, 4), (5, 4), (4, 4)
    ]
    
    flags_list = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        None,
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

def collect_chessboard_corners():
    """
    모든 이미지에서 체스보드 코너 수집
    각 이미지가 다른 패턴 크기를 가져도 OK
    """
    cam1_images = sorted(glob.glob("chessboard_images/Cam1_*.bmp"))
    cam2_images = sorted(glob.glob("chessboard_images/Cam2_*.bmp"))
    
    print("="*70)
    print("부분 체스보드 허용 캘리브레이션")
    print("="*70)
    print("각 이미지에서 가능한 최대 패턴 검출")
    print()
    
    cam1_data = []
    cam2_data = []
    
    # Camera 1
    print("Camera 1 처리 중...")
    for img_path in cam1_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        basename = os.path.basename(img_path)
        
        methods = enhance_for_detection(img)
        found = False
        
        for method_name, gray in methods:
            pattern, corners, success = try_chessboard_detection(gray)
            if success:
                # 3D 객체 포인트 생성
                objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
                objp *= 25.0  # 25mm 정사각형
                
                cam1_data.append({
                    'basename': basename,
                    'pattern': pattern,
                    'corners': corners,
                    'objpoints': objp,
                    'gray': gray,
                    'method': method_name
                })
                print(f"  ✓ {basename}: {pattern} ({method_name})")
                found = True
                break
        
        if not found:
            print(f"  ✗ {basename}: 실패")
    
    # Camera 2
    print("\nCamera 2 처리 중...")
    for img_path in cam2_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        basename = os.path.basename(img_path)
        
        methods = enhance_for_detection(img)
        found = False
        
        for method_name, gray in methods:
            pattern, corners, success = try_chessboard_detection(gray)
            if success:
                objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
                objp *= 25.0
                
                cam2_data.append({
                    'basename': basename,
                    'pattern': pattern,
                    'corners': corners,
                    'objpoints': objp,
                    'gray': gray,
                    'method': method_name
                })
                print(f"  ✓ {basename}: {pattern} ({method_name})")
                found = True
                break
        
        if not found:
            print(f"  ✗ {basename}: 실패")
    
    print(f"\n검출 결과:")
    print(f"  Camera 1: {len(cam1_data)}개")
    print(f"  Camera 2: {len(cam2_data)}개")
    
    return cam1_data, cam2_data

def calibrate_flexible_pattern():
    """
    유연한 패턴 크기로 캘리브레이션
    """
    cam1_data, cam2_data = collect_chessboard_corners()
    
    if len(cam1_data) < 10 or len(cam2_data) < 10:
        print(f"\n❌ 충분한 이미지가 없습니다")
        print(f"   Camera 1: {len(cam1_data)}/10")
        print(f"   Camera 2: {len(cam2_data)}/10")
        return
    
    print(f"\n{'='*70}")
    print("개별 카메라 캘리브레이션")
    print(f"{'='*70}")
    
    # Camera 1 캘리브레이션
    print("\nCamera 1 캘리브레이션...")
    objpoints1 = [d['objpoints'] for d in cam1_data]
    imgpoints1 = [d['corners'] for d in cam1_data]
    
    img_size = (1440, 1080)
    
    ret1, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(
        objpoints1, imgpoints1, img_size, None, None,
        flags=cv2.CALIB_RATIONAL_MODEL
    )
    
    print(f"  RMS 재투영 오차: {ret1:.4f} pixels")
    print(f"  초점거리: fx={K1[0,0]:.2f}, fy={K1[1,1]:.2f}")
    print(f"  주점: cx={K1[0,2]:.2f}, cy={K1[1,2]:.2f}")
    print(f"  왜곡: k1={D1[0,0]:.6f}, k2={D1[0,1]:.6f}")
    
    # Camera 2 캘리브레이션
    print("\nCamera 2 캘리브레이션...")
    objpoints2 = [d['objpoints'] for d in cam2_data]
    imgpoints2 = [d['corners'] for d in cam2_data]
    
    ret2, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(
        objpoints2, imgpoints2, img_size, None, None,
        flags=cv2.CALIB_RATIONAL_MODEL
    )
    
    print(f"  RMS 재투영 오차: {ret2:.4f} pixels")
    print(f"  초점거리: fx={K2[0,0]:.2f}, fy={K2[1,1]:.2f}")
    print(f"  주점: cx={K2[0,2]:.2f}, cy={K2[1,2]:.2f}")
    print(f"  왜곡: k1={D2[0,0]:.6f}, k2={D2[0,1]:.6f}")
    
    # 스테레오 캘리브레이션 - 공통 이미지만 사용
    print(f"\n{'='*70}")
    print("스테레오 캘리브레이션")
    print(f"{'='*70}")
    
    # 공통 이미지 찾기
    cam1_dict = {d['basename'].split('_')[1].split('.')[0]: d for d in cam1_data}
    cam2_dict = {d['basename'].split('_')[1].split('.')[0]: d for d in cam2_data}
    
    common_numbers = sorted(set(cam1_dict.keys()) & set(cam2_dict.keys()))
    
    print(f"\n공통 이미지 번호: {len(common_numbers)}개")
    if len(common_numbers) < 10:
        print(f"⚠️ 스테레오 캘리브레이션에 부족 ({len(common_numbers)}/10)")
        print(f"개별 카메라 캘리브레이션만 저장합니다")
        
        # 개별 캘리브레이션만 저장
        calibration = {
            "camera1": {
                "K": K1.tolist(),
                "D": D1.tolist(),
                "image_size": list(img_size),
                "rms_error": ret1,
                "num_images": len(cam1_data)
            },
            "camera2": {
                "K": K2.tolist(),
                "D": D2.tolist(),
                "image_size": list(img_size),
                "rms_error": ret2,
                "num_images": len(cam2_data)
            },
            "note": "Stereo calibration failed - not enough common images"
        }
        
        with open("individual_camera_calibration.json", "w") as f:
            json.dump(calibration, f, indent=2)
        
        print(f"\n✓ 개별 캘리브레이션 저장: individual_camera_calibration.json")
        return
    
    print(f"공통 이미지 목록:")
    for num in common_numbers[:10]:
        print(f"  [{num}] Cam1: {cam1_dict[num]['pattern']}, Cam2: {cam2_dict[num]['pattern']}")
    if len(common_numbers) > 10:
        print(f"  ... 외 {len(common_numbers)-10}개")
    
    # 스테레오용 데이터 준비
    stereo_objpoints = []
    stereo_imgpoints1 = []
    stereo_imgpoints2 = []
    
    print(f"\n스테레오 페어 데이터 준비 중...")
    for num in common_numbers:
        d1 = cam1_dict[num]
        d2 = cam2_dict[num]
        
        # 두 패턴 중 작은 것 사용
        pattern1 = d1['pattern']
        pattern2 = d2['pattern']
        
        min_pattern = (min(pattern1[0], pattern2[0]), min(pattern1[1], pattern2[1]))
        
        # 양쪽 코너를 작은 패턴 크기로 자름
        corners1 = d1['corners'][:min_pattern[0]*min_pattern[1]]
        corners2 = d2['corners'][:min_pattern[0]*min_pattern[1]]
        
        objp = np.zeros((min_pattern[0] * min_pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:min_pattern[0], 0:min_pattern[1]].T.reshape(-1, 2)
        objp *= 25.0
        
        stereo_objpoints.append(objp)
        stereo_imgpoints1.append(corners1)
        stereo_imgpoints2.append(corners2)
    
    print(f"스테레오 페어: {len(stereo_objpoints)}개")
    
    # 스테레오 캘리브레이션 실행
    print(f"\n스테레오 캘리브레이션 실행 중...")
    
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    ret_stereo, K1_stereo, D1_stereo, K2_stereo, D2_stereo, R, T, E, F = cv2.stereoCalibrate(
        stereo_objpoints, stereo_imgpoints1, stereo_imgpoints2,
        K1, D1, K2, D2, img_size,
        criteria=criteria,
        flags=stereo_flags
    )
    
    print(f"✓ 스테레오 캘리브레이션 성공!")
    print(f"  RMS 오차: {ret_stereo:.4f} pixels")
    print(f"\n회전 행렬 R:")
    print(R)
    print(f"\n변환 벡터 T (mm):")
    print(T.ravel())
    print(f"  Camera 간 거리: {np.linalg.norm(T):.2f} mm")
    
    # 정류화 (Rectification)
    print(f"\n정류화 계산 중...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )
    
    print(f"✓ 정류화 완료")
    
    # 결과 저장
    calibration = {
        "camera1": {
            "K": K1.tolist(),
            "D": D1.tolist(),
            "image_size": list(img_size),
            "rms_error": ret1,
            "num_images": len(cam1_data),
            "R": R1.tolist(),
            "P": P1.tolist()
        },
        "camera2": {
            "K": K2.tolist(),
            "D": D2.tolist(),
            "image_size": list(img_size),
            "rms_error": ret2,
            "num_images": len(cam2_data),
            "R": R2.tolist(),
            "P": P2.tolist()
        },
        "stereo": {
            "R": R.tolist(),
            "T": T.tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
            "Q": Q.tolist(),
            "rms_error": ret_stereo,
            "num_pairs": len(stereo_objpoints),
            "baseline_mm": float(np.linalg.norm(T))
        },
        "note": "Flexible pattern size calibration - each image may have different pattern size"
    }
    
    output_file = "flexible_stereo_calibration.json"
    with open(output_file, "w") as f:
        json.dump(calibration, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✓✓✓ 캘리브레이션 완료!")
    print(f"{'='*70}")
    print(f"저장: {output_file}")
    print(f"\nCamera 1: {len(cam1_data)}개 이미지, RMS {ret1:.4f} px")
    print(f"Camera 2: {len(cam2_data)}개 이미지, RMS {ret2:.4f} px")
    print(f"Stereo: {len(stereo_objpoints)}개 페어, RMS {ret_stereo:.4f} px")
    print(f"Baseline: {np.linalg.norm(T):.2f} mm")
    
    return calibration

if __name__ == "__main__":
    calibration = calibrate_flexible_pattern()
