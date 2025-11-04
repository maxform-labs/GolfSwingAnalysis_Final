"""
체스보드 이미지를 사용하여 스테레오 캘리브레이션 재수행
기존 캘리브레이션의 문제점 검증 및 올바른 파라미터 획득
"""

import cv2
import numpy as np
import glob
import json
import os
from pathlib import Path

def find_chessboard_corners(images, pattern_size=(9, 6), show_progress=True):
    """
    체스보드 코너 검출
    
    Args:
        images: 이미지 파일 경로 리스트
        pattern_size: 체스보드 내부 코너 개수 (columns, rows)
        show_progress: 진행 상황 출력 여부
    
    Returns:
        objpoints: 3D 점들의 리스트
        imgpoints: 2D 이미지 점들의 리스트
        valid_images: 성공적으로 검출된 이미지 경로 리스트
    """
    # 체스보드 실제 좌표 (z=0 평면)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    # 체스보드 정사각형 크기 (mm 단위)
    square_size = 25.0  # 25mm = 2.5cm
    objp *= square_size
    
    objpoints = []  # 3D 점
    imgpoints = []  # 2D 점
    valid_images = []
    
    print(f"\n체스보드 코너 검출 시작...")
    print(f"패턴 크기: {pattern_size}")
    print(f"정사각형 크기: {square_size}mm")
    print(f"처리할 이미지 수: {len(images)}")
    
    # 코너 검출 기준 개선
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"  [{i+1}/{len(images)}] ✗ 이미지 로드 실패: {os.path.basename(fname)}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 체스보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            objpoints.append(objp)
            
            # 서브픽셀 정확도로 코너 위치 개선
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(corners_refined)
            valid_images.append(fname)
            
            if show_progress:
                print(f"  [{i+1}/{len(images)}] ✓ 성공: {os.path.basename(fname)}")
        else:
            if show_progress:
                print(f"  [{i+1}/{len(images)}] ✗ 실패: {os.path.basename(fname)}")
    
    print(f"\n검출 완료: {len(valid_images)}/{len(images)} 성공")
    
    return objpoints, imgpoints, valid_images

def calibrate_single_camera(objpoints, imgpoints, image_size):
    """
    단일 카메라 캘리브레이션
    
    Returns:
        ret: RMS re-projection error
        camera_matrix: 카메라 내부 파라미터 행렬
        dist_coeffs: 왜곡 계수
        rvecs: 회전 벡터들
        tvecs: 이동 벡터들
    """
    print("\n단일 카메라 캘리브레이션 수행 중...")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None,
        flags=cv2.CALIB_RATIONAL_MODEL
    )
    
    print(f"RMS re-projection error: {ret:.4f} pixels")
    print(f"카메라 행렬:\n{camera_matrix}")
    print(f"왜곡 계수: {dist_coeffs.ravel()}")
    
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

def stereo_calibrate(objpoints, imgpoints1, imgpoints2, camera_matrix1, dist_coeffs1,
                     camera_matrix2, dist_coeffs2, image_size):
    """
    스테레오 캘리브레이션
    
    Returns:
        ret: RMS re-projection error
        camera_matrix1, camera_matrix2: 개선된 카메라 행렬들
        dist_coeffs1, dist_coeffs2: 개선된 왜곡 계수들
        R: 회전 행렬 (카메라1에서 카메라2로)
        T: 이동 벡터 (카메라1에서 카메라2로)
        E: Essential 행렬
        F: Fundamental 행렬
    """
    print("\n스테레오 캘리브레이션 수행 중...")
    
    # 스테레오 캘리브레이션 기준
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F = \
        cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            camera_matrix1, dist_coeffs1,
            camera_matrix2, dist_coeffs2,
            image_size,
            criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
    
    print(f"\nRMS re-projection error: {ret:.4f} pixels")
    print(f"\n회전 행렬 R:")
    print(R)
    print(f"\n이동 벡터 T (mm):")
    print(T.ravel())
    
    # 베이스라인 계산
    baseline = np.linalg.norm(T)
    print(f"\n베이스라인 거리: {baseline:.2f} mm")
    
    return ret, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F

def compute_rectification(camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2,
                          image_size, R, T):
    """
    스테레오 정류(rectification) 계산
    
    Returns:
        R1, R2: 각 카메라의 정류 회전 행렬
        P1, P2: 정류된 좌표계에서의 투영 행렬
        Q: disparity-to-depth 매핑 행렬
    """
    print("\n스테레오 정류 계산 중...")
    
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        camera_matrix1, dist_coeffs1,
        camera_matrix2, dist_coeffs2,
        image_size, R, T,
        alpha=0  # 0 = 유효 픽셀만, 1 = 모든 픽셀 포함
    )
    
    print(f"정류 회전 행렬 R1:")
    print(R1)
    print(f"\n정류 회전 행렬 R2:")
    print(R2)
    print(f"\nQ 행렬 (disparity-to-depth):")
    print(Q)
    
    return R1, R2, P1, P2, Q

def analyze_calibration_quality(objpoints, imgpoints, camera_matrix, dist_coeffs, 
                                rvecs, tvecs, camera_name):
    """
    캘리브레이션 품질 분석
    """
    print(f"\n{'='*60}")
    print(f"{camera_name} 캘리브레이션 품질 분석")
    print(f"{'='*60}")
    
    total_error = 0
    errors_per_image = []
    
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors_per_image.append(error)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    max_error = max(errors_per_image)
    min_error = min(errors_per_image)
    
    print(f"평균 re-projection error: {mean_error:.4f} pixels")
    print(f"최대 error: {max_error:.4f} pixels")
    print(f"최소 error: {min_error:.4f} pixels")
    
    # 초점 거리 분석
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print(f"\n카메라 내부 파라미터:")
    print(f"  fx = {fx:.2f} pixels")
    print(f"  fy = {fy:.2f} pixels")
    print(f"  cx = {cx:.2f} pixels")
    print(f"  cy = {cy:.2f} pixels")
    print(f"  종횡비 (fx/fy) = {fx/fy:.4f}")
    
    # 왜곡 계수 분석
    print(f"\n왜곡 계수:")
    if len(dist_coeffs) >= 5:
        k1, k2, p1, p2, k3 = dist_coeffs.ravel()[:5]
        print(f"  k1 (radial) = {k1:.6f}")
        print(f"  k2 (radial) = {k2:.6f}")
        print(f"  p1 (tangential) = {p1:.6f}")
        print(f"  p2 (tangential) = {p2:.6f}")
        print(f"  k3 (radial) = {k3:.6f}")
        
        if abs(k1) < 1e-6 and abs(k2) < 1e-6 and abs(p1) < 1e-6 and abs(p2) < 1e-6:
            print("  ⚠️ 경고: 모든 왜곡 계수가 거의 0 - 비현실적!")
    
    return mean_error, errors_per_image

def save_calibration_results(output_file, camera_matrix1, dist_coeffs1, camera_matrix2, 
                             dist_coeffs2, R, T, E, F, R1, R2, P1, P2, Q, 
                             image_size, rms_error, cam1_error, cam2_error):
    """
    캘리브레이션 결과를 JSON 파일로 저장
    """
    print(f"\n캘리브레이션 결과 저장 중: {output_file}")
    
    calibration_data = {
        "calibration_method": "OpenCV stereoCalibrate with CALIB_FIX_INTRINSIC",
        "chessboard_pattern": "9x6 inner corners",
        "square_size_mm": 25.0,
        "image_size": list(image_size),
        "rms_reprojection_error": float(rms_error),
        "camera1_mean_error": float(cam1_error),
        "camera2_mean_error": float(cam2_error),
        
        "camera_matrix_1": camera_matrix1.tolist(),
        "distortion_coeffs_1": dist_coeffs1.ravel().tolist(),
        "camera_matrix_2": camera_matrix2.tolist(),
        "distortion_coeffs_2": dist_coeffs2.ravel().tolist(),
        
        "rotation_matrix": R.tolist(),
        "translation_vector": T.ravel().tolist(),
        "baseline_mm": float(np.linalg.norm(T)),
        
        "essential_matrix": E.tolist(),
        "fundamental_matrix": F.tolist(),
        
        "rectification_R1": R1.tolist(),
        "rectification_R2": R2.tolist(),
        "projection_P1": P1.tolist(),
        "projection_P2": P2.tolist(),
        "disparity_to_depth_Q": Q.tolist(),
        
        "notes": [
            "Recalibrated on 2025-11-03",
            "Using 17 stereo pairs of chessboard images",
            "Rational distortion model used",
            "ROI offsets: Camera1 Y=396, Camera2 Y=372"
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"✓ 저장 완료")

def compare_with_old_calibration(old_file, new_camera_matrix1, new_dist_coeffs1,
                                 new_camera_matrix2, new_dist_coeffs2, new_R, new_T):
    """
    기존 캘리브레이션과 새 캘리브레이션 비교
    """
    print(f"\n{'='*60}")
    print("기존 vs 새 캘리브레이션 비교")
    print(f"{'='*60}")
    
    if not os.path.exists(old_file):
        print(f"⚠️ 기존 캘리브레이션 파일이 없습니다: {old_file}")
        return
    
    with open(old_file, 'r') as f:
        old_calib = json.load(f)
    
    old_K1 = np.array(old_calib.get('camera_matrix_1', old_calib.get('K1', [[1500, 0, 720], [0, 1500, 540], [0, 0, 1]])))
    old_D1 = np.array(old_calib.get('distortion_coeffs_1', old_calib.get('D1', [0, 0, 0, 0, 0])))
    old_K2 = np.array(old_calib.get('camera_matrix_2', old_calib.get('K2', [[1500, 0, 720], [0, 1500, 540], [0, 0, 1]])))
    old_D2 = np.array(old_calib.get('distortion_coeffs_2', old_calib.get('D2', [0, 0, 0, 0, 0])))
    old_R = np.array(old_calib.get('rotation_matrix', old_calib.get('R', [[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
    old_T = np.array(old_calib.get('translation_vector', old_calib.get('T', [[0], [470], [0]])))
    
    print("\n[Camera 1 비교]")
    print(f"초점 거리 (fx):")
    print(f"  기존: {old_K1[0, 0]:.2f} pixels")
    print(f"  새로: {new_camera_matrix1[0, 0]:.2f} pixels")
    print(f"  차이: {abs(new_camera_matrix1[0, 0] - old_K1[0, 0]):.2f} pixels")
    
    print(f"\n주점 (cx, cy):")
    print(f"  기존: ({old_K1[0, 2]:.2f}, {old_K1[1, 2]:.2f})")
    print(f"  새로: ({new_camera_matrix1[0, 2]:.2f}, {new_camera_matrix1[1, 2]:.2f})")
    
    print(f"\n왜곡 계수 k1:")
    print(f"  기존: {old_D1.ravel()[0]:.6f}")
    print(f"  새로: {new_dist_coeffs1.ravel()[0]:.6f}")
    if abs(old_D1.ravel()[0]) < 1e-6:
        print(f"  ⚠️ 기존 값이 0 - 왜곡 보정 안 됨!")
    
    print("\n[회전 행렬 R 비교]")
    print("기존 R:")
    print(old_R)
    print("\n새로운 R:")
    print(new_R)
    
    if np.allclose(old_R, np.eye(3), atol=0.01):
        print("⚠️ 기존 R이 단위 행렬 - 카메라 회전 미반영!")
    
    rotation_diff = np.linalg.norm(new_R - old_R, 'fro')
    print(f"\nR 차이 (Frobenius norm): {rotation_diff:.4f}")
    
    print("\n[이동 벡터 T 비교]")
    print(f"기존 T: {old_T.ravel()}")
    print(f"새로 T: {new_T.ravel()}")
    print(f"기존 베이스라인: {np.linalg.norm(old_T):.2f} mm")
    print(f"새로 베이스라인: {np.linalg.norm(new_T):.2f} mm")
    print(f"차이: {abs(np.linalg.norm(new_T) - np.linalg.norm(old_T)):.2f} mm")

def main():
    """
    메인 실행 함수
    """
    print("="*60)
    print("스테레오 캘리브레이션 재수행")
    print("="*60)
    
    # 체스보드 이미지 경로
    cam1_images = sorted(glob.glob("chessboard_images/Cam1_*.bmp"))
    cam2_images = sorted(glob.glob("chessboard_images/Cam2_*.bmp"))
    
    if len(cam1_images) == 0 or len(cam2_images) == 0:
        print("❌ 체스보드 이미지를 찾을 수 없습니다!")
        print(f"   Camera1: {len(cam1_images)}개")
        print(f"   Camera2: {len(cam2_images)}개")
        return
    
    print(f"\n발견된 이미지:")
    print(f"  Camera1: {len(cam1_images)}개")
    print(f"  Camera2: {len(cam2_images)}개")
    
    # 이미지 크기 확인
    sample_img = cv2.imread(cam1_images[0])
    if sample_img is None:
        print(f"❌ 이미지 로드 실패: {cam1_images[0]}")
        return
    
    image_size = (sample_img.shape[1], sample_img.shape[0])
    print(f"\n이미지 크기: {image_size[0]} × {image_size[1]}")
    
    # 체스보드 패턴 크기 (내부 코너 개수)
    pattern_size = (9, 6)  # columns × rows
    
    # ========== Camera 1 코너 검출 ==========
    print(f"\n{'='*60}")
    print("Camera 1 체스보드 코너 검출")
    print(f"{'='*60}")
    objpoints1, imgpoints1, valid_images1 = find_chessboard_corners(
        cam1_images, pattern_size, show_progress=True
    )
    
    if len(valid_images1) < 10:
        print(f"❌ Camera 1: 충분한 이미지가 없습니다 ({len(valid_images1)}/10)")
        return
    
    # ========== Camera 2 코너 검출 ==========
    print(f"\n{'='*60}")
    print("Camera 2 체스보드 코너 검출")
    print(f"{'='*60}")
    objpoints2, imgpoints2, valid_images2 = find_chessboard_corners(
        cam2_images, pattern_size, show_progress=True
    )
    
    if len(valid_images2) < 10:
        print(f"❌ Camera 2: 충분한 이미지가 없습니다 ({len(valid_images2)}/10)")
        return
    
    # ========== 매칭되는 이미지 쌍 찾기 ==========
    print(f"\n{'='*60}")
    print("스테레오 이미지 쌍 매칭")
    print(f"{'='*60}")
    
    # 파일명 기반으로 매칭
    valid_pairs = []
    objpoints_paired = []
    imgpoints1_paired = []
    imgpoints2_paired = []
    
    for i, img1 in enumerate(valid_images1):
        basename1 = os.path.basename(img1).replace('Cam1_', '').replace('.bmp', '')
        for j, img2 in enumerate(valid_images2):
            basename2 = os.path.basename(img2).replace('Cam2_', '').replace('.bmp', '')
            if basename1 == basename2:
                valid_pairs.append((img1, img2))
                objpoints_paired.append(objpoints1[i])
                imgpoints1_paired.append(imgpoints1[i])
                imgpoints2_paired.append(imgpoints2[j])
                print(f"  ✓ 쌍 {len(valid_pairs)}: {basename1}")
                break
    
    print(f"\n매칭된 스테레오 쌍: {len(valid_pairs)}개")
    
    if len(valid_pairs) < 10:
        print(f"❌ 충분한 스테레오 쌍이 없습니다 ({len(valid_pairs)}/10)")
        return
    
    # ========== Camera 1 캘리브레이션 ==========
    print(f"\n{'='*60}")
    print("Camera 1 단일 캘리브레이션")
    print(f"{'='*60}")
    ret1, camera_matrix1, dist_coeffs1, rvecs1, tvecs1 = calibrate_single_camera(
        objpoints_paired, imgpoints1_paired, image_size
    )
    cam1_error, errors1 = analyze_calibration_quality(
        objpoints_paired, imgpoints1_paired, camera_matrix1, dist_coeffs1,
        rvecs1, tvecs1, "Camera 1"
    )
    
    # ========== Camera 2 캘리브레이션 ==========
    print(f"\n{'='*60}")
    print("Camera 2 단일 캘리브레이션")
    print(f"{'='*60}")
    ret2, camera_matrix2, dist_coeffs2, rvecs2, tvecs2 = calibrate_single_camera(
        objpoints_paired, imgpoints2_paired, image_size
    )
    cam2_error, errors2 = analyze_calibration_quality(
        objpoints_paired, imgpoints2_paired, camera_matrix2, dist_coeffs2,
        rvecs2, tvecs2, "Camera 2"
    )
    
    # ========== 스테레오 캘리브레이션 ==========
    print(f"\n{'='*60}")
    print("스테레오 캘리브레이션")
    print(f"{'='*60}")
    ret_stereo, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F = \
        stereo_calibrate(
            objpoints_paired, imgpoints1_paired, imgpoints2_paired,
            camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2,
            image_size
        )
    
    # ========== 스테레오 정류 ==========
    R1, R2, P1, P2, Q = compute_rectification(
        camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2,
        image_size, R, T
    )
    
    # ========== 결과 저장 ==========
    output_file = "recalibrated_stereo_calibration.json"
    save_calibration_results(
        output_file, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2,
        R, T, E, F, R1, R2, P1, P2, Q, image_size,
        ret_stereo, cam1_error, cam2_error
    )
    
    # ========== 기존 캘리브레이션과 비교 ==========
    old_calib_files = [
        "vertical_stereo_calibration_z_axis.json",
        "precise_vertical_stereo_calibration.json"
    ]
    
    for old_file in old_calib_files:
        if os.path.exists(old_file):
            compare_with_old_calibration(
                old_file, camera_matrix1, dist_coeffs1,
                camera_matrix2, dist_coeffs2, R, T
            )
            break
    
    # ========== 최종 요약 ==========
    print(f"\n{'='*60}")
    print("캘리브레이션 완료!")
    print(f"{'='*60}")
    print(f"✓ 사용된 스테레오 쌍: {len(valid_pairs)}개")
    print(f"✓ RMS re-projection error: {ret_stereo:.4f} pixels")
    print(f"✓ Camera 1 mean error: {cam1_error:.4f} pixels")
    print(f"✓ Camera 2 mean error: {cam2_error:.4f} pixels")
    print(f"✓ 베이스라인: {np.linalg.norm(T):.2f} mm")
    print(f"✓ 결과 파일: {output_file}")
    
    # 품질 평가
    print(f"\n품질 평가:")
    if ret_stereo < 0.5:
        print(f"  ✅ 우수 (< 0.5 pixels)")
    elif ret_stereo < 1.0:
        print(f"  ✓ 양호 (< 1.0 pixels)")
    elif ret_stereo < 2.0:
        print(f"  ⚠️ 보통 (< 2.0 pixels)")
    else:
        print(f"  ❌ 불량 (≥ 2.0 pixels)")
    
    # R = Identity 체크
    if np.allclose(R, np.eye(3), atol=0.01):
        print(f"  ⚠️ 경고: 회전 행렬이 단위 행렬에 가까움 - 카메라 정렬 확인 필요")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
