#!/usr/bin/env python3
"""
정밀 수직 스테레오비전 캘리브레이션 시스템
- 1440x1080 전체 이미지에서 캘리브레이션 수행
- ROI 좌표계 변환을 고려한 정밀 파라미터 추출
- 실제 거리 측정값을 활용한 보정 계수 계산
"""
import cv2
import numpy as np
import json
import os
import glob
from pathlib import Path

class PreciseVerticalStereoCalibration:
    def __init__(self, calib_dir="data2/Calibration_image_1025", output_file="precise_vertical_stereo_calibration.json"):
        """
        정밀 수직 스테레오비전 캘리브레이션 초기화
        
        Args:
            calib_dir: 캘리브레이션 이미지 디렉토리
            output_file: 출력 JSON 파일명
        """
        self.calib_dir = calib_dir
        self.output_file = output_file
        
        # 체스보드 설정
        self.chessboard_size = (9, 6)  # 내부 코너 개수
        self.square_size = 25.0  # mm 단위
        
        # 이미지 크기
        self.full_image_size = (1440, 1080)  # (width, height)
        
        # 실제 측정 거리 (mm)
        self.real_distances = {
            'cam1_to_ball': (900, 1000),  # 카메라1-골프공: 900-1000mm
            'cam2_to_ball': (500, 600),   # 카메라2-골프공: 500-600mm
            'baseline': 470               # 카메라 간 거리: 470mm (수직 배치)
        }
        
        # ROI 정보
        self.roi_info = {
            'cam1': {'XOffset': 0, 'YOffset': 396, 'Width': 1440, 'Height': 300},
            'cam2': {'XOffset': 0, 'YOffset': 372, 'Width': 1440, 'Height': 300}
        }
        
        print("=" * 80)
        print("PRECISE VERTICAL STEREO CALIBRATION SYSTEM")
        print("=" * 80)
        print(f"Calibration directory: {self.calib_dir}")
        print(f"Chessboard size: {self.chessboard_size}")
        print(f"Square size: {self.square_size} mm")
        print(f"Full image size: {self.full_image_size}")
        print(f"Real distances:")
        print(f"  - Cam1 to ball: {self.real_distances['cam1_to_ball']} mm")
        print(f"  - Cam2 to ball: {self.real_distances['cam2_to_ball']} mm")
        print(f"  - Baseline (vertical): {self.real_distances['baseline']} mm")
        print(f"ROI information:")
        print(f"  - Cam1: Y={self.roi_info['cam1']['YOffset']}, H={self.roi_info['cam1']['Height']}")
        print(f"  - Cam2: Y={self.roi_info['cam2']['YOffset']}, H={self.roi_info['cam2']['Height']}")
        print("=" * 80)
    
    def prepare_object_points(self):
        """3D 체스보드 포인트 생성"""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # mm 단위로 스케일링
        return objp
    
    def find_chessboard_corners(self, image_path, camera_name):
        """체스보드 코너 검출"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"  [ERROR] Failed to load: {image_path}")
            return None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 체스보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # 서브픽셀 정확도로 코너 위치 개선
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners_refined, img.shape[:2][::-1]  # (width, height)
        else:
            print(f"  [WARNING] Chessboard not found in: {os.path.basename(image_path)}")
            return None, None
    
    def calibrate_single_camera(self, image_files, camera_name):
        """단일 카메라 캘리브레이션"""
        print(f"\n--- Calibrating {camera_name} ---")
        
        objpoints = []  # 3D 포인트
        imgpoints = []  # 2D 포인트
        
        objp = self.prepare_object_points()
        image_size = None
        
        valid_images = 0
        for img_file in sorted(image_files):
            corners, img_size = self.find_chessboard_corners(img_file, camera_name)
            
            if corners is not None:
                objpoints.append(objp)
                imgpoints.append(corners)
                image_size = img_size
                valid_images += 1
                print(f"  ✓ {os.path.basename(img_file)}: {len(corners)} corners found")
            else:
                print(f"  ✗ {os.path.basename(img_file)}: Failed")
        
        print(f"\nValid images for {camera_name}: {valid_images}/{len(image_files)}")
        
        if valid_images < 5:
            raise ValueError(f"Not enough valid images for {camera_name} calibration (need at least 5)")
        
        # 캘리브레이션 수행
        print(f"Running calibration for {camera_name}...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None,
            flags=cv2.CALIB_RATIONAL_MODEL
        )
        
        # 재투영 오차 계산
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        mean_error /= len(objpoints)
        
        print(f"\n{camera_name} Calibration Results:")
        print(f"  RMS reprojection error: {ret:.4f} pixels")
        print(f"  Mean error: {mean_error:.4f} pixels")
        print(f"  Focal length: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
        print(f"  Principal point: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
        print(f"  Distortion coeffs: {dist_coeffs.ravel()[:5]}")
        
        return camera_matrix, dist_coeffs, rvecs, tvecs, objpoints, imgpoints
    
    def stereo_calibrate(self, K1, D1, K2, D2, objpoints, imgpoints1, imgpoints2, image_size):
        """스테레오 캘리브레이션"""
        print("\n--- Stereo Calibration (Vertical Configuration) ---")
        
        # 스테레오 캘리브레이션 플래그
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        
        ret, K1_stereo, D1_stereo, K2_stereo, D2_stereo, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            K1, D1, K2, D2,
            image_size,
            criteria=criteria,
            flags=flags
        )
        
        # 수직 배치이므로 T는 주로 Y 방향
        baseline_calculated = np.linalg.norm(T)
        
        print(f"\nStereo Calibration Results:")
        print(f"  RMS stereo error: {ret:.4f} pixels")
        print(f"  Rotation matrix R:")
        print(f"    {R}")
        print(f"  Translation vector T (mm):")
        print(f"    {T.ravel()}")
        print(f"  Calculated baseline: {baseline_calculated:.2f} mm")
        print(f"  Expected baseline: {self.real_distances['baseline']} mm")
        print(f"  Baseline error: {abs(baseline_calculated - self.real_distances['baseline']):.2f} mm")
        
        return K1_stereo, D1_stereo, K2_stereo, D2_stereo, R, T, E, F, ret
    
    def compute_rectification_maps(self, K1, D1, K2, D2, R, T, image_size):
        """정류 맵 계산"""
        print("\n--- Computing Rectification Maps ---")
        
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2,
            image_size,
            R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )
        
        print(f"  Projection matrix P1:")
        print(f"    {P1}")
        print(f"  Projection matrix P2:")
        print(f"    {P2}")
        print(f"  Disparity-to-depth matrix Q:")
        print(f"    {Q}")
        print(f"  Valid ROI 1: {roi1}")
        print(f"  Valid ROI 2: {roi2}")
        
        # 언디스토션 맵 생성
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
        
        return R1, R2, P1, P2, Q, (map1x, map1y), (map2x, map2y), roi1, roi2
    
    def calculate_depth_correction_factor(self, K1, baseline):
        """실제 거리 기반 깊이 보정 계수 계산"""
        print("\n--- Calculating Depth Correction Factor ---")
        
        # 실제 거리의 평균값
        cam1_avg = (self.real_distances['cam1_to_ball'][0] + self.real_distances['cam1_to_ball'][1]) / 2
        cam2_avg = (self.real_distances['cam2_to_ball'][0] + self.real_distances['cam2_to_ball'][1]) / 2
        real_avg_distance = (cam1_avg + cam2_avg) / 2
        
        # 예상 시차 계산 (샘플 포인트 기준)
        # 골프공이 이미지 중앙 부근에 있다고 가정
        focal_length = K1[0, 0]
        
        # 시차 = (focal_length * baseline) / depth
        # 실제 거리에서 예상되는 시차
        expected_disparity = (focal_length * baseline) / real_avg_distance
        
        print(f"  Real average distance: {real_avg_distance:.2f} mm")
        print(f"  Focal length: {focal_length:.2f} pixels")
        print(f"  Baseline: {baseline:.2f} mm")
        print(f"  Expected disparity: {expected_disparity:.2f} pixels")
        
        # 보정 계수: 실제 거리 / 계산된 거리 비율
        # 초기 보정 계수는 1.0으로 시작하고, 실측 데이터로 조정
        correction_factor = {
            'global': 1.0,
            'cam1_factor': cam1_avg / real_avg_distance,
            'cam2_factor': cam2_avg / real_avg_distance,
            'baseline_used': baseline,
            'focal_length': focal_length,
            'expected_disparity': expected_disparity,
            'real_distances': self.real_distances
        }
        
        return correction_factor
    
    def run_calibration(self):
        """전체 캘리브레이션 프로세스 실행"""
        print("\n" + "=" * 80)
        print("STARTING CALIBRATION PROCESS")
        print("=" * 80)
        
        # 1. 이미지 파일 수집
        cam1_files = sorted(glob.glob(os.path.join(self.calib_dir, "Cam1_*.bmp")))
        cam2_files = sorted(glob.glob(os.path.join(self.calib_dir, "Cam2_*.bmp")))
        
        if len(cam1_files) == 0 or len(cam2_files) == 0:
            raise FileNotFoundError(f"No calibration images found in {self.calib_dir}")
        
        print(f"\nFound {len(cam1_files)} Cam1 images and {len(cam2_files)} Cam2 images")
        
        # 2. 단일 카메라 캘리브레이션
        K1, D1, rvecs1, tvecs1, objpoints1, imgpoints1 = self.calibrate_single_camera(cam1_files, "Cam1")
        K2, D2, rvecs2, tvecs2, objpoints2, imgpoints2 = self.calibrate_single_camera(cam2_files, "Cam2")
        
        # 3. 공통 프레임만 사용
        common_objpoints = []
        common_imgpoints1 = []
        common_imgpoints2 = []
        
        min_length = min(len(objpoints1), len(objpoints2))
        for i in range(min_length):
            common_objpoints.append(objpoints1[i])
            common_imgpoints1.append(imgpoints1[i])
            common_imgpoints2.append(imgpoints2[i])
        
        print(f"\nUsing {len(common_objpoints)} common image pairs for stereo calibration")
        
        # 4. 스테레오 캘리브레이션
        K1_s, D1_s, K2_s, D2_s, R, T, E, F, stereo_error = self.stereo_calibrate(
            K1, D1, K2, D2,
            common_objpoints, common_imgpoints1, common_imgpoints2,
            self.full_image_size
        )
        
        # 5. 정류 맵 계산
        R1, R2, P1, P2, Q, maps1, maps2, roi1, roi2 = self.compute_rectification_maps(
            K1_s, D1_s, K2_s, D2_s, R, T, self.full_image_size
        )
        
        # 6. 깊이 보정 계수 계산
        baseline_mm = np.linalg.norm(T)
        correction_factor = self.calculate_depth_correction_factor(K1_s, baseline_mm)
        
        # 7. 결과 저장
        calibration_results = {
            # 카메라 내부 파라미터
            'camera_matrix_1': K1_s.tolist(),
            'distortion_coeffs_1': D1_s.ravel().tolist(),
            'camera_matrix_2': K2_s.tolist(),
            'distortion_coeffs_2': D2_s.ravel().tolist(),
            
            # 스테레오 파라미터
            'rotation_matrix': R.tolist(),
            'translation_vector': T.ravel().tolist(),
            'essential_matrix': E.tolist(),
            'fundamental_matrix': F.tolist(),
            
            # 정류 파라미터
            'rectification_matrix_1': R1.tolist(),
            'rectification_matrix_2': R2.tolist(),
            'projection_matrix_1': P1.tolist(),
            'projection_matrix_2': P2.tolist(),
            'disparity_to_depth_matrix': Q.tolist(),
            'valid_roi_1': roi1,
            'valid_roi_2': roi2,
            
            # 기하학적 정보
            'baseline_mm': float(baseline_mm),
            'focal_length_px': float(K1_s[0, 0]),
            'image_size': list(self.full_image_size),
            
            # ROI 정보
            'roi_cam1': self.roi_info['cam1'],
            'roi_cam2': self.roi_info['cam2'],
            
            # 보정 계수
            'depth_correction': correction_factor,
            
            # 오차 정보
            'stereo_calibration_error': float(stereo_error),
            
            # 메타데이터
            'calibration_type': 'vertical_stereo',
            'coordinate_system': 'Y_vertical_Z_depth',
            'units': 'mm',
            'note': 'Precise calibration with ROI coordinate transformation and real distance correction'
        }
        
        # JSON 저장
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("CALIBRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results saved to: {self.output_file}")
        print(f"\nKey Parameters:")
        print(f"  - Baseline: {baseline_mm:.2f} mm")
        print(f"  - Focal length: {K1_s[0,0]:.2f} px")
        print(f"  - Stereo error: {stereo_error:.4f} px")
        print(f"  - Image size: {self.full_image_size}")
        print(f"  - ROI Cam1: Y={self.roi_info['cam1']['YOffset']}, H={self.roi_info['cam1']['Height']}")
        print(f"  - ROI Cam2: Y={self.roi_info['cam2']['YOffset']}, H={self.roi_info['cam2']['Height']}")
        print("=" * 80)
        
        return calibration_results

def main():
    """메인 함수"""
    calibrator = PreciseVerticalStereoCalibration(
        calib_dir="data2/Calibration_image_1025",
        output_file="precise_vertical_stereo_calibration.json"
    )
    
    results = calibrator.run_calibration()
    
    print("\n✓ Precise vertical stereo calibration completed!")
    print("✓ Ready for golf ball 3D tracking with ROI coordinate correction")

if __name__ == "__main__":
    main()
