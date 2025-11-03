#!/usr/bin/env python3
"""
ROI 기반 새로운 캘리브레이션
드라이버 이미지와 동일한 ROI로 스테레오 캘리브레이션 수행
"""

import cv2
import numpy as np
import json
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

class ROIBasedCalibration:
    def __init__(self):
        """ROI 기반 캘리브레이션 초기화"""
        self.calibration_images_dir = "data2/Calibration_image_1025"
        self.driver_dir = "data2/driver"
        
        # 드라이버 이미지의 ROI 정보 수집
        self.roi_info = self.collect_roi_info()
        
        print("ROI Based Calibration Initialized")
        print(f"Driver ROI Info: {self.roi_info}")
    
    def collect_roi_info(self):
        """드라이버 이미지들의 ROI 정보 수집"""
        roi_info = {
            'cam1_rois': [],
            'cam2_rois': [],
            'common_roi': None
        }
        
        # 샘플 드라이버 디렉토리들에서 ROI 정보 수집
        sample_dirs = ['1', '2', '3', '4', '5']
        
        for shot_dir in sample_dirs:
            shot_path = os.path.join(self.driver_dir, shot_dir)
            
            # ROI 파일 로드
            roi_cam1_file = os.path.join(shot_path, "roi_cam1.json")
            roi_cam2_file = os.path.join(shot_path, "roi_cam2.json")
            
            if os.path.exists(roi_cam1_file):
                with open(roi_cam1_file, 'r', encoding='utf-8') as f:
                    roi_cam1 = json.load(f)
                    roi_info['cam1_rois'].append(roi_cam1)
            
            if os.path.exists(roi_cam2_file):
                with open(roi_cam2_file, 'r', encoding='utf-8') as f:
                    roi_cam2 = json.load(f)
                    roi_info['cam2_rois'].append(roi_cam2)
        
        # 공통 ROI 계산
        if roi_info['cam1_rois'] and roi_info['cam2_rois']:
            # 모든 ROI가 동일하다고 가정 (첫 번째 것 사용)
            roi_info['common_roi'] = {
                'cam1': roi_info['cam1_rois'][0],
                'cam2': roi_info['cam2_rois'][0]
            }
        
        return roi_info
    
    def apply_roi_to_calibration_images(self, image, roi_info, camera_num):
        """캘리브레이션 이미지에 ROI 적용"""
        if roi_info is None:
            return image
        
        roi = roi_info['cam1'] if camera_num == 1 else roi_info['cam2']
        
        # ROI 영역 추출
        x_offset = roi['XOffset']
        y_offset = roi['YOffset']
        width = roi['Width']
        height = roi['Height']
        
        # ROI 적용
        roi_image = image[y_offset:y_offset+height, x_offset:x_offset+width]
        
        return roi_image
    
    def detect_chessboard_corners(self, image, pattern_size=(9, 6)):
        """체스보드 코너 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 체스보드 코너 검출
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # 코너 정제
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return ret, corners
    
    def perform_roi_calibration(self, output_dir="roi_calibration_result"):
        """ROI 기반 스테레오 캘리브레이션 수행"""
        print(f"\n=== PERFORMING ROI-BASED STEREO CALIBRATION ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.roi_info['common_roi'] is None:
            print("ERROR: No ROI information found")
            return None
        
        # 캘리브레이션 이미지 파일 목록
        pattern_cam1 = os.path.join(self.calibration_images_dir, "Cam1_*.bmp")
        pattern_cam2 = os.path.join(self.calibration_images_dir, "Cam2_*.bmp")
        files_cam1 = sorted(glob.glob(pattern_cam1), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        files_cam2 = sorted(glob.glob(pattern_cam2), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        print(f"Found {len(files_cam1)} Cam1 calibration images")
        print(f"Found {len(files_cam2)} Cam2 calibration images")
        
        # 체스보드 패턴 크기
        pattern_size = (9, 6)
        
        # 3D 점 생성 (체스보드의 실제 좌표)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        # 이미지 포인트와 객체 포인트 저장
        objpoints = []  # 3D 점
        imgpoints_cam1 = []  # Cam1 2D 점
        imgpoints_cam2 = []  # Cam2 2D 점
        
        # 각 이미지에서 체스보드 검출
        for i in range(min(len(files_cam1), len(files_cam2))):
            filename1 = os.path.basename(files_cam1[i])
            filename2 = os.path.basename(files_cam2[i])
            frame_num1 = int(filename1.split('_')[1].split('.')[0])
            frame_num2 = int(filename2.split('_')[1].split('.')[0])
            
            if frame_num1 != frame_num2:
                continue
            
            print(f"Processing frame {frame_num1}...")
            
            # 이미지 로드
            img1 = cv2.imread(files_cam1[i])
            img2 = cv2.imread(files_cam2[i])
            
            if img1 is None or img2 is None:
                continue
            
            # ROI 적용
            roi_img1 = self.apply_roi_to_calibration_images(img1, self.roi_info['common_roi'], 1)
            roi_img2 = self.apply_roi_to_calibration_images(img2, self.roi_info['common_roi'], 2)
            
            if roi_img1 is None or roi_img2 is None:
                print(f"  Failed to apply ROI to frame {frame_num1}")
                continue
            
            # 체스보드 코너 검출
            ret1, corners1 = self.detect_chessboard_corners(roi_img1, pattern_size)
            ret2, corners2 = self.detect_chessboard_corners(roi_img2, pattern_size)
            
            if ret1 and ret2:
                objpoints.append(objp)
                imgpoints_cam1.append(corners1)
                imgpoints_cam2.append(corners2)
                print(f"  Found chessboard in both cameras")
                
                # 시각화 저장
                vis_img1 = roi_img1.copy()
                vis_img2 = roi_img2.copy()
                cv2.drawChessboardCorners(vis_img1, pattern_size, corners1, ret1)
                cv2.drawChessboardCorners(vis_img2, pattern_size, corners2, ret2)
                
                # 결과 이미지 저장
                result_img = np.hstack([vis_img1, vis_img2])
                cv2.imwrite(os.path.join(output_dir, f"roi_chessboard_frame_{frame_num1:02d}.jpg"), result_img)
            else:
                print(f"  Failed to find chessboard in frame {frame_num1}")
        
        if len(objpoints) < 5:
            print(f"ERROR: Not enough valid frames for calibration ({len(objpoints)} frames)")
            return None
        
        print(f"\nUsing {len(objpoints)} frames for calibration")
        
        # ROI 크기 정보
        roi_cam1 = self.roi_info['common_roi']['cam1']
        roi_cam2 = self.roi_info['common_roi']['cam2']
        roi_size = (roi_cam1['Width'], roi_cam1['Height'])
        
        print(f"ROI size: {roi_size}")
        
        # 개별 카메라 캘리브레이션
        print("\nPerforming individual camera calibration...")
        
        ret1, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(
            objpoints, imgpoints_cam1, roi_size, None, None
        )
        
        ret2, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(
            objpoints, imgpoints_cam2, roi_size, None, None
        )
        
        if not ret1 or not ret2:
            print("ERROR: Individual camera calibration failed")
            return None
        
        print(f"Cam1 calibration RMS error: {ret1:.3f}")
        print(f"Cam2 calibration RMS error: {ret2:.3f}")
        
        # 스테레오 캘리브레이션
        print("\nPerforming stereo calibration...")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        ret_stereo, K1_new, D1_new, K2_new, D2_new, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_cam1, imgpoints_cam2,
            K1, D1, K2, D2, roi_size,
            criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        if not ret_stereo:
            print("ERROR: Stereo calibration failed")
            return None
        
        print(f"Stereo calibration RMS error: {ret_stereo:.3f}")
        
        # 베이스라인 계산
        baseline = np.linalg.norm(T)
        baseline_mm = baseline * 1000  # mm 단위
        
        print(f"Baseline: {baseline_mm:.1f} mm")
        
        # 결과 저장
        calibration_result = {
            'camera_matrix_1': K1_new.tolist(),
            'camera_matrix_2': K2_new.tolist(),
            'distortion_coeffs_1': D1_new.tolist(),
            'distortion_coeffs_2': D2_new.tolist(),
            'rotation_matrix': R.tolist(),
            'translation_vector': T.tolist(),
            'essential_matrix': E.tolist(),
            'fundamental_matrix': F.tolist(),
            'baseline_mm': float(baseline_mm),
            'image_size': roi_size,
            'rms_error': float(ret_stereo),
            'roi_info': self.roi_info['common_roi'],
            'coordinate_system': {
                'X_axis': 'Forward direction',
                'Y_axis': 'Side direction', 
                'Z_axis': 'Height direction',
                'baseline_direction': 'Z-axis'
            }
        }
        
        # JSON 파일로 저장
        result_file = os.path.join(output_dir, "roi_based_stereo_calibration.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_result, f, indent=2, ensure_ascii=False)
        
        print(f"\nROI-based calibration completed!")
        print(f"Results saved to: {result_file}")
        
        return calibration_result
    
    def test_roi_calibration(self, calibration_result, test_shot_dir="data2/driver/2"):
        """ROI 캘리브레이션 테스트"""
        if calibration_result is None:
            return
        
        print(f"\n=== TESTING ROI CALIBRATION ===")
        
        # 테스트 샷 분석
        pattern_cam1 = os.path.join(test_shot_dir, "Cam1_*.bmp")
        pattern_cam2 = os.path.join(test_shot_dir, "Cam2_*.bmp")
        files_cam1 = sorted(glob.glob(pattern_cam1), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        files_cam2 = sorted(glob.glob(pattern_cam2), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        print(f"Testing with {len(files_cam1)} Cam1 images, {len(files_cam2)} Cam2 images")
        
        # 첫 번째 프레임으로 테스트
        if files_cam1 and files_cam2:
            img1 = cv2.imread(files_cam1[0])
            img2 = cv2.imread(files_cam2[0])
            
            if img1 is not None and img2 is not None:
                # ROI 적용
                roi_img1 = self.apply_roi_to_calibration_images(img1, self.roi_info['common_roi'], 1)
                roi_img2 = self.apply_roi_to_calibration_images(img2, self.roi_info['common_roi'], 2)
                
                print(f"Original image size: {img1.shape}")
                print(f"ROI image size: {roi_img1.shape}")
                
                # ROI 이미지 저장
                cv2.imwrite("roi_calibration_result/test_roi_cam1.jpg", roi_img1)
                cv2.imwrite("roi_calibration_result/test_roi_cam2.jpg", roi_img2)
                print("Test ROI images saved")

def main():
    """메인 함수"""
    calibrator = ROIBasedCalibration()
    
    # ROI 기반 캘리브레이션 수행
    calibration_result = calibrator.perform_roi_calibration()
    
    # 캘리브레이션 테스트
    if calibration_result:
        calibrator.test_roi_calibration(calibration_result)
    
    print("\n=== ROI-BASED CALIBRATION COMPLETE ===")

if __name__ == "__main__":
    main()

