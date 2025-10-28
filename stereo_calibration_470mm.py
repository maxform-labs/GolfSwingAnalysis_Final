#!/usr/bin/env python3
"""
새로운 베이스라인(470.0mm)으로 스테레오 캘리브레이션
"""

import cv2
import numpy as np
import glob
import json
from datetime import datetime

class StereoCalibration:
    def __init__(self):
        # 새로운 베이스라인
        self.baseline = 470.0  # mm (수정된 값)
        self.focal_length = 1440  # pixels
        self.image_size = (1440, 300)  # pixels
        
        # 체스보드 크기
        self.chessboard_size = (9, 6)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        print(f"스테레오 캘리브레이션 시작")
        print(f"베이스라인: {self.baseline}mm")
        print(f"초점거리: {self.focal_length} pixels")
        print(f"이미지 크기: {self.image_size}")
    
    def prepare_object_points(self):
        """3D 점 준비"""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        return objp
    
    def find_chessboard_corners(self, images):
        """체스보드 코너 검출"""
        objpoints = []  # 3D 점
        imgpoints = []  # 2D 점
        
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                objpoints.append(self.prepare_object_points())
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners2)
                print(f"  이미지 {i+1}: 체스보드 검출 성공")
            else:
                print(f"  이미지 {i+1}: 체스보드 검출 실패")
        
        return objpoints, imgpoints
    
    def calibrate_camera(self, images, camera_name):
        """개별 카메라 캘리브레이션"""
        print(f"\n{camera_name} 캘리브레이션 중...")
        
        objpoints, imgpoints = self.find_chessboard_corners(images)
        
        if len(objpoints) < 10:
            print(f"❌ {camera_name}: 충분한 이미지가 없습니다 (최소 10장 필요)")
            return None, None
        
        # 카메라 캘리브레이션
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_size, None, None
        )
        
        # 재투영 오차 계산
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        
        print(f"✅ {camera_name} 캘리브레이션 완료")
        print(f"  재투영 오차: {mean_error:.3f} pixels")
        
        return mtx, dist
    
    def stereo_calibrate(self, cam1_images, cam2_images):
        """스테레오 캘리브레이션"""
        print("\n스테레오 캘리브레이션 중...")
        
        # 각 카메라 개별 캘리브레이션
        mtx1, dist1 = self.calibrate_camera(cam1_images, "카메라1")
        mtx2, dist2 = self.calibrate_camera(cam2_images, "카메라2")
        
        if mtx1 is None or mtx2 is None:
            print("❌ 개별 캘리브레이션 실패")
            return None
        
        # 스테레오 캘리브레이션
        objpoints1, imgpoints1 = self.find_chessboard_corners(cam1_images)
        objpoints2, imgpoints2 = self.find_chessboard_corners(cam2_images)
        
        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints1, imgpoints1, imgpoints2,
            mtx1, dist1, mtx2, dist2,
            self.image_size
        )
        
        print(f"✅ 스테레오 캘리브레이션 완료")
        print(f"  베이스라인: {self.baseline}mm")
        print(f"  회전 행렬: {R.shape}")
        print(f"  이동 벡터: {T.shape}")
        
        return {
            'camera_matrix_1': mtx1.tolist(),
            'distortion_coeffs_1': dist1.tolist(),
            'camera_matrix_2': mtx2.tolist(),
            'distortion_coeffs_2': dist2.tolist(),
            'rotation_matrix': R.tolist(),
            'translation_vector': T.tolist(),
            'baseline': self.baseline,
            'focal_length': self.focal_length,
            'image_size': self.image_size,
            'calibration_date': datetime.now().isoformat()
        }
    
    def save_calibration(self, calibration_data, filename="calibration_470mm.json"):
        """캘리브레이션 데이터 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 캘리브레이션 데이터 저장: {filename}")
    
    def load_calibration(self, filename="calibration_470mm.json"):
        """캘리브레이션 데이터 로드"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ 캘리브레이션 데이터 로드: {filename}")
            print(f"  베이스라인: {data['baseline']}mm")
            print(f"  캘리브레이션 날짜: {data['calibration_date']}")
            
            return data
        except Exception as e:
            print(f"❌ 캘리브레이션 데이터 로드 실패: {e}")
            return None

def main():
    calibrator = StereoCalibration()
    
    # 체스보드 이미지 로드 (실제 경로로 수정 필요)
    print("\n체스보드 이미지 로드 중...")
    # cam1_images = [cv2.imread(f) for f in glob.glob("calibration_images/cam1/*.jpg")]
    # cam2_images = [cv2.imread(f) for f in glob.glob("calibration_images/cam2/*.jpg")]
    
    print("⚠️ 실제 체스보드 이미지 경로를 설정하고 실행하세요")
    print("예시:")
    print("  cam1_images = [cv2.imread(f) for f in glob.glob('calibration_images/cam1/*.jpg')]")
    print("  cam2_images = [cv2.imread(f) for f in glob.glob('calibration_images/cam2/*.jpg')]")
    
    # 캘리브레이션 수행
    # calibration_data = calibrator.stereo_calibrate(cam1_images, cam2_images)
    # if calibration_data:
    #     calibrator.save_calibration(calibration_data)

if __name__ == "__main__":
    main()
