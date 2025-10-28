#!/usr/bin/env python3
"""
실제 캘리브레이션 이미지로 새로운 베이스라인(470.0mm) 스테레오 캘리브레이션
"""

import cv2
import numpy as np
import glob
import json
import os
from datetime import datetime

class RealStereoCalibration:
    def __init__(self):
        # 새로운 베이스라인
        self.baseline = 470.0  # mm (수정된 값)
        self.focal_length = 1440  # pixels
        self.image_size = (1440, 300)  # pixels
        
        # 체스보드 크기
        self.chessboard_size = (9, 6)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        print("실제 캘리브레이션 이미지로 스테레오 캘리브레이션")
        print("=" * 60)
        print(f"베이스라인: {self.baseline}mm")
        print(f"초점거리: {self.focal_length} pixels")
        print(f"이미지 크기: {self.image_size}")
        print(f"체스보드 크기: {self.chessboard_size}")
        print()
    
    def prepare_object_points(self):
        """3D 점 준비"""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        return objp
    
    def load_calibration_images(self):
        """캘리브레이션 이미지 로드"""
        print("캘리브레이션 이미지 로드 중...")
        
        # 이미지 경로
        calibration_path = "data/calibrations"
        
        # Cam1, Cam2 이미지 분리
        cam1_images = []
        cam2_images = []
        
        # BMP 파일들 로드
        image_files = glob.glob(os.path.join(calibration_path, "*.bmp"))
        image_files.sort()  # 시간순 정렬
        
        print(f"발견된 이미지 파일: {len(image_files)}개")
        
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ❌ 이미지 로드 실패: {img_path}")
                continue
            
            # 이미지 크기 조정 (1440x300으로)
            img_resized = cv2.resize(img, self.image_size)
            
            # 파일명에서 카메라 구분
            filename = os.path.basename(img_path)
            if "_Cam1.bmp" in filename:
                cam1_images.append(img_resized)
                print(f"  ✅ Cam1: {filename}")
            elif "_Cam2.bmp" in filename:
                cam2_images.append(img_resized)
                print(f"  ✅ Cam2: {filename}")
        
        print(f"\n로드된 이미지:")
        print(f"  Cam1: {len(cam1_images)}개")
        print(f"  Cam2: {len(cam2_images)}개")
        
        return cam1_images, cam2_images
    
    def find_chessboard_corners(self, images, camera_name):
        """체스보드 코너 검출"""
        print(f"\n{camera_name} 체스보드 코너 검출 중...")
        
        objpoints = []  # 3D 점
        imgpoints = []  # 2D 점
        successful_images = []
        
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                objpoints.append(self.prepare_object_points())
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners2)
                successful_images.append(i)
                
                # 검출 결과 시각화
                img_with_corners = img.copy()
                cv2.drawChessboardCorners(img_with_corners, self.chessboard_size, corners2, ret)
                
                # 결과 이미지 저장
                output_path = f"chessboard_detected_{i+1}_{camera_name.lower()}.jpg"
                cv2.imwrite(output_path, img_with_corners)
                
                print(f"  ✅ 이미지 {i+1}: 체스보드 검출 성공")
            else:
                print(f"  ❌ 이미지 {i+1}: 체스보드 검출 실패")
        
        print(f"  총 {len(successful_images)}개 이미지에서 체스보드 검출 성공")
        return objpoints, imgpoints, successful_images
    
    def calibrate_camera(self, images, camera_name):
        """개별 카메라 캘리브레이션"""
        print(f"\n{camera_name} 개별 캘리브레이션 중...")
        
        objpoints, imgpoints, successful_images = self.find_chessboard_corners(images, camera_name)
        
        if len(objpoints) < 5:
            print(f"❌ {camera_name}: 충분한 이미지가 없습니다 (최소 5장 필요)")
            return None, None, None
        
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
        print(f"  성공한 이미지: {len(successful_images)}개")
        print(f"  재투영 오차: {mean_error:.3f} pixels")
        print(f"  카메라 행렬 크기: {mtx.shape}")
        print(f"  왜곡 계수: {dist.shape}")
        
        return mtx, dist, (objpoints, imgpoints)
    
    def stereo_calibrate(self, cam1_images, cam2_images):
        """스테레오 캘리브레이션"""
        print("\n" + "=" * 60)
        print("스테레오 캘리브레이션 시작")
        print("=" * 60)
        
        # 각 카메라 개별 캘리브레이션
        mtx1, dist1, cam1_data = self.calibrate_camera(cam1_images, "Cam1")
        mtx2, dist2, cam2_data = self.calibrate_camera(cam2_images, "Cam2")
        
        if mtx1 is None or mtx2 is None:
            print("❌ 개별 캘리브레이션 실패")
            return None
        
        # 스테레오 캘리브레이션
        print("\n스테레오 캘리브레이션 수행 중...")
        
        objpoints1, imgpoints1 = cam1_data
        objpoints2, imgpoints2 = cam2_data
        
        # 공통 이미지 쌍 찾기
        min_pairs = min(len(objpoints1), len(objpoints2))
        print(f"공통 이미지 쌍: {min_pairs}개")
        
        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints1[:min_pairs], imgpoints1[:min_pairs], imgpoints2[:min_pairs],
            mtx1, dist1, mtx2, dist2,
            self.image_size
        )
        
        print(f"✅ 스테레오 캘리브레이션 완료")
        print(f"  베이스라인: {self.baseline}mm")
        print(f"  회전 행렬 크기: {R.shape}")
        print(f"  이동 벡터: {T.flatten()}")
        print(f"  이동 벡터 크기: {np.linalg.norm(T):.2f}mm")
        
        # 실제 베이스라인과 비교
        calculated_baseline = np.linalg.norm(T)
        baseline_error = abs(calculated_baseline - self.baseline)
        baseline_error_percent = (baseline_error / self.baseline) * 100
        
        print(f"\n베이스라인 검증:")
        print(f"  설정된 베이스라인: {self.baseline}mm")
        print(f"  계산된 베이스라인: {calculated_baseline:.2f}mm")
        print(f"  오차: {baseline_error:.2f}mm ({baseline_error_percent:.2f}%)")
        
        if baseline_error_percent < 5:
            print("  ✅ 베이스라인 검증 통과")
        else:
            print("  ⚠️ 베이스라인 검증 실패 - 재캘리브레이션 권장")
        
        return {
            'camera_matrix_1': mtx1.tolist(),
            'distortion_coeffs_1': dist1.tolist(),
            'camera_matrix_2': mtx2.tolist(),
            'distortion_coeffs_2': dist2.tolist(),
            'rotation_matrix': R.tolist(),
            'translation_vector': T.tolist(),
            'essential_matrix': E.tolist(),
            'fundamental_matrix': F.tolist(),
            'baseline': self.baseline,
            'calculated_baseline': float(calculated_baseline),
            'baseline_error': float(baseline_error),
            'baseline_error_percent': float(baseline_error_percent),
            'focal_length': self.focal_length,
            'image_size': self.image_size,
            'chessboard_size': self.chessboard_size,
            'calibration_date': datetime.now().isoformat(),
            'successful_images_cam1': len(objpoints1),
            'successful_images_cam2': len(objpoints2)
        }
    
    def save_calibration(self, calibration_data, filename="real_calibration_470mm.json"):
        """캘리브레이션 데이터 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 캘리브레이션 데이터 저장: {filename}")
        
        # 요약 정보 출력
        print(f"\n📊 캘리브레이션 요약:")
        print(f"  베이스라인: {calibration_data['baseline']}mm")
        print(f"  계산된 베이스라인: {calibration_data['calculated_baseline']:.2f}mm")
        print(f"  베이스라인 오차: {calibration_data['baseline_error_percent']:.2f}%")
        print(f"  성공한 이미지 (Cam1): {calibration_data['successful_images_cam1']}개")
        print(f"  성공한 이미지 (Cam2): {calibration_data['successful_images_cam2']}개")
        print(f"  캘리브레이션 날짜: {calibration_data['calibration_date']}")
    
    def load_calibration(self, filename="real_calibration_470mm.json"):
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
    calibrator = RealStereoCalibration()
    
    # 캘리브레이션 이미지 로드
    cam1_images, cam2_images = calibrator.load_calibration_images()
    
    if len(cam1_images) == 0 or len(cam2_images) == 0:
        print("❌ 캘리브레이션 이미지를 찾을 수 없습니다.")
        return
    
    # 스테레오 캘리브레이션 수행
    calibration_data = calibrator.stereo_calibrate(cam1_images, cam2_images)
    
    if calibration_data:
        # 캘리브레이션 데이터 저장
        calibrator.save_calibration(calibration_data)
        
        print("\n🎉 실제 캘리브레이션 완료!")
        print("이제 새로운 베이스라인으로 골프 스윙 분석을 수행할 수 있습니다.")
    else:
        print("❌ 캘리브레이션 실패")

if __name__ == "__main__":
    main()
