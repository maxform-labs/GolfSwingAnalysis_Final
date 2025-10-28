#!/usr/bin/env python3
"""
수직 배치된 카메라를 위한 스테레오비전 캘리브레이션
Calibration_image_1025 폴더의 이미지들을 사용
"""

import cv2
import numpy as np
import glob
import json
import os
from datetime import datetime
from pathlib import Path

class VerticalStereoCalibration:
    def __init__(self, calibration_images_path="data2/Calibration_image_1025"):
        self.calibration_images_path = calibration_images_path
        self.image_size = (1440, 1080)  # 실제 이미지 크기
        
        # 체스보드 패턴 크기 (여러 패턴 시도)
        self.chessboard_patterns = [
            (7, 5), (8, 6), (9, 6), (10, 7), (11, 8),
            (6, 4), (7, 6), (8, 5), (9, 7), (10, 6),
            (5, 4), (6, 5), (7, 4), (8, 4), (9, 5)
        ]
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        print(f"수직 스테레오 캘리브레이션 초기화")
        print(f"이미지 경로: {calibration_images_path}")
        print(f"이미지 크기: {self.image_size}")
    
    def preprocess_image(self, img):
        """이미지 전처리로 체스보드 검출 개선"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # 이진화
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return enhanced, binary
    
    def find_chessboard_corners(self, images, camera_name):
        """체스보드 코너 검출 (여러 패턴 시도)"""
        print(f"\n{camera_name} 체스보드 코너 검출 중...")
        
        objpoints = []  # 3D 점
        imgpoints = []  # 2D 점
        successful_pattern = None
        
        for pattern in self.chessboard_patterns:
            print(f"  패턴 {pattern} 시도 중...")
            pattern_objpoints = []
            pattern_imgpoints = []
            
            # 3D 점 준비
            objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
            
            success_count = 0
            
            for i, img in enumerate(images):
                # 이미지 전처리
                enhanced, binary = self.preprocess_image(img)
                
                # 체스보드 코너 검출
                ret, corners = cv2.findChessboardCorners(enhanced, pattern, None)
                
                if ret:
                    # 코너 개선
                    corners2 = cv2.cornerSubPix(enhanced, corners, (11, 11), (-1, -1), self.criteria)
                    pattern_objpoints.append(objp)
                    pattern_imgpoints.append(corners2)
                    success_count += 1
                    
                    # 시각화 (첫 번째 성공한 이미지만)
                    if success_count == 1:
                        img_with_corners = img.copy()
                        cv2.drawChessboardCorners(img_with_corners, pattern, corners2, ret)
                        output_path = f"chessboard_detection_{camera_name}_{pattern[0]}x{pattern[1]}.jpg"
                        cv2.imwrite(output_path, img_with_corners)
                        print(f"    시각화 결과 저장: {output_path}")
            
            print(f"  패턴 {pattern}: {success_count}/{len(images)} 이미지 성공")
            
            # 충분한 이미지가 성공한 패턴 선택
            if success_count >= 10 and success_count > len(pattern_objpoints) * 0.6:
                objpoints = pattern_objpoints
                imgpoints = pattern_imgpoints
                successful_pattern = pattern
                print(f"  OK 패턴 {pattern} 선택됨 ({success_count}개 이미지)")
                break
        
        if successful_pattern is None:
            print(f"  X {camera_name}: 적절한 체스보드 패턴을 찾을 수 없습니다")
            return None, None, None
        
        return objpoints, imgpoints, successful_pattern
    
    def calibrate_individual_camera(self, images, camera_name):
        """개별 카메라 캘리브레이션"""
        print(f"\n{camera_name} 개별 캘리브레이션 중...")
        
        objpoints, imgpoints, pattern = self.find_chessboard_corners(images, camera_name)
        
        if objpoints is None:
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
        
        print(f"OK {camera_name} 캘리브레이션 완료")
        print(f"  재투영 오차: {mean_error:.3f} pixels")
        print(f"  사용된 패턴: {pattern}")
        print(f"  성공한 이미지: {len(objpoints)}개")
        
        return mtx, dist, (objpoints, imgpoints)
    
    def calibrate_stereo(self, cam1_images, cam2_images):
        """스테레오 캘리브레이션"""
        print("\n=== 스테레오 캘리브레이션 시작 ===")
        
        # 각 카메라 개별 캘리브레이션
        mtx1, dist1, cam1_data = self.calibrate_individual_camera(cam1_images, "Cam1")
        mtx2, dist2, cam2_data = self.calibrate_individual_camera(cam2_images, "Cam2")
        
        if mtx1 is None or mtx2 is None:
            print("X 개별 캘리브레이션 실패")
            return None
        
        # 스테레오 캘리브레이션
        print("\n스테레오 캘리브레이션 수행 중...")
        
        objpoints1, imgpoints1 = cam1_data
        objpoints2, imgpoints2 = cam2_data
        
        # 스테레오 캘리브레이션
        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints1, imgpoints1, imgpoints2,
            mtx1, dist1, mtx2, dist2,
            self.image_size,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        # 스테레오 정규화
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx1, dist1, mtx2, dist2, self.image_size, R, T
        )
        
        # 베이스라인 계산
        baseline = np.linalg.norm(T)
        
        print(f"OK 스테레오 캘리브레이션 완료")
        print(f"  재투영 오차: {ret:.3f}")
        print(f"  베이스라인: {baseline:.2f} (픽셀 단위)")
        print(f"  베이스라인: {baseline * 0.1:.2f}mm (추정, 픽셀당 0.1mm 가정)")
        
        return {
            'camera_matrix_1': mtx1.tolist(),
            'distortion_coeffs_1': dist1.tolist(),
            'camera_matrix_2': mtx2.tolist(),
            'distortion_coeffs_2': dist2.tolist(),
            'rotation_matrix': R.tolist(),
            'translation_vector': T.tolist(),
            'rectification_rotation_1': R1.tolist(),
            'rectification_rotation_2': R2.tolist(),
            'projection_matrix_1': P1.tolist(),
            'projection_matrix_2': P2.tolist(),
            'disparity_to_depth_matrix': Q.tolist(),
            'roi_1': roi1.tolist(),
            'roi_2': roi2.tolist(),
            'baseline_pixels': baseline,
            'baseline_mm': baseline * 0.1,  # 추정값
            'reprojection_error': ret,
            'image_size': self.image_size,
            'calibration_date': datetime.now().isoformat(),
            'calibration_images_path': self.calibration_images_path
        }
    
    def load_calibration_images(self):
        """캘리브레이션 이미지 로드"""
        print(f"\n캘리브레이션 이미지 로드 중...")
        
        cam1_pattern = os.path.join(self.calibration_images_path, "Cam1_*.bmp")
        cam2_pattern = os.path.join(self.calibration_images_path, "Cam2_*.bmp")
        
        cam1_files = sorted(glob.glob(cam1_pattern))
        cam2_files = sorted(glob.glob(cam2_pattern))
        
        print(f"Cam1 이미지: {len(cam1_files)}개")
        print(f"Cam2 이미지: {len(cam2_files)}개")
        
        if len(cam1_files) == 0 or len(cam2_files) == 0:
            print("❌ 캘리브레이션 이미지를 찾을 수 없습니다")
            return None, None
        
        # 이미지 로드
        cam1_images = []
        cam2_images = []
        
        for file in cam1_files:
            img = cv2.imread(file)
            if img is not None:
                cam1_images.append(img)
        
        for file in cam2_files:
            img = cv2.imread(file)
            if img is not None:
                cam2_images.append(img)
        
        print(f"로드된 Cam1 이미지: {len(cam1_images)}개")
        print(f"로드된 Cam2 이미지: {len(cam2_images)}개")
        
        return cam1_images, cam2_images
    
    def save_calibration(self, calibration_data, filename="vertical_stereo_calibration_1025.json"):
        """캘리브레이션 데이터 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nOK 캘리브레이션 데이터 저장: {filename}")
    
    def run_calibration(self):
        """전체 캘리브레이션 실행"""
        print("=== 수직 스테레오 캘리브레이션 시작 ===")
        
        # 1. 이미지 로드
        cam1_images, cam2_images = self.load_calibration_images()
        
        if cam1_images is None:
            return None
        
        # 2. 스테레오 캘리브레이션
        calibration_data = self.calibrate_stereo(cam1_images, cam2_images)
        
        if calibration_data is None:
            print("X 캘리브레이션 실패")
            return None
        
        # 3. 결과 저장
        self.save_calibration(calibration_data)
        
        # 4. 결과 요약 출력
        self.print_calibration_summary(calibration_data)
        
        return calibration_data
    
    def print_calibration_summary(self, calibration_data):
        """캘리브레이션 결과 요약 출력"""
        print("\n=== 캘리브레이션 결과 요약 ===")
        print(f"재투영 오차: {calibration_data['reprojection_error']:.3f} pixels")
        print(f"베이스라인 (픽셀): {calibration_data['baseline_pixels']:.2f}")
        print(f"베이스라인 (추정 mm): {calibration_data['baseline_mm']:.2f}")
        print(f"이미지 크기: {calibration_data['image_size']}")
        print(f"캘리브레이션 날짜: {calibration_data['calibration_date']}")
        
        # 카메라 매트릭스 정보
        K1 = np.array(calibration_data['camera_matrix_1'])
        K2 = np.array(calibration_data['camera_matrix_2'])
        
        print(f"\nCam1 초점거리: fx={K1[0,0]:.2f}, fy={K1[1,1]:.2f}")
        print(f"Cam1 주점: cx={K1[0,2]:.2f}, cy={K1[1,2]:.2f}")
        print(f"Cam2 초점거리: fx={K2[0,0]:.2f}, fy={K2[1,1]:.2f}")
        print(f"Cam2 주점: cx={K2[0,2]:.2f}, cy={K2[1,2]:.2f}")

def main():
    """메인 함수"""
    calibrator = VerticalStereoCalibration()
    calibration_data = calibrator.run_calibration()
    
    if calibration_data:
        print("\nOK 캘리브레이션이 성공적으로 완료되었습니다!")
        print("결과 파일: vertical_stereo_calibration_1025.json")
    else:
        print("\nX 캘리브레이션이 실패했습니다.")

if __name__ == "__main__":
    main()
