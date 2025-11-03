#!/usr/bin/env python3
"""
강화된 스테레오 캘리브레이션
체스보드 검출을 위한 다양한 전처리 방법 적용
"""

import cv2
import numpy as np
import glob
import json
import os
from datetime import datetime

class EnhancedStereoCalibration:
    def __init__(self, calibration_images_path="data2/Calibration_image_1025"):
        self.calibration_images_path = calibration_images_path
        self.image_size = (1440, 1080)  # 실제 이미지 크기
        self.baseline_mm = 470.0  # 실제 베이스라인 (mm)
        
        # 더 많은 체스보드 패턴 크기 시도
        self.chessboard_patterns = [
            (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4),
            (5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5),
            (6, 6), (7, 6), (8, 6), (9, 6), (10, 6),
            (7, 7), (8, 7), (9, 7), (10, 7),
            (8, 8), (9, 8), (10, 8), (11, 8)
        ]
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        print(f"강화된 스테레오 캘리브레이션 초기화")
        print(f"이미지 경로: {calibration_images_path}")
        print(f"이미지 크기: {self.image_size}")
        print(f"베이스라인: {self.baseline_mm}mm")
    
    def ultra_enhance_image(self, img):
        """초강력 이미지 향상"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 여러 향상 방법 시도
        enhanced_images = []
        
        # 1. 원본 그레이스케일
        enhanced_images.append(gray)
        
        # 2. CLAHE (다양한 파라미터)
        for clip_limit in [2.0, 4.0, 8.0]:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced_images.append(enhanced)
        
        # 3. 감마 보정
        for gamma in [0.5, 0.7, 1.5, 2.0]:
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                    for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(gray, lookup_table)
            enhanced_images.append(gamma_corrected)
        
        # 4. 히스토그램 균등화
        equalized = cv2.equalizeHist(gray)
        enhanced_images.append(equalized)
        
        # 5. 가우시안 블러 후 CLAHE
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        blurred_enhanced = clahe.apply(blurred)
        enhanced_images.append(blurred_enhanced)
        
        # 6. 언샤프 마스킹
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        enhanced_images.append(unsharp_mask)
        
        return enhanced_images
    
    def find_best_chessboard_pattern(self, images, camera_name):
        """최적의 체스보드 패턴 찾기 (강화된 방법)"""
        print(f"\n{camera_name} 최적 패턴 검색 중...")
        
        best_pattern = None
        best_success_count = 0
        best_objpoints = None
        best_imgpoints = None
        best_method = 0
        
        for pattern in self.chessboard_patterns:
            print(f"  패턴 {pattern} 테스트 중...")
            
            # 3D 점 준비
            objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
            
            objpoints = []
            imgpoints = []
            success_count = 0
            method_success = 0
            
            for i, img in enumerate(images):
                enhanced_images = self.ultra_enhance_image(img)
                
                # 여러 향상 방법 시도
                for method_idx, enhanced in enumerate(enhanced_images):
                    # 체스보드 코너 검출 (다양한 플래그 시도)
                    flags = [
                        None,
                        cv2.CALIB_CB_ADAPTIVE_THRESH,
                        cv2.CALIB_CB_NORMALIZE_IMAGE,
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                        cv2.CALIB_CB_FILTER_QUADS
                    ]
                    
                    for flag in flags:
                        if flag is None:
                            ret, corners = cv2.findChessboardCorners(enhanced, pattern, None)
                        else:
                            ret, corners = cv2.findChessboardCorners(enhanced, pattern, flag)
                        
                        if ret:
                            # 코너 개선
                            corners2 = cv2.cornerSubPix(enhanced, corners, (11, 11), (-1, -1), self.criteria)
                            objpoints.append(objp)
                            imgpoints.append(corners2)
                            success_count += 1
                            method_success = method_idx
                            
                            # 시각화 (첫 번째 성공한 이미지만)
                            if success_count == 1:
                                img_with_corners = img.copy()
                                cv2.drawChessboardCorners(img_with_corners, pattern, corners2, ret)
                                output_path = f"chessboard_detection_{camera_name}_{pattern[0]}x{pattern[1]}_method{method_idx}.jpg"
                                cv2.imwrite(output_path, img_with_corners)
                                print(f"    시각화 결과 저장: {output_path}")
                            break
                    
                    if ret:
                        break  # 성공하면 다른 방법은 시도하지 않음
            
            print(f"    성공: {success_count}/{len(images)} 이미지 (방법 {method_success})")
            
            # 최고 성공률 패턴 선택 (기준 완화)
            if success_count > best_success_count and success_count >= 3:
                best_pattern = pattern
                best_success_count = success_count
                best_objpoints = objpoints
                best_imgpoints = imgpoints
                best_method = method_success
        
        if best_pattern is None:
            print(f"  X {camera_name}: 적절한 패턴을 찾을 수 없습니다")
            return None, None, None
        
        print(f"  OK {camera_name} 최적 패턴: {best_pattern} ({best_success_count}개 이미지, 방법 {best_method})")
        return best_objpoints, best_imgpoints, best_pattern
    
    def calibrate_individual_camera(self, images, camera_name):
        """개별 카메라 캘리브레이션"""
        print(f"\n{camera_name} 개별 캘리브레이션 중...")
        
        objpoints, imgpoints, pattern = self.find_best_chessboard_pattern(images, camera_name)
        
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
        baseline_pixels = np.linalg.norm(T)
        
        # 픽셀당 실제 크기 계산 (470mm 베이스라인 기준)
        pixel_to_mm = self.baseline_mm / baseline_pixels
        
        print(f"OK 스테레오 캘리브레이션 완료")
        print(f"  재투영 오차: {ret:.3f}")
        print(f"  베이스라인 (픽셀): {baseline_pixels:.2f}")
        print(f"  베이스라인 (실제): {self.baseline_mm}mm")
        print(f"  픽셀당 크기: {pixel_to_mm:.3f}mm")
        
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
            'baseline_pixels': baseline_pixels,
            'baseline_mm': self.baseline_mm,
            'pixel_to_mm_ratio': pixel_to_mm,
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
            print("X 캘리브레이션 이미지를 찾을 수 없습니다")
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
    
    def save_calibration(self, calibration_data, filename="enhanced_stereo_calibration_470mm.json"):
        """캘리브레이션 데이터 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nOK 캘리브레이션 데이터 저장: {filename}")
    
    def run_calibration(self):
        """전체 캘리브레이션 실행"""
        print("=== 강화된 스테레오 캘리브레이션 시작 ===")
        
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
        print(f"베이스라인 (실제): {calibration_data['baseline_mm']}mm")
        print(f"픽셀당 크기: {calibration_data['pixel_to_mm_ratio']:.3f}mm")
        print(f"이미지 크기: {calibration_data['image_size']}")
        print(f"캘리브레이션 날짜: {calibration_data['calibration_date']}")
        
        # 카메라 매트릭스 정보
        K1 = np.array(calibration_data['camera_matrix_1'])
        K2 = np.array(calibration_data['camera_matrix_2'])
        
        print(f"\nCam1 내부 파라미터:")
        print(f"  초점거리: fx={K1[0,0]:.2f}, fy={K1[1,1]:.2f}")
        print(f"  주점: cx={K1[0,2]:.2f}, cy={K1[1,2]:.2f}")
        
        print(f"\nCam2 내부 파라미터:")
        print(f"  초점거리: fx={K2[0,0]:.2f}, fy={K2[1,1]:.2f}")
        print(f"  주점: cx={K2[0,2]:.2f}, cy={K2[1,2]:.2f}")
        
        # 스테레오 변환 정보
        T = np.array(calibration_data['translation_vector'])
        print(f"\n스테레오 변환:")
        print(f"  변위 벡터 T: [{T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}]")

def main():
    """메인 함수"""
    calibrator = EnhancedStereoCalibration()
    calibration_data = calibrator.run_calibration()
    
    if calibration_data:
        print("\nOK 캘리브레이션이 성공적으로 완료되었습니다!")
        print("결과 파일: enhanced_stereo_calibration_470mm.json")
        print("\n다음 단계:")
        print("1. 캘리브레이션 결과 검증")
        print("2. 스테레오 정합 및 깊이 맵 생성")
        print("3. 3D 위치 추정 테스트")
    else:
        print("\nX 캘리브레이션이 실패했습니다.")

if __name__ == "__main__":
    main()
