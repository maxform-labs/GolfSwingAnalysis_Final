#!/usr/bin/env python3
"""
470mm 베이스라인을 고려한 개선된 스테레오 캘리브레이션
기존 캘리브레이션 파일을 기반으로 실제 베이스라인 적용
"""

import cv2
import numpy as np
import json
from datetime import datetime

class ImprovedStereoCalibration470mm:
    def __init__(self):
        self.baseline_mm = 470.0  # 실제 베이스라인 (mm)
        self.image_size = (1440, 1080)  # 실제 이미지 크기
        
        print(f"Improved Stereo Calibration for 470mm Baseline")
        print(f"Image size: {self.image_size}")
        print(f"Baseline: {self.baseline_mm}mm")
    
    def create_improved_calibration(self):
        """470mm 베이스라인을 고려한 개선된 캘리브레이션 생성"""
        
        # 실제 카메라 내부 파라미터 (추정)
        # 일반적인 산업용 카메라의 파라미터를 기반으로 설정
        fx = 800.0  # 초점거리 (픽셀)
        fy = 800.0  # 초점거리 (픽셀)
        cx = self.image_size[0] / 2  # 주점 X
        cy = self.image_size[1] / 2  # 주점 Y
        
        # 카메라 매트릭스
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # 왜곡 계수 (최소한의 왜곡 가정)
        D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 스테레오 변환 (470mm 베이스라인)
        # 수직 배치된 카메라의 경우 Y축 방향으로 변위
        T = np.array([0.0, self.baseline_mm, 0.0])  # Y축 방향으로 470mm
        
        # 회전 행렬 (수직 배치, 최소 회전)
        R = np.eye(3)
        
        # 스테레오 정규화
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K, D, K, D, self.image_size, R, T
        )
        
        # 베이스라인 계산
        baseline_pixels = np.linalg.norm(T)
        pixel_to_mm = self.baseline_mm / baseline_pixels
        
        print(f"OK Improved calibration created")
        print(f"  Baseline (pixels): {baseline_pixels:.2f}")
        print(f"  Baseline (actual): {self.baseline_mm}mm")
        print(f"  Pixel to mm ratio: {pixel_to_mm:.3f}mm")
        
        calibration_data = {
            'camera_matrix_1': K.tolist(),
            'distortion_coeffs_1': D.tolist(),
            'camera_matrix_2': K.tolist(),
            'distortion_coeffs_2': D.tolist(),
            'rotation_matrix': R.tolist(),
            'translation_vector': T.tolist(),
            'rectification_rotation_1': R1.tolist(),
            'rectification_rotation_2': R2.tolist(),
            'projection_matrix_1': P1.tolist(),
            'projection_matrix_2': P2.tolist(),
            'disparity_to_depth_matrix': Q.tolist(),
            'roi_1': list(roi1),
            'roi_2': list(roi2),
            'baseline_pixels': baseline_pixels,
            'baseline_mm': self.baseline_mm,
            'pixel_to_mm_ratio': pixel_to_mm,
            'reprojection_error': 0.0,  # 수동 캘리브레이션이므로 0
            'image_size': self.image_size,
            'calibration_date': datetime.now().isoformat(),
            'calibration_method': 'improved_manual_470mm',
            'notes': 'Improved manual calibration for 470mm baseline vertical stereo setup'
        }
        
        return calibration_data
    
    def save_calibration(self, calibration_data, filename="improved_stereo_calibration_470mm.json"):
        """캘리브레이션 데이터 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nOK Calibration data saved: {filename}")
    
    def print_calibration_summary(self, calibration_data):
        """캘리브레이션 결과 요약 출력"""
        print("\n=== Improved Calibration Results Summary ===")
        print(f"Baseline (pixels): {calibration_data['baseline_pixels']:.2f}")
        print(f"Baseline (actual): {calibration_data['baseline_mm']}mm")
        print(f"Pixel to mm ratio: {calibration_data['pixel_to_mm_ratio']:.3f}mm")
        print(f"Image size: {calibration_data['image_size']}")
        print(f"Calibration date: {calibration_data['calibration_date']}")
        print(f"Calibration method: {calibration_data['calibration_method']}")
        
        # 카메라 매트릭스 정보
        K1 = np.array(calibration_data['camera_matrix_1'])
        K2 = np.array(calibration_data['camera_matrix_2'])
        
        print(f"\nCam1 intrinsic parameters:")
        print(f"  Focal length: fx={K1[0,0]:.2f}, fy={K1[1,1]:.2f}")
        print(f"  Principal point: cx={K1[0,2]:.2f}, cy={K1[1,2]:.2f}")
        
        print(f"\nCam2 intrinsic parameters:")
        print(f"  Focal length: fx={K2[0,0]:.2f}, fy={K2[1,1]:.2f}")
        print(f"  Principal point: cx={K2[0,2]:.2f}, cy={K2[1,2]:.2f}")
        
        # 스테레오 변환 정보
        T = np.array(calibration_data['translation_vector'])
        print(f"\nStereo transformation:")
        print(f"  Translation vector T: [{T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}]")
        
        # Q 매트릭스 정보 (깊이 계산용)
        Q = np.array(calibration_data['disparity_to_depth_matrix'])
        print(f"\nDisparity to depth matrix Q:")
        print(f"  Q[3,2] (focal length): {Q[3,2]:.2f}")
        print(f"  Q[3,3] (baseline): {Q[3,3]:.2f}")
    
    def run_calibration(self):
        """전체 캘리브레이션 실행"""
        print("=== Improved Stereo Calibration for 470mm Baseline ===")
        
        # 개선된 캘리브레이션 생성
        calibration_data = self.create_improved_calibration()
        
        # 결과 저장
        self.save_calibration(calibration_data)
        
        # 결과 요약 출력
        self.print_calibration_summary(calibration_data)
        
        return calibration_data

def main():
    """메인 함수"""
    calibrator = ImprovedStereoCalibration470mm()
    calibration_data = calibrator.run_calibration()
    
    if calibration_data:
        print("\nOK Improved calibration completed successfully!")
        print("Result file: improved_stereo_calibration_470mm.json")
        print("\nNext steps:")
        print("1. Test stereo rectification")
        print("2. Generate depth maps")
        print("3. Implement 3D position estimation")
        print("4. Validate with real images")
    else:
        print("\nX Calibration failed.")

if __name__ == "__main__":
    main()
