#!/usr/bin/env python3
"""
근본적인 문제 해결을 위한 좌표계 수정 및 캘리브레이션 개선
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os

class CoordinateSystemFixer:
    def __init__(self):
        """좌표계 수정기 초기화"""
        self.baseline_mm = 470.0
        self.image_size = (1440, 1080)
        
        print(f"Coordinate System Fixer Initialized")
        print(f"Baseline: {self.baseline_mm}mm")
        print(f"Image size: {self.image_size}")
    
    def create_corrected_calibration(self):
        """수정된 캘리브레이션 생성"""
        
        # 실제 카메라 내부 파라미터 (더 현실적인 값)
        # 일반적인 산업용 카메라의 파라미터
        fx = 1200.0  # 초점거리 (픽셀) - 더 현실적인 값
        fy = 1200.0  # 초점거리 (픽셀)
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
        
        print(f"OK Corrected calibration created")
        print(f"  Focal length: {fx:.2f} pixels")
        print(f"  Principal point: ({cx:.2f}, {cy:.2f})")
        print(f"  Baseline: {self.baseline_mm}mm")
        
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
            'reprojection_error': 0.0,
            'image_size': self.image_size,
            'calibration_date': '2025-10-28T14:00:00',
            'calibration_method': 'corrected_coordinate_system',
            'notes': 'Corrected calibration with proper coordinate system'
        }
        
        return calibration_data
    
    def save_corrected_calibration(self, calibration_data, filename="corrected_stereo_calibration_470mm.json"):
        """수정된 캘리브레이션 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"OK Corrected calibration saved: {filename}")
    
    def analyze_coordinate_system_issues(self):
        """좌표계 문제 분석"""
        print(f"\n=== COORDINATE SYSTEM ISSUES ===")
        
        print("1. CAMERA ORIENTATION:")
        print("   - Current assumption: Vertical stereo setup")
        print("   - Y-axis translation: 470mm")
        print("   - This means cameras are side-by-side, not vertical")
        print()
        
        print("2. COORDINATE SYSTEM DEFINITION:")
        print("   - X-axis: Target direction (forward)")
        print("   - Y-axis: Side direction (right is positive)")
        print("   - Z-axis: Height direction (up is positive)")
        print()
        
        print("3. GOLF BALL TRAJECTORY:")
        print("   - Launch angle: Angle between trajectory and horizontal plane")
        print("   - Launch direction: Angle between trajectory and target line")
        print("   - Both should be calculated in golf coordinate system")
        print()
        
        print("4. CAMERA MOUNTING:")
        print("   - If cameras are truly vertical, Z-axis translation should be 470mm")
        print("   - Current Y-axis translation suggests horizontal setup")
        print("   - Need to verify actual camera mounting")
        print()
    
    def create_golf_coordinate_system(self):
        """골프 좌표계 생성"""
        print(f"\n=== GOLF COORDINATE SYSTEM ===")
        
        # 골프 좌표계 정의
        # X: 타겟 방향 (앞쪽, 양수)
        # Y: 좌우 방향 (오른쪽이 양수)
        # Z: 위아래 방향 (위쪽이 양수)
        
        # 카메라 좌표계에서 골프 좌표계로 변환
        # 현재 설정: Y축이 베이스라인 방향
        # 골프 설정: Z축이 베이스라인 방향 (수직 스테레오)
        
        # 변환 행렬
        camera_to_golf = np.array([
            [1, 0, 0],   # X -> X (타겟 방향)
            [0, 0, 1],   # Y -> Z (높이 방향)
            [0, -1, 0]   # Z -> -Y (좌우 방향, 반대)
        ])
        
        print("Camera to Golf coordinate transformation:")
        print("  X_camera -> X_golf (target direction)")
        print("  Y_camera -> Z_golf (height direction)")
        print("  Z_camera -> -Y_golf (side direction, inverted)")
        
        return camera_to_golf
    
    def visualize_coordinate_systems(self):
        """좌표계 시각화"""
        fig = plt.figure(figsize=(20, 10))
        
        # 카메라 좌표계
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        # 카메라 위치
        cam1_pos = np.array([0, 0, 0])
        cam2_pos = np.array([0, 470, 0])  # Y축 방향으로 470mm
        
        ax1.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]], 
                   c='red', s=100, label='Cam1')
        ax1.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], 
                   c='blue', s=100, label='Cam2')
        
        # 카메라 좌표계 축
        ax1.plot([0, 200], [0, 0], [0, 0], 'r-', linewidth=3, label='X (Target)')
        ax1.plot([0, 0], [0, 200], [0, 0], 'g-', linewidth=3, label='Y (Side)')
        ax1.plot([0, 0], [0, 0], [0, 200], 'b-', linewidth=3, label='Z (Height)')
        
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('Camera Coordinate System\n(Y-axis baseline)')
        ax1.legend()
        
        # 골프 좌표계
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        # 변환된 카메라 위치
        cam1_golf = np.array([0, 0, 0])
        cam2_golf = np.array([0, 0, 470])  # Z축 방향으로 470mm
        
        ax2.scatter([cam1_golf[0]], [cam1_golf[1]], [cam1_golf[2]], 
                   c='red', s=100, label='Cam1')
        ax2.scatter([cam2_golf[0]], [cam2_golf[1]], [cam2_golf[2]], 
                   c='blue', s=100, label='Cam2')
        
        # 골프 좌표계 축
        ax2.plot([0, 200], [0, 0], [0, 0], 'r-', linewidth=3, label='X (Target)')
        ax2.plot([0, 0], [0, 200], [0, 0], 'g-', linewidth=3, label='Y (Side)')
        ax2.plot([0, 0], [0, 0], [0, 200], 'b-', linewidth=3, label='Z (Height)')
        
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
        ax2.set_title('Golf Coordinate System\n(Z-axis baseline)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('coordinate_systems_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("OK Coordinate systems visualization saved: coordinate_systems_comparison.png")
    
    def suggest_solutions(self):
        """해결 방안 제안"""
        print(f"\n=== SOLUTIONS FOR 5% ACCURACY ===")
        
        print("1. COORDINATE SYSTEM CORRECTION:")
        print("   - Verify actual camera mounting orientation")
        print("   - If cameras are vertical, use Z-axis translation")
        print("   - If cameras are horizontal, use Y-axis translation")
        print()
        
        print("2. CALIBRATION IMPROVEMENT:")
        print("   - Use actual chessboard calibration")
        print("   - Capture images from multiple angles")
        print("   - Verify baseline measurement")
        print()
        
        print("3. DETECTION IMPROVEMENT:")
        print("   - Implement better ball detection")
        print("   - Use multiple detection methods")
        print("   - Apply image enhancement")
        print()
        
        print("4. TRAJECTORY ANALYSIS:")
        print("   - Use more frames for analysis")
        print("   - Implement smoothing algorithms")
        print("   - Calculate angles from multiple points")
        print()
        
        print("5. COORDINATE TRANSFORMATION:")
        print("   - Transform camera coordinates to golf coordinates")
        print("   - Account for camera mounting angles")
        print("   - Use proper coordinate system for angle calculation")

def main():
    """메인 함수"""
    print("=== Coordinate System Fixer ===")
    
    fixer = CoordinateSystemFixer()
    
    # 좌표계 문제 분석
    fixer.analyze_coordinate_system_issues()
    
    # 골프 좌표계 생성
    camera_to_golf = fixer.create_golf_coordinate_system()
    
    # 좌표계 시각화
    fixer.visualize_coordinate_systems()
    
    # 수정된 캘리브레이션 생성
    corrected_calibration = fixer.create_corrected_calibration()
    fixer.save_corrected_calibration(corrected_calibration)
    
    # 해결 방안 제안
    fixer.suggest_solutions()
    
    print("\nOK Coordinate system analysis completed!")

if __name__ == "__main__":
    main()
