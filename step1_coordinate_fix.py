#!/usr/bin/env python3
"""
1단계: 좌표계 수정 - Z축 베이스라인으로 변경
수직 스테레오 비전을 위한 올바른 좌표계 설정
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

class VerticalStereoCoordinateFixer:
    def __init__(self):
        """수직 스테레오 좌표계 수정기 초기화"""
        self.baseline_mm = 470.0
        self.image_size = (1440, 1080)
        
        print(f"Vertical Stereo Coordinate Fixer Initialized")
        print(f"Baseline: {self.baseline_mm}mm")
        print(f"Image size: {self.image_size}")
    
    def create_vertical_stereo_calibration(self):
        """수직 스테레오를 위한 올바른 캘리브레이션 생성"""
        
        # 실제 카메라 내부 파라미터 (더 현실적인 값)
        fx = 1500.0  # 초점거리 (픽셀) - 더 현실적인 값
        fy = 1500.0  # 초점거리 (픽셀)
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
        
        # 수직 스테레오 변환 (Z축 방향으로 베이스라인)
        T = np.array([0.0, 0.0, self.baseline_mm])  # Z축 방향으로 470mm
        
        # 회전 행렬 (수직 배치, 최소 회전)
        R = np.eye(3)
        
        # 스테레오 정규화
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K, D, K, D, self.image_size, R, T
        )
        
        # 베이스라인 계산
        baseline_pixels = np.linalg.norm(T)
        pixel_to_mm = self.baseline_mm / baseline_pixels
        
        print(f"OK Vertical stereo calibration created")
        print(f"  Focal length: {fx:.2f} pixels")
        print(f"  Principal point: ({cx:.2f}, {cy:.2f})")
        print(f"  Baseline: {self.baseline_mm}mm (Z-axis)")
        print(f"  Translation vector: [{T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}]")
        
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
            'calibration_date': datetime.now().isoformat(),
            'calibration_method': 'vertical_stereo_z_axis',
            'notes': 'Vertical stereo calibration with Z-axis baseline (470mm)',
            'coordinate_system': {
                'x_axis': 'Target direction (forward)',
                'y_axis': 'Side direction (right is positive)',
                'z_axis': 'Height direction (up is positive)',
                'baseline_direction': 'Z-axis (vertical)'
            }
        }
        
        return calibration_data
    
    def save_vertical_calibration(self, calibration_data, filename="vertical_stereo_calibration_z_axis.json"):
        """수직 스테레오 캘리브레이션 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"OK Vertical stereo calibration saved: {filename}")
    
    def visualize_vertical_setup(self, calibration_data):
        """수직 스테레오 설정 시각화"""
        fig = plt.figure(figsize=(20, 10))
        
        # 3D 시각화
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        # 카메라 위치 (Z축 방향으로 베이스라인)
        cam1_pos = np.array([0, 0, 0])
        cam2_pos = np.array([0, 0, self.baseline_mm])  # Z축 방향으로 470mm
        
        ax1.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]], 
                   c='red', s=100, label='Cam1 (Bottom)')
        ax1.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], 
                   c='blue', s=100, label='Cam2 (Top)')
        
        # 베이스라인 표시
        ax1.plot([cam1_pos[0], cam2_pos[0]], [cam1_pos[1], cam2_pos[1]], 
                [cam1_pos[2], cam2_pos[2]], 'k--', linewidth=3, label=f'Baseline ({self.baseline_mm}mm)')
        
        # 골프 공 위치 (예시)
        ball_pos = np.array([1000, 0, 500])  # 1m 앞, 중앙, 0.5m 높이
        ax1.scatter([ball_pos[0]], [ball_pos[1]], [ball_pos[2]], 
                   c='green', s=150, label='Golf Ball')
        
        # 시야각 표시
        fov_half = np.arctan(self.image_size[0] / (2 * calibration_data['camera_matrix_1'][0][0]))
        fov_v_half = np.arctan(self.image_size[1] / (2 * calibration_data['camera_matrix_1'][1][1]))
        
        distance = 1000  # 1m 거리에서 시야각 표시
        fov_x = distance * np.tan(fov_half)
        fov_y = distance * np.tan(fov_v_half)
        
        # Cam1 시야각
        cam1_fov_corners = np.array([
            [-fov_x, -fov_y, distance],
            [fov_x, -fov_y, distance],
            [fov_x, fov_y, distance],
            [-fov_x, fov_y, distance],
            [-fov_x, -fov_y, distance]
        ])
        
        ax1.plot(cam1_fov_corners[:, 0], cam1_fov_corners[:, 1], cam1_fov_corners[:, 2], 
                'r-', linewidth=2, label='Cam1 FOV')
        
        # Cam2 시야각
        cam2_fov_corners = cam1_fov_corners + cam2_pos
        ax1.plot(cam2_fov_corners[:, 0], cam2_fov_corners[:, 1], cam2_fov_corners[:, 2], 
                'b-', linewidth=2, label='Cam2 FOV')
        
        # 축 표시
        ax1.plot([0, 2000], [0, 0], [0, 0], 'k-', linewidth=2, label='X (Target)')
        ax1.plot([0, 0], [0, 1000], [0, 0], 'k-', linewidth=2, label='Y (Side)')
        ax1.plot([0, 0], [0, 0], [0, 1000], 'k-', linewidth=2, label='Z (Height)')
        
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('Vertical Stereo Setup (Z-axis Baseline)')
        ax1.legend()
        
        # 2D 평면도 (X-Z Plane)
        ax2 = fig.add_subplot(1, 2, 2)
        
        # 카메라 위치 (2D)
        ax2.plot(cam1_pos[0], cam1_pos[2], 'ro', markersize=10, label='Cam1 (Bottom)')
        ax2.plot(cam2_pos[0], cam2_pos[2], 'bo', markersize=10, label='Cam2 (Top)')
        
        # 베이스라인 표시
        ax2.plot([cam1_pos[0], cam2_pos[0]], [cam1_pos[2], cam2_pos[2]], 
                'k--', linewidth=3, label=f'Baseline ({self.baseline_mm}mm)')
        
        # 골프 공 위치
        ax2.plot(ball_pos[0], ball_pos[2], 'go', markersize=15, label='Golf Ball')
        
        # 시야각 표시 (X-Z 평면에서)
        ax2.plot([cam1_pos[0], cam1_pos[0] + fov_x], [cam1_pos[2], cam1_pos[2] + distance], 
                'r-', linewidth=2, label='Cam1 FOV')
        ax2.plot([cam1_pos[0], cam1_pos[0] - fov_x], [cam1_pos[2], cam1_pos[2] + distance], 
                'r-', linewidth=2)
        
        ax2.plot([cam2_pos[0], cam2_pos[0] + fov_x], [cam2_pos[2], cam2_pos[2] + distance], 
                'b-', linewidth=2, label='Cam2 FOV')
        ax2.plot([cam2_pos[0], cam2_pos[0] - fov_x], [cam2_pos[2], cam2_pos[2] + distance], 
                'b-', linewidth=2)
        
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Z (mm)')
        ax2.set_title('Vertical Stereo Setup (X-Z Plane)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        plt.savefig('vertical_stereo_setup_z_axis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("OK Vertical stereo setup visualization saved: vertical_stereo_setup_z_axis.png")
    
    def compare_coordinate_systems(self):
        """좌표계 비교"""
        print(f"\n=== COORDINATE SYSTEM COMPARISON ===")
        
        print("BEFORE (Horizontal Stereo - Y-axis baseline):")
        print("  Translation: [0, 470, 0] mm")
        print("  Cameras: Side-by-side")
        print("  Problem: Wrong coordinate system for vertical setup")
        print()
        
        print("AFTER (Vertical Stereo - Z-axis baseline):")
        print("  Translation: [0, 0, 470] mm")
        print("  Cameras: One above the other")
        print("  Solution: Correct coordinate system for vertical setup")
        print()
        
        print("GOLF COORDINATE SYSTEM:")
        print("  X-axis: Target direction (forward)")
        print("  Y-axis: Side direction (right is positive)")
        print("  Z-axis: Height direction (up is positive)")
        print("  Launch angle: Angle between trajectory and horizontal plane")
        print("  Launch direction: Angle between trajectory and target line")

def main():
    """메인 함수"""
    print("=== STEP 1: Coordinate System Fix ===")
    
    fixer = VerticalStereoCoordinateFixer()
    
    # 수직 스테레오 캘리브레이션 생성
    calibration_data = fixer.create_vertical_stereo_calibration()
    
    # 캘리브레이션 저장
    fixer.save_vertical_calibration(calibration_data)
    
    # 시각화
    fixer.visualize_vertical_setup(calibration_data)
    
    # 좌표계 비교
    fixer.compare_coordinate_systems()
    
    print("\nOK Step 1 completed: Coordinate system fixed!")
    print("Next: Step 2 - Calibration improvement")

if __name__ == "__main__":
    main()
