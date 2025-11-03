#!/usr/bin/env python3
"""
골프 공 3D 위치 추정 시스템
470mm 베이스라인을 고려한 실용적인 스테레오 비전
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

class GolfBall3DTracker:
    def __init__(self, calibration_file="improved_stereo_calibration_470mm.json"):
        """골프 공 3D 추적기 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        print(f"Golf Ball 3D Tracker Initialized")
        print(f"Baseline: {self.baseline_mm}mm")
        print(f"Image size: {self.image_size}")
    
    def load_calibration(self):
        """캘리브레이션 데이터 로드"""
        with open(self.calibration_file, 'r', encoding='utf-8') as f:
            self.calibration_data = json.load(f)
        
        # 카메라 매트릭스
        self.K1 = np.array(self.calibration_data['camera_matrix_1'])
        self.K2 = np.array(self.calibration_data['camera_matrix_2'])
        
        # 왜곡 계수
        self.D1 = np.array(self.calibration_data['distortion_coeffs_1'])
        self.D2 = np.array(self.calibration_data['distortion_coeffs_2'])
        
        # 스테레오 변환
        self.R = np.array(self.calibration_data['rotation_matrix'])
        self.T = np.array(self.calibration_data['translation_vector'])
        
        # 기타 정보
        self.baseline_mm = self.calibration_data['baseline_mm']
        self.image_size = tuple(self.calibration_data['image_size'])
        
        # 초점거리 (픽셀)
        self.focal_length = self.K1[0, 0]
        
        print(f"OK Calibration loaded")
        print(f"  Focal length: {self.focal_length:.2f} pixels")
        print(f"  Principal point: ({self.K1[0,2]:.2f}, {self.K1[1,2]:.2f})")
    
    def undistort_images(self, img1, img2):
        """이미지 왜곡 보정"""
        undistorted1 = cv2.undistort(img1, self.K1, self.D1)
        undistorted2 = cv2.undistort(img2, self.K2, self.D2)
        return undistorted1, undistorted2
    
    def calculate_3d_position_simple(self, u1, v1, u2, v2):
        """간단한 3D 위치 계산 (삼각측량)"""
        # 시차 계산
        disparity = u1 - u2
        
        if disparity <= 0:
            return None
        
        # 깊이 계산 (Z = (f * B) / disparity)
        depth = (self.focal_length * self.baseline_mm) / disparity
        
        # 3D 좌표 계산
        x = (u1 - self.K1[0, 2]) * depth / self.focal_length
        y = (v1 - self.K1[1, 2]) * depth / self.focal_length
        z = depth
        
        return np.array([x, y, z])
    
    def detect_ball_center(self, img, method='hough'):
        """골프 공 중심 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if method == 'hough':
            # 허프 원 검출
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                param1=50, param2=30, minRadius=10, maxRadius=100
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # 가장 큰 원 선택
                largest_circle = max(circles, key=lambda x: x[2])
                return (largest_circle[0], largest_circle[1]), largest_circle[2]
        
        elif method == 'contour':
            # 컨투어 기반 검출
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            edges = cv2.Canny(blurred, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 컨투어 선택
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 원형도 확인
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.7:  # 원형도가 높은 경우
                        # 중심과 반지름 계산
                        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                        return (int(x), int(y)), int(radius)
        
        return None, None
    
    def process_stereo_pair(self, img1_path, img2_path, output_prefix="golf_ball_3d"):
        """스테레오 이미지 쌍에서 골프 공 3D 위치 계산"""
        print(f"\nProcessing golf ball 3D position:")
        print(f"  Image 1: {img1_path}")
        print(f"  Image 2: {img2_path}")
        
        # 이미지 로드
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print("X Failed to load images")
            return None
        
        # 왜곡 보정
        undistorted1, undistorted2 = self.undistort_images(img1, img2)
        
        # 골프 공 검출
        center1, radius1 = self.detect_ball_center(undistorted1)
        center2, radius2 = self.detect_ball_center(undistorted2)
        
        if center1 is None or center2 is None:
            print("X Golf ball not detected in one or both images")
            return None
        
        print(f"  Ball detected:")
        print(f"    Cam1: center=({center1[0]}, {center1[1]}), radius={radius1}")
        print(f"    Cam2: center=({center2[0]}, {center2[1]}), radius={radius2}")
        
        # 3D 위치 계산
        u1, v1 = center1
        u2, v2 = center2
        
        # 시차 계산 (Y 좌표는 거의 동일해야 함)
        disparity_x = u1 - u2
        disparity_y = v1 - v2
        
        print(f"  Disparity: dx={disparity_x:.2f}, dy={disparity_y:.2f}")
        
        # 3D 위치 계산
        position_3d = self.calculate_3d_position_simple(u1, v1, u2, v2)
        
        if position_3d is not None:
            print(f"  OK 3D position: ({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f}) mm")
            
            # 결과 시각화
            self.visualize_detection(undistorted1, undistorted2, center1, center2, 
                                   radius1, radius2, position_3d, output_prefix)
            
            return {
                'position_3d': position_3d,
                'center1': center1,
                'center2': center2,
                'radius1': radius1,
                'radius2': radius2,
                'disparity': (disparity_x, disparity_y)
            }
        else:
            print("X Failed to calculate 3D position")
            return None
    
    def visualize_detection(self, img1, img2, center1, center2, radius1, radius2, 
                           position_3d, output_prefix):
        """검출 결과 시각화"""
        # 이미지에 원 그리기
        img1_vis = img1.copy()
        img2_vis = img2.copy()
        
        cv2.circle(img1_vis, center1, radius1, (0, 255, 0), 2)
        cv2.circle(img1_vis, center1, 2, (0, 0, 255), -1)
        
        cv2.circle(img2_vis, center2, radius2, (0, 255, 0), 2)
        cv2.circle(img2_vis, center2, 2, (0, 0, 255), -1)
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 원본 이미지들
        axes[0].imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Camera 1\nBall at ({center1[0]}, {center1[1]})')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Camera 2\nBall at ({center2[0]}, {center2[1]})')
        axes[1].axis('off')
        
        # 3D 위치 시각화
        ax = axes[2]
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        
        # 골프 공 위치
        ax.scatter([position_3d[0]], [position_3d[1]], [position_3d[2]], 
                  c='red', s=100, label='Golf Ball')
        
        # 카메라 위치 (원점과 베이스라인 거리)
        ax.scatter([0], [0], [0], c='blue', s=50, label='Cam1')
        ax.scatter([0], [self.baseline_mm], [0], c='green', s=50, label='Cam2')
        
        # 축 설정
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Position\n({position_3d[0]:.1f}, {position_3d[1]:.1f}, {position_3d[2]:.1f}) mm')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_detection.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"OK Visualization saved: {output_prefix}_detection.png")
    
    def test_with_calibration_images(self):
        """캘리브레이션 이미지로 테스트"""
        calibration_path = "data2/Calibration_image_1025"
        
        # 여러 이미지 쌍으로 테스트
        test_pairs = [
            ("Cam1_1.bmp", "Cam2_1.bmp"),
            ("Cam1_5.bmp", "Cam2_5.bmp"),
            ("Cam1_10.bmp", "Cam2_10.bmp")
        ]
        
        results = []
        
        for img1_name, img2_name in test_pairs:
            img1_path = f"{calibration_path}/{img1_name}"
            img2_path = f"{calibration_path}/{img2_name}"
            
            if Path(img1_path).exists() and Path(img2_path).exists():
                result = self.process_stereo_pair(img1_path, img2_path, f"test_{img1_name.split('_')[1].split('.')[0]}")
                
                if result:
                    results.append(result)
        
        if results:
            print(f"\nOK Successfully processed {len(results)} image pairs")
            
            # 평균 위치 계산
            positions = [r['position_3d'] for r in results]
            avg_position = np.mean(positions, axis=0)
            
            print(f"Average 3D position: ({avg_position[0]:.2f}, {avg_position[1]:.2f}, {avg_position[2]:.2f}) mm")
            
            # 위치 변화 분석
            if len(positions) > 1:
                positions_array = np.array(positions)
                std_position = np.std(positions_array, axis=0)
                print(f"Position std: ({std_position[0]:.2f}, {std_position[1]:.2f}, {std_position[2]:.2f}) mm")
        else:
            print("X No successful detections")

def main():
    """메인 함수"""
    print("=== Golf Ball 3D Tracking System ===")
    
    # 골프 공 3D 추적기 초기화
    tracker = GolfBall3DTracker()
    
    # 캘리브레이션 이미지로 테스트
    tracker.test_with_calibration_images()
    
    print("\nOK Golf ball 3D tracking system ready!")
    print("\nNext steps:")
    print("1. Test with real golf swing sequences")
    print("2. Implement trajectory tracking")
    print("3. Calculate ball speed and launch angle")
    print("4. Validate accuracy with known measurements")

if __name__ == "__main__":
    main()
