#!/usr/bin/env python3
"""
스테레오 정합 및 깊이 맵 생성
470mm 베이스라인을 고려한 스테레오 비전 시스템
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

class StereoDepthEstimation:
    def __init__(self, calibration_file="improved_stereo_calibration_470mm.json"):
        """스테레오 깊이 추정 클래스 초기화"""
        self.calibration_file = calibration_file
        self.load_calibration()
        
        print(f"Stereo Depth Estimation Initialized")
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
        
        # 정규화 매트릭스
        self.R1 = np.array(self.calibration_data['rectification_rotation_1'])
        self.R2 = np.array(self.calibration_data['rectification_rotation_2'])
        self.P1 = np.array(self.calibration_data['projection_matrix_1'])
        self.P2 = np.array(self.calibration_data['projection_matrix_2'])
        self.Q = np.array(self.calibration_data['disparity_to_depth_matrix'])
        
        # ROI
        self.roi1 = tuple(self.calibration_data['roi_1'])
        self.roi2 = tuple(self.calibration_data['roi_2'])
        
        # 기타 정보
        self.baseline_mm = self.calibration_data['baseline_mm']
        self.image_size = tuple(self.calibration_data['image_size'])
        
        # 스테레오 정합기 초기화
        self.init_stereo_matcher()
    
    def init_stereo_matcher(self):
        """스테레오 정합기 초기화"""
        # SGBM (Semi-Global Block Matching) 파라미터 설정
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,  # 16의 배수여야 함
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # WLS 필터 초기화 (깊이 맵 품질 향상)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.5)
        
        print("OK Stereo matcher initialized")
    
    def rectify_images(self, img1, img2):
        """이미지 정규화"""
        # 스테레오 정규화 맵 생성
        map1x, map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, self.image_size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, self.image_size, cv2.CV_32FC1
        )
        
        # 이미지 정규화
        rectified1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        rectified2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
        
        return rectified1, rectified2
    
    def compute_disparity(self, img1, img2):
        """시차 계산"""
        # 그레이스케일 변환
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 시차 계산
        disparity = self.stereo.compute(gray1, gray2).astype(np.float32) / 16.0
        
        # WLS 필터 적용
        disparity_filtered = self.wls_filter.filter(disparity, gray1, None, gray2)
        
        return disparity_filtered
    
    def disparity_to_depth(self, disparity):
        """시차를 깊이로 변환"""
        # Q 매트릭스를 사용한 깊이 계산
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        
        # 깊이 맵 추출 (Z 좌표)
        depth_map = points_3d[:, :, 2]
        
        # 무효한 깊이 값 제거
        depth_map[depth_map <= 0] = 0
        depth_map[depth_map > 10000] = 0  # 너무 먼 거리 제거
        
        return depth_map
    
    def calculate_3d_position(self, u1, v1, u2, v2):
        """특정 픽셀의 3D 위치 계산"""
        # 시차 계산
        disparity = u1 - u2
        
        if disparity <= 0:
            return None
        
        # 깊이 계산 (Z = (f * B) / disparity)
        focal_length = self.K1[0, 0]  # fx
        depth = (focal_length * self.baseline_mm) / disparity
        
        # 3D 좌표 계산
        x = (u1 - self.K1[0, 2]) * depth / focal_length
        y = (v1 - self.K1[1, 2]) * depth / focal_length
        z = depth
        
        return np.array([x, y, z])
    
    def process_stereo_pair(self, img1_path, img2_path, output_prefix="stereo_result"):
        """스테레오 이미지 쌍 처리"""
        print(f"\nProcessing stereo pair:")
        print(f"  Image 1: {img1_path}")
        print(f"  Image 2: {img2_path}")
        
        # 이미지 로드
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print("X Failed to load images")
            return None
        
        # 이미지 정규화
        rectified1, rectified2 = self.rectify_images(img1, img2)
        
        # 시차 계산
        disparity = self.compute_disparity(rectified1, rectified2)
        
        # 깊이 맵 생성
        depth_map = self.disparity_to_depth(disparity)
        
        # 결과 시각화
        self.visualize_results(rectified1, rectified2, disparity, depth_map, output_prefix)
        
        return {
            'rectified1': rectified1,
            'rectified2': rectified2,
            'disparity': disparity,
            'depth_map': depth_map
        }
    
    def visualize_results(self, rectified1, rectified2, disparity, depth_map, output_prefix):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 정규화된 이미지들
        axes[0, 0].imshow(cv2.cvtColor(rectified1, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Rectified Image 1')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(rectified2, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Rectified Image 2')
        axes[0, 1].axis('off')
        
        # 시차 맵
        disparity_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(disparity, alpha=255/disparity.max()),
            cv2.COLORMAP_JET
        )
        axes[1, 0].imshow(cv2.cvtColor(disparity_vis, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Disparity Map')
        axes[1, 0].axis('off')
        
        # 깊이 맵
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_map, alpha=255/depth_map.max()),
            cv2.COLORMAP_JET
        )
        axes[1, 1].imshow(cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Depth Map')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"OK Visualization saved: {output_prefix}_visualization.png")
    
    def test_with_calibration_images(self):
        """캘리브레이션 이미지로 테스트"""
        calibration_path = "data2/Calibration_image_1025"
        
        # 첫 번째 이미지 쌍으로 테스트
        img1_path = f"{calibration_path}/Cam1_1.bmp"
        img2_path = f"{calibration_path}/Cam2_1.bmp"
        
        if Path(img1_path).exists() and Path(img2_path).exists():
            result = self.process_stereo_pair(img1_path, img2_path, "calibration_test")
            
            if result:
                print("\nOK Calibration test completed")
                
                # 깊이 맵 통계
                depth_map = result['depth_map']
                valid_depths = depth_map[depth_map > 0]
                
                if len(valid_depths) > 0:
                    print(f"Depth map statistics:")
                    print(f"  Valid pixels: {len(valid_depths)}")
                    print(f"  Min depth: {valid_depths.min():.2f}mm")
                    print(f"  Max depth: {valid_depths.max():.2f}mm")
                    print(f"  Mean depth: {valid_depths.mean():.2f}mm")
                    print(f"  Std depth: {valid_depths.std():.2f}mm")
                else:
                    print("X No valid depth values found")
            else:
                print("X Calibration test failed")
        else:
            print("X Calibration images not found")

def main():
    """메인 함수"""
    print("=== Stereo Depth Estimation Test ===")
    
    # 스테레오 깊이 추정 초기화
    stereo_depth = StereoDepthEstimation()
    
    # 캘리브레이션 이미지로 테스트
    stereo_depth.test_with_calibration_images()
    
    print("\nOK Stereo depth estimation system ready!")
    print("\nNext steps:")
    print("1. Test with real golf swing images")
    print("2. Implement ball tracking in 3D")
    print("3. Calculate trajectory and speed")
    print("4. Validate accuracy with known distances")

if __name__ == "__main__":
    main()
