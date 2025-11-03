#!/usr/bin/env python3
"""
골프공 검출 결과 시각화 시스템
검출된 골프공을 이미지에 표시하고 저장
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

class GolfBallVisualizer:
    def __init__(self):
        """골프공 시각화기 초기화"""
        print("Golf Ball Visualizer Initialized")
        print("Creating visualizations of detected golf balls")
    
    def detect_golf_ball_hough(self, img):
        """허프 원을 사용한 골프공 검출 (최적화된 파라미터)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 최적화된 허프 원 파라미터
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 
            dp=1, minDist=10,
            param1=30, param2=20, 
            minRadius=2, maxRadius=20
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # 가장 밝은 원 선택
            best_circle = self.find_brightest_circle(circles, gray)
            if best_circle is not None:
                center = (best_circle[0], best_circle[1])
                radius = best_circle[2]
                return center, radius
        
        return None, None
    
    def find_brightest_circle(self, circles, gray_img):
        """가장 밝은 원 찾기"""
        if len(circles) == 0:
            return None
        
        best_circle = None
        best_brightness = 0
        
        for circle in circles:
            center = (circle[0], circle[1])
            radius = circle[2]
            
            # 원 내부의 평균 밝기 계산
            mask = np.zeros(gray_img.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            mean_brightness = cv2.mean(gray_img, mask)[0]
            
            if mean_brightness > best_brightness:
                best_brightness = mean_brightness
                best_circle = circle
        
        return best_circle
    
    def create_detection_visualization(self, img1, img2, center1, center2, radius1, radius2, 
                                     frame_num, disparity=None, position_3d=None):
        """검출 결과 시각화 생성"""
        # 이미지 복사
        img1_vis = img1.copy()
        img2_vis = img2.copy()
        
        # 골프공 위치에 원 그리기
        if center1 is not None and radius1 is not None:
            cv2.circle(img1_vis, center1, radius1, (0, 255, 0), 2)  # 녹색 원
            cv2.circle(img1_vis, center1, 2, (0, 0, 255), -1)       # 빨간색 중심점
            cv2.putText(img1_vis, f"({center1[0]}, {center1[1]})", 
                       (center1[0] + 10, center1[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if center2 is not None and radius2 is not None:
            cv2.circle(img2_vis, center2, radius2, (0, 255, 0), 2)  # 녹색 원
            cv2.circle(img2_vis, center2, 2, (0, 0, 255), -1)       # 빨간색 중심점
            cv2.putText(img2_vis, f"({center2[0]}, {center2[1]})", 
                       (center2[0] + 10, center2[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # matplotlib을 사용한 시각화
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 카메라 1 이미지
        axes[0].imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Camera 1 - Frame {frame_num}\nGolf Ball Detection', fontsize=14, fontweight='bold')
        if center1 is not None:
            axes[0].text(0.02, 0.98, f'Center: ({center1[0]}, {center1[1]})\nRadius: {radius1}px', 
                        transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[0].axis('off')
        
        # 카메라 2 이미지
        axes[1].imshow(cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Camera 2 - Frame {frame_num}\nGolf Ball Detection', fontsize=14, fontweight='bold')
        if center2 is not None:
            axes[1].text(0.02, 0.98, f'Center: ({center2[0]}, {center2[1]})\nRadius: {radius2}px', 
                        transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[1].axis('off')
        
        # 전체 제목
        title = f'Golf Ball Detection - Frame {frame_num}'
        if disparity is not None:
            title += f' (Disparity: {disparity:.1f}px)'
        if position_3d is not None:
            title += f' - 3D: ({position_3d[0]:.1f}, {position_3d[1]:.1f}, {position_3d[2]:.1f})mm'
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # 이미지 저장
        filename = f'golf_ball_detection_frame_{frame_num:02d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization saved: {filename}")
        return filename
    
    def process_all_frames(self, image_folder="data2/driver/2", max_frames=10):
        """모든 프레임 처리 및 시각화"""
        print(f"\n=== PROCESSING ALL FRAMES ===")
        print(f"Image folder: {image_folder}")
        print(f"Max frames: {max_frames}")
        
        # 이미지 파일 목록
        gamma1_files = sorted(glob.glob(f"{image_folder}/Gamma_1_*.bmp"))
        gamma2_files = sorted(glob.glob(f"{image_folder}/Gamma_2_*.bmp"))
        
        print(f"Found {len(gamma1_files)} Gamma_1 images")
        print(f"Found {len(gamma2_files)} Gamma_2 images")
        
        if len(gamma1_files) == 0:
            print("ERROR: No images found!")
            return
        
        successful_detections = 0
        total_frames = min(len(gamma1_files), len(gamma2_files), max_frames)
        
        print(f"\nProcessing {total_frames} frames...")
        print("=" * 60)
        
        for i in range(total_frames):
            print(f"\nFrame {i+1}/{total_frames}:")
            
            img1_path = gamma1_files[i]
            img2_path = gamma2_files[i]
            
            # 이미지 로드
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                print(f"  ERROR: Failed to load images")
                continue
            
            # 골프공 검출
            center1, radius1 = self.detect_golf_ball_hough(img1)
            center2, radius2 = self.detect_golf_ball_hough(img2)
            
            if center1 is not None and center2 is not None:
                successful_detections += 1
                
                # 시차 계산
                disparity = abs(center1[1] - center2[1])
                
                # 3D 위치 계산 (간단한 버전)
                if disparity > 2:
                    focal_length = 1800.0  # 캘리브레이션에서 가져온 값
                    baseline_mm = 470.0
                    depth = (focal_length * baseline_mm) / disparity
                    
                    if 100 < depth < 1500:  # 유효한 깊이 범위
                        x = (center1[0] - 720) * depth / focal_length  # 720은 주점
                        y = ((center1[1] + center2[1]) / 2 - 540) * depth / focal_length  # 540은 주점
                        z = depth
                        position_3d = np.array([x, y, z])
                    else:
                        position_3d = None
                else:
                    position_3d = None
                
                print(f"  SUCCESS: Cam1=({center1[0]}, {center1[1]}, r={radius1}), Cam2=({center2[0]}, {center2[1]}, r={radius2})")
                print(f"  Disparity: {disparity:.1f}px")
                if position_3d is not None:
                    print(f"  3D Position: ({position_3d[0]:.1f}, {position_3d[1]:.1f}, {position_3d[2]:.1f})mm")
                
                # 시각화 생성
                self.create_detection_visualization(img1, img2, center1, center2, radius1, radius2, 
                                                  i+1, disparity, position_3d)
            else:
                print(f"  FAILED: Ball not detected")
                if center1 is None:
                    print(f"    Camera 1: No ball detected")
                if center2 is None:
                    print(f"    Camera 2: No ball detected")
        
        # 최종 결과
        detection_rate = (successful_detections / total_frames) * 100
        
        print(f"\n" + "=" * 60)
        print(f"=== FINAL RESULTS ===")
        print(f"Total frames processed: {total_frames}")
        print(f"Successful detections: {successful_detections}")
        print(f"Detection rate: {detection_rate:.1f}%")
        print(f"Visualization images saved in current directory")
        print(f"=" * 60)
        
        return detection_rate, successful_detections, total_frames
    
    def create_summary_visualization(self, detection_results):
        """검출 결과 요약 시각화"""
        if not detection_results:
            return
        
        # 성공한 프레임들의 정보 수집
        successful_frames = [r for r in detection_results if r['success']]
        
        if not successful_frames:
            print("No successful detections to summarize")
            return
        
        # 시차 분포 히스토그램
        disparities = [r['disparity'] for r in successful_frames]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 시차 분포
        axes[0, 0].hist(disparities, bins=10, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Disparity Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Disparity (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 프레임별 시차
        frame_nums = [r['frame'] for r in successful_frames]
        axes[0, 1].plot(frame_nums, disparities, 'o-', color='red', linewidth=2, markersize=6)
        axes[0, 1].set_title('Disparity by Frame', fontweight='bold')
        axes[0, 1].set_xlabel('Frame Number')
        axes[0, 1].set_ylabel('Disparity (pixels)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3D 위치 (Z 좌표)
        z_coords = [r['position_3d'][2] for r in successful_frames if r['position_3d'] is not None]
        if z_coords:
            axes[1, 0].plot(frame_nums[:len(z_coords)], z_coords, 'o-', color='green', linewidth=2, markersize=6)
            axes[1, 0].set_title('Depth (Z) by Frame', fontweight='bold')
            axes[1, 0].set_xlabel('Frame Number')
            axes[1, 0].set_ylabel('Depth (mm)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 검출 통계
        total_frames = len(detection_results)
        successful_count = len(successful_frames)
        detection_rate = (successful_count / total_frames) * 100
        
        stats_text = f"""Detection Statistics:
        
Total Frames: {total_frames}
Successful: {successful_count}
Detection Rate: {detection_rate:.1f}%

Disparity Stats:
Min: {min(disparities):.1f}px
Max: {max(disparities):.1f}px
Mean: {np.mean(disparities):.1f}px
Std: {np.std(disparities):.1f}px"""
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('Detection Statistics', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('golf_ball_detection_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Summary visualization saved: golf_ball_detection_summary.png")

def main():
    """메인 함수"""
    print("=== GOLF BALL DETECTION VISUALIZER ===")
    print("Creating visualizations of detected golf balls")
    
    visualizer = GolfBallVisualizer()
    
    # 모든 프레임 처리 및 시각화
    detection_rate, successful_detections, total_frames = visualizer.process_all_frames(max_frames=10)
    
    print(f"\n🎯 Golf ball detection completed!")
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"Check the generated PNG files to see the results!")

if __name__ == "__main__":
    main()
