#!/usr/bin/env python3
"""
다중 디렉토리 골프공 검출 시스템
디렉토리 3, 4, 5의 모든 이미지에 대해 골프공 검출 수행
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

class MultiDirectoryGolfDetector:
    def __init__(self, base_dir="data2/driver"):
        """다중 디렉토리 골프공 검출기 초기화"""
        self.base_dir = base_dir
        self.directories = ["3", "4", "5"]
        
        print("Multi-Directory Golf Ball Detector Initialized")
        print(f"Base directory: {self.base_dir}")
        print(f"Target directories: {self.directories}")
    
    def detect_golf_ball_optimized(self, img):
        """최적화된 골프공 검출"""
        # 1. 허프 원 검출 (가장 효과적인 방법)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 최적화된 허프 원 파라미터
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 
            dp=1, minDist=5,
            param1=20, param2=15, 
            minRadius=2, maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # 가장 밝은 원 선택
            best_circle = self.find_brightest_circle(circles, gray)
            if best_circle is not None:
                center = (best_circle[0], best_circle[1])
                radius = best_circle[2]
                
                # 밝기 확인
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, radius, 255, -1)
                mean_brightness = cv2.mean(gray, mask)[0]
                
                if mean_brightness > 30:
                    return center, radius, mask
        
        # 2. HSV 색상 기반 검출 (보조 방법)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 매우 넓은 흰색 범위
        white_ranges = [
            ([0, 0, 160], [180, 80, 255]),  # Very wide white
            ([0, 0, 140], [180, 100, 255]), # Ultra wide white
        ]
        
        for lower, upper in white_ranges:
            lower_white = np.array(lower)
            upper_white = np.array(upper)
            
            # 흰색 마스크 생성
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            
            # 컨투어 검출
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # 가장 원형에 가까운 컨투어 선택
            best_contour = self.find_best_circular_contour(contours, min_area=5)
            
            if best_contour is not None:
                (x, y), radius = cv2.minEnclosingCircle(best_contour)
                center = (int(x), int(y))
                
                # 밝기 확인
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, int(radius), 255, -1)
                mean_brightness = cv2.mean(gray, mask)[0]
                
                if mean_brightness > 30:
                    return center, int(radius), white_mask
        
        return None, None, None
    
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
    
    def find_best_circular_contour(self, contours, min_area=5):
        """가장 원형에 가까운 컨투어 찾기"""
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.2:
                        score = circularity * area
                        if score > best_score:
                            best_score = score
                            best_contour = contour
        
        return best_contour
    
    def create_detection_visualization(self, img, center, radius, frame_num, camera_name, 
                                     detection_success, output_dir):
        """검출 결과 시각화"""
        # 이미지 복사
        img_vis = img.copy()
        
        # 골프공 위치에 원 그리기
        if center is not None and radius is not None:
            color = (0, 255, 0) if detection_success else (0, 0, 255)  # 성공: 녹색, 실패: 빨간색
            cv2.circle(img_vis, center, radius, color, 2)
            cv2.circle(img_vis, center, 2, (0, 0, 255), -1)
            cv2.putText(img_vis, f"({center[0]}, {center[1]})", 
                       (center[0] + 10, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # matplotlib을 사용한 시각화
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 상태에 따른 색상 결정
        if detection_success:
            title_color = 'green'
            status_text = 'GOLF BALL DETECTED'
        else:
            title_color = 'blue'
            status_text = 'NO GOLF BALL DETECTED'
        
        # 이미지 표시
        ax.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{camera_name} - Frame {frame_num}\n{status_text}', 
                     fontsize=14, fontweight='bold', color=title_color)
        
        if center is not None:
            ax.text(0.02, 0.98, f'Center: ({center[0]}, {center[1]})\nRadius: {radius}px', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axis('off')
        
        plt.tight_layout()
        
        # 이미지 저장
        filename = f'{camera_name}_{frame_num:02d}_detect.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def process_directory(self, directory_name):
        """단일 디렉토리 처리"""
        input_dir = os.path.join(self.base_dir, directory_name)
        output_dir = f"driver_{directory_name}_result"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== PROCESSING DIRECTORY {directory_name} ===")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        # 이미지 파일 목록
        cam1_files = sorted(glob.glob(os.path.join(input_dir, "Cam1_*.bmp")))
        cam2_files = sorted(glob.glob(os.path.join(input_dir, "Cam2_*.bmp")))
        
        print(f"Found {len(cam1_files)} Cam1 images")
        print(f"Found {len(cam2_files)} Cam2 images")
        
        if len(cam1_files) == 0:
            print("ERROR: No images found!")
            return
        
        # 결과 통계
        total_frames = len(cam1_files)
        cam1_detections = 0
        cam2_detections = 0
        
        detection_results = []
        
        print(f"\nProcessing {total_frames} frames...")
        print("=" * 80)
        
        for i in range(total_frames):
            frame_num = i + 1
            img1_path = cam1_files[i] if i < len(cam1_files) else None
            img2_path = cam2_files[i] if i < len(cam2_files) else None
            
            print(f"\nFrame {frame_num:2d}: ", end="")
            
            # Cam1 처리
            if img1_path and os.path.exists(img1_path):
                img1 = cv2.imread(img1_path)
                if img1 is not None:
                    center1, radius1, mask1 = self.detect_golf_ball_optimized(img1)
                    detection1_success = center1 is not None
                    
                    if detection1_success:
                        cam1_detections += 1
                        print(f"Cam1: DETECTED ", end="")
                    else:
                        print(f"Cam1: NOT DETECTED ", end="")
                    
                    # 시각화 생성
                    filepath1 = self.create_detection_visualization(
                        img1, center1, radius1, frame_num, "Cam1", 
                        detection1_success, output_dir
                    )
                    
                    # 결과 저장
                    result1 = {
                        'camera': 'Cam1',
                        'frame': frame_num,
                        'detection_success': detection1_success,
                        'center': center1,
                        'radius': radius1,
                        'visualization_file': filepath1
                    }
                    detection_results.append(result1)
                else:
                    print(f"Cam1: ERROR loading image ", end="")
            else:
                print(f"Cam1: NO IMAGE ", end="")
            
            # Cam2 처리
            if img2_path and os.path.exists(img2_path):
                img2 = cv2.imread(img2_path)
                if img2 is not None:
                    center2, radius2, mask2 = self.detect_golf_ball_optimized(img2)
                    detection2_success = center2 is not None
                    
                    if detection2_success:
                        cam2_detections += 1
                        print(f"Cam2: DETECTED")
                    else:
                        print(f"Cam2: NOT DETECTED")
                    
                    # 시각화 생성
                    filepath2 = self.create_detection_visualization(
                        img2, center2, radius2, frame_num, "Cam2", 
                        detection2_success, output_dir
                    )
                    
                    # 결과 저장
                    result2 = {
                        'camera': 'Cam2',
                        'frame': frame_num,
                        'detection_success': detection2_success,
                        'center': center2,
                        'radius': radius2,
                        'visualization_file': filepath2
                    }
                    detection_results.append(result2)
                else:
                    print(f"Cam2: ERROR loading image")
            else:
                print(f"Cam2: NO IMAGE")
        
        # 통계 출력
        print(f"\n" + "=" * 80)
        print(f"=== DIRECTORY {directory_name} DETECTION RESULTS ===")
        print(f"Total frames processed: {total_frames}")
        print(f"")
        print(f"Cam1 detections: {cam1_detections}/{total_frames} ({cam1_detections/total_frames*100:.1f}%)")
        print(f"Cam2 detections: {cam2_detections}/{total_frames} ({cam2_detections/total_frames*100:.1f}%)")
        print(f"")
        print(f"Visualization files saved in: {output_dir}")
        print(f"=" * 80)
        
        return detection_results, {
            'total_frames': total_frames,
            'cam1_detections': cam1_detections,
            'cam2_detections': cam2_detections,
            'cam1_detection_rate': cam1_detections/total_frames*100,
            'cam2_detection_rate': cam2_detections/total_frames*100
        }
    
    def process_all_directories(self):
        """모든 디렉토리 처리"""
        print("=== MULTI-DIRECTORY GOLF BALL DETECTION SYSTEM ===")
        print("Processing directories 3, 4, 5 with golf ball detection")
        
        all_results = {}
        all_stats = {}
        
        for directory in self.directories:
            results, stats = self.process_directory(directory)
            all_results[directory] = results
            all_stats[directory] = stats
        
        # 전체 통계
        print(f"\n" + "=" * 80)
        print(f"=== OVERALL DETECTION RESULTS ===")
        for directory in self.directories:
            stats = all_stats[directory]
            print(f"Directory {directory}: Cam1 {stats['cam1_detection_rate']:.1f}%, Cam2 {stats['cam2_detection_rate']:.1f}%")
        print(f"=" * 80)
        
        return all_results, all_stats

def main():
    """메인 함수"""
    detector = MultiDirectoryGolfDetector()
    
    # 모든 디렉토리 처리
    results, stats = detector.process_all_directories()
    
    print(f"\nMulti-directory golf ball detection completed!")
    print(f"Check the driver_X_result directories for visualization files!")

if __name__ == "__main__":
    main()
