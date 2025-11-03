#!/usr/bin/env python3
"""
Driver 4 골프공 검출 시스템
골프공이 검출된 사진만 저장
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

class Driver4GolfDetector:
    def __init__(self, input_dir="data2/driver/4", output_dir="driver_4_detected_result"):
        """Driver 4 골프공 검출기 초기화"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Driver 4 Golf Ball Detector Initialized")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print("Only saving images with detected golf balls")
    
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
    
    def create_detection_visualization(self, img, center, radius, frame_num, camera_name):
        """검출 결과 시각화 (골프공이 검출된 경우만)"""
        # 이미지 복사
        img_vis = img.copy()
        
        # 골프공 위치에 원 그리기
        if center is not None and radius is not None:
            color = (0, 255, 0)  # 녹색
            cv2.circle(img_vis, center, radius, color, 2)
            cv2.circle(img_vis, center, 2, (0, 0, 255), -1)
            cv2.putText(img_vis, f"({center[0]}, {center[1]})", 
                       (center[0] + 10, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # matplotlib을 사용한 시각화
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 이미지 표시
        ax.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{camera_name} - Frame {frame_num}\nGOLF BALL DETECTED', 
                     fontsize=14, fontweight='bold', color='green')
        
        if center is not None:
            ax.text(0.02, 0.98, f'Center: ({center[0]}, {center[1]})\nRadius: {radius}px', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axis('off')
        
        plt.tight_layout()
        
        # 이미지 저장
        filename = f'{camera_name}_{frame_num:02d}_detect.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def process_camera_sequence(self, camera_name):
        """카메라 시퀀스 처리"""
        print(f"\n=== PROCESSING {camera_name} SEQUENCE ===")
        
        # 결과 통계
        total_frames = 0
        detected_frames = 0
        saved_images = 0
        
        detection_results = []
        
        # 파일명 패턴으로 정렬된 파일 목록 가져오기
        pattern = os.path.join(self.input_dir, f"{camera_name}_*.bmp")
        files = sorted(glob.glob(pattern))
        
        print(f"Found {len(files)} {camera_name} images")
        
        for file_path in files:
            # 파일명에서 프레임 번호 추출
            filename = os.path.basename(file_path)
            frame_num = int(filename.split('_')[1].split('.')[0])
            
            total_frames += 1
            
            print(f"\n{camera_name} Frame {frame_num:2d}: ", end="")
            
            # 이미지 로드
            img = cv2.imread(file_path)
            
            if img is None:
                print(f"ERROR - Failed to load image")
                continue
            
            # 골프공 검출
            center, radius, mask = self.detect_golf_ball_optimized(img)
            detection_success = center is not None
            
            if detection_success:
                detected_frames += 1
                print(f"DETECTED - Saving image")
                
                # 시각화 생성 및 저장
                filepath = self.create_detection_visualization(
                    img, center, radius, frame_num, camera_name
                )
                saved_images += 1
                
                # 결과 저장
                result = {
                    'camera': camera_name,
                    'frame': frame_num,
                    'center': center,
                    'radius': radius,
                    'visualization_file': filepath
                }
                detection_results.append(result)
            else:
                print(f"NOT DETECTED - Skipping")
        
        # 통계 출력
        print(f"\n" + "=" * 60)
        print(f"=== {camera_name} DETECTION RESULTS ===")
        print(f"Total frames processed: {total_frames}")
        print(f"Frames with golf ball detected: {detected_frames}")
        print(f"Images saved: {saved_images}")
        print(f"Detection rate: {detected_frames/total_frames*100:.1f}%")
        print(f"")
        print(f"Saved images in: {self.output_dir}")
        print(f"=" * 60)
        
        return detection_results, {
            'total_frames': total_frames,
            'detected_frames': detected_frames,
            'saved_images': saved_images,
            'detection_rate': detected_frames/total_frames*100
        }
    
    def process_all_cameras(self):
        """모든 카메라 처리"""
        print("=== DRIVER 4 GOLF BALL DETECTION SYSTEM ===")
        print("Processing Cam1 and Cam2 sequences - saving only detected images")
        
        all_results = {}
        all_stats = {}
        
        # Cam1 처리
        cam1_results, cam1_stats = self.process_camera_sequence('Cam1')
        all_results['Cam1'] = cam1_results
        all_stats['Cam1'] = cam1_stats
        
        # Cam2 처리
        cam2_results, cam2_stats = self.process_camera_sequence('Cam2')
        all_results['Cam2'] = cam2_results
        all_stats['Cam2'] = cam2_stats
        
        # 전체 통계
        print(f"\n" + "=" * 80)
        print(f"=== OVERALL DETECTION RESULTS ===")
        print(f"Cam1 Detection rate: {cam1_stats['detection_rate']:.1f}% ({cam1_stats['saved_images']} images saved)")
        print(f"Cam2 Detection rate: {cam2_stats['detection_rate']:.1f}% ({cam2_stats['saved_images']} images saved)")
        print(f"")
        print(f"Total images saved: {cam1_stats['saved_images'] + cam2_stats['saved_images']}")
        print(f"All saved images in: {self.output_dir}")
        print(f"=" * 80)
        
        return all_results, all_stats

def main():
    """메인 함수"""
    detector = Driver4GolfDetector()
    
    # 모든 카메라 처리
    results, stats = detector.process_all_cameras()
    
    print(f"\nDriver 4 golf ball detection completed!")
    print(f"Only images with detected golf balls were saved!")

if __name__ == "__main__":
    main()
