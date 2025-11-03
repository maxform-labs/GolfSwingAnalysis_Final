#!/usr/bin/env python3
"""
종합 골프공 검출 시스템
모든 프레임을 처리하고 실제 골프공 존재 여부와 검출 결과를 비교
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

class ComprehensiveGolfBallDetector:
    def __init__(self, output_dir="driver_2_result"):
        """종합 골프공 검출기 초기화"""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Comprehensive Golf Ball Detector Initialized")
        print(f"Output directory: {self.output_dir}")
        
        # 실제 골프공이 있는 프레임 정의
        self.golf_ball_frames = {
            'Gamma_1': list(range(1, 12)),  # 1~11
            'Gamma_2': list(range(1, 8))    # 1~7
        }
        
        print(f"Frames with actual golf balls:")
        print(f"  Gamma_1: {self.golf_ball_frames['Gamma_1']}")
        print(f"  Gamma_2: {self.golf_ball_frames['Gamma_2']}")
    
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
    
    def create_detection_visualization(self, img1, img2, center1, center2, radius1, radius2, 
                                     frame_num, has_golf_ball, detection_success, 
                                     disparity=None, position_3d=None):
        """검출 결과 시각화"""
        # 이미지 복사
        img1_vis = img1.copy()
        img2_vis = img2.copy()
        
        # 골프공 위치에 원 그리기
        if center1 is not None and radius1 is not None:
            color = (0, 255, 0) if detection_success else (0, 0, 255)  # 성공: 녹색, 실패: 빨간색
            cv2.circle(img1_vis, center1, radius1, color, 2)
            cv2.circle(img1_vis, center1, 2, (0, 0, 255), -1)
            cv2.putText(img1_vis, f"({center1[0]}, {center1[1]})", 
                       (center1[0] + 10, center1[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if center2 is not None and radius2 is not None:
            color = (0, 255, 0) if detection_success else (0, 0, 255)
            cv2.circle(img2_vis, center2, radius2, color, 2)
            cv2.circle(img2_vis, center2, 2, (0, 0, 255), -1)
            cv2.putText(img2_vis, f"({center2[0]}, {center2[1]})", 
                       (center2[0] + 10, center2[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # matplotlib을 사용한 시각화
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 상태에 따른 색상 결정
        if has_golf_ball and detection_success:
            title_color = 'green'
            status_text = 'CORRECT DETECTION'
        elif has_golf_ball and not detection_success:
            title_color = 'red'
            status_text = 'MISSED DETECTION'
        elif not has_golf_ball and detection_success:
            title_color = 'orange'
            status_text = 'FALSE POSITIVE'
        else:
            title_color = 'blue'
            status_text = 'CORRECT NO DETECTION'
        
        # 카메라 1 이미지
        axes[0].imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Camera 1 - Frame {frame_num}\n{status_text}', 
                         fontsize=14, fontweight='bold', color=title_color)
        if center1 is not None:
            axes[0].text(0.02, 0.98, f'Center: ({center1[0]}, {center1[1]})\nRadius: {radius1}px', 
                        transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[0].axis('off')
        
        # 카메라 2 이미지
        axes[1].imshow(cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Camera 2 - Frame {frame_num}\n{status_text}', 
                         fontsize=14, fontweight='bold', color=title_color)
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
        
        fig.suptitle(title, fontsize=16, fontweight='bold', color=title_color)
        
        plt.tight_layout()
        
        # 이미지 저장
        filename = f'frame_{frame_num:02d}_detection.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def process_all_frames(self, image_folder="data2/driver/2"):
        """모든 프레임 처리"""
        print(f"\n=== PROCESSING ALL FRAMES ===")
        print(f"Image folder: {image_folder}")
        print(f"Output directory: {self.output_dir}")
        
        # 이미지 파일 목록
        gamma1_files = sorted(glob.glob(f"{image_folder}/Gamma_1_*.bmp"))
        gamma2_files = sorted(glob.glob(f"{image_folder}/Gamma_2_*.bmp"))
        
        print(f"Found {len(gamma1_files)} Gamma_1 images")
        print(f"Found {len(gamma2_files)} Gamma_2 images")
        
        if len(gamma1_files) == 0:
            print("ERROR: No images found!")
            return
        
        # 결과 통계
        total_frames = len(gamma1_files)
        correct_detections = 0
        missed_detections = 0
        false_positives = 0
        correct_no_detections = 0
        
        detection_results = []
        
        print(f"\nProcessing {total_frames} frames...")
        print("=" * 80)
        
        for i in range(total_frames):
            frame_num = i + 1
            img1_path = gamma1_files[i]
            img2_path = gamma2_files[i]
            
            # 실제 골프공 존재 여부 확인
            has_golf_ball = frame_num in self.golf_ball_frames['Gamma_1']
            
            print(f"\nFrame {frame_num:2d}: ", end="")
            
            # 이미지 로드
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                print(f"ERROR - Failed to load images")
                continue
            
            # 골프공 검출
            center1, radius1, mask1 = self.detect_golf_ball_optimized(img1)
            center2, radius2, mask2 = self.detect_golf_ball_optimized(img2)
            
            detection_success = center1 is not None and center2 is not None
            
            # 시차 및 3D 계산
            disparity = None
            position_3d = None
            
            if detection_success:
                disparity = abs(center1[1] - center2[1])
                
                if disparity > 2:
                    focal_length = 1800.0
                    baseline_mm = 470.0
                    depth = (focal_length * baseline_mm) / disparity
                    
                    if 100 < depth < 1500:
                        x = (center1[0] - 720) * depth / focal_length
                        y = ((center1[1] + center2[1]) / 2 - 540) * depth / focal_length
                        z = depth
                        position_3d = np.array([x, y, z])
            
            # 결과 분류 및 로그
            if has_golf_ball and detection_success:
                correct_detections += 1
                print(f"SUCCESS - Correct detection (Golf ball exists and detected)")
                if disparity is not None and position_3d is not None:
                    print(f"         Disparity: {disparity:.1f}px, 3D: ({position_3d[0]:.1f}, {position_3d[1]:.1f}, {position_3d[2]:.1f})mm")
                elif disparity is not None:
                    print(f"         Disparity: {disparity:.1f}px, 3D: Invalid depth")
            elif has_golf_ball and not detection_success:
                missed_detections += 1
                print(f"FAILED  - Missed detection (Golf ball exists but not detected)")
            elif not has_golf_ball and detection_success:
                false_positives += 1
                print(f"ERROR   - False positive (No golf ball but detected)")
            else:
                correct_no_detections += 1
                print(f"CORRECT - No detection (No golf ball and correctly not detected)")
            
            # 시각화 생성
            filepath = self.create_detection_visualization(
                img1, img2, center1, center2, radius1, radius2, 
                frame_num, has_golf_ball, detection_success, 
                disparity, position_3d
            )
            
            # 결과 저장
            result = {
                'frame': frame_num,
                'has_golf_ball': has_golf_ball,
                'detection_success': detection_success,
                'center1': center1,
                'center2': center2,
                'radius1': radius1,
                'radius2': radius2,
                'disparity': disparity,
                'position_3d': position_3d,
                'visualization_file': filepath
            }
            detection_results.append(result)
        
        # 최종 통계
        print(f"\n" + "=" * 80)
        print(f"=== COMPREHENSIVE DETECTION RESULTS ===")
        print(f"Total frames processed: {total_frames}")
        print(f"")
        print(f"Correct detections: {correct_detections}")
        print(f"Missed detections: {missed_detections}")
        print(f"False positives: {false_positives}")
        print(f"Correct no detections: {correct_no_detections}")
        print(f"")
        
        # 정확도 계산
        total_with_golf_ball = len(self.golf_ball_frames['Gamma_1'])
        total_without_golf_ball = total_frames - total_with_golf_ball
        
        detection_accuracy = (correct_detections / total_with_golf_ball) * 100 if total_with_golf_ball > 0 else 0
        no_detection_accuracy = (correct_no_detections / total_without_golf_ball) * 100 if total_without_golf_ball > 0 else 0
        overall_accuracy = ((correct_detections + correct_no_detections) / total_frames) * 100
        
        print(f"Detection accuracy (golf ball present): {detection_accuracy:.1f}%")
        print(f"No detection accuracy (no golf ball): {no_detection_accuracy:.1f}%")
        print(f"Overall accuracy: {overall_accuracy:.1f}%")
        print(f"")
        print(f"Visualization files saved in: {self.output_dir}")
        print(f"=" * 80)
        
        return detection_results, {
            'total_frames': total_frames,
            'correct_detections': correct_detections,
            'missed_detections': missed_detections,
            'false_positives': false_positives,
            'correct_no_detections': correct_no_detections,
            'detection_accuracy': detection_accuracy,
            'no_detection_accuracy': no_detection_accuracy,
            'overall_accuracy': overall_accuracy
        }

def main():
    """메인 함수"""
    print("=== COMPREHENSIVE GOLF BALL DETECTION SYSTEM ===")
    print("Processing all frames with detailed logging and visualization")
    
    detector = ComprehensiveGolfBallDetector()
    
    # 모든 프레임 처리
    results, stats = detector.process_all_frames()
    
    print(f"\nComprehensive golf ball detection completed!")
    print(f"Overall accuracy: {stats['overall_accuracy']:.1f}%")
    print(f"Check the {detector.output_dir} directory for visualization files!")

if __name__ == "__main__":
    main()
