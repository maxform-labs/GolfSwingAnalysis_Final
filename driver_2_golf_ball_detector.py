#!/usr/bin/env python3
"""
Driver 2 골프공 검출 시스템
Cam1: 1~11번에 골프공 있음
Cam2: 1~7번에 골프공 있음
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class Driver2GolfBallDetector:
    def __init__(self, input_dir="data2/driver/2", output_dir="driver_2_result"):
        """Driver 2 골프공 검출기 초기화"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Driver 2 Golf Ball Detector Initialized")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # 실제 골프공이 있는 프레임 정의
        self.golf_ball_frames = {
            'Cam1': list(range(1, 12)),  # 1~11
            'Cam2': list(range(1, 8))    # 1~7
        }
        
        print(f"Frames with actual golf balls:")
        print(f"  Cam1: {self.golf_ball_frames['Cam1']}")
        print(f"  Cam2: {self.golf_ball_frames['Cam2']}")
    
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
                                     has_golf_ball, detection_success):
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
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def process_camera_sequence(self, camera_name):
        """카메라 시퀀스 처리"""
        print(f"\n=== PROCESSING {camera_name} SEQUENCE ===")
        
        # 해당 카메라의 골프공이 있는 프레임
        golf_ball_frames = self.golf_ball_frames[camera_name]
        
        # 결과 통계
        total_frames = 0
        correct_detections = 0
        missed_detections = 0
        false_positives = 0
        correct_no_detections = 0
        
        detection_results = []
        
        # 모든 이미지 파일 처리 (1~14번까지)
        for frame_num in range(1, 15):
            img_path = os.path.join(self.input_dir, f"{camera_name}_{frame_num}.bmp")
            
            if not os.path.exists(img_path):
                continue
            
            total_frames += 1
            has_golf_ball = frame_num in golf_ball_frames
            
            print(f"\n{camera_name} Frame {frame_num:2d}: ", end="")
            
            # 이미지 로드
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"ERROR - Failed to load image")
                continue
            
            # 골프공 검출
            center, radius, mask = self.detect_golf_ball_optimized(img)
            detection_success = center is not None
            
            # 결과 분류 및 로그
            if has_golf_ball and detection_success:
                correct_detections += 1
                print(f"SUCCESS - Correct detection (Golf ball exists and detected)")
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
                img, center, radius, frame_num, camera_name, 
                has_golf_ball, detection_success
            )
            
            # 결과 저장
            result = {
                'camera': camera_name,
                'frame': frame_num,
                'has_golf_ball': has_golf_ball,
                'detection_success': detection_success,
                'center': center,
                'radius': radius,
                'visualization_file': filepath
            }
            detection_results.append(result)
        
        # 통계 출력
        print(f"\n" + "=" * 60)
        print(f"=== {camera_name} DETECTION RESULTS ===")
        print(f"Total frames processed: {total_frames}")
        print(f"")
        print(f"Correct detections: {correct_detections}")
        print(f"Missed detections: {missed_detections}")
        print(f"False positives: {false_positives}")
        print(f"Correct no detections: {correct_no_detections}")
        print(f"")
        
        # 정확도 계산
        total_with_golf_ball = len(golf_ball_frames)
        total_without_golf_ball = total_frames - total_with_golf_ball
        
        detection_accuracy = (correct_detections / total_with_golf_ball) * 100 if total_with_golf_ball > 0 else 0
        no_detection_accuracy = (correct_no_detections / total_without_golf_ball) * 100 if total_without_golf_ball > 0 else 0
        overall_accuracy = ((correct_detections + correct_no_detections) / total_frames) * 100
        
        print(f"Detection accuracy (golf ball present): {detection_accuracy:.1f}%")
        print(f"No detection accuracy (no golf ball): {no_detection_accuracy:.1f}%")
        print(f"Overall accuracy: {overall_accuracy:.1f}%")
        print(f"")
        print(f"Visualization files saved in: {self.output_dir}")
        print(f"=" * 60)
        
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
    
    def process_all_cameras(self):
        """모든 카메라 처리"""
        print("=== DRIVER 2 GOLF BALL DETECTION SYSTEM ===")
        print("Processing Cam1 and Cam2 sequences with detailed logging and visualization")
        
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
        print(f"Cam1 Overall accuracy: {cam1_stats['overall_accuracy']:.1f}%")
        print(f"Cam2 Overall accuracy: {cam2_stats['overall_accuracy']:.1f}%")
        print(f"")
        print(f"All visualization files saved in: {self.output_dir}")
        print(f"=" * 80)
        
        return all_results, all_stats

def main():
    """메인 함수"""
    detector = Driver2GolfBallDetector()
    
    # 모든 카메라 처리
    results, stats = detector.process_all_cameras()
    
    print(f"\nDriver 2 golf ball detection completed!")
    print(f"Check the {detector.output_dir} directory for visualization files!")

if __name__ == "__main__":
    main()
