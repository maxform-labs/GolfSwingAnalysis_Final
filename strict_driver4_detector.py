#!/usr/bin/env python3
"""
엄격한 Driver 4 골프공 검출 시스템
더 엄격한 조건으로 실제 골프공만 검출
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

class StrictDriver4Detector:
    def __init__(self, input_dir="data2/driver/4", output_dir="driver_4_strict_result"):
        """엄격한 Driver 4 골프공 검출기 초기화"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Strict Driver 4 Golf Ball Detector Initialized")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print("Using strict detection criteria")
    
    def detect_golf_ball_strict(self, frame: np.ndarray) -> tuple:
        """
        엄격한 골프공 검출
        
        Args:
            frame: 입력 프레임
            
        Returns:
            (center, radius, mask) 또는 (None, None, None)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. IR 조명 기반 이진화 (더 엄격한 임계값)
        ir_binary = cv2.inRange(gray, 220, 255)  # 더 높은 임계값
        
        # 2. 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ir_binary = cv2.morphologyEx(ir_binary, cv2.MORPH_OPEN, kernel)
        ir_binary = cv2.morphologyEx(ir_binary, cv2.MORPH_CLOSE, kernel)
        
        # 3. 컨투어 검출
        contours, _ = cv2.findContours(ir_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 엄격한 면적 필터링
            if 50 < area < 1500:  # 더 좁은 면적 범위
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # 엄격한 원형도 검사
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.8:  # 더 높은 원형도 요구
                        # 밝기 검사
                        x, y, w, h = cv2.boundingRect(contour)
                        roi = gray[y:y+h, x:x+w]
                        mean_brightness = np.mean(roi)
                        
                        if mean_brightness > 200:  # 높은 밝기 요구
                            # 종합 점수 계산
                            score = circularity * (area / 100) * (mean_brightness / 255)
                            
                            if score > best_score:
                                best_score = score
                                best_contour = contour
        
        if best_contour is not None and best_score > 0.5:  # 높은 점수 임계값
            # 최소 외접원 계산
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center = (int(x), int(y))
            
            # 마스크 생성
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, int(radius), 255, -1)
            
            return center, int(radius), mask
        
        # 4. HoughCircles 보조 검출 (더 엄격한 조건)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 
            dp=1, minDist=30,  # 더 큰 최소 거리
            param1=50, param2=40,  # 더 높은 임계값
            minRadius=8, maxRadius=20  # 더 좁은 반지름 범위
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for circle in circles:
                center = (circle[0], circle[1])
                radius = circle[2]
                
                # 밝기와 크기 검사
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, radius, 255, -1)
                mean_brightness = cv2.mean(gray, mask)[0]
                
                # 엄격한 조건
                if mean_brightness > 180 and 8 <= radius <= 20:
                    return center, radius, mask
        
        return None, None, None
    
    def create_detection_visualization(self, img, center, radius, frame_num, camera_name, 
                                     detection_success, confidence_score=0):
        """검출 결과 시각화"""
        # 이미지 복사
        img_vis = img.copy()
        
        # 골프공 위치에 원 그리기
        if center is not None and radius is not None:
            color = (0, 255, 0) if detection_success else (0, 0, 255)
            cv2.circle(img_vis, center, radius, color, 2)
            cv2.circle(img_vis, center, 2, (0, 0, 255), -1)
            cv2.putText(img_vis, f"({center[0]}, {center[1]})", 
                       (center[0] + 10, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img_vis, f"Score: {confidence_score:.2f}", 
                       (center[0] + 10, center[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
            ax.text(0.02, 0.98, f'Center: ({center[0]}, {center[1]})\nRadius: {radius}px\nScore: {confidence_score:.2f}', 
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
            
            # 엄격한 골프공 검출
            center, radius, mask = self.detect_golf_ball_strict(img)
            detection_success = center is not None
            
            if detection_success:
                detected_frames += 1
                print(f"DETECTED - Saving image")
                
                # 신뢰도 점수 계산
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask_score = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask_score, center, radius, 255, -1)
                mean_brightness = cv2.mean(gray, mask_score)[0]
                
                # 원형도 계산
                contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                confidence_score = circularity * (mean_brightness / 255) * (area / 100)
                
                # 시각화 생성 및 저장
                filepath = self.create_detection_visualization(
                    img, center, radius, frame_num, camera_name, 
                    detection_success, confidence_score
                )
                saved_images += 1
                
                # 결과 저장
                result = {
                    'camera': camera_name,
                    'frame': frame_num,
                    'center': center,
                    'radius': radius,
                    'confidence_score': confidence_score,
                    'circularity': circularity,
                    'brightness': mean_brightness,
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
        print("=== STRICT DRIVER 4 GOLF BALL DETECTION SYSTEM ===")
        print("Using strict detection criteria for realistic results")
        print("Saving only images with detected golf balls")
        
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
    detector = StrictDriver4Detector()
    
    # 모든 카메라 처리
    results, stats = detector.process_all_cameras()
    
    print(f"\nStrict Driver 4 golf ball detection completed!")
    print(f"Only images with detected golf balls were saved!")

if __name__ == "__main__":
    main()
