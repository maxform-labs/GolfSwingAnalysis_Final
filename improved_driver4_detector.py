#!/usr/bin/env python3
"""
개선된 Driver 4 골프공 검출 시스템
object_tracker.py의 정교한 검출 방법 활용
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

class ImprovedDriver4Detector:
    def __init__(self, input_dir="data2/driver/4", output_dir="driver_4_improved_result"):
        """개선된 Driver 4 골프공 검출기 초기화"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Improved Driver 4 Golf Ball Detector Initialized")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print("Using object_tracker.py's refined detection methods")
    
    def detect_golf_ball_ir_optimized(self, frame: np.ndarray) -> list:
        """
        IR 조명 최적화된 골프공 검출 (object_tracker.py 방식)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            검출된 골프공 바운딩 박스 리스트 (x, y, w, h)
        """
        # IR 조명 기반 이진화
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        
        # 원형 객체 검출 (HoughCircles)
        circles = cv2.HoughCircles(
            binary, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=25
        )
        
        ball_candidates = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # 골프공 크기 필터링
                if 5 <= r <= 25:
                    # 바운딩 박스 계산
                    bbox = (x - r, y - r, 2 * r, 2 * r)
                    ball_candidates.append(bbox)
        
        return ball_candidates
    
    def detect_golf_ball_shot_detector_method(self, frame: np.ndarray) -> list:
        """
        ShotDetector 방식의 골프공 검출
        
        Args:
            frame: 입력 프레임
            
        Returns:
            검출된 골프공 바운딩 박스 리스트 (x, y, w, h)
        """
        # IR 조명 기반 이진화
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # IR 조명에서 골프공은 매우 밝게 나타남
        ir_binary = cv2.inRange(gray, 200, 255)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ir_binary = cv2.morphologyEx(ir_binary, cv2.MORPH_OPEN, kernel)
        ir_binary = cv2.morphologyEx(ir_binary, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(ir_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 골프공 크기 및 원형도 필터링
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # 골프공 예상 면적 범위
                # 원형도 검사
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:  # 원형에 가까운 객체만
                        x, y, w, h = cv2.boundingRect(contour)
                        valid_contours.append((x, y, w, h))
        
        return valid_contours
    
    def detect_golf_ball_comprehensive(self, frame: np.ndarray) -> tuple:
        """
        종합적인 골프공 검출 (여러 방법 결합)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            (center, radius, mask) 또는 (None, None, None)
        """
        # 방법 1: IR 조명 기반 HoughCircles
        ir_candidates = self.detect_golf_ball_ir_optimized(frame)
        
        # 방법 2: ShotDetector 방식
        shot_candidates = self.detect_golf_ball_shot_detector_method(frame)
        
        # 방법 3: 기존 허프 원 방법 (보조)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 
            dp=1, minDist=5,
            param1=20, param2=15, 
            minRadius=2, maxRadius=25
        )
        
        hough_candidates = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for circle in circles:
                center = (circle[0], circle[1])
                radius = circle[2]
                
                # 밝기 확인
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, radius, 255, -1)
                mean_brightness = cv2.mean(gray, mask)[0]
                
                if mean_brightness > 30:
                    hough_candidates.append((center, radius))
        
        # 가장 신뢰할 만한 검출 결과 선택
        best_detection = None
        best_score = 0
        
        # IR 방법 결과 평가
        for bbox in ir_candidates:
            x, y, w, h = bbox
            center = (x + w//2, y + h//2)
            radius = w//2
            
            # 밝기와 크기 기반 점수
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            mean_brightness = cv2.mean(gray, mask)[0]
            
            score = mean_brightness * (radius / 10)  # 밝기와 크기 가중치
            
            if score > best_score and mean_brightness > 150:
                best_score = score
                best_detection = (center, radius, mask)
        
        # ShotDetector 방법 결과 평가
        for bbox in shot_candidates:
            x, y, w, h = bbox
            center = (x + w//2, y + h//2)
            radius = min(w, h) // 2
            
            # 밝기와 크기 기반 점수
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            mean_brightness = cv2.mean(gray, mask)[0]
            
            score = mean_brightness * (radius / 10)
            
            if score > best_score and mean_brightness > 150:
                best_score = score
                best_detection = (center, radius, mask)
        
        # Hough 방법 결과 평가
        for center, radius in hough_candidates:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            mean_brightness = cv2.mean(gray, mask)[0]
            
            score = mean_brightness * (radius / 10)
            
            if score > best_score and mean_brightness > 30:
                best_score = score
                best_detection = (center, radius, mask)
        
        if best_detection:
            return best_detection
        else:
            return None, None, None
    
    def create_detection_visualization(self, img, center, radius, frame_num, camera_name, 
                                     detection_success, confidence_score=0):
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
            cv2.putText(img_vis, f"Score: {confidence_score:.1f}", 
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
            ax.text(0.02, 0.98, f'Center: ({center[0]}, {center[1]})\nRadius: {radius}px\nScore: {confidence_score:.1f}', 
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
            
            # 종합적인 골프공 검출
            center, radius, mask = self.detect_golf_ball_comprehensive(img)
            detection_success = center is not None
            
            if detection_success:
                detected_frames += 1
                print(f"DETECTED - Saving image")
                
                # 신뢰도 점수 계산
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask_score = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask_score, center, radius, 255, -1)
                mean_brightness = cv2.mean(gray, mask_score)[0]
                confidence_score = mean_brightness * (radius / 10)
                
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
        print("=== IMPROVED DRIVER 4 GOLF BALL DETECTION SYSTEM ===")
        print("Using object_tracker.py's refined detection methods")
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
    detector = ImprovedDriver4Detector()
    
    # 모든 카메라 처리
    results, stats = detector.process_all_cameras()
    
    print(f"\nImproved Driver 4 golf ball detection completed!")
    print(f"Only images with detected golf balls were saved!")

if __name__ == "__main__":
    main()
