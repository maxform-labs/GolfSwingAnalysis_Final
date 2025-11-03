#!/usr/bin/env python3
"""
디버그용 Driver 2, 3 골프공 검출 시스템
적응형 검출 방식으로 골프공 검출
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

class DebugDriver23Detector:
    def __init__(self, input_dir, output_dir):
        """디버그용 Driver 2, 3 골프공 검출기 초기화"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Debug Driver {input_dir.split('/')[-1]} Golf Ball Detector Initialized")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print("Using adaptive detection without labeling")
    
    def detect_golf_ball_adaptive(self, frame: np.ndarray, frame_num: int, camera_name: str) -> tuple:
        """
        적응형 골프공 검출 (이미지별로 조건 조정)
        
        Args:
            frame: 입력 프레임
            frame_num: 프레임 번호
            camera_name: 카메라 이름
            
        Returns:
            (center, radius, mask) 또는 (None, None, None)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 이미지별 적응형 임계값 설정
        mean_brightness = np.mean(gray)
        max_brightness = np.max(gray)
        
        # 적응형 임계값 계산
        if mean_brightness > 150:
            # 밝은 이미지
            threshold_low = max(180, mean_brightness - 30)
            threshold_high = 255
        else:
            # 어두운 이미지
            threshold_low = max(120, mean_brightness + 20)
            threshold_high = 255
        
        # 1. 적응형 이진화
        binary = cv2.inRange(gray, threshold_low, threshold_high)
        
        # 2. 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 3. 컨투어 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_score = 0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 면적 필터링 (더 관대하게)
            if 20 < area < 3000:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # 원형도 필터링 (더 관대하게)
                    if circularity > 0.5:
                        x, y, w, h = cv2.boundingRect(contour)
                        roi = gray[y:y+h, x:x+w]
                        mean_brightness_roi = np.mean(roi)
                        
                        # 밝기 필터링 (더 관대하게)
                        if mean_brightness_roi > threshold_low * 0.8:
                            # 종합 점수 계산
                            score = circularity * (area / 100) * (mean_brightness_roi / 255)
                            
                            if score > best_score:
                                best_score = score
                                best_contour = contour
        
        if best_contour is not None and best_score > 0.1:  # 낮은 점수 임계값
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center = (int(x), int(y))
            
            # 마스크 생성
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, int(radius), 255, -1)
            
            return center, int(radius), mask
        
        # 4. HoughCircles 보조 검출
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 
            dp=1, minDist=20,
            param1=30, param2=20,
            minRadius=5, maxRadius=30
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for circle in circles:
                center = (circle[0], circle[1])
                radius = circle[2]
                
                # 밝기 검사
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, radius, 255, -1)
                mean_brightness = cv2.mean(gray, mask)[0]
                
                if mean_brightness > threshold_low * 0.8:
                    return center, radius, mask
        
        return None, None, None
    
    def create_debug_visualization(self, img, center, radius, frame_num, camera_name, 
                                 detection_success, confidence_score=0, binary_img=None):
        """디버그용 검출 결과 시각화"""
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
            cv2.putText(img_vis, f"Score: {confidence_score:.2f}", 
                       (center[0] + 10, center[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # matplotlib을 사용한 시각화
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 원본 이미지
        axes[0].imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'{camera_name} Frame {frame_num}\n{"DETECTED" if detection_success else "NOT DETECTED"}', 
                         color='green' if detection_success else 'red')
        axes[0].axis('off')
        
        # 그레이스케일 이미지
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        axes[1].imshow(gray, cmap='gray')
        axes[1].set_title('Grayscale')
        axes[1].axis('off')
        
        # 이진화 이미지
        if binary_img is not None:
            axes[2].imshow(binary_img, cmap='gray')
            axes[2].set_title('Binary')
        else:
            axes[2].text(0.5, 0.5, 'No binary image', ha='center', va='center')
            axes[2].set_title('Binary')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 이미지 저장 (파일명 + 'debug')
        filename = f'{camera_name}_{frame_num:02d}_debug.png'
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
            
            # 적응형 골프공 검출
            center, radius, mask = self.detect_golf_ball_adaptive(img, frame_num, camera_name)
            detection_success = center is not None
            
            if detection_success:
                detected_frames += 1
                print(f"DETECTED - Saving image")
                
                # 신뢰도 점수 계산
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask_score = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask_score, center, radius, 255, -1)
                mean_brightness_roi = cv2.mean(gray, mask_score)[0]
                
                confidence_score = (mean_brightness_roi / 255) * (radius / 10)
                
                # 이진화 이미지 생성 (디버그용)
                mean_brightness = np.mean(gray)
                threshold_low = max(180, mean_brightness - 30) if mean_brightness > 150 else max(120, mean_brightness + 20)
                binary_img = cv2.inRange(gray, threshold_low, 255)
                
                # 시각화 생성 및 저장
                filepath = self.create_debug_visualization(
                    img, center, radius, frame_num, camera_name, 
                    detection_success, confidence_score, binary_img
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
                
                # 디버그용 시각화 (검출되지 않은 경우도)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                threshold_low = max(180, mean_brightness - 30) if mean_brightness > 150 else max(120, mean_brightness + 20)
                binary_img = cv2.inRange(gray, threshold_low, 255)
                
                filepath = self.create_debug_visualization(
                    img, None, None, frame_num, camera_name, 
                    detection_success, 0, binary_img
                )
        
        # 통계 출력
        print(f"\n" + "=" * 60)
        print(f"=== {camera_name} DEBUG RESULTS ===")
        print(f"Total frames processed: {total_frames}")
        print(f"Frames with golf ball detected: {detected_frames}")
        print(f"Images saved: {saved_images}")
        print(f"Detection rate: {detected_frames/total_frames*100:.1f}%")
        print(f"")
        print(f"Debug images saved in: {self.output_dir}")
        print(f"=" * 60)
        
        return detection_results, {
            'total_frames': total_frames,
            'detected_frames': detected_frames,
            'saved_images': saved_images,
            'detection_rate': detected_frames/total_frames*100
        }
    
    def process_all_cameras(self):
        """모든 카메라 처리"""
        print(f"=== DEBUG DRIVER {self.input_dir.split('/')[-1]} GOLF BALL DETECTION SYSTEM ===")
        print("Using adaptive detection without labeling")
        
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
        print(f"=== OVERALL DEBUG RESULTS ===")
        print(f"Cam1 Detection rate: {cam1_stats['detection_rate']:.1f}% ({cam1_stats['saved_images']} images saved)")
        print(f"Cam2 Detection rate: {cam2_stats['detection_rate']:.1f}% ({cam2_stats['saved_images']} images saved)")
        print(f"")
        print(f"Total images saved: {cam1_stats['saved_images'] + cam2_stats['saved_images']}")
        print(f"All debug images in: {self.output_dir}")
        print(f"=" * 80)
        
        return all_results, all_stats

def main():
    """메인 함수"""
    # Driver 2 처리
    print("=" * 80)
    print("PROCESSING DRIVER 2")
    print("=" * 80)
    detector2 = DebugDriver23Detector("data2/driver/2", "driver_2_debug_result")
    results2, stats2 = detector2.process_all_cameras()
    
    print("\n" + "=" * 80)
    print("PROCESSING DRIVER 3")
    print("=" * 80)
    detector3 = DebugDriver23Detector("data2/driver/3", "driver_3_debug_result")
    results3, stats3 = detector3.process_all_cameras()
    
    print(f"\n" + "=" * 80)
    print(f"=== FINAL SUMMARY ===")
    print(f"Driver 2 - Cam1: {stats2['Cam1']['detection_rate']:.1f}%, Cam2: {stats2['Cam2']['detection_rate']:.1f}%")
    print(f"Driver 3 - Cam1: {stats3['Cam1']['detection_rate']:.1f}%, Cam2: {stats3['Cam2']['detection_rate']:.1f}%")
    print(f"All debug images saved with '_debug' suffix!")
    print("=" * 80)

if __name__ == "__main__":
    main()
