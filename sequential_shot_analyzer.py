#!/usr/bin/env python3
"""
순차적 샷 분석 시스템
- 각 샷별로 Gamma 이미지를 순서대로 분석
- 골프공과 골프채 검출 과정을 명확하게 시각화
- 검출 결과를 순서대로 저장
"""

import cv2
import numpy as np
import json
import os
import glob
import time
from pathlib import Path

class SequentialShotAnalyzer:
    def __init__(self, calibration_file="manual_calibration_470mm.json"):
        """순차적 샷 분석기 초기화"""
        self.load_calibration_data(calibration_file)
        self.setup_detection_parameters()
        self.setup_output_directories()
        
    def load_calibration_data(self, calibration_file):
        """캘리브레이션 데이터 로드"""
        try:
            with open(calibration_file, 'r', encoding='utf-8') as f:
                self.calibration_data = json.load(f)
            print("✅ 캘리브레이션 데이터 로드 완료")
        except FileNotFoundError:
            print("❌ 캘리브레이션 파일을 찾을 수 없습니다.")
            self.calibration_data = None
    
    def setup_detection_parameters(self):
        """검출 파라미터 설정"""
        # 골프공 검출 파라미터 (최적화됨)
        self.ball_params = {
            'dp': 1,
            'minDist': 25,
            'param1': 40,
            'param2': 25,
            'minRadius': 3,
            'maxRadius': 30
        }
        
        # 골프채 검출 파라미터
        self.club_params = {
            'rho': 1,
            'theta': np.pi/180,
            'threshold': 40,
            'minLineLength': 80,
            'maxLineGap': 15
        }
    
    def setup_output_directories(self):
        """출력 디렉토리 설정"""
        self.base_output_dir = "sequential_shot_results"
        self.dirs = {
            'ball_detection': os.path.join(self.base_output_dir, "ball_detection"),
            'club_detection': os.path.join(self.base_output_dir, "club_detection"),
            'combined_detection': os.path.join(self.base_output_dir, "combined_detection"),
            'analysis_results': os.path.join(self.base_output_dir, "analysis_results")
        }
        
        # 디렉토리 생성
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        print("✅ 출력 디렉토리 설정 완료")
    
    def preprocess_image(self, image):
        """이미지 전처리"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 감마 보정
        gamma = 1.5
        gamma_corrected = np.power(gray / 255.0, gamma) * 255.0
        gamma_corrected = np.uint8(gamma_corrected)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gamma_corrected)
        
        return enhanced
    
    def detect_golf_ball(self, image):
        """골프공 검출"""
        processed = self.preprocess_image(image)
        blurred = cv2.GaussianBlur(processed, (5, 5), 0)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            **self.ball_params
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # 골프공 후보 필터링
            filtered_circles = []
            for (x, y, r) in circles:
                if 5 <= r <= 25 and 100 <= x <= 1820 and 100 <= y <= 980:
                    filtered_circles.append((x, y, r))
            
            return filtered_circles
        
        return []
    
    def detect_golf_club(self, image):
        """골프채 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        lines = cv2.HoughLinesP(cleaned, **self.club_params)
        
        if lines is not None:
            filtered_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 60:
                    angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                    if abs(angle) > 45:
                        filtered_lines.append(line[0])
            return filtered_lines
        
        return []
    
    def create_detection_visualization(self, image, balls, clubs, frame_info):
        """검출 결과 시각화"""
        result_image = image.copy()
        
        # 골프공 검출 결과 그리기
        for i, (x, y, r) in enumerate(balls):
            # 원 그리기
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 3)
            cv2.circle(result_image, (x, y), 2, (0, 0, 255), 5)
            
            # 라벨 추가
            label = f"Ball {i+1}"
            cv2.putText(result_image, label, (x-20, y-r-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 골프채 검출 결과 그리기
        for i, line in enumerate(clubs):
            x1, y1, x2, y2 = line
            # 선 그리기
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # 라벨 추가
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            label = f"Club {i+1}"
            cv2.putText(result_image, label, (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 프레임 정보 추가
        info_text = f"Shot {frame_info['shot']} - Frame {frame_info['frame']} - {frame_info['camera']}"
        cv2.putText(result_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 검출 개수 정보
        detection_text = f"Balls: {len(balls)}, Clubs: {len(clubs)}"
        cv2.putText(result_image, detection_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result_image
    
    def analyze_shot_sequence(self, shot_num):
        """개별 샷의 순차적 분석"""
        print(f"🏌️ 샷 {shot_num} 순차 분석 중...")
        
        shot_dir = f"data/video_ballData_20250930/video_ballData_20250930/5Iron_0930/{shot_num}"
        
        if not os.path.exists(shot_dir):
            print(f"❌ 샷 {shot_num} 디렉토리를 찾을 수 없습니다.")
            return None
        
        # Gamma 이미지 파일들 가져오기
        gamma_files = glob.glob(os.path.join(shot_dir, "Gamma_*.bmp"))
        
        if len(gamma_files) == 0:
            print(f"❌ 샷 {shot_num}: Gamma 이미지가 없습니다.")
            return None
        
        # 카메라별로 분류
        cam1_images = sorted([f for f in gamma_files if "Gamma_1_" in f])
        cam2_images = sorted([f for f in gamma_files if "Gamma_2_" in f])
        
        if not cam1_images or not cam2_images:
            print(f"❌ 샷 {shot_num}: 카메라별 이미지가 없습니다.")
            return None
        
        # 분석 결과 저장
        shot_results = {
            'shot_number': shot_num,
            'total_frames': min(len(cam1_images), len(cam2_images)),
            'frame_analysis': []
        }
        
        # 각 프레임 순차적으로 분석
        for frame_idx in range(min(len(cam1_images), len(cam2_images))):
            frame_num = frame_idx + 1
            
            # 카메라1 분석
            cam1_image = cv2.imread(cam1_images[frame_idx])
            if cam1_image is not None:
                balls1 = self.detect_golf_ball(cam1_image)
                clubs1 = self.detect_golf_club(cam1_image)
                
                # 시각화 생성
                frame_info = {'shot': shot_num, 'frame': frame_num, 'camera': 'Cam1'}
                vis1 = self.create_detection_visualization(cam1_image, balls1, clubs1, frame_info)
                
                # 이미지 저장
                filename = f"shot_{shot_num}_frame_{frame_num:02d}_cam1.jpg"
                cv2.imwrite(os.path.join(self.dirs['combined_detection'], filename), vis1)
                
                # 개별 검출 이미지도 저장
                ball_vis1 = self.create_detection_visualization(cam1_image, balls1, [], frame_info)
                club_vis1 = self.create_detection_visualization(cam1_image, [], clubs1, frame_info)
                
                cv2.imwrite(os.path.join(self.dirs['ball_detection'], f"shot_{shot_num}_frame_{frame_num:02d}_cam1_ball.jpg"), ball_vis1)
                cv2.imwrite(os.path.join(self.dirs['club_detection'], f"shot_{shot_num}_frame_{frame_num:02d}_cam1_club.jpg"), club_vis1)
            
            # 카메라2 분석
            cam2_image = cv2.imread(cam2_images[frame_idx])
            if cam2_image is not None:
                balls2 = self.detect_golf_ball(cam2_image)
                clubs2 = self.detect_golf_club(cam2_image)
                
                # 시각화 생성
                frame_info = {'shot': shot_num, 'frame': frame_num, 'camera': 'Cam2'}
                vis2 = self.create_detection_visualization(cam2_image, balls2, clubs2, frame_info)
                
                # 이미지 저장
                filename = f"shot_{shot_num}_frame_{frame_num:02d}_cam2.jpg"
                cv2.imwrite(os.path.join(self.dirs['combined_detection'], filename), vis2)
                
                # 개별 검출 이미지도 저장
                ball_vis2 = self.create_detection_visualization(cam2_image, balls2, [], frame_info)
                club_vis2 = self.create_detection_visualization(cam2_image, [], clubs2, frame_info)
                
                cv2.imwrite(os.path.join(self.dirs['ball_detection'], f"shot_{shot_num}_frame_{frame_num:02d}_cam2_ball.jpg"), ball_vis2)
                cv2.imwrite(os.path.join(self.dirs['club_detection'], f"shot_{shot_num}_frame_{frame_num:02d}_cam2_club.jpg"), club_vis2)
            
            # 프레임별 분석 결과 저장
            frame_result = {
                'frame_number': frame_num,
                'cam1_balls': len(balls1) if 'balls1' in locals() else 0,
                'cam1_clubs': len(clubs1) if 'clubs1' in locals() else 0,
                'cam2_balls': len(balls2) if 'balls2' in locals() else 0,
                'cam2_clubs': len(clubs2) if 'clubs2' in locals() else 0
            }
            shot_results['frame_analysis'].append(frame_result)
            
            print(f"  프레임 {frame_num}: Cam1(공:{len(balls1) if 'balls1' in locals() else 0}, 채:{len(clubs1) if 'clubs1' in locals() else 0}) | Cam2(공:{len(balls2) if 'balls2' in locals() else 0}, 채:{len(clubs2) if 'clubs2' in locals() else 0})")
        
        # 샷별 결과 저장
        with open(os.path.join(self.dirs['analysis_results'], f"shot_{shot_num}_sequence_analysis.json"), 'w') as f:
            json.dump(shot_results, f, indent=2)
        
        return shot_results
    
    def run_sequential_analysis(self):
        """순차적 분석 실행"""
        print("🚀 순차적 샷 분석 시작")
        print("=" * 50)
        
        if self.calibration_data is None:
            print("❌ 캘리브레이션 데이터가 없습니다.")
            return
        
        all_results = []
        
        # 각 샷 순차적으로 분석
        for shot_num in range(1, 11):  # 샷 1~10
            result = self.analyze_shot_sequence(shot_num)
            if result:
                all_results.append(result)
                print(f"✅ 샷 {shot_num} 완료")
        
        # 전체 결과 요약 저장
        summary = {
            'total_shots': len(all_results),
            'shots': all_results,
            'analysis_summary': {
                'total_frames_analyzed': sum(r['total_frames'] for r in all_results),
                'avg_frames_per_shot': np.mean([r['total_frames'] for r in all_results]) if all_results else 0
            }
        }
        
        with open(os.path.join(self.dirs['analysis_results'], "complete_sequential_analysis.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n🎉 순차적 샷 분석 완료!")
        print(f"📁 결과 저장 위치: {self.base_output_dir}/")
        print("📊 생성된 파일들:")
        print("  - combined_detection/: 골프공+골프채 통합 검출 이미지")
        print("  - ball_detection/: 골프공 검출 이미지")
        print("  - club_detection/: 골프채 검출 이미지")
        print("  - analysis_results/: 순차 분석 결과 JSON 파일들")
        print("\n📋 파일 명명 규칙:")
        print("  - shot_X_frame_YY_camZ.jpg (통합 검출)")
        print("  - shot_X_frame_YY_camZ_ball.jpg (골프공만)")
        print("  - shot_X_frame_YY_camZ_club.jpg (골프채만)")

def main():
    analyzer = SequentialShotAnalyzer()
    analyzer.run_sequential_analysis()

if __name__ == "__main__":
    main()
