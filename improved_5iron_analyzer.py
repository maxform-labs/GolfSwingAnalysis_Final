#!/usr/bin/env python3
"""
개선된 5번 아이언 샷 분석 시스템
- 다양한 전처리 방법 적용
- 골프공 검출 파라미터 최적화
- 이미지 품질 분석 및 디버깅
"""

import cv2
import numpy as np
import json
import os
import glob
import time
from pathlib import Path
import matplotlib.pyplot as plt

class ImprovedFiveIronAnalyzer:
    def __init__(self, calibration_file="manual_calibration_470mm.json"):
        """개선된 5번 아이언 분석기 초기화"""
        self.load_calibration_data(calibration_file)
        self.setup_detection_parameters()
        self.setup_output_directories()
        
        # CSV 데이터 로드
        self.csv_data = self.load_csv_data()
        
    def load_calibration_data(self, calibration_file):
        """캘리브레이션 데이터 로드"""
        try:
            with open(calibration_file, 'r', encoding='utf-8') as f:
                self.calibration_data = json.load(f)
            print("✅ 캘리브레이션 데이터 로드 완료")
        except FileNotFoundError:
            print("❌ 캘리브레이션 파일을 찾을 수 없습니다.")
            self.calibration_data = None
    
    def load_csv_data(self):
        """CSV 데이터 로드"""
        csv_path = "data/video_ballData_20250930/video_ballData_20250930/5Iron_0930/shotdata_20250930.csv"
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"✅ CSV 데이터 로드 완료: {len(df)}개 샷")
            return df
        except ImportError:
            print("❌ pandas가 설치되지 않았습니다.")
            return None
        except FileNotFoundError:
            print("❌ CSV 파일을 찾을 수 없습니다.")
            return None
    
    def setup_detection_parameters(self):
        """검출 파라미터 설정"""
        # 다양한 골프공 검출 파라미터 세트
        self.ball_param_sets = [
            # 기본 파라미터
            {'dp': 1, 'minDist': 30, 'param1': 50, 'param2': 30, 'minRadius': 5, 'maxRadius': 25},
            # 민감한 파라미터
            {'dp': 1, 'minDist': 20, 'param1': 30, 'param2': 20, 'minRadius': 3, 'maxRadius': 30},
            # 보수적인 파라미터
            {'dp': 1, 'minDist': 40, 'param1': 70, 'param2': 40, 'minRadius': 8, 'maxRadius': 20},
            # 매우 민감한 파라미터
            {'dp': 1, 'minDist': 15, 'param1': 20, 'param2': 15, 'minRadius': 2, 'maxRadius': 35}
        ]
        
        # 골프채 검출 파라미터
        self.club_params = {
            'rho': 1,
            'theta': np.pi/180,
            'threshold': 50,
            'minLineLength': 100,
            'maxLineGap': 20
        }
    
    def setup_output_directories(self):
        """출력 디렉토리 설정"""
        self.base_output_dir = "final_results_improved"
        self.dirs = {
            'ball_detection': os.path.join(self.base_output_dir, "ball_detection"),
            'club_detection': os.path.join(self.base_output_dir, "club_detection"),
            'calibration': os.path.join(self.base_output_dir, "calibration"),
            'analysis_results': os.path.join(self.base_output_dir, "analysis_results"),
            'debug_images': os.path.join(self.base_output_dir, "debug_images")
        }
        
        # 디렉토리 생성
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        print("✅ 출력 디렉토리 설정 완료")
    
    def analyze_image_quality(self, image_path):
        """이미지 품질 분석"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 이미지 품질 지표
        quality_metrics = {
            'brightness': np.mean(gray),
            'contrast': np.std(gray),
            'min_pixel': np.min(gray),
            'max_pixel': np.max(gray),
            'histogram': cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        }
        
        return quality_metrics
    
    def preprocess_image_advanced(self, image, method="multi_enhance"):
        """고급 이미지 전처리"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == "multi_enhance":
            # 다단계 향상
            # 1. 감마 보정
            gamma = 1.5
            gamma_corrected = np.power(gray / 255.0, gamma) * 255.0
            gamma_corrected = np.uint8(gamma_corrected)
            
            # 2. CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            clahe_result = clahe.apply(gamma_corrected)
            
            # 3. 가우시안 블러
            blurred = cv2.GaussianBlur(clahe_result, (5, 5), 0)
            
            return blurred
            
        elif method == "brightness_boost":
            # 밝기 대폭 향상
            return cv2.convertScaleAbs(gray, alpha=2.0, beta=100)
            
        elif method == "contrast_enhance":
            # 대비 향상
            return cv2.convertScaleAbs(gray, alpha=2.5, beta=0)
            
        elif method == "histogram_eq":
            # 히스토그램 균등화
            return cv2.equalizeHist(gray)
            
        elif method == "adaptive_threshold":
            # 적응적 임계값
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        return gray
    
    def detect_golf_ball_advanced(self, image):
        """고급 골프공 검출"""
        best_circles = []
        best_score = 0
        
        # 다양한 전처리 방법 시도
        preprocessing_methods = ["multi_enhance", "brightness_boost", "contrast_enhance", "histogram_eq"]
        
        for method in preprocessing_methods:
            processed = self.preprocess_image_advanced(image, method)
            
            # 다양한 파라미터 세트 시도
            for i, params in enumerate(self.ball_param_sets):
                circles = cv2.HoughCircles(
                    processed,
                    cv2.HOUGH_GRADIENT,
                    **params
                )
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    
                    # 골프공 후보 필터링 및 점수 계산
                    filtered_circles = []
                    total_score = 0
                    
                    for (x, y, r) in circles:
                        # 크기 필터링
                        if 5 <= r <= 25:
                            # 위치 필터링 (골프공이 나타날 가능성이 높은 영역)
                            if 100 <= x <= 1820 and 100 <= y <= 980:
                                # 점수 계산 (크기, 위치, 원형도 등)
                                score = self.calculate_circle_score(processed, x, y, r)
                                if score > 0.3:  # 임계값
                                    filtered_circles.append((x, y, r, score))
                                    total_score += score
                    
                    # 최고 점수 업데이트
                    if total_score > best_score:
                        best_score = total_score
                        best_circles = filtered_circles
        
        # 최고 점수 원들 반환 (점수 제거)
        return [(x, y, r) for (x, y, r, score) in best_circles]
    
    def calculate_circle_score(self, image, x, y, r):
        """원의 품질 점수 계산"""
        try:
            # 원 주변 영역 추출
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # 원 내부와 외부의 대비 계산
            inside = cv2.bitwise_and(image, mask)
            outside_mask = cv2.bitwise_not(mask)
            outside = cv2.bitwise_and(image, outside_mask)
            
            inside_mean = np.mean(inside[inside > 0])
            outside_mean = np.mean(outside[outside > 0])
            
            # 대비 점수
            contrast_score = abs(inside_mean - outside_mean) / 255.0
            
            # 크기 점수 (적절한 크기일수록 높은 점수)
            size_score = 1.0 - abs(r - 12) / 12.0  # 12픽셀을 이상적 크기로 가정
            
            # 위치 점수 (중앙에 가까울수록 높은 점수)
            center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            position_score = 1.0 - distance / max_distance
            
            # 전체 점수 (가중 평균)
            total_score = 0.5 * contrast_score + 0.3 * size_score + 0.2 * position_score
            
            return total_score
            
        except:
            return 0.0
    
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
                if length > 80:
                    angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                    if abs(angle) > 60:
                        filtered_lines.append(line[0])
            return filtered_lines
        
        return []
    
    def calculate_3d_coordinates(self, point1, point2):
        """3D 좌표 계산"""
        if self.calibration_data is None:
            return None
        
        # 카메라 행렬을 3x4 형식으로 변환
        K1 = np.array(self.calibration_data['camera_matrix_1'])
        K2 = np.array(self.calibration_data['camera_matrix_2'])
        R = np.array(self.calibration_data['rotation_matrix'])
        T = np.array(self.calibration_data['translation_vector']).reshape(3, 1)
        
        # 프로젝션 행렬 생성 (3x4)
        P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K2 @ np.hstack([R, T])
        
        points_3d = cv2.triangulatePoints(
            P1, P2,
            point1.reshape(-1, 1, 2),
            point2.reshape(-1, 1, 2)
        )
        
        points_3d = points_3d[:3] / points_3d[3]
        return points_3d
    
    def match_golf_balls(self, balls1, balls2):
        """골프공 매칭"""
        if self.calibration_data is None:
            return []
        
        matched_pairs = []
        
        for ball1 in balls1:
            x1, y1, r1 = ball1
            best_match = None
            min_distance = float('inf')
            
            for ball2 in balls2:
                x2, y2, r2 = ball2
                
                point1 = np.array([[x1, y1]], dtype=np.float32)
                point2 = np.array([[x2, y2]], dtype=np.float32)
                
                line2 = cv2.computeCorrespondEpilines(point1, 1, 
                    np.array(self.calibration_data['fundamental_matrix']))
                line2 = line2.reshape(-1, 3)
                
                distance = abs(line2[0][0] * x2 + line2[0][1] * y2 + line2[0][2]) / \
                          np.sqrt(line2[0][0]**2 + line2[0][1]**2)
                
                if distance < 15 and distance < min_distance:  # 임계값 증가
                    min_distance = distance
                    best_match = ball2
            
            if best_match is not None:
                matched_pairs.append((ball1, best_match))
        
        return matched_pairs
    
    def calculate_ball_speed(self, positions_3d, time_interval=0.033):
        """골프공 속도 계산"""
        if len(positions_3d) < 2:
            return 0
        
        distances = []
        for i in range(1, len(positions_3d)):
            dist = np.linalg.norm(positions_3d[i] - positions_3d[i-1])
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        speed_mm_per_s = avg_distance / time_interval
        speed_mph = speed_mm_per_s * 0.002237
        
        return speed_mph
    
    def calculate_launch_angle(self, positions_3d):
        """발사각 계산"""
        if len(positions_3d) < 2:
            return 0
        
        start_pos = positions_3d[0]
        end_pos = positions_3d[min(2, len(positions_3d)-1)]
        
        horizontal_dist = np.sqrt((end_pos[0] - start_pos[0])**2 + 
                                 (end_pos[2] - start_pos[2])**2)
        vertical_dist = end_pos[1] - start_pos[1]
        
        launch_angle = np.degrees(np.arctan2(vertical_dist, horizontal_dist))
        return launch_angle
    
    def calculate_direction_angle(self, positions_3d):
        """방향각 계산"""
        if len(positions_3d) < 2:
            return 0
        
        start_pos = positions_3d[0]
        end_pos = positions_3d[-1]
        
        direction_vector = end_pos - start_pos
        direction_angle = np.degrees(np.arctan2(direction_vector[2], direction_vector[0]))
        
        return direction_angle
    
    def save_detection_images(self, shot_num, frame1, frame2, balls1, balls2, clubs1, clubs2):
        """검출 결과 이미지 저장"""
        # 골프공 검출 이미지
        ball_img1 = frame1.copy()
        ball_img2 = frame2.copy()
        
        for (x, y, r) in balls1:
            cv2.circle(ball_img1, (x, y), r, (0, 255, 0), 2)
            cv2.circle(ball_img1, (x, y), 2, (0, 0, 255), 3)
        
        for (x, y, r) in balls2:
            cv2.circle(ball_img2, (x, y), r, (0, 255, 0), 2)
            cv2.circle(ball_img2, (x, y), 2, (0, 0, 255), 3)
        
        cv2.imwrite(os.path.join(self.dirs['ball_detection'], f"shot_{shot_num}_cam1_ball.jpg"), ball_img1)
        cv2.imwrite(os.path.join(self.dirs['ball_detection'], f"shot_{shot_num}_cam2_ball.jpg"), ball_img2)
        
        # 골프채 검출 이미지
        club_img1 = frame1.copy()
        club_img2 = frame2.copy()
        
        for line in clubs1:
            x1, y1, x2, y2 = line
            cv2.line(club_img1, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        for line in clubs2:
            x1, y1, x2, y2 = line
            cv2.line(club_img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(self.dirs['club_detection'], f"shot_{shot_num}_cam1_club.jpg"), club_img1)
        cv2.imwrite(os.path.join(self.dirs['club_detection'], f"shot_{shot_num}_cam2_club.jpg"), club_img2)
    
    def save_debug_images(self, shot_num, frame1, frame2):
        """디버그 이미지 저장"""
        # 원본 이미지
        cv2.imwrite(os.path.join(self.dirs['debug_images'], f"shot_{shot_num}_cam1_original.jpg"), frame1)
        cv2.imwrite(os.path.join(self.dirs['debug_images'], f"shot_{shot_num}_cam2_original.jpg"), frame2)
        
        # 전처리된 이미지들
        methods = ["multi_enhance", "brightness_boost", "contrast_enhance", "histogram_eq"]
        
        for method in methods:
            processed1 = self.preprocess_image_advanced(frame1, method)
            processed2 = self.preprocess_image_advanced(frame2, method)
            
            cv2.imwrite(os.path.join(self.dirs['debug_images'], f"shot_{shot_num}_cam1_{method}.jpg"), processed1)
            cv2.imwrite(os.path.join(self.dirs['debug_images'], f"shot_{shot_num}_cam2_{method}.jpg"), processed2)
    
    def save_calibration_data(self):
        """캘리브레이션 데이터 저장"""
        if self.calibration_data:
            with open(os.path.join(self.dirs['calibration'], "calibration_data.json"), 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
            
            # 캘리브레이션 요약 정보
            summary = {
                "baseline_mm": self.calibration_data['baseline'],
                "focal_length": self.calibration_data['focal_length'],
                "image_size": self.calibration_data['image_size'],
                "calibration_method": self.calibration_data['calibration_method'],
                "calibration_date": self.calibration_data['calibration_date']
            }
            
            with open(os.path.join(self.dirs['calibration'], "calibration_summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
    
    def analyze_shot(self, shot_num):
        """개별 샷 분석"""
        print(f"🏌️ 샷 {shot_num} 분석 중...")
        
        shot_dir = f"data/video_ballData_20250930/video_ballData_20250930/5Iron_0930/{shot_num}"
        
        if not os.path.exists(shot_dir):
            print(f"❌ 샷 {shot_num} 디렉토리를 찾을 수 없습니다.")
            return None
        
        # Gamma 이미지 사용
        gamma_files = glob.glob(os.path.join(shot_dir, "Gamma_*.bmp"))
        
        if len(gamma_files) < 2:
            print(f"❌ 샷 {shot_num}: 충분한 Gamma 이미지가 없습니다.")
            return None
        
        # 카메라별 이미지 분류
        cam1_images = [f for f in gamma_files if "Gamma_1_" in f]
        cam2_images = [f for f in gamma_files if "Gamma_2_" in f]
        
        if not cam1_images or not cam2_images:
            print(f"❌ 샷 {shot_num}: 카메라별 이미지가 없습니다.")
            return None
        
        # 분석 결과 초기화
        shot_analysis = {
            'shot_number': shot_num,
            'total_frames': min(len(cam1_images), len(cam2_images)),
            'ball_detections': [],
            'club_detections': [],
            '3d_positions': [],
            'speeds': [],
            'launch_angles': [],
            'direction_angles': [],
            'processing_times': []
        }
        
        # 각 프레임 쌍 분석
        for i in range(min(len(cam1_images), len(cam2_images))):
            start_time = time.time()
            
            # 이미지 로드
            frame1 = cv2.imread(cam1_images[i])
            frame2 = cv2.imread(cam2_images[i])
            
            if frame1 is None or frame2 is None:
                continue
            
            # 골프공 검출 (개선된 방법)
            balls1 = self.detect_golf_ball_advanced(frame1)
            balls2 = self.detect_golf_ball_advanced(frame2)
            
            # 골프채 검출
            clubs1 = self.detect_golf_club(frame1)
            clubs2 = self.detect_golf_club(frame2)
            
            # 골프공 매칭 및 3D 계산
            matched_pairs = self.match_golf_balls(balls1, balls2)
            
            if matched_pairs:
                ball1, ball2 = matched_pairs[0]
                point1 = np.array([[ball1[0], ball1[1]]], dtype=np.float32)
                point2 = np.array([[ball2[0], ball2[1]]], dtype=np.float32)
                
                position_3d = self.calculate_3d_coordinates(point1, point2)
                
                if position_3d is not None:
                    shot_analysis['ball_detections'].append(len(balls1) + len(balls2))
                    shot_analysis['3d_positions'].append(position_3d.flatten().tolist())
                    
                    # 첫 번째 프레임에서 검출 이미지 저장
                    if i == 0:
                        self.save_detection_images(shot_num, frame1, frame2, balls1, balls2, clubs1, clubs2)
                        self.save_debug_images(shot_num, frame1, frame2)
            else:
                shot_analysis['ball_detections'].append(0)
            
            shot_analysis['club_detections'].append(len(clubs1) + len(clubs2))
            shot_analysis['processing_times'].append((time.time() - start_time) * 1000)
        
        # 전체 분석 결과 계산
        if shot_analysis['3d_positions']:
            positions_3d = np.array(shot_analysis['3d_positions'])
            
            # 속도 계산
            if len(positions_3d) >= 2:
                shot_analysis['avg_speed_mph'] = self.calculate_ball_speed(positions_3d)
                shot_analysis['avg_launch_angle'] = self.calculate_launch_angle(positions_3d)
                shot_analysis['avg_direction_angle'] = self.calculate_direction_angle(positions_3d)
            else:
                shot_analysis['avg_speed_mph'] = 0
                shot_analysis['avg_launch_angle'] = 0
                shot_analysis['avg_direction_angle'] = 0
        else:
            shot_analysis['avg_speed_mph'] = 0
            shot_analysis['avg_launch_angle'] = 0
            shot_analysis['avg_direction_angle'] = 0
        
        # 통계 계산
        shot_analysis['ball_detection_rate'] = len([x for x in shot_analysis['ball_detections'] if x > 0]) / len(shot_analysis['ball_detections']) * 100 if shot_analysis['ball_detections'] else 0
        shot_analysis['club_detection_rate'] = len([x for x in shot_analysis['club_detections'] if x > 0]) / len(shot_analysis['club_detections']) * 100 if shot_analysis['club_detections'] else 0
        shot_analysis['avg_processing_time'] = np.mean(shot_analysis['processing_times']) if shot_analysis['processing_times'] else 0
        
        # CSV 데이터와 비교
        if self.csv_data is not None and shot_num <= len(self.csv_data):
            csv_row = self.csv_data.iloc[shot_num - 1]
            shot_analysis['csv_speed_mps'] = csv_row['BallSpeed(m/s)']
            shot_analysis['csv_launch_angle'] = csv_row['LaunchAngle(deg)']
            shot_analysis['csv_direction_angle'] = csv_row['LaunchDirection(deg)']
            
            # 오차 계산
            speed_mps = shot_analysis['avg_speed_mph'] * 0.44704  # mph to m/s
            shot_analysis['speed_error_percent'] = abs(speed_mps - shot_analysis['csv_speed_mps']) / shot_analysis['csv_speed_mps'] * 100
            shot_analysis['launch_angle_error'] = abs(shot_analysis['avg_launch_angle'] - shot_analysis['csv_launch_angle'])
            shot_analysis['direction_angle_error'] = abs(shot_analysis['avg_direction_angle'] - shot_analysis['csv_direction_angle'])
        
        return shot_analysis
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("🚀 개선된 5번 아이언 샷 분석 시작")
        print("=" * 50)
        
        if self.calibration_data is None:
            print("❌ 캘리브레이션 데이터가 없습니다.")
            return
        
        # 캘리브레이션 데이터 저장
        self.save_calibration_data()
        
        # 각 샷 분석
        all_results = []
        
        for shot_num in range(1, 11):  # 샷 1~10
            result = self.analyze_shot(shot_num)
            if result:
                all_results.append(result)
                
                # 개별 샷 결과 저장
                with open(os.path.join(self.dirs['analysis_results'], f"shot_{shot_num}_analysis.json"), 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"✅ 샷 {shot_num} 완료:")
                print(f"  - 골프공 검출률: {result['ball_detection_rate']:.1f}%")
                print(f"  - 골프채 검출률: {result['club_detection_rate']:.1f}%")
                print(f"  - 평균 속도: {result['avg_speed_mph']:.1f} mph")
                print(f"  - 발사각: {result['avg_launch_angle']:.1f}°")
                print(f"  - 처리 시간: {result['avg_processing_time']:.1f} ms")
        
        # 전체 결과 요약 저장
        summary = {
            'total_shots': len(all_results),
            'shots': all_results,
            'overall_stats': {
                'avg_ball_detection_rate': np.mean([r['ball_detection_rate'] for r in all_results]),
                'avg_club_detection_rate': np.mean([r['club_detection_rate'] for r in all_results]),
                'avg_processing_time': np.mean([r['avg_processing_time'] for r in all_results]),
                'avg_speed_mph': np.mean([r['avg_speed_mph'] for r in all_results]),
                'avg_launch_angle': np.mean([r['avg_launch_angle'] for r in all_results])
            }
        }
        
        with open(os.path.join(self.dirs['analysis_results'], "complete_analysis_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n🎉 개선된 5번 아이언 샷 분석 완료!")
        print(f"📁 결과 저장 위치: {self.base_output_dir}/")
        print("📊 생성된 파일들:")
        print("  - ball_detection/: 골프공 검출 이미지")
        print("  - club_detection/: 골프채 검출 이미지")
        print("  - calibration/: 캘리브레이션 데이터")
        print("  - analysis_results/: 분석 결과 JSON 파일들")
        print("  - debug_images/: 디버그 이미지들")

def main():
    analyzer = ImprovedFiveIronAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
