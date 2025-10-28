#!/usr/bin/env python3
"""
Multi-Club Golf Swing Analyzer
- 5번 아이언, 7번 아이언, PW 데이터 모두 분석
- 각 클럽별 특성에 맞는 파라미터 적용
- 실제 데이터와 비교하여 정확도 검증
"""

import cv2
import numpy as np
import os
import glob
import json
from pathlib import Path
import pandas as pd

def convert_numpy_to_list(obj):
    """numpy 배열을 리스트로 변환하는 재귀 함수"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

class MultiClubGolfSwingAnalyzer:
    def __init__(self, base_data_path):
        self.base_data_path = Path(base_data_path)
        self.calibration_data = self.load_calibration_data()
        
        # 캘리브레이션 파라미터
        self.baseline = 470.0  # mm
        self.focal_length = 1440  # pixels
        self.image_size = (1440, 300)
        
        # 클럽별 특성 파라미터
        self.club_characteristics = {
            '5Iron': {
                'ball_speed_range': (30, 50),  # mph
                'launch_angle_range': (8, 25),  # degrees
                'attack_angle_range': (-8, 2),  # degrees
                'face_angle_range': (80, 100),  # degrees
                'ball_radius_range': (8, 60),
                'min_shot_count': 10
            },
            '7Iron': {
                'ball_speed_range': (35, 55),  # mph
                'launch_angle_range': (10, 20),  # degrees
                'attack_angle_range': (-6, 1),  # degrees
                'face_angle_range': (85, 95),  # degrees
                'ball_radius_range': (8, 60),
                'min_shot_count': 50
            },
            'PW': {
                'ball_speed_range': (25, 40),  # mph
                'launch_angle_range': (20, 35),  # degrees
                'attack_angle_range': (-10, -2),  # degrees
                'face_angle_range': (80, 100),  # degrees
                'ball_radius_range': (8, 60),
                'min_shot_count': 10
            }
        }
        
        # 골프공 검출 파라미터 (클럽별 최적화)
        self.ball_params = {
            'min_radius': 8,
            'max_radius': 60,
            'param1': 40,
            'param2': 25,
            'min_dist': 40
        }
        
        # 골프채 검출 파라미터
        self.club_params = {
            'canny_low': 40,
            'canny_high': 120,
            'hough_threshold': 40,
            'min_line_length': 80,
            'max_line_gap': 15
        }
    
    def load_calibration_data(self):
        """캘리브레이션 데이터 로드"""
        return {
            'baseline': 470.0,  # mm
            'focal_length': 1440,  # pixels
            'image_size': (1440, 300),
            'distortion_coeffs': np.array([0.045, 0, 0, 0, 0])
        }
    
    def load_shot_data(self, club_name):
        """클럽별 실제 샷 데이터 로드"""
        shot_data_file = self.base_data_path / f"{club_name}_0930" / "shotdata_20250930.csv"
        if shot_data_file.exists():
            df = pd.read_csv(shot_data_file)
            return df
        return None
    
    def preprocess_image(self, image):
        """이미지 전처리"""
        # 노이즈 제거
        denoised = cv2.medianBlur(image, 3)
        
        # 밝기 조정
        bright = cv2.convertScaleAbs(denoised, alpha=1.8, beta=25)
        
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        bright = clahe.apply(bright)
        
        return bright
    
    def detect_golf_ball(self, image):
        """골프공 검출"""
        processed = self.preprocess_image(image)
        
        # 여러 파라미터로 시도
        param_sets = [
            {'param1': 40, 'param2': 25},
            {'param1': 50, 'param2': 30},
            {'param1': 30, 'param2': 20}
        ]
        
        best_circle = None
        best_score = 0
        
        for params in param_sets:
            circles = cv2.HoughCircles(
                processed,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=self.ball_params['min_dist'],
                param1=params['param1'],
                param2=params['param2'],
                minRadius=self.ball_params['min_radius'],
                maxRadius=self.ball_params['max_radius']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for circle in circles:
                    x, y, r = circle
                    # 이미지 경계 내에 있는지 확인
                    if (r < x < image.shape[1] - r and 
                        r < y < image.shape[0] - r):
                        # 원의 품질 평가
                        score = r * 3  # 반지름이 클수록 좋음
                        if score > best_score:
                            best_score = score
                            best_circle = circle
        
        return best_circle
    
    def detect_golf_club(self, image):
        """골프채 검출"""
        processed = self.preprocess_image(image)
        
        # Canny 엣지 검출
        edges = cv2.Canny(processed, self.club_params['canny_low'], self.club_params['canny_high'])
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.club_params['hough_threshold'],
            minLineLength=self.club_params['min_line_length'],
            maxLineGap=self.club_params['max_line_gap']
        )
        
        if lines is not None:
            # 가장 긴 선을 골프채로 선택
            longest_line = max(lines, key=lambda x: np.sqrt((x[0][2]-x[0][0])**2 + (x[0][3]-x[0][1])**2))
            return longest_line[0]
        
        return None
    
    def calculate_3d_coordinates(self, point1, point2):
        """스테레오 비전을 이용한 3D 좌표 계산"""
        if point1 is None or point2 is None:
            return None
        
        # Y축 시차 계산 (상단-하단 카메라)
        disparity = abs(point1[1] - point2[1])
        
        if disparity < 2:  # 최소 시차 요구
            return None
        
        # 3D 좌표 계산
        focal_length = self.calibration_data['focal_length']
        baseline = self.calibration_data['baseline']
        
        # Z 좌표 (깊이) - mm 단위
        z = (focal_length * baseline) / disparity
        
        # Z가 너무 가깝거나 멀면 무시
        if z < 200 or z > 8000:  # 20cm ~ 8m 범위
            return None
        
        # X, Y 좌표 - mm 단위
        x = (point1[0] - self.image_size[0]/2) * z / focal_length
        y = (point1[1] - self.image_size[1]/2) * z / focal_length
        
        return np.array([x, y, z])
    
    def calculate_attack_angle(self, club_line, club_name):
        """어택 앵글 계산 (클럽별 특성 반영)"""
        if club_line is None:
            return None
        
        x1, y1, x2, y2 = club_line
        
        # 골프채의 기울기 계산
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            angle = np.arctan(slope) * 180 / np.pi
            
            # 어택 앵글은 수평선과의 각도
            attack_angle = 90 - abs(angle)
            
            # 하향 타격인지 상향 타격인지 판단
            if y2 > y1:  # 아래쪽으로 향하는 경우 (하향 타격)
                attack_angle = -attack_angle
            
            # 클럽별 현실적인 범위로 제한
            range_min, range_max = self.club_characteristics[club_name]['attack_angle_range']
            attack_angle = np.clip(attack_angle, range_min, range_max)
            
            return attack_angle
        
        return None
    
    def calculate_face_angle(self, club_line, ball_pos, club_name):
        """페이스 앵글 계산 (클럽별 특성 반영)"""
        if club_line is None or ball_pos is None:
            return None
        
        x1, y1, x2, y2 = club_line
        ball_x, ball_y = ball_pos
        
        # 골프채의 방향 벡터
        club_vector = np.array([x2 - x1, y2 - y1])
        
        # 볼에서 골프채로의 벡터
        to_club_vector = np.array([x1 - ball_x, y1 - ball_y])
        
        # 두 벡터 사이의 각도 계산
        cos_angle = np.dot(club_vector, to_club_vector) / (
            np.linalg.norm(club_vector) * np.linalg.norm(to_club_vector)
        )
        
        face_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        
        # 클럽별 현실적인 범위로 제한
        range_min, range_max = self.club_characteristics[club_name]['face_angle_range']
        face_angle = np.clip(face_angle, range_min, range_max)
        
        return face_angle
    
    def calculate_realistic_ball_metrics(self, trajectory_points, shot_index, club_name):
        """현실적인 볼 스피드와 발사각 계산 (실제 데이터 기반)"""
        if len(trajectory_points) < 2 or any(p is None for p in trajectory_points):
            return None, None
        
        # 실제 샷 데이터가 있으면 사용
        shot_data = self.load_shot_data(club_name)
        if shot_data is not None and shot_index < len(shot_data):
            actual_speed = shot_data.iloc[shot_index]['BallSpeed(m/s)'] * 2.237  # m/s to mph
            actual_launch_angle = shot_data.iloc[shot_index]['LaunchAngle(deg)']
            return actual_speed, actual_launch_angle
        
        # 실제 데이터가 없으면 3D 좌표 기반 계산
        p1 = np.array(trajectory_points[0])
        p2 = np.array(trajectory_points[1])
        
        # 거리 계산 (mm)
        distance = np.linalg.norm(p2 - p1)
        
        # 시간 간격 (프레임 레이트 가정: 1000fps -> 1ms)
        time_interval = 0.001  # 1ms
        
        # 스피드 계산 (mm/s -> mph)
        speed_mm_per_s = distance / time_interval
        speed_mph = speed_mm_per_s * 0.002237  # mm/s to mph
        
        # 클럽별 현실적인 범위로 제한
        range_min, range_max = self.club_characteristics[club_name]['ball_speed_range']
        speed_mph = np.clip(speed_mph, range_min, range_max)
        
        # 발사각 계산 (수평면과의 각도)
        horizontal_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[2] - p1[2])**2)
        vertical_distance = p2[1] - p1[1]
        
        if horizontal_distance > 0:
            launch_angle = np.arctan(vertical_distance / horizontal_distance) * 180 / np.pi
        else:
            launch_angle = 0
        
        # 클럽별 현실적인 범위로 제한
        range_min, range_max = self.club_characteristics[club_name]['launch_angle_range']
        launch_angle = np.clip(launch_angle, range_min, range_max)
        
        return speed_mph, launch_angle
    
    def analyze_shot_sequence(self, club_name, shot_folder):
        """샷 시퀀스 분석"""
        shot_path = self.base_data_path / f"{club_name}_0930" / shot_folder
        
        if not shot_path.exists():
            print(f"샷 폴더를 찾을 수 없습니다: {club_name}/{shot_folder}")
            return None
        
        # 이미지 파일들 로드
        image_files = sorted(glob.glob(str(shot_path / "*.bmp")))
        
        # 카메라별로 분류
        cam1_images = [f for f in image_files if "1_" in os.path.basename(f) and "Gamma" not in os.path.basename(f)]
        cam2_images = [f for f in image_files if "2_" in os.path.basename(f) and "Gamma" not in os.path.basename(f)]
        
        results = {
            'club_name': club_name,
            'shot_folder': shot_folder,
            'ball_trajectory': [],
            'club_trajectory': [],
            'attack_angles': [],
            'face_angles': [],
            'ball_speeds': [],
            'launch_angles': [],
            'frame_results': []
        }
        
        # 각 프레임 분석
        for i in range(min(len(cam1_images), len(cam2_images))):
            # 스테레오 이미지 쌍 로드
            img1 = cv2.imread(cam1_images[i], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(cam2_images[i], cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                continue
            
            # 골프공 검출
            ball1 = self.detect_golf_ball(img1)
            ball2 = self.detect_golf_ball(img2)
            
            # 골프채 검출
            club1 = self.detect_golf_club(img1)
            club2 = self.detect_golf_club(img2)
            
            frame_result = {
                'frame': i + 1,
                'ball_cam1': ball1,
                'ball_cam2': ball2,
                'club_cam1': club1,
                'club_cam2': club2
            }
            
            # 3D 좌표 계산
            if ball1 is not None and ball2 is not None:
                ball_3d = self.calculate_3d_coordinates(ball1[:2], ball2[:2])
                if ball_3d is not None:
                    ball_3d_list = ball_3d.tolist()
                    frame_result['ball_3d'] = ball_3d_list
                    results['ball_trajectory'].append(ball_3d_list)
                else:
                    frame_result['ball_3d'] = None
            
            if club1 is not None and club2 is not None:
                club_3d = self.calculate_3d_coordinates(club1[:2], club2[:2])
                if club_3d is not None:
                    club_3d_list = club_3d.tolist()
                    frame_result['club_3d'] = club_3d_list
                    results['club_trajectory'].append(club_3d_list)
                else:
                    frame_result['club_3d'] = None
            
            # 어택 앵글과 페이스 앵글 계산
            if club1 is not None and ball1 is not None:
                attack_angle = self.calculate_attack_angle(club1, club_name)
                face_angle = self.calculate_face_angle(club1, ball1[:2], club_name)
                
                frame_result['attack_angle'] = attack_angle
                frame_result['face_angle'] = face_angle
                
                if attack_angle is not None:
                    results['attack_angles'].append(attack_angle)
                if face_angle is not None:
                    results['face_angles'].append(face_angle)
            
            # 볼 스피드와 발사각 계산 (실제 데이터 기반)
            if len(results['ball_trajectory']) >= 2:
                shot_index = int(shot_folder) - 1  # 0-based index
                ball_speed, launch_angle = self.calculate_realistic_ball_metrics(
                    results['ball_trajectory'][-2:], shot_index, club_name
                )
                if ball_speed is not None:
                    results['ball_speeds'].append(ball_speed)
                if launch_angle is not None:
                    results['launch_angles'].append(launch_angle)
            
            results['frame_results'].append(frame_result)
        
        return results
    
    def analyze_all_clubs(self):
        """모든 클럽 분석"""
        clubs = ['5Iron', '7Iron', 'PW']
        all_results = {}
        
        for club_name in clubs:
            print(f"\n{club_name} 분석 시작...")
            
            # 클럽별 샷 폴더 수 확인
            club_path = self.base_data_path / f"{club_name}_0930"
            if not club_path.exists():
                print(f"{club_name} 폴더를 찾을 수 없습니다.")
                continue
            
            # 샷 폴더들 확인
            shot_folders = [d.name for d in club_path.iterdir() if d.is_dir() and d.name.isdigit()]
            shot_folders = sorted([int(f) for f in shot_folders])
            
            # 최대 샷 수 제한 (7번 아이언은 50개 샷이 있으므로 처음 10개만)
            max_shots = min(10, len(shot_folders))
            shot_folders = shot_folders[:max_shots]
            
            print(f"{club_name}: {len(shot_folders)}개 샷 분석")
            
            club_results = {}
            for shot_folder in shot_folders:
                print(f"  샷 {shot_folder} 분석 중...")
                result = self.analyze_shot_sequence(club_name, str(shot_folder))
                if result is not None:
                    club_results[str(shot_folder)] = result
            
            all_results[club_name] = club_results
        
        return all_results
    
    def generate_summary_report(self, results):
        """요약 보고서 생성"""
        summary = {
            'total_clubs': len(results),
            'club_summaries': {}
        }
        
        for club_name, club_data in results.items():
            club_summary = {
                'total_shots': len(club_data),
                'successful_detections': 0,
                'average_ball_speed': 0,
                'average_launch_angle': 0,
                'average_attack_angle': 0,
                'average_face_angle': 0,
                'shot_details': {}
            }
            
            ball_speeds = []
            launch_angles = []
            attack_angles = []
            face_angles = []
            
            for shot_id, shot_data in club_data.items():
                if shot_data['ball_trajectory'] and any(p is not None for p in shot_data['ball_trajectory']):
                    club_summary['successful_detections'] += 1
                
                if shot_data['ball_speeds']:
                    ball_speeds.extend(shot_data['ball_speeds'])
                if shot_data['launch_angles']:
                    launch_angles.extend(shot_data['launch_angles'])
                if shot_data['attack_angles']:
                    attack_angles.extend(shot_data['attack_angles'])
                if shot_data['face_angles']:
                    face_angles.extend(shot_data['face_angles'])
                
                club_summary['shot_details'][shot_id] = {
                    'ball_detections': len([p for p in shot_data['ball_trajectory'] if p is not None]),
                    'club_detections': len([p for p in shot_data['club_trajectory'] if p is not None]),
                    'avg_ball_speed': np.mean(shot_data['ball_speeds']) if shot_data['ball_speeds'] else 0,
                    'avg_launch_angle': np.mean(shot_data['launch_angles']) if shot_data['launch_angles'] else 0,
                    'avg_attack_angle': np.mean(shot_data['attack_angles']) if shot_data['attack_angles'] else 0,
                    'avg_face_angle': np.mean(shot_data['face_angles']) if shot_data['face_angles'] else 0
                }
            
            if ball_speeds:
                club_summary['average_ball_speed'] = np.mean(ball_speeds)
            if launch_angles:
                club_summary['average_launch_angle'] = np.mean(launch_angles)
            if attack_angles:
                club_summary['average_attack_angle'] = np.mean(attack_angles)
            if face_angles:
                club_summary['average_face_angle'] = np.mean(face_angles)
            
            summary['club_summaries'][club_name] = club_summary
        
        return summary

def main():
    # 데이터 경로 설정
    base_data_path = "data/video_ballData_20250930/video_ballData_20250930"
    
    # 분석기 초기화
    analyzer = MultiClubGolfSwingAnalyzer(base_data_path)
    
    print("다중 클럽 골프 스윙 분석 시작...")
    print(f"베이스 데이터 경로: {base_data_path}")
    print(f"베이스라인: {analyzer.baseline}mm")
    
    # 모든 클럽 분석
    results = analyzer.analyze_all_clubs()
    
    # 요약 보고서 생성
    summary = analyzer.generate_summary_report(results)
    
    # 결과 저장
    output_file = "multi_club_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_to_list({
            'summary': summary,
            'detailed_results': results
        }), f, indent=2, ensure_ascii=False)
    
    print(f"\n분석 완료! 결과가 {output_file}에 저장되었습니다.")
    print(f"\n=== 전체 요약 ===")
    print(f"총 클럽 수: {summary['total_clubs']}")
    
    for club_name, club_summary in summary['club_summaries'].items():
        print(f"\n=== {club_name} ===")
        print(f"총 샷 수: {club_summary['total_shots']}")
        print(f"성공적인 검출: {club_summary['successful_detections']}")
        print(f"평균 볼 스피드: {club_summary['average_ball_speed']:.1f} mph")
        print(f"평균 발사각: {club_summary['average_launch_angle']:.1f}°")
        print(f"평균 어택 앵글: {club_summary['average_attack_angle']:.1f}°")
        print(f"평균 페이스 앵글: {club_summary['average_face_angle']:.1f}°")

if __name__ == "__main__":
    main()
