#!/usr/bin/env python3
"""
골프공과 골프채 검출을 위한 스테레오 비전 시스템
실제 구현 예제
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path

class GolfSwingAnalyzer:
    def __init__(self, calibration_file="stereo_calibration_results.json"):
        """골프 스윙 분석기 초기화"""
        self.load_calibration_data(calibration_file)
        self.setup_detection_parameters()
        self.ball_trajectory = []
        self.club_trajectory = []
        
    def load_calibration_data(self, calibration_file):
        """캘리브레이션 데이터 로드"""
        try:
            with open(calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            print("✅ 캘리브레이션 데이터 로드 완료")
        except FileNotFoundError:
            print("❌ 캘리브레이션 파일을 찾을 수 없습니다.")
            print("먼저 캘리브레이션을 수행해주세요.")
            self.calibration_data = None
    
    def setup_detection_parameters(self):
        """검출 파라미터 설정"""
        # 골프공 검출 파라미터
        self.ball_params = {
            'dp': 1,
            'minDist': 30,
            'param1': 50,
            'param2': 30,
            'minRadius': 5,
            'maxRadius': 25
        }
        
        # 골프채 검출 파라미터
        self.club_params = {
            'rho': 1,
            'theta': np.pi/180,
            'threshold': 50,
            'minLineLength': 100,
            'maxLineGap': 20
        }
    
    def preprocess_image(self, image, method="gamma_correct"):
        """이미지 전처리"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == "gamma_correct":
            gamma = 1.5
            gamma_corrected = np.power(gray / 255.0, gamma) * 255.0
            return np.uint8(gamma_corrected)
        elif method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(gray)
        else:
            return gray
    
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
                if 8 <= r <= 20 and 200 <= x <= 1720 and 200 <= y <= 880:
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
        
        points_3d = cv2.triangulatePoints(
            np.array(self.calibration_data['camera_matrix_1']),
            np.array(self.calibration_data['camera_matrix_2']),
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
                
                distance = abs(line2[0][0] * x2 + line2[0][1] * y2 + line2[0][2]) /                           np.sqrt(line2[0][0]**2 + line2[0][1]**2)
                
                if distance < 10 and distance < min_distance:
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
    
    def process_frame_pair(self, frame1, frame2):
        """프레임 쌍 처리"""
        start_time = time.time()
        
        # 골프공 검출
        balls1 = self.detect_golf_ball(frame1)
        balls2 = self.detect_golf_ball(frame2)
        
        result = {
            'processing_time': 0,
            'ball_detected': False,
            'club_detected': False,
            'position_3d': None,
            'speed_mph': 0,
            'launch_angle': 0,
            'direction_angle': 0
        }
        
        if balls1 and balls2:
            # 골프공 매칭
            matched_pairs = self.match_golf_balls(balls1, balls2)
            
            if matched_pairs:
                ball1, ball2 = matched_pairs[0]
                point1 = np.array([[ball1[0], ball1[1]]], dtype=np.float32)
                point2 = np.array([[ball2[0], ball2[1]]], dtype=np.float32)
                
                # 3D 좌표 계산
                position_3d = self.calculate_3d_coordinates(point1, point2)
                
                if position_3d is not None:
                    result['ball_detected'] = True
                    result['position_3d'] = position_3d.flatten()
                    
                    # 궤적 업데이트
                    self.ball_trajectory.append(position_3d.flatten())
                    
                    # 최대 10개 위치만 유지
                    if len(self.ball_trajectory) > 10:
                        self.ball_trajectory.pop(0)
                    
                    # 분석 결과 계산
                    if len(self.ball_trajectory) >= 3:
                        result['speed_mph'] = self.calculate_ball_speed(self.ball_trajectory)
                        result['launch_angle'] = self.calculate_launch_angle(self.ball_trajectory)
                        result['direction_angle'] = self.calculate_direction_angle(self.ball_trajectory)
        
        # 골프채 검출
        club1 = self.detect_golf_club(frame1)
        club2 = self.detect_golf_club(frame2)
        
        if club1 or club2:
            result['club_detected'] = True
        
        result['processing_time'] = (time.time() - start_time) * 1000  # ms
        
        return result
    
    def visualize_results(self, frame1, frame2, result):
        """결과 시각화"""
        # 골프공 검출 결과 그리기
        if result['ball_detected']:
            # 프레임1에 골프공 표시
            cv2.circle(frame1, (int(result['position_3d'][0]), int(result['position_3d'][1])), 
                      10, (0, 255, 0), 2)
            
            # 프레임2에 골프공 표시
            cv2.circle(frame2, (int(result['position_3d'][0]), int(result['position_3d'][1])), 
                      10, (0, 255, 0), 2)
        
        # 정보 텍스트 표시
        info_text = f"Speed: {result['speed_mph']:.1f} mph"
        cv2.putText(frame1, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        info_text = f"Launch: {result['launch_angle']:.1f}°"
        cv2.putText(frame1, info_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        info_text = f"Direction: {result['direction_angle']:.1f}°"
        cv2.putText(frame1, info_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        info_text = f"Time: {result['processing_time']:.1f} ms"
        cv2.putText(frame1, info_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame1, frame2

def main():
    """메인 함수"""
    print("🏌️ 골프 스윙 분석기 시작")
    
    # 분석기 초기화
    analyzer = GolfSwingAnalyzer()
    
    if analyzer.calibration_data is None:
        print("❌ 캘리브레이션 데이터가 없습니다.")
        print("먼저 캘리브레이션을 수행해주세요.")
        return
    
    # 예제 사용법
    print("📋 사용법:")
    print("1. 두 카메라에서 동시에 촬영한 이미지를 준비하세요.")
    print("2. process_frame_pair(frame1, frame2) 메서드를 호출하세요.")
    print("3. 결과를 visualize_results()로 시각화하세요.")
    
    # 예제 코드
    print("\n💻 예제 코드:")
    print("""
    # 이미지 로드
    frame1 = cv2.imread('cam1_image.jpg')
    frame2 = cv2.imread('cam2_image.jpg')
    
    # 분석 수행
    result = analyzer.process_frame_pair(frame1, frame2)
    
    # 결과 시각화
    frame1_vis, frame2_vis = analyzer.visualize_results(frame1, frame2, result)
    
    # 결과 출력
    print(f"골프공 검출: {result['ball_detected']}")
    print(f"속도: {result['speed_mph']:.1f} mph")
    print(f"발사각: {result['launch_angle']:.1f}°")
    print(f"처리 시간: {result['processing_time']:.1f} ms")
    """)

if __name__ == "__main__":
    main()
