#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
골프 샷 이미지 분석 프로그램
BMP 이미지를 JPG로 변환하고 볼/클럽 데이터를 추출
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
import shutil

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 기존 모듈 임포트 (선택적)
try:
    from GolfSwingAnalysis_Final_ver8.golf_image_analyzer import GolfImageAnalyzer
    from GolfSwingAnalysis_Final_ver8.stereo_vision_vertical import VerticalStereoVision
    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False

class GolfShotAnalyzer:
    def __init__(self):
        # 모듈이 있으면 사용, 없으면 내장 메서드 사용
        if HAS_MODULES:
            self.image_analyzer = GolfImageAnalyzer()
            # VerticalStereoVision을 위한 설정 객체 생성
            class Config:
                def __init__(self):
                    self.camera_settings = {
                        'fps': 820,
                        'resolution': [1440, 300],
                        'vertical_baseline': 500.0,
                        'camera1_height': 400.0,
                        'camera2_height': 900.0,
                        'inward_angle': 12.0
                    }
                    self.processing_threads = 4
                    self.app_name = "GolfSwingAnalysis"
                    self.app_version = "4.0"
            
            config = Config()
            self.stereo_system = VerticalStereoVision(config)
        else:
            self.image_analyzer = None
            self.stereo_system = None
        self.results = {
            'ball_positions': [],
            'club_positions': [],
            'stereo_depth': [],
            'ball_speed': None,
            'launch_angle': None,
            'club_speed': None
        }
        
    def convert_bmp_to_jpg(self, source_dir, output_dir):
        """BMP 이미지를 JPG로 변환"""
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        converted_files = []
        
        # BMP 파일 찾기 및 변환
        for bmp_file in source_path.glob('*.bmp'):
            try:
                # 이미지 열기
                img = Image.open(bmp_file)
                img = img.convert('RGB')
                
                # JPG로 저장
                jpg_filename = bmp_file.stem + '.jpg'
                jpg_path = output_path / jpg_filename
                img.save(jpg_path, 'JPEG', quality=95)
                
                converted_files.append(str(jpg_path))
                print(f"변환 완료: {bmp_file.name} → {jpg_filename}")
                
            except Exception as e:
                print(f"변환 실패 {bmp_file.name}: {e}")
                
        return converted_files
        
    def analyze_frame_sequence(self, image_dir, camera_position='top'):
        """프레임 시퀀스 분석"""
        image_path = Path(image_dir)
        
        # 카메라 위치에 따른 파일 패턴
        if camera_position == 'top':
            pattern = '1_*.jpg'  # 상단 카메라
        else:
            pattern = '2_*.jpg'  # 하단 카메라
            
        # 이미지 파일 정렬
        image_files = sorted(image_path.glob(pattern), 
                           key=lambda x: int(x.stem.split('_')[1]))
        
        ball_positions = []
        club_positions = []
        
        for idx, img_file in enumerate(image_files):
            print(f"\n프레임 {idx+1}/{len(image_files)} 분석 중: {img_file.name}")
            
            # 이미지 읽기
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"이미지 읽기 실패: {img_file}")
                continue
                
            # 골프공 검출
            ball_result = self.detect_golf_ball(frame)
            if ball_result:
                ball_positions.append({
                    'frame': idx + 1,
                    'position': ball_result['center'],
                    'radius': ball_result['radius'],
                    'confidence': ball_result.get('confidence', 0.9)
                })
                print(f"  볼 검출: 위치={ball_result['center']}, 반경={ball_result['radius']}")
            
            # 클럽 헤드 검출
            club_result = self.detect_club_head(frame)
            if club_result:
                club_positions.append({
                    'frame': idx + 1,
                    'position': club_result['center'],
                    'angle': club_result.get('angle', 0),
                    'confidence': club_result.get('confidence', 0.85)
                })
                print(f"  클럽 검출: 위치={club_result['center']}")
                
        return ball_positions, club_positions
        
    def detect_golf_ball(self, frame):
        """골프공 검출 - 향상된 알고리즘"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 적응형 히스토그램 균일화
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # Hough Circle 검출
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=30
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 가장 큰 원을 골프공으로 선택
            best_circle = None
            max_radius = 0
            
            for circle in circles[0, :]:
                x, y, r = circle
                # 흰색 영역 비율 확인
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]
                
                # 골프공은 일반적으로 밝은색
                if mean_val > 180 and r > max_radius:
                    best_circle = circle
                    max_radius = r
                    
            if best_circle is not None:
                return {
                    'center': (int(best_circle[0]), int(best_circle[1])),
                    'radius': int(best_circle[2]),
                    'confidence': 0.9
                }
                
        # 대체 방법: 컨투어 기반 검출
        _, thresh = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 3000:  # 골프공 크기 범위
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
                
                if circularity > 0.7:  # 원형성 체크
                    return {
                        'center': (int(x), int(y)),
                        'radius': int(radius),
                        'confidence': 0.8
                    }
                    
        return None
        
    def detect_club_head(self, frame):
        """클럽 헤드 검출"""
        # HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 금속성 클럽 헤드 검출을 위한 범위
        # 은색/회색 범위
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 30, 200])
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # 모폴로지 연산
        kernel = np.ones((5, 5), np.uint8)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 클럽 헤드 후보 찾기
        best_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:  # 클럽 헤드 크기 범위
                # 종횡비 확인
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                if 0.5 < aspect_ratio < 3.0 and area > max_area:
                    best_contour = contour
                    max_area = area
                    
        if best_contour is not None:
            # 클럽 헤드 중심 및 각도 계산
            M = cv2.moments(best_contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # 최소 면적 사각형으로 각도 계산
            rect = cv2.minAreaRect(best_contour)
            angle = rect[2]
            
            return {
                'center': (cx, cy),
                'angle': angle,
                'area': max_area,
                'confidence': 0.85
            }
            
        return None
        
    def perform_stereo_matching(self, top_positions, bottom_positions):
        """스테레오 매칭으로 3D 위치 계산"""
        matched_positions = []
        
        for top_ball in top_positions:
            frame_num = top_ball['frame']
            
            # 같은 프레임의 하단 카메라 데이터 찾기
            bottom_ball = next((b for b in bottom_positions if b['frame'] == frame_num), None)
            
            if bottom_ball:
                # Y축 시차 계산
                disparity_y = top_ball['position'][1] - bottom_ball['position'][1]
                
                # 깊이 계산 (수직 스테레오 비전)
                baseline = 500  # mm
                focal_length = 1000  # 픽셀 (추정값)
                
                if abs(disparity_y) > 0:
                    depth = (focal_length * baseline) / abs(disparity_y)
                    
                    # 3D 위치 계산
                    x_3d = (top_ball['position'][0] * depth) / focal_length
                    y_3d = (top_ball['position'][1] * depth) / focal_length
                    z_3d = depth
                    
                    matched_positions.append({
                        'frame': frame_num,
                        'position_2d_top': top_ball['position'],
                        'position_2d_bottom': bottom_ball['position'],
                        'position_3d': (x_3d, y_3d, z_3d),
                        'disparity_y': disparity_y,
                        'depth': depth
                    })
                    
        return matched_positions
        
    def calculate_velocities(self, positions_3d, fps=820):
        """속도 계산"""
        if len(positions_3d) < 2:
            return None
            
        velocities = []
        dt = 1.0 / fps  # 프레임 간 시간 간격
        
        for i in range(1, len(positions_3d)):
            prev_pos = np.array(positions_3d[i-1]['position_3d'])
            curr_pos = np.array(positions_3d[i]['position_3d'])
            
            # 속도 벡터 계산 (mm/s)
            velocity = (curr_pos - prev_pos) / dt / 1000  # m/s로 변환
            speed = np.linalg.norm(velocity)
            
            # mph로 변환
            speed_mph = speed * 2.237
            
            velocities.append({
                'frame': positions_3d[i]['frame'],
                'velocity_vector': velocity.tolist(),
                'speed_ms': speed,
                'speed_mph': speed_mph
            })
            
        return velocities
        
    def save_results(self, output_file, results):
        """분석 결과 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n결과 저장 완료: {output_file}")
        
    def run_complete_analysis(self, source_dir, shot_name):
        """전체 분석 실행"""
        print(f"\n{'='*60}")
        print(f"골프 샷 분석 시작: {shot_name}")
        print(f"{'='*60}")
        
        # 1. BMP → JPG 변환
        output_dir = str(PROJECT_ROOT / f"shot-image-jpg/{shot_name}")
        print(f"\n1. 이미지 변환 중...")
        converted_files = self.convert_bmp_to_jpg(source_dir, output_dir)
        print(f"   변환 완료: {len(converted_files)}개 파일")
        
        # 2. 상단 카메라 분석
        print(f"\n2. 상단 카메라 이미지 분석...")
        top_balls, top_clubs = self.analyze_frame_sequence(output_dir, 'top')
        print(f"   볼 검출: {len(top_balls)}개 프레임")
        print(f"   클럽 검출: {len(top_clubs)}개 프레임")
        
        # 3. 하단 카메라 분석
        print(f"\n3. 하단 카메라 이미지 분석...")
        bottom_balls, bottom_clubs = self.analyze_frame_sequence(output_dir, 'bottom')
        print(f"   볼 검출: {len(bottom_balls)}개 프레임")
        print(f"   클럽 검출: {len(bottom_clubs)}개 프레임")
        
        # 4. 스테레오 매칭
        print(f"\n4. 스테레오 매칭 수행...")
        ball_3d_positions = self.perform_stereo_matching(top_balls, bottom_balls)
        club_3d_positions = self.perform_stereo_matching(top_clubs, bottom_clubs)
        print(f"   3D 볼 위치: {len(ball_3d_positions)}개")
        print(f"   3D 클럽 위치: {len(club_3d_positions)}개")
        
        # 5. 속도 계산
        print(f"\n5. 속도 계산...")
        ball_velocities = self.calculate_velocities(ball_3d_positions)
        club_velocities = self.calculate_velocities(club_3d_positions)
        
        # 최대 속도 찾기
        max_ball_speed = 0
        max_club_speed = 0
        
        if ball_velocities:
            max_ball_speed = max(v['speed_mph'] for v in ball_velocities)
            print(f"   최대 볼 스피드: {max_ball_speed:.1f} mph")
            
        if club_velocities:
            max_club_speed = max(v['speed_mph'] for v in club_velocities)
            print(f"   최대 클럽 스피드: {max_club_speed:.1f} mph")
            
        # 6. 발사각 계산
        launch_angle = None
        if len(ball_3d_positions) >= 2:
            # 임팩트 후 첫 두 프레임으로 발사각 계산
            p1 = np.array(ball_3d_positions[0]['position_3d'])
            p2 = np.array(ball_3d_positions[1]['position_3d'])
            
            delta = p2 - p1
            horizontal_dist = np.sqrt(delta[0]**2 + delta[2]**2)
            vertical_dist = delta[1]
            
            launch_angle = np.degrees(np.arctan2(vertical_dist, horizontal_dist))
            print(f"   발사각: {launch_angle:.1f}°")
            
        # 결과 정리
        results = {
            'shot_info': {
                'name': shot_name,
                'source_dir': source_dir,
                'total_frames': len(converted_files) // 2,  # 상/하 카메라
                'fps': 820
            },
            'ball_data': {
                'detected_frames': len(top_balls),
                '3d_positions': ball_3d_positions,
                'velocities': ball_velocities,
                'max_speed_mph': max_ball_speed,
                'launch_angle': launch_angle
            },
            'club_data': {
                'detected_frames': len(top_clubs),
                '3d_positions': club_3d_positions,
                'velocities': club_velocities,
                'max_speed_mph': max_club_speed
            },
            'raw_detections': {
                'top_camera_balls': top_balls,
                'top_camera_clubs': top_clubs,
                'bottom_camera_balls': bottom_balls,
                'bottom_camera_clubs': bottom_clubs
            }
        }
        
        # 결과 저장
        output_file = PROJECT_ROOT / f"analysis_results/{shot_name}_analysis.json"
        output_file.parent.mkdir(exist_ok=True)
        self.save_results(str(output_file), results)
        
        return results


def main():
    """메인 실행 함수"""
    analyzer = GolfShotAnalyzer()
    
    # 분석할 샷 목록
    shots_to_analyze = [
        {
            'source_dir': r'C:\src\GolfSwingAnalysis_Final_ver8\shot-image\7번 아이언\로고, 마커없는 볼-1',
            'name': '7번아이언_로고마커없는볼_샷1'
        },
        # 추가 샷이 필요하면 여기에 추가
    ]
    
    all_results = []
    
    for shot in shots_to_analyze:
        try:
            results = analyzer.run_complete_analysis(shot['source_dir'], shot['name'])
            all_results.append(results)
            
            # 주요 결과 출력
            print(f"\n{'='*60}")
            print(f"분석 완료: {shot['name']}")
            print(f"{'='*60}")
            print(f"최대 볼 스피드: {results['ball_data']['max_speed_mph']:.1f} mph")
            print(f"발사각: {results['ball_data']['launch_angle']:.1f}°")
            print(f"최대 클럽 스피드: {results['club_data']['max_speed_mph']:.1f} mph")
            
        except Exception as e:
            print(f"분석 실패 {shot['name']}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\n전체 분석 완료. 총 {len(all_results)}개 샷 분석됨.")
    

if __name__ == "__main__":
    main()