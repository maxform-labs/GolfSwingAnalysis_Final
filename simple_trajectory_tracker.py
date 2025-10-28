#!/usr/bin/env python3
"""
간단한 볼 궤적 추적 시스템
연속된 스테레오 비전 이미지에서 볼의 움직임을 추적하여 속도 계산
"""

import cv2
import numpy as np
import os
import glob
import logging
from datetime import datetime
import json
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTrajectoryTracker:
    def __init__(self):
        """볼 궤적 추적기 초기화"""
        self.output_base_path = "ball_trajectory_results"
        
        # 궤적 추적을 위한 파라미터
        self.max_displacement = 150  # 프레임 간 최대 이동 거리 (픽셀)
        self.min_trajectory_length = 3  # 최소 궤적 길이
        
        # 캘리브레이션 데이터 (간단한 값 사용)
        self.baseline = 303.19  # mm
        self.focal_length = 1000  # 픽셀
        self.cx1, self.cy1 = 720, 150  # 카메라1 주점
        self.cx2, self.cy2 = 720, 150  # 카메라2 주점
        
    def create_output_directories(self):
        """출력 디렉토리 생성"""
        clubs = ['5iron', '7iron', 'driver', 'pw']
        
        for club in clubs:
            club_output_path = os.path.join(self.output_base_path, club)
            os.makedirs(club_output_path, exist_ok=True)
            
            # 각 샷별 디렉토리 생성
            for shot_num in range(1, 11):
                shot_path = os.path.join(club_output_path, f"shot_{shot_num}")
                os.makedirs(shot_path, exist_ok=True)
        
        logger.info(f"Created output directories under {self.output_base_path}")
    
    def detect_ball_simple(self, img):
        """간단한 볼 검출"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 가우시안 블러
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # 허프 원 검출
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=8,
                maxRadius=30
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # 가장 큰 원 선택
                if len(circles) > 0:
                    # 반지름 기준으로 정렬
                    circles = sorted(circles, key=lambda x: x[2], reverse=True)
                    x, y, r = circles[0]
                    return (x, y, r)
            
            return None
            
        except Exception as e:
            logger.warning(f"Ball detection error: {str(e)}")
            return None
    
    def detect_ball_in_frame(self, img, previous_ball=None):
        """단일 프레임에서 볼 검출 (이전 위치 힌트 사용)"""
        try:
            # 기본 검출
            ball = self.detect_ball_simple(img)
            
            # 이전 위치가 있으면 거리 기반 필터링
            if ball is not None and previous_ball is not None:
                prev_x, prev_y = previous_ball[:2]
                curr_x, curr_y = ball[:2]
                
                distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                
                if distance > self.max_displacement:
                    logger.warning(f"Ball displacement too large: {distance:.1f} pixels, using previous position")
                    return None
            
            return ball
            
        except Exception as e:
            logger.warning(f"Ball detection error: {str(e)}")
            return None
    
    def calculate_3d_position(self, x1, y1, x2, y2):
        """3D 위치 계산 (Y축 disparity 기반)"""
        try:
            # Y축 disparity 계산
            disparity_y = abs(y1 - y2)
            
            # 3D 좌표 계산
            if disparity_y > 0:
                z_3d = (self.baseline * self.focal_length) / disparity_y
                x_3d = (x1 - self.cx1) * z_3d / self.focal_length
                y_3d = (y1 - self.cy1) * z_3d / self.focal_length
                
                return x_3d, y_3d, z_3d
            else:
                return None, None, None
                
        except Exception as e:
            logger.error(f"3D calculation error: {str(e)}")
            return None, None, None
    
    def track_ball_trajectory(self, img1_files, img2_files):
        """볼 궤적 추적"""
        trajectories = {'camera1': [], 'camera2': []}
        
        # 프레임 수 확인
        num_frames = min(len(img1_files), len(img2_files))
        logger.info(f"Tracking {num_frames} frames")
        
        prev_ball1 = None
        prev_ball2 = None
        
        for frame_idx in range(num_frames):
            try:
                # 이미지 로드
                img1 = cv2.imread(img1_files[frame_idx])
                img2 = cv2.imread(img2_files[frame_idx])
                
                if img1 is None or img2 is None:
                    logger.warning(f"Could not load frame {frame_idx}")
                    continue
                
                # 볼 검출 (이전 위치 힌트 사용)
                ball1 = self.detect_ball_in_frame(img1, prev_ball1)
                ball2 = self.detect_ball_in_frame(img2, prev_ball2)
                
                # 궤적에 추가
                if ball1 is not None:
                    trajectories['camera1'].append({
                        'frame': frame_idx,
                        'x': ball1[0],
                        'y': ball1[1],
                        'radius': ball1[2],
                        'timestamp': frame_idx / 30.0  # 30fps 가정
                    })
                    prev_ball1 = ball1
                else:
                    # 검출 실패 시 이전 위치 유지 (보간)
                    if prev_ball1 is not None:
                        trajectories['camera1'].append({
                            'frame': frame_idx,
                            'x': prev_ball1[0],
                            'y': prev_ball1[1],
                            'radius': prev_ball1[2],
                            'timestamp': frame_idx / 30.0,
                            'interpolated': True
                        })
                
                if ball2 is not None:
                    trajectories['camera2'].append({
                        'frame': frame_idx,
                        'x': ball2[0],
                        'y': ball2[1],
                        'radius': ball2[2],
                        'timestamp': frame_idx / 30.0
                    })
                    prev_ball2 = ball2
                else:
                    # 검출 실패 시 이전 위치 유지 (보간)
                    if prev_ball2 is not None:
                        trajectories['camera2'].append({
                            'frame': frame_idx,
                            'x': prev_ball2[0],
                            'y': prev_ball2[1],
                            'radius': prev_ball2[2],
                            'timestamp': frame_idx / 30.0,
                            'interpolated': True
                        })
                
                logger.info(f"Frame {frame_idx}: Camera1={ball1 is not None}, Camera2={ball2 is not None}")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                continue
        
        return trajectories
    
    def calculate_3d_trajectory(self, trajectory1, trajectory2):
        """3D 궤적 계산"""
        if not trajectory1 or not trajectory2:
            return []
        
        min_length = min(len(trajectory1), len(trajectory2))
        trajectory_3d = []
        
        for i in range(min_length):
            try:
                point1 = trajectory1[i]
                point2 = trajectory2[i]
                
                # 3D 좌표 계산
                x_3d, y_3d, z_3d = self.calculate_3d_position(
                    point1['x'], point1['y'], point2['x'], point2['y']
                )
                
                if x_3d is not None:
                    trajectory_3d.append({
                        'frame': point1['frame'],
                        'timestamp': point1['timestamp'],
                        'x_3d': x_3d,
                        'y_3d': y_3d,
                        'z_3d': z_3d,
                        'x1_pixel': point1['x'],
                        'y1_pixel': point1['y'],
                        'x2_pixel': point2['x'],
                        'y2_pixel': point2['y'],
                        'interpolated': point1.get('interpolated', False) or point2.get('interpolated', False)
                    })
                else:
                    logger.warning(f"3D calculation failed for frame {point1['frame']}")
                    
            except Exception as e:
                logger.error(f"Error calculating 3D for frame {i}: {str(e)}")
                continue
        
        return trajectory_3d
    
    def calculate_ball_speed(self, trajectory_3d):
        """볼 속도 계산"""
        if len(trajectory_3d) < 2:
            return None
        
        # 시간 간격 계산 (초)
        time_intervals = []
        distances = []
        
        for i in range(1, len(trajectory_3d)):
            dt = trajectory_3d[i]['timestamp'] - trajectory_3d[i-1]['timestamp']
            if dt > 0:
                # 3D 거리 계산 (mm)
                dx = trajectory_3d[i]['x_3d'] - trajectory_3d[i-1]['x_3d']
                dy = trajectory_3d[i]['y_3d'] - trajectory_3d[i-1]['y_3d']
                dz = trajectory_3d[i]['z_3d'] - trajectory_3d[i-1]['z_3d']
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                
                time_intervals.append(dt)
                distances.append(distance)
        
        if not time_intervals:
            return None
        
        # 평균 속도 계산 (mm/s)
        speeds = [d/t for d, t in zip(distances, time_intervals)]
        avg_speed = np.mean(speeds)
        
        # m/s로 변환
        speed_ms = avg_speed / 1000.0
        
        # mph로 변환
        speed_mph = speed_ms * 2.237
        
        return {
            'speed_mph': speed_mph,
            'speed_ms': speed_ms,
            'speed_mmps': avg_speed,
            'trajectory_length': len(trajectory_3d),
            'time_span': trajectory_3d[-1]['timestamp'] - trajectory_3d[0]['timestamp']
        }
    
    def visualize_trajectory(self, img1_files, img2_files, trajectory1, trajectory2, output_path):
        """궤적 시각화"""
        try:
            # 첫 번째 프레임으로 시각화
            img1 = cv2.imread(img1_files[0])
            img2 = cv2.imread(img2_files[0])
            
            if img1 is None or img2 is None:
                return
            
            # 궤적 그리기
            for i, point in enumerate(trajectory1):
                if i < len(trajectory1) - 1:
                    next_point = trajectory1[i + 1]
                    cv2.line(img1, 
                            (int(point['x']), int(point['y'])), 
                            (int(next_point['x']), int(next_point['y'])), 
                            (0, 255, 0), 2)
                
                # 점 표시
                color = (0, 255, 0) if not point.get('interpolated', False) else (0, 255, 255)
                cv2.circle(img1, (int(point['x']), int(point['y'])), 3, color, -1)
            
            for i, point in enumerate(trajectory2):
                if i < len(trajectory2) - 1:
                    next_point = trajectory2[i + 1]
                    cv2.line(img2, 
                            (int(point['x']), int(point['y'])), 
                            (int(next_point['x']), int(next_point['y'])), 
                            (0, 255, 0), 2)
                
                # 점 표시
                color = (0, 255, 0) if not point.get('interpolated', False) else (0, 255, 255)
                cv2.circle(img2, (int(point['x']), int(point['y'])), 3, color, -1)
            
            # 결과 저장
            cv2.imwrite(os.path.join(output_path, "camera1_trajectory.jpg"), img1)
            cv2.imwrite(os.path.join(output_path, "camera2_trajectory.jpg"), img2)
            
        except Exception as e:
            logger.error(f"Error visualizing trajectory: {str(e)}")
    
    def process_single_shot(self, shot_path, club_name, shot_num):
        """단일 샷 궤적 추적 처리"""
        # 이미지 파일 찾기
        img1_files = sorted(glob.glob(os.path.join(shot_path, "1_*.bmp")))
        img2_files = sorted(glob.glob(os.path.join(shot_path, "2_*.bmp")))
        
        if not img1_files or not img2_files:
            logger.warning(f"No image files found in {shot_path}")
            return None
        
        logger.info(f"Processing {club_name} shot {shot_num} with {len(img1_files)} frames")
        
        # 궤적 추적
        trajectories = self.track_ball_trajectory(img1_files, img2_files)
        
        # 3D 궤적 계산
        trajectory_3d = self.calculate_3d_trajectory(trajectories['camera1'], trajectories['camera2'])
        
        if not trajectory_3d:
            logger.warning(f"No 3D trajectory calculated for {club_name} shot {shot_num}")
            return None
        
        # 속도 계산
        speed_info = self.calculate_ball_speed(trajectory_3d)
        
        # 출력 디렉토리
        output_dir = os.path.join(self.output_base_path, club_name, f"shot_{shot_num}")
        
        # 궤적 시각화
        self.visualize_trajectory(img1_files, img2_files, 
                                 trajectories['camera1'], trajectories['camera2'], output_dir)
        
        # 결과 저장
        result = {
            'club': club_name,
            'shot_number': shot_num,
            'trajectory_2d_camera1': trajectories['camera1'],
            'trajectory_2d_camera2': trajectories['camera2'],
            'trajectory_3d': trajectory_3d,
            'speed_info': speed_info,
            'num_frames': len(img1_files),
            'timestamp': datetime.now().isoformat()
        }
        
        # JSON 저장
        with open(os.path.join(output_dir, "trajectory_data.json"), 'w') as f:
            json.dump(result, f, indent=2)
        
        # CSV 저장
        if trajectory_3d:
            df = pd.DataFrame(trajectory_3d)
            df.to_csv(os.path.join(output_dir, "trajectory_3d.csv"), index=False)
        
        logger.info(f"Shot {shot_num}: Ball speed = {speed_info['speed_mph']:.1f} mph" if speed_info else "No speed calculated")
        
        return result
    
    def process_all_shots(self, data_path):
        """모든 샷 처리"""
        clubs = {
            '5Iron_0930': '5iron',
            '7Iron_0930': '7iron', 
            'driver_0930': 'driver',
            'PW_0930': 'pw'
        }
        
        all_results = []
        
        for club_folder, club_name in clubs.items():
            club_path = os.path.join(data_path, club_folder)
            
            if not os.path.exists(club_path):
                logger.warning(f"Club path not found: {club_path}")
                continue
            
            logger.info(f"Processing {club_name} shots for trajectory tracking...")
            
            # 샷 디렉토리 찾기
            shot_dirs = [d for d in os.listdir(club_path) if d.isdigit()]
            shot_dirs.sort(key=int)
            
            for shot_num in shot_dirs[:3]:  # 처음 3개 샷만 처리 (테스트용)
                shot_path = os.path.join(club_path, shot_num)
                
                result = self.process_single_shot(shot_path, club_name, int(shot_num))
                if result:
                    all_results.append(result)
                
                logger.info(f"Completed {club_name} shot {shot_num}")
        
        return all_results

def main():
    """메인 실행 함수"""
    logger.info("Starting simple ball trajectory tracking...")
    
    tracker = SimpleTrajectoryTracker()
    
    # 출력 디렉토리 생성
    tracker.create_output_directories()
    
    # 모든 샷 처리
    data_path = "data/video_ballData_20250930/video_ballData_20250930"
    if os.path.exists(data_path):
        results = tracker.process_all_shots(data_path)
        
        logger.info(f"Trajectory tracking completed! Processed {len(results)} shots")
        logger.info(f"Results saved to: {tracker.output_base_path}")
        
        # 속도 요약 출력
        print(f"\n=== 궤적 추적 결과 요약 ===")
        for result in results:
            speed_info = result.get('speed_info')
            if speed_info:
                print(f"{result['club']} Shot {result['shot_number']}: {speed_info['speed_mph']:.1f} mph")
            else:
                print(f"{result['club']} Shot {result['shot_number']}: No speed calculated")
        
    else:
        logger.error(f"Data path not found: {data_path}")

if __name__ == "__main__":
    main()





