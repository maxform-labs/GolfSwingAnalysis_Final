#!/usr/bin/env python3
"""
향상된 골프공 3D 분석 시스템 v2
- 방향각 계산 로직 개선
- 검출 파라미터 최적화
- Kalman 필터 튜닝
- 좌표계 정렬 확인
"""
import cv2
import numpy as np
import json
import os
import glob
import math
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class SimpleKalmanFilter:
    """간단한 칼만 필터 구현 (튜닝 버전)"""
    def __init__(self, dim_x, dim_z, process_noise=0.01, measurement_noise=100):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))  # 상태
        self.P = np.eye(dim_x) * 1000   # 초기 공분산 (불확실성 높음)
        self.F = np.eye(dim_x)         # 전이 행렬
        self.H = np.zeros((dim_z, dim_x))  # 관측 행렬
        self.Q = np.eye(dim_x) * process_noise   # 프로세스 노이즈 (낮게)
        self.R = np.eye(dim_z) * measurement_noise    # 측정 노이즈 (높게 - 검출 불안정)
    
    def predict(self):
        """예측 단계"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        """업데이트 단계"""
        if len(z.shape) == 1:
            z = z.reshape(-1, 1)
        
        y = z - self.H @ self.x  # 혁신
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 칼만 이득
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

class EnhancedGolfBall3DAnalyzer:
    def __init__(self, calibration_file="precise_vertical_stereo_calibration.json",
                 real_data_file="data2/driver/shotdata_20251020.csv"):
        """
        향상된 골프공 3D 분석기 초기화
        
        Args:
            calibration_file: 캘리브레이션 파일
            real_data_file: 실측 데이터 CSV 파일
        """
        self.calibration_file = calibration_file
        self.real_data_file = real_data_file
        
        self.load_calibration()
        self.load_real_data()
        
        # 고속 카메라 설정
        self.fps = 820
        self.frame_interval = 1.0 / self.fps
        
        # 검출 히스토리 (시간적 연속성)
        self.detection_history = {'cam1': [], 'cam2': []}
        
        # 최적화된 보정 계수
        self.depth_scale_factor = 1.0
        
        # 검출 파라미터 (튜닝)
        self.detection_params = {
            'min_radius': 3,
            'max_radius': 25,
            'hough_param1': 50,  # Canny edge threshold (낮춤 - 더 많은 엣지)
            'hough_param2': 10,  # Accumulator threshold (낮춤 - 더 많은 원 검출)
            'min_circularity': 0.5,  # 최소 원형도 (낮춤)
            'brightness_threshold': 200,  # 밝기 임계값
            'temporal_window': 5,  # 시간적 연속성 윈도우
            'max_position_change': 200  # 최대 위치 변화 (픽셀)
        }
        
        print("=" * 80)
        print("ENHANCED GOLF BALL 3D ANALYZER V2")
        print("=" * 80)
        print(f"Calibration: {self.calibration_file}")
        print(f"Real data: {self.real_data_file}")
        print(f"Camera: {self.fps} fps ({self.frame_interval*1000:.3f} ms/frame)")
        print(f"Baseline: {self.baseline_mm:.2f} mm")
        print(f"Focal length: {self.focal_length:.2f} px")
        print(f"Real shots available: {len(self.real_data)}")
        print(f"Depth scale factor: {self.depth_scale_factor}")
        print("=" * 80)
    
    def load_calibration(self):
        """캘리브레이션 데이터 로드"""
        with open(self.calibration_file, 'r', encoding='utf-8') as f:
            calib = json.load(f)
        
        self.K1 = np.array(calib['camera_matrix_1'])
        self.K2 = np.array(calib['camera_matrix_2'])
        self.D1 = np.array(calib['distortion_coeffs_1'])
        self.D2 = np.array(calib['distortion_coeffs_2'])
        self.R = np.array(calib['rotation_matrix'])
        self.T = np.array(calib['translation_vector'])
        
        self.baseline_mm = calib['baseline_mm']
        self.focal_length = calib['focal_length_px']
        self.image_size = tuple(calib['image_size'])
        
        self.roi_cam1 = calib['roi_cam1']
        self.roi_cam2 = calib['roi_cam2']
    
    def load_real_data(self):
        """실측 데이터 로드"""
        df = pd.read_csv(self.real_data_file)
        
        # 샷 번호는 1부터 시작 (CSV 행 인덱스 + 1)
        self.real_data = {}
        for idx, row in df.iterrows():
            shot_num = idx + 1
            self.real_data[shot_num] = {
                'ball_speed_ms': row['BallSpeed(m/s)'],
                'launch_angle_deg': row['LaunchAngle(deg)'],
                'launch_direction_deg': row['LaunchDirection(deg)'],
                'datetime': row['DateTime']
            }
        
        print(f"\n[OK] Loaded {len(self.real_data)} real measurement data points")
    
    def roi_to_full_coords(self, roi_x: float, roi_y: float, 
                          roi_info: Dict) -> Tuple[float, float]:
        """ROI 좌표를 전체 이미지 좌표로 변환"""
        full_x = roi_x + roi_info['XOffset']
        full_y = roi_y + roi_info['YOffset']
        return full_x, full_y
    
    def detect_golf_ball_enhanced(self, img: np.ndarray, 
                                 prev_detection: Optional[Tuple] = None,
                                 camera_id: str = 'cam1') -> Optional[Tuple]:
        """
        향상된 골프공 검출 (파라미터 최적화)
        
        Returns:
            (center_x, center_y, radius) 또는 None
        """
        params = self.detection_params
        
        # 1. 밝은 영역 검출 (적응형 임계값)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 적응형 임계값 (더 넓은 윈도우)
        _, binary = cv2.threshold(blurred, params['brightness_threshold'], 255, cv2.THRESH_BINARY)
        
        # 2. 형태학적 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        candidates = []
        
        # 3. Hough Circle Detection (파라미터 완화)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=params['hough_param1'],
            param2=params['hough_param2'],
            minRadius=params['min_radius'],
            maxRadius=params['max_radius']
        )
        
        if circles is not None:
            circles = circles[0]
            for circle in circles:
                x, y, r = circle
                
                # 원형도 체크 (완화된 기준)
                mask = np.zeros_like(gray)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
                overlap = cv2.bitwise_and(binary, mask)
                circularity = np.sum(overlap > 0) / (np.pi * r * r) if r > 0 else 0
                
                if circularity >= params['min_circularity']:
                    # 평균 밝기 체크
                    roi_mask = np.zeros_like(gray)
                    cv2.circle(roi_mask, (int(x), int(y)), int(r), 255, -1)
                    mean_brightness = cv2.mean(gray, mask=roi_mask)[0]
                    
                    score = circularity * (mean_brightness / 255.0)
                    candidates.append((x, y, r, score))
        
        # 4. Contour 기반 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:  # 최소 면적 (완화)
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity >= params['min_circularity']:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if params['min_radius'] <= radius <= params['max_radius']:
                    mean_brightness = cv2.mean(gray, mask=binary)[0]
                    score = circularity * (mean_brightness / 255.0)
                    candidates.append((x, y, radius, score))
        
        if not candidates:
            return None
        
        # 5. 시간적 연속성 기반 선택
        if prev_detection is not None:
            prev_x, prev_y = prev_detection[0], prev_detection[1]
            
            # 이전 검출과 가까운 후보에 보너스
            for i, (x, y, r, score) in enumerate(candidates):
                dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if dist < params['max_position_change']:
                    temporal_bonus = 1.0 - (dist / params['max_position_change'])
                    candidates[i] = (x, y, r, score + temporal_bonus)
        
        # 최고 점수 후보 선택
        candidates.sort(key=lambda x: x[3], reverse=True)
        best = candidates[0]
        
        return (best[0], best[1], best[2])
    
    def init_kalman_filter(self):
        """칼만 필터 초기화 (튜닝된 파라미터)"""
        kf = SimpleKalmanFilter(
            dim_x=6,  # 상태: [x, y, z, vx, vy, vz]
            dim_z=3,  # 관측: [x, y, z]
            process_noise=0.1,    # 프로세스 노이즈 (속도 변화 작음)
            measurement_noise=50  # 측정 노이즈 (검출 불안정성)
        )
        
        # 전이 행렬 (등속 운동 모델)
        dt = self.frame_interval
        kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 관측 행렬 (위치만 관측)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        return kf
    
    def calculate_3d_position_hybrid(self, center1: Tuple[float, float],
                                    center2: Tuple[float, float]) -> Optional[Dict]:
        """
        하이브리드 3D 위치 계산 (X, Y 시차 모두 고려)
        """
        # ROI 좌표를 전체 이미지 좌표로 변환
        x1_full, y1_full = self.roi_to_full_coords(center1[0], center1[1], self.roi_cam1)
        x2_full, y2_full = self.roi_to_full_coords(center2[0], center2[1], self.roi_cam2)
        
        # 이미지 중심점 (주점)
        cx1, cy1 = self.K1[0, 2], self.K1[1, 2]
        cx2, cy2 = self.K2[0, 2], self.K2[1, 2]
        
        # 정규화 좌표
        x1_norm = (x1_full - cx1) / self.focal_length
        y1_norm = (y1_full - cy1) / self.focal_length
        x2_norm = (x2_full - cx2) / self.focal_length
        y2_norm = (y2_full - cy2) / self.focal_length
        
        # X, Y 시차 계산
        disparity_x = abs(x1_norm - x2_norm)
        disparity_y = abs(y1_norm - y2_norm)
        
        # 수직 스테레오: Y 시차가 주요 (카메라가 수직 배치)
        # 하지만 X 시차도 발생할 수 있음 (정렬 불완전)
        
        # 더 큰 시차 선택 (더 신뢰성 있음)
        if disparity_y > disparity_x:
            disparity = disparity_y
            disparity_type = "vertical"
        else:
            disparity = disparity_x
            disparity_type = "horizontal"
        
        if disparity < 0.001:  # 너무 작은 시차
            return None
        
        # 깊이 계산 (보정 계수 적용)
        Z = (self.baseline_mm / disparity) * self.depth_scale_factor
        
        # 깊이 범위 체크 (100mm ~ 5000mm)
        if Z < 100 or Z > 5000:
            return None
        
        # X, Y 계산 (평균 정규화 좌표 사용)
        x_norm_avg = (x1_norm + x2_norm) / 2
        y_norm_avg = (y1_norm + y2_norm) / 2
        
        X = x_norm_avg * Z
        Y = y_norm_avg * Z
        
        position = np.array([X, Y, Z])
        
        return {
            'position': position,
            'disparity': float(disparity),
            'disparity_type': disparity_type,
            'disparity_x': float(disparity_x),
            'disparity_y': float(disparity_y)
        }
    
    def calculate_physics_parameters_improved(self, positions: List[np.ndarray], 
                                             frame_numbers: List[int]) -> Dict:
        """
        개선된 물리량 계산
        
        좌표계 정의:
        - X: 수평 방향 (좌우, 카메라 센서 가로축)
        - Y: 수직 방향 (위아래, 카메라 센서 세로축)
        - Z: 깊이 방향 (전후, 타겟 방향)
        
        방향각 정의 (골프 표준):
        - 0°: 정면 (타겟 방향, +Z)
        - +각도: 우측 (+X)
        - -각도: 좌측 (-X)
        """
        if len(positions) < 2:
            return {'success': False, 'error': 'Not enough data'}
        
        velocities = []
        for i in range(1, len(positions)):
            dt = (frame_numbers[i] - frame_numbers[i-1]) * self.frame_interval
            if dt > 0:
                dpos = positions[i] - positions[i-1]
                velocity = dpos / dt
                velocities.append(velocity)
        
        if not velocities:
            return {'success': False, 'error': 'Cannot calculate velocity'}
        
        # 평균 속도 (초기 몇 프레임 중요)
        # 골프공은 발사 직후가 최대 속도
        avg_velocity = np.mean(velocities[:3], axis=0) if len(velocities) >= 3 else np.mean(velocities, axis=0)
        vx, vy, vz = avg_velocity
        
        speed_mm_s = np.linalg.norm(avg_velocity)
        speed_m_s = speed_mm_s / 1000.0
        
        # 발사각 계산
        # 수평 성분: sqrt(vx^2 + vz^2)
        # 수직 성분: vy
        horizontal_speed = np.sqrt(vx**2 + vz**2)
        if horizontal_speed > 0:
            launch_angle = math.degrees(math.atan2(vy, horizontal_speed))
        else:
            launch_angle = 90.0 if vy > 0 else -90.0
        
        # 방향각 계산 개선
        # atan2(X, Z): X가 양수면 양각(우측), Z가 양수면 전방
        # 주의: 카메라 좌표계와 골프 좌표계의 관계 확인 필요
        
        # 방법 1: 표준 골프 좌표계 (타겟이 +Z 방향)
        direction_angle = math.degrees(math.atan2(vx, vz))
        
        # 방법 2: 카메라 정렬 보정 (필요 시)
        # 만약 카메라가 90도 회전되어 있다면:
        # direction_angle = math.degrees(math.atan2(vz, vx))
        
        # 방법 3: 절대값 기준 (방향 확인)
        # 실측 데이터와 비교하여 부호 확인
        
        return {
            'success': True,
            'velocity': {
                'vx': float(vx), 
                'vy': float(vy), 
                'vz': float(vz), 
                'unit': 'mm/s',
                'horizontal': float(horizontal_speed)
            },
            'speed': {
                'mm_s': float(speed_mm_s), 
                'm_s': float(speed_m_s), 
                'km_h': float(speed_m_s * 3.6)
            },
            'launch_angle': {
                'degrees': float(launch_angle), 
                'radians': float(math.radians(launch_angle))
            },
            'direction_angle': {
                'degrees': float(direction_angle), 
                'radians': float(math.radians(direction_angle))
            }
        }
    
    def analyze_shot_enhanced(self, shot_dir: str, shot_number: int) -> Dict:
        """
        향상된 샷 분석
        """
        print(f"\n{'='*60}")
        print(f"Analyzing Shot {shot_number} (Enhanced)")
        print(f"{'='*60}")
        
        cam1_pattern = os.path.join(shot_dir, "Cam1_*.bmp")
        cam2_pattern = os.path.join(shot_dir, "Cam2_*.bmp")
        
        cam1_files = sorted(glob.glob(cam1_pattern))
        cam2_files = sorted(glob.glob(cam2_pattern))
        
        print(f"Found {len(cam1_files)} Cam1, {len(cam2_files)} Cam2 images")
        
        # 칼만 필터 초기화
        kf = None
        kalman_initialized = False
        
        positions_3d = []
        positions_3d_filtered = []
        frame_numbers = []
        detections = []
        
        prev_det1 = None
        prev_det2 = None
        
        min_frames = min(len(cam1_files), len(cam2_files))
        
        for i in range(min_frames):
            frame_num = int(os.path.basename(cam1_files[i]).split('_')[1].split('.')[0])
            
            img1 = cv2.imread(cam1_files[i])
            img2 = cv2.imread(cam2_files[i])
            
            if img1 is None or img2 is None:
                continue
            
            # 향상된 검출
            det1 = self.detect_golf_ball_enhanced(img1, prev_det1, 'cam1')
            det2 = self.detect_golf_ball_enhanced(img2, prev_det2, 'cam2')
            
            if det1 is None or det2 is None:
                print(f"  Frame {frame_num:2d}: Detection failed")
                continue
            
            center1 = (det1[0], det1[1])
            center2 = (det2[0], det2[1])
            
            prev_det1 = det1
            prev_det2 = det2
            
            # 하이브리드 3D 계산
            pos_3d_info = self.calculate_3d_position_hybrid(center1, center2)
            
            if pos_3d_info is None:
                print(f"  Frame {frame_num:2d}: 3D calculation failed")
                continue
            
            pos_3d = pos_3d_info['position']
            
            # 칼만 필터 적용
            if not kalman_initialized:
                kf = self.init_kalman_filter()
                kf.x = np.array([[pos_3d[0]], [pos_3d[1]], [pos_3d[2]], [0], [0], [0]])
                kalman_initialized = True
                pos_3d_filtered = pos_3d
            else:
                kf.predict()
                kf.update(pos_3d)
                pos_3d_filtered = kf.x[:3].flatten()
            
            positions_3d.append(pos_3d)
            positions_3d_filtered.append(pos_3d_filtered)
            frame_numbers.append(frame_num)
            
            detections.append({
                'frame': frame_num,
                'cam1': {'x': det1[0], 'y': det1[1], 'r': det1[2]},
                'cam2': {'x': det2[0], 'y': det2[1], 'r': det2[2]},
                'pos_3d_raw': pos_3d.tolist(),
                'pos_3d_filtered': pos_3d_filtered.tolist(),
                'disparity_type': pos_3d_info['disparity_type'],
                'disparity': pos_3d_info['disparity']
            })
            
            print(f"  Frame {frame_num:2d}: "
                  f"Cam1=({det1[0]:.1f},{det1[1]:.1f}) "
                  f"Cam2=({det2[0]:.1f},{det2[1]:.1f}) "
                  f"3D=({pos_3d[0]:.1f},{pos_3d[1]:.1f},{pos_3d[2]:.1f}mm) "
                  f"Filtered=({pos_3d_filtered[0]:.1f},{pos_3d_filtered[1]:.1f},{pos_3d_filtered[2]:.1f}mm) "
                  f"{pos_3d_info['disparity_type']}")
        
        print(f"\nTracked {len(positions_3d)} frames")
        
        # 물리량 계산 (필터링된 위치 사용)
        if len(positions_3d_filtered) >= 2:
            physics = self.calculate_physics_parameters_improved(positions_3d_filtered, frame_numbers)
        else:
            physics = {'success': False, 'error': 'Not enough tracking data'}
        
        result = {
            'shot_number': shot_number,
            'shot_dir': shot_dir,
            'total_frames': min_frames,
            'tracked_frames': len(positions_3d),
            'detections': detections,
            'physics': physics,
            'real_data': self.real_data.get(shot_number, None)
        }
        
        # 결과 출력
        if physics['success']:
            print(f"\n{'='*60}")
            print(f"SHOT {shot_number} CALCULATED RESULTS")
            print(f"{'='*60}")
            print(f"Speed: {physics['speed']['m_s']:.2f} m/s")
            print(f"Launch Angle: {physics['launch_angle']['degrees']:.2f}°")
            print(f"Direction Angle: {physics['direction_angle']['degrees']:.2f}°")
            print(f"Velocity: vx={physics['velocity']['vx']:.1f}, "
                  f"vy={physics['velocity']['vy']:.1f}, "
                  f"vz={physics['velocity']['vz']:.1f} mm/s")
            
            if shot_number in self.real_data:
                real = self.real_data[shot_number]
                print(f"\nREAL MEASUREMENT")
                print(f"Speed: {real['ball_speed_ms']:.2f} m/s")
                print(f"Launch Angle: {real['launch_angle_deg']:.2f}°")
                print(f"Direction Angle: {real['launch_direction_deg']:.2f}°")
                
                speed_error = abs(physics['speed']['m_s'] - real['ball_speed_ms'])
                speed_error_pct = (speed_error / real['ball_speed_ms']) * 100
                launch_error = abs(physics['launch_angle']['degrees'] - real['launch_angle_deg'])
                direction_error = abs(physics['direction_angle']['degrees'] - real['launch_direction_deg'])
                
                print(f"\nERRORS")
                print(f"Speed error: {speed_error:.2f} m/s ({speed_error_pct:.1f}%)")
                print(f"Launch angle error: {launch_error:.2f}°")
                print(f"Direction angle error: {direction_error:.2f}°")
            print(f"{'='*60}")
        
        return result

def main():
    """메인 함수"""
    analyzer = EnhancedGolfBall3DAnalyzer()
    
    # 테스트: 샷 1, 4, 17, 20 (최적 샷들)
    test_shots = [1, 4, 17, 20]
    
    results = []
    for shot_num in test_shots:
        shot_dir = f"data2/driver/{shot_num}"
        if os.path.exists(shot_dir):
            result = analyzer.analyze_shot_enhanced(shot_dir, shot_num)
            results.append(result)
    
    # 통계 출력
    print("\n" + "="*80)
    print("TEST SUMMARY (BEST SHOTS)")
    print("="*80)
    
    speed_errors = []
    launch_errors = []
    direction_errors = []
    
    for result in results:
        if result['physics']['success'] and result['shot_number'] in analyzer.real_data:
            real = analyzer.real_data[result['shot_number']]
            calc = result['physics']
            
            speed_error = abs(calc['speed']['m_s'] - real['ball_speed_ms'])
            speed_error_pct = (speed_error / real['ball_speed_ms']) * 100
            launch_error = abs(calc['launch_angle']['degrees'] - real['launch_angle_deg'])
            direction_error = abs(calc['direction_angle']['degrees'] - real['launch_direction_deg'])
            
            speed_errors.append(speed_error_pct)
            launch_errors.append(launch_error)
            direction_errors.append(direction_error)
            
            print(f"Shot {result['shot_number']:2d}: "
                  f"Speed={speed_error_pct:5.1f}%, "
                  f"Launch={launch_error:5.2f}°, "
                  f"Direction={direction_error:6.2f}°")
    
    if speed_errors:
        print("\nAVERAGES:")
        print(f"  Speed Error: {np.mean(speed_errors):.1f}% ± {np.std(speed_errors):.1f}%")
        print(f"  Launch Error: {np.mean(launch_errors):.2f}° ± {np.std(launch_errors):.2f}°")
        print(f"  Direction Error: {np.mean(direction_errors):.2f}° ± {np.std(direction_errors):.2f}°")
    
    # 결과 저장
    output_file = "enhanced_golf_ball_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
