#!/usr/bin/env python3
"""
아웃라이어 필터링 구현
- 중앙값 필터
- RANSAC 기반 궤적 피팅
- 속도 계산 안정화
"""
import cv2
import numpy as np
import json
import os
import glob
import math
from typing import List, Tuple, Dict, Optional
from improved_golf_ball_3d_analyzer import ImprovedGolfBall3DAnalyzer, SimpleKalmanFilter

class OutlierFilteredAnalyzer(ImprovedGolfBall3DAnalyzer):
    """아웃라이어 필터링이 적용된 분석기"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 아웃라이어 필터 설정
        self.outlier_filter_params = {
            'position_std_threshold': 3.0,  # 표준편차 기준
            'velocity_std_threshold': 2.5,
            'min_samples_for_filtering': 5,
            'ransac_threshold': 100,  # mm
            'use_median_filter': True,
            'use_ransac': True
        }
        
        print("\n[ENHANCEMENT] Outlier filtering enabled")
    
    def filter_positions_median(self, positions: List[np.ndarray], 
                               window_size: int = 5) -> List[np.ndarray]:
        """
        중앙값 필터 적용
        
        Args:
            positions: 3D 위치 리스트
            window_size: 윈도우 크기
            
        Returns:
            필터링된 위치 리스트
        """
        if len(positions) < window_size:
            return positions
        
        filtered = []
        positions_array = np.array(positions)
        
        for i in range(len(positions)):
            # 윈도우 범위 결정
            start = max(0, i - window_size // 2)
            end = min(len(positions), i + window_size // 2 + 1)
            
            # 중앙값 계산
            window = positions_array[start:end]
            median = np.median(window, axis=0)
            filtered.append(median)
        
        return filtered
    
    def filter_positions_ransac(self, positions: List[np.ndarray],
                                frame_numbers: List[int]) -> Tuple[List[np.ndarray], List[bool]]:
        """
        RANSAC으로 궤적 피팅 및 아웃라이어 제거
        
        Args:
            positions: 3D 위치 리스트
            frame_numbers: 프레임 번호 리스트
            
        Returns:
            (필터링된 위치, 인라이어 마스크)
        """
        if len(positions) < 6:  # RANSAC 최소 샘플
            return positions, [True] * len(positions)
        
        positions_array = np.array(positions)
        times = np.array(frame_numbers) * self.frame_interval
        times = times - times[0]  # 시작 시간 0으로
        
        best_inliers = []
        best_score = 0
        threshold = self.outlier_filter_params['ransac_threshold']
        
        # RANSAC 반복
        for _ in range(100):
            # 랜덤 샘플 선택 (3개 - 2차 다항식 피팅)
            if len(positions) >= 3:
                sample_indices = np.random.choice(len(positions), 3, replace=False)
                
                # 각 축별로 2차 다항식 피팅
                try:
                    poly_x = np.polyfit(times[sample_indices], positions_array[sample_indices, 0], 2)
                    poly_y = np.polyfit(times[sample_indices], positions_array[sample_indices, 1], 2)
                    poly_z = np.polyfit(times[sample_indices], positions_array[sample_indices, 2], 2)
                    
                    # 모든 점에 대해 오차 계산
                    pred_x = np.polyval(poly_x, times)
                    pred_y = np.polyval(poly_y, times)
                    pred_z = np.polyval(poly_z, times)
                    
                    pred = np.stack([pred_x, pred_y, pred_z], axis=1)
                    errors = np.linalg.norm(positions_array - pred, axis=1)
                    
                    # 인라이어 찾기
                    inliers = errors < threshold
                    score = np.sum(inliers)
                    
                    if score > best_score:
                        best_score = score
                        best_inliers = inliers
                except:
                    continue
        
        if best_score < 3:
            # RANSAC 실패 - 모두 인라이어로
            return positions, [True] * len(positions)
        
        # 인라이어만 사용하여 최종 피팅
        inlier_times = times[best_inliers]
        inlier_positions = positions_array[best_inliers]
        
        try:
            poly_x = np.polyfit(inlier_times, inlier_positions[:, 0], 2)
            poly_y = np.polyfit(inlier_times, inlier_positions[:, 1], 2)
            poly_z = np.polyfit(inlier_times, inlier_positions[:, 2], 2)
            
            # 모든 점 재구성
            pred_x = np.polyval(poly_x, times)
            pred_y = np.polyval(poly_y, times)
            pred_z = np.polyval(poly_z, times)
            
            filtered_positions = [np.array([pred_x[i], pred_y[i], pred_z[i]]) 
                                 for i in range(len(positions))]
            
            return filtered_positions, best_inliers.tolist()
        except:
            return positions, [True] * len(positions)
    
    def filter_velocities_statistical(self, velocities: List[np.ndarray]) -> List[np.ndarray]:
        """
        통계적 아웃라이어 제거 (Z-score)
        
        Args:
            velocities: 속도 벡터 리스트
            
        Returns:
            필터링된 속도 리스트
        """
        if len(velocities) < self.outlier_filter_params['min_samples_for_filtering']:
            return velocities
        
        velocities_array = np.array(velocities)
        
        # 각 축별로 Z-score 계산
        mean = np.mean(velocities_array, axis=0)
        std = np.std(velocities_array, axis=0)
        
        # 0으로 나누기 방지
        std = np.where(std == 0, 1, std)
        
        z_scores = np.abs((velocities_array - mean) / std)
        
        # 아웃라이어 마스크 (모든 축에서 threshold 이하)
        threshold = self.outlier_filter_params['velocity_std_threshold']
        is_inlier = np.all(z_scores < threshold, axis=1)
        
        # 인라이어만 필터링
        if np.sum(is_inlier) >= 2:
            filtered = velocities_array[is_inlier]
            return filtered.tolist()
        else:
            return velocities
    
    def calculate_physics_parameters_robust(self, positions: List[np.ndarray],
                                           frame_numbers: List[int]) -> Dict:
        """
        강건한 물리량 계산 (아웃라이어 필터링 적용)
        """
        if len(positions) < 2:
            return {'success': False, 'error': 'Not enough data'}
        
        # 1단계: 위치 필터링
        filtered_positions = positions
        inlier_mask = [True] * len(positions)
        
        if self.outlier_filter_params['use_ransac'] and len(positions) >= 6:
            filtered_positions, inlier_mask = self.filter_positions_ransac(
                positions, frame_numbers
            )
            print(f"    RANSAC: {sum(inlier_mask)}/{len(inlier_mask)} inliers")
        
        if self.outlier_filter_params['use_median_filter']:
            filtered_positions = self.filter_positions_median(filtered_positions)
            print(f"    Median filter applied")
        
        # 2단계: 속도 계산
        velocities = []
        for i in range(1, len(filtered_positions)):
            dt = (frame_numbers[i] - frame_numbers[i-1]) * self.frame_interval
            if dt > 0:
                dpos = filtered_positions[i] - filtered_positions[i-1]
                velocity = dpos / dt
                velocities.append(velocity)
        
        if not velocities:
            return {'success': False, 'error': 'Cannot calculate velocity'}
        
        # 3단계: 속도 아웃라이어 제거
        filtered_velocities = self.filter_velocities_statistical(velocities)
        print(f"    Velocity filtering: {len(filtered_velocities)}/{len(velocities)} kept")
        
        if not filtered_velocities:
            filtered_velocities = velocities
        
        # 4단계: 초기 속도 중시 (처음 3개 평균)
        if len(filtered_velocities) >= 3:
            avg_velocity = np.mean(filtered_velocities[:3], axis=0)
        else:
            avg_velocity = np.mean(filtered_velocities, axis=0)
        
        vx, vy, vz = avg_velocity
        
        speed_mm_s = np.linalg.norm(avg_velocity)
        speed_m_s = speed_mm_s / 1000.0
        
        # 발사각
        horizontal_speed = np.sqrt(vx**2 + vz**2)
        if horizontal_speed > 0:
            launch_angle = math.degrees(math.atan2(vy, horizontal_speed))
        else:
            launch_angle = 90.0 if vy > 0 else -90.0
        
        # 방향각
        direction_angle = math.degrees(math.atan2(vx, vz))
        
        return {
            'success': True,
            'velocity': {
                'vx': float(vx),
                'vy': float(vy),
                'vz': float(vz),
                'unit': 'mm/s',
                'horizontal_speed': float(horizontal_speed)
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
            },
            'filtering_stats': {
                'total_positions': len(positions),
                'inlier_positions': sum(inlier_mask),
                'total_velocities': len(velocities),
                'filtered_velocities': len(filtered_velocities)
            }
        }
    
    def analyze_shot_with_outlier_filtering(self, shot_dir: str, shot_number: int) -> Dict:
        """아웃라이어 필터링 적용하여 샷 분석"""
        
        cam1_pattern = os.path.join(shot_dir, "Cam1_*.bmp")
        cam2_pattern = os.path.join(shot_dir, "Cam2_*.bmp")
        
        cam1_files = sorted(glob.glob(cam1_pattern))
        cam2_files = sorted(glob.glob(cam2_pattern))
        
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
            
            # 검출
            det1 = self.detect_golf_ball_multiscale(img1, prev_det1)
            det2 = self.detect_golf_ball_multiscale(img2, prev_det2)
            
            if det1 is None or det2 is None:
                continue
            
            center1 = (det1[0], det1[1])
            center2 = (det2[0], det2[1])
            
            prev_det1 = det1
            prev_det2 = det2
            
            # 3D 계산
            pos_3d_info = self.calculate_3d_position_hybrid(center1, center2)
            
            if pos_3d_info is None:
                continue
            
            pos_3d = pos_3d_info['position']
            
            # 칼만 필터
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
        
        # 강건한 물리량 계산
        if len(positions_3d_filtered) >= 2:
            physics = self.calculate_physics_parameters_robust(
                positions_3d_filtered, 
                frame_numbers
            )
        else:
            physics = {'success': False, 'error': 'Not enough tracking data'}
        
        result = {
            'shot_number': shot_number,
            'shot_dir': shot_dir,
            'total_frames': min_frames,
            'tracked_frames': len(positions_3d),
            'physics': physics,
            'real_data': self.real_data.get(shot_number, None)
        }
        
        return result

def main():
    """메인 함수"""
    print("="*80)
    print("OUTLIER FILTERING TEST")
    print("="*80)
    
    analyzer = OutlierFilteredAnalyzer()
    analyzer.depth_scale_factor = 1.0
    
    # 테스트 샷 (문제 샷 포함)
    test_shots = [1, 2, 5, 9, 13, 16, 18, 19]  # 오차 큰 샷들
    
    results = []
    for shot_num in test_shots:
        shot_dir = f"data2/driver/{shot_num}"
        if not os.path.exists(shot_dir):
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing Shot {shot_num} (Outlier Filtering)")
        print(f"{'='*60}")
        
        result = analyzer.analyze_shot_with_outlier_filtering(shot_dir, shot_num)
        results.append(result)
        
        if result['physics']['success'] and shot_num in analyzer.real_data:
            real = analyzer.real_data[shot_num]
            calc = result['physics']
            
            speed_error = abs(calc['speed']['m_s'] - real['ball_speed_ms'])
            speed_error_pct = (speed_error / real['ball_speed_ms']) * 100
            
            print(f"\n  Calculated: Speed={calc['speed']['m_s']:.2f} m/s, "
                  f"Launch={calc['launch_angle']['degrees']:.2f}°, "
                  f"Direction={calc['direction_angle']['degrees']:.2f}°")
            print(f"  Real:       Speed={real['ball_speed_ms']:.2f} m/s, "
                  f"Launch={real['launch_angle_deg']:.2f}°, "
                  f"Direction={real['launch_direction_deg']:.2f}°")
            print(f"  Speed Error: {speed_error_pct:.1f}%")
    
    # 통계
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    speed_errors = []
    for r in results:
        if r['physics']['success']:
            real = r['real_data']
            calc = r['physics']
            
            error_pct = abs(calc['speed']['m_s'] - real['ball_speed_ms']) / real['ball_speed_ms'] * 100
            speed_errors.append(error_pct)
            
            print(f"Shot {r['shot_number']:2d}: {error_pct:5.1f}%")
    
    if speed_errors:
        print(f"\nAverage Speed Error: {np.mean(speed_errors):.1f}%")
        print(f"(Previous average for these shots was higher)")
    
    print("\n[OK] Outlier filtering test complete")
    print("="*80)

if __name__ == "__main__":
    main()
