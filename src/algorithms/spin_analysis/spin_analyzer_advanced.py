#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고급 820fps 볼 스핀 분석 시스템

실제 820fps 볼 이미지를 활용한 정밀 스핀 측정:
1. 볼 표면 패턴 추적을 통한 백스핀 측정
2. 볼의 그림자 변화를 통한 사이드스핀 측정  
3. 볼 형태 변화를 통한 스핀축 계산

물리적 기반 알고리즘으로 95% 정확도 달성 목표
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math
from scipy import signal
from scipy.optimize import minimize
import threading
from concurrent.futures import ThreadPoolExecutor


@dataclass
class Ball820fpsFrame:
    """820fps 볼 프레임 데이터"""
    image: np.ndarray         # 볼 이미지
    timestamp: float          # 타임스탬프 (ms)
    frame_number: int         # 프레임 번호
    ball_center: Tuple[int, int]  # 볼 중심 좌표
    ball_radius: float        # 볼 반지름 (픽셀)
    lighting_angle: float     # 조명 각도


@dataclass
class SpinMeasurement:
    """스핀 측정 결과"""
    backspin: float           # RPM
    sidespin: float          # RPM  
    spin_axis: float         # 도(°)
    total_spin: float        # 총 스핀 RPM
    confidence: float        # 신뢰도 (0-1)
    method_used: str         # 사용된 측정 방법
    frame_analysis_count: int # 분석에 사용된 프레임 수


class OpticalFlow820fpsTracker:
    """820fps 전용 광학 흐름 추적기"""
    
    def __init__(self):
        # Lucas-Kanade 광학 흐름 매개변수 (820fps 최적화)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 특징점 검출 매개변수
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
    def track_surface_features(self, frame1: np.ndarray, frame2: np.ndarray, 
                              ball_mask: np.ndarray) -> Optional[np.ndarray]:
        """볼 표면 특징점 추적"""
        # 그레이스케일 변환
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 볼 영역에서 특징점 검출
        corners = cv2.goodFeaturesToTrack(gray1, mask=ball_mask, **self.feature_params)
        
        if corners is None or len(corners) < 10:
            return None
            
        # 광학 흐름으로 특징점 추적
        new_corners, status, error = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, corners, None, **self.lk_params
        )
        
        # 양질의 추적 결과만 선택
        good_corners = corners[status.flatten() == 1]
        good_new_corners = new_corners[status.flatten() == 1]
        
        if len(good_corners) < 8:
            return None
            
        # 움직임 벡터 계산
        motion_vectors = good_new_corners - good_corners
        
        return motion_vectors


class ShadowAnalyzer820fps:
    """820fps 볼 그림자 분석기 (사이드스핀 전용)"""
    
    def __init__(self):
        self.shadow_history = []
        self.light_direction = np.array([0, -1, 1])  # 조명 방향 벡터
        
    def analyze_shadow_movement(self, frames: List[Ball820fpsFrame]) -> float:
        """그림자 움직임을 통한 사이드스핀 계산"""
        if len(frames) < 5:
            return 0.0
            
        shadow_positions = []
        
        for frame in frames:
            shadow_pos = self._detect_shadow_center(frame.image, frame.ball_center, frame.ball_radius)
            if shadow_pos:
                shadow_positions.append(shadow_pos)
        
        if len(shadow_positions) < 3:
            return 0.0
            
        # 그림자 중심의 움직임 분석
        shadow_motion = self._calculate_shadow_motion(shadow_positions, frames)
        
        # 사이드스핀 RPM으로 변환
        sidespin_rpm = self._convert_shadow_to_sidespin(shadow_motion, frames[0].ball_radius)
        
        return sidespin_rpm
    
    def _detect_shadow_center(self, image: np.ndarray, ball_center: Tuple[int, int], 
                            ball_radius: float) -> Optional[Tuple[float, float]]:
        """볼 그림자 중심점 검출"""
        # HSV 색상 공간으로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 그림자 영역 검출 (낮은 명도 영역)
        shadow_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 100))
        
        # 볼 영역 내부만 고려
        ball_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(ball_mask, ball_center, int(ball_radius), 255, -1)
        
        shadow_in_ball = cv2.bitwise_and(shadow_mask, ball_mask)
        
        # 그림자 중심 계산
        moments = cv2.moments(shadow_in_ball)
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            return (cx, cy)
        
        return None
    
    def _calculate_shadow_motion(self, shadow_positions: List[Tuple[float, float]], 
                               frames: List[Ball820fpsFrame]) -> np.ndarray:
        """그림자 움직임 벡터 계산"""
        if len(shadow_positions) < 2:
            return np.array([0, 0])
            
        # 연속 프레임 간 그림자 이동 벡터 계산
        motion_vectors = []
        for i in range(1, len(shadow_positions)):
            dx = shadow_positions[i][0] - shadow_positions[i-1][0]
            dy = shadow_positions[i][1] - shadow_positions[i-1][1]
            motion_vectors.append([dx, dy])
        
        # 평균 움직임 벡터
        avg_motion = np.mean(motion_vectors, axis=0)
        
        return avg_motion
    
    def _convert_shadow_to_sidespin(self, shadow_motion: np.ndarray, ball_radius: float) -> float:
        """그림자 움직임을 사이드스핀 RPM으로 변환"""
        # 그림자 움직임의 수평 성분 (사이드스핀과 관련)
        horizontal_motion = shadow_motion[0]  # 픽셀/프레임
        
        # 820fps 기준 각속도 계산
        frame_interval = 1.0 / 820  # 1.22ms
        angular_velocity = horizontal_motion / ball_radius / frame_interval  # rad/s
        
        # RPM으로 변환
        sidespin_rpm = angular_velocity * 60 / (2 * np.pi)
        
        # 물리적 제한 적용
        sidespin_rpm = np.clip(sidespin_rpm, -4000, 4000)
        
        return sidespin_rpm


class BallShapeAnalyzer820fps:
    """820fps 볼 형태 분석기 (백스핀 + 스핀축)"""
    
    def __init__(self):
        self.shape_history = []
        
    def analyze_ball_deformation(self, frames: List[Ball820fpsFrame]) -> Tuple[float, float]:
        """볼 형태 변화를 통한 백스핀 및 스핀축 분석"""
        if len(frames) < 8:
            return 0.0, 0.0
            
        # 각 프레임에서 볼의 형태 특성 추출
        shape_features = []
        for frame in frames:
            features = self._extract_shape_features(frame)
            if features:
                shape_features.append(features)
        
        if len(shape_features) < 5:
            return 0.0, 0.0
            
        # 형태 변화 패턴 분석
        backspin = self._calculate_backspin_from_shape(shape_features, frames)
        spin_axis = self._calculate_spin_axis_from_shape(shape_features, frames)
        
        return backspin, spin_axis
    
    def _extract_shape_features(self, frame: Ball820fpsFrame) -> Optional[Dict]:
        """볼 형태 특성 추출"""
        image = frame.image
        center = frame.ball_center
        radius = frame.ball_radius
        
        # 볼 영역 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, int(radius), 255, -1)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 볼 윤곽선 검출
        masked_gray = cv2.bitwise_and(gray, mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        contour = max(contours, key=cv2.contourArea)
        
        # 형태 특성 계산
        # 1. 타원 피팅
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_ratio = ellipse[1][1] / ellipse[1][0]  # 장축/단축 비율
            ellipse_angle = ellipse[2]  # 타원 각도
        else:
            ellipse_ratio = 1.0
            ellipse_angle = 0.0
        
        # 2. 중심으로부터 거리 분포
        center_distances = []
        for point in contour.reshape(-1, 2):
            dist = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
            center_distances.append(dist)
        
        distance_std = np.std(center_distances)
        
        # 3. 밝기 분포 (볼 표면 패턴)
        brightness_profile = self._calculate_brightness_profile(masked_gray, center, radius)
        
        return {
            'ellipse_ratio': ellipse_ratio,
            'ellipse_angle': ellipse_angle,
            'distance_std': distance_std,
            'brightness_profile': brightness_profile,
            'timestamp': frame.timestamp
        }
    
    def _calculate_brightness_profile(self, gray_image: np.ndarray, 
                                    center: Tuple[int, int], radius: float) -> np.ndarray:
        """볼 중심으로부터의 방사형 밝기 프로파일 계산"""
        profile = []
        num_rays = 36  # 10도 간격
        
        for angle in range(0, 360, 10):
            rad = np.radians(angle)
            end_x = int(center[0] + radius * 0.8 * np.cos(rad))
            end_y = int(center[1] + radius * 0.8 * np.sin(rad))
            
            # 중심에서 가장자리까지의 밝기 값들
            line_coords = self._get_line_coordinates(center, (end_x, end_y))
            brightness_values = []
            
            for x, y in line_coords:
                if 0 <= x < gray_image.shape[1] and 0 <= y < gray_image.shape[0]:
                    brightness_values.append(gray_image[y, x])
            
            if brightness_values:
                profile.append(np.mean(brightness_values))
        
        return np.array(profile)
    
    def _get_line_coordinates(self, start: Tuple[int, int], 
                            end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """두 점 사이의 선 좌표들 계산 (브레젠함 알고리즘)"""
        x1, y1 = start
        x2, y2 = end
        
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        x, y = x1, y1
        x_inc = 1 if x1 < x2 else -1
        y_inc = 1 if y1 < y2 else -1
        
        error = dx - dy
        
        while True:
            points.append((x, y))
            
            if x == x2 and y == y2:
                break
                
            error2 = 2 * error
            
            if error2 > -dy:
                error -= dy
                x += x_inc
                
            if error2 < dx:
                error += dx
                y += y_inc
        
        return points
    
    def _calculate_backspin_from_shape(self, shape_features: List[Dict], 
                                     frames: List[Ball820fpsFrame]) -> float:
        """형태 변화로부터 백스핀 계산"""
        # 타원 비율의 변화 패턴 분석 (백스핀으로 인한 볼 변형)
        ellipse_ratios = [f['ellipse_ratio'] for f in shape_features]
        
        # 시간에 따른 비율 변화의 주기성 분석
        time_intervals = [frames[i].timestamp - frames[0].timestamp for i in range(len(frames))]
        
        # FFT를 사용한 주기 분석
        if len(ellipse_ratios) >= 8:
            # 신호 전처리
            detrended = signal.detrend(ellipse_ratios)
            windowed = detrended * signal.windows.hann(len(detrended))
            
            # FFT 계산
            fft = np.fft.rfft(windowed)
            freqs = np.fft.rfftfreq(len(windowed), d=(time_intervals[1] - time_intervals[0])/1000)
            
            # 피크 주파수 찾기
            peak_idx = np.argmax(np.abs(fft[1:]))  # DC 성분 제외
            peak_freq = freqs[peak_idx + 1]  # Hz
            
            # 백스핀 RPM으로 변환
            backspin_rpm = peak_freq * 60
            
            # 물리적 제한 적용
            backspin_rpm = np.clip(backspin_rpm, 0, 15000)
            
            return backspin_rpm
        
        return 0.0
    
    def _calculate_spin_axis_from_shape(self, shape_features: List[Dict], 
                                      frames: List[Ball820fpsFrame]) -> float:
        """형태 변화로부터 스핀축 계산"""
        # 타원 각도의 변화 패턴 분석
        ellipse_angles = [f['ellipse_angle'] for f in shape_features]
        
        # 각도 변화의 방향성 분석
        angle_changes = []
        for i in range(1, len(ellipse_angles)):
            angle_diff = ellipse_angles[i] - ellipse_angles[i-1]
            
            # 각도 차이 정규화 (-180 ~ 180도)
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
                
            angle_changes.append(angle_diff)
        
        if angle_changes:
            # 평균 각도 변화로 스핀축 추정
            mean_angle_change = np.mean(angle_changes)
            
            # 스핀축 각도로 변환 (도)
            spin_axis = mean_angle_change * 10  # 경험적 스케일 팩터
            
            # 유효 범위 제한
            spin_axis = np.clip(spin_axis, -45, 45)
            
            return spin_axis
        
        return 0.0


class IntegratedSpinAnalyzer820fps:
    """통합 820fps 스핀 분석기"""
    
    def __init__(self):
        self.optical_tracker = OpticalFlow820fpsTracker()
        self.shadow_analyzer = ShadowAnalyzer820fps()
        self.shape_analyzer = BallShapeAnalyzer820fps()
        
        # 분석 결과 가중치
        self.method_weights = {
            'optical_flow_backspin': 0.4,
            'shape_backspin': 0.6,
            'shadow_sidespin': 0.7,
            'optical_sidespin': 0.3,
            'shape_spin_axis': 1.0
        }
        
    def analyze_ball_spin(self, frames: List[Ball820fpsFrame]) -> SpinMeasurement:
        """종합적인 볼 스핀 분석"""
        if len(frames) < 5:
            return SpinMeasurement(0, 0, 0, 0, 0, "insufficient_data", len(frames))
        
        start_time = time.perf_counter()
        
        # 1. 광학 흐름 기반 분석
        optical_results = self._analyze_with_optical_flow(frames)
        
        # 2. 그림자 기반 사이드스핀 분석
        shadow_sidespin = self.shadow_analyzer.analyze_shadow_movement(frames)
        
        # 3. 볼 형태 기반 백스핀 및 스핀축 분석
        shape_backspin, shape_spin_axis = self.shape_analyzer.analyze_ball_deformation(frames)
        
        # 4. 결과 통합 및 가중 평균
        final_backspin = self._integrate_backspin_measurements(
            optical_results.get('backspin', 0), shape_backspin
        )
        
        final_sidespin = self._integrate_sidespin_measurements(
            optical_results.get('sidespin', 0), shadow_sidespin
        )
        
        final_spin_axis = shape_spin_axis  # 형태 분석 결과 사용
        
        # 5. 총 스핀 계산
        total_spin = np.sqrt(final_backspin**2 + final_sidespin**2)
        
        # 6. 신뢰도 계산
        confidence = self._calculate_measurement_confidence(
            optical_results, shadow_sidespin, shape_backspin, shape_spin_axis
        )
        
        # 7. 처리 시간 계산
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return SpinMeasurement(
            backspin=final_backspin,
            sidespin=final_sidespin,
            spin_axis=final_spin_axis,
            total_spin=total_spin,
            confidence=confidence,
            method_used="integrated_820fps",
            frame_analysis_count=len(frames)
        )
    
    def _analyze_with_optical_flow(self, frames: List[Ball820fpsFrame]) -> Dict:
        """광학 흐름 기반 스핀 분석"""
        if len(frames) < 2:
            return {}
        
        motion_data = []
        
        for i in range(1, len(frames)):
            frame1 = frames[i-1]
            frame2 = frames[i]
            
            # 볼 마스크 생성
            mask = np.zeros(frame1.image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, frame1.ball_center, int(frame1.ball_radius), 255, -1)
            
            # 광학 흐름 추적
            motion_vectors = self.optical_tracker.track_surface_features(
                frame1.image, frame2.image, mask
            )
            
            if motion_vectors is not None:
                motion_data.append(motion_vectors)
        
        if not motion_data:
            return {}
        
        # 움직임 벡터들로부터 스핀 계산
        backspin, sidespin = self._calculate_spin_from_motion(motion_data, frames[0].ball_radius)
        
        return {'backspin': backspin, 'sidespin': sidespin}
    
    def _calculate_spin_from_motion(self, motion_data: List[np.ndarray], 
                                   ball_radius: float) -> Tuple[float, float]:
        """움직임 벡터로부터 백스핀/사이드스핀 계산"""
        if not motion_data:
            return 0.0, 0.0
        
        # 모든 움직임 벡터 수집
        all_vectors = np.vstack(motion_data)
        
        # 중심에서의 평균 움직임 (전체 볼의 움직임)
        global_motion = np.mean(all_vectors, axis=0)
        
        # 전체 움직임을 제거한 상대적 움직임 (회전으로 인한 움직임)
        relative_motions = all_vectors - global_motion
        
        # 수직 성분 (백스핀과 관련)
        vertical_motion = np.mean(relative_motions[:, 1])
        
        # 수평 성분 (사이드스핀과 관련)
        horizontal_motion = np.mean(relative_motions[:, 0])
        
        # 820fps 기준 각속도로 변환
        frame_interval = 1.0 / 820  # 1.22ms
        
        # RPM 계산
        backspin_rpm = abs(vertical_motion) / ball_radius / frame_interval * 60 / (2 * np.pi)
        sidespin_rpm = horizontal_motion / ball_radius / frame_interval * 60 / (2 * np.pi)
        
        # 물리적 제한 적용
        backspin_rpm = np.clip(backspin_rpm, 0, 15000)
        sidespin_rpm = np.clip(sidespin_rpm, -4000, 4000)
        
        return backspin_rpm, sidespin_rpm
    
    def _integrate_backspin_measurements(self, optical_backspin: float, 
                                       shape_backspin: float) -> float:
        """백스핀 측정 결과 통합"""
        w1 = self.method_weights['optical_flow_backspin']
        w2 = self.method_weights['shape_backspin']
        
        weighted_backspin = (optical_backspin * w1 + shape_backspin * w2) / (w1 + w2)
        
        return weighted_backspin
    
    def _integrate_sidespin_measurements(self, optical_sidespin: float, 
                                       shadow_sidespin: float) -> float:
        """사이드스핀 측정 결과 통합"""
        w1 = self.method_weights['optical_sidespin']
        w2 = self.method_weights['shadow_sidespin']
        
        weighted_sidespin = (optical_sidespin * w1 + shadow_sidespin * w2) / (w1 + w2)
        
        return weighted_sidespin
    
    def _calculate_measurement_confidence(self, optical_results: Dict, 
                                        shadow_sidespin: float,
                                        shape_backspin: float, 
                                        shape_spin_axis: float) -> float:
        """측정 신뢰도 계산"""
        confidence_factors = []
        
        # 1. 측정 방법들 간의 일치도
        if 'backspin' in optical_results and shape_backspin > 0:
            backspin_consistency = 1.0 - abs(optical_results['backspin'] - shape_backspin) / max(optical_results['backspin'], shape_backspin, 1)
            confidence_factors.append(max(0, backspin_consistency))
        
        # 2. 사이드스핀 신뢰도
        if 'sidespin' in optical_results and abs(shadow_sidespin) > 10:
            sidespin_consistency = 1.0 - abs(optical_results['sidespin'] - shadow_sidespin) / max(abs(optical_results['sidespin']), abs(shadow_sidespin), 1)
            confidence_factors.append(max(0, sidespin_consistency))
        
        # 3. 물리적 합리성 검사
        total_spin = np.sqrt(shape_backspin**2 + shadow_sidespin**2)
        if 100 <= total_spin <= 12000:  # 합리적인 스핀 범위
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # 4. 스핀축 합리성
        if -45 <= shape_spin_axis <= 45:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.2)
        
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.1  # 최소 신뢰도


# 테스트 및 검증 함수
def test_integrated_spin_analyzer():
    """통합 스핀 분석기 테스트"""
    analyzer = IntegratedSpinAnalyzer820fps()
    
    # 가상 볼 프레임 생성 (테스트용)
    test_frames = []
    for i in range(10):
        # 가상 볼 이미지 생성
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        center = (100 + i, 100)
        cv2.circle(image, center, 40, (255, 255, 255), -1)
        
        # 볼 프레임 객체 생성
        frame = Ball820fpsFrame(
            image=image,
            timestamp=i * 1.22,  # 820fps 간격
            frame_number=i,
            ball_center=center,
            ball_radius=40.0,
            lighting_angle=30.0
        )
        test_frames.append(frame)
    
    # 스핀 분석 실행
    print("820fps 통합 스핀 분석 테스트 시작...")
    result = analyzer.analyze_ball_spin(test_frames)
    
    print(f"백스핀: {result.backspin:.1f} RPM")
    print(f"사이드스핀: {result.sidespin:.1f} RPM")
    print(f"스핀축: {result.spin_axis:.1f}°")
    print(f"총 스핀: {result.total_spin:.1f} RPM")
    print(f"신뢰도: {result.confidence:.2f}")
    print(f"사용 방법: {result.method_used}")
    print(f"분석 프레임 수: {result.frame_analysis_count}")


if __name__ == "__main__":
    test_integrated_spin_analyzer()