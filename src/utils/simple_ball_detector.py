#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Ball Detector v1.0
간단하지만 실용적인 볼 검출 시스템
"""

import cv2
import numpy as np
import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class BallDetectionResult:
    """볼 검출 결과"""
    center_x: float
    center_y: float
    radius: float
    confidence: float
    motion_state: str = "static"

class SimpleBallDetector:
    """간단한 볼 검출기"""
    
    def __init__(self):
        self.previous_positions = []
        self.motion_threshold = 5.0  # 픽셀 단위 움직임 임계값
        self.min_radius = 5
        self.max_radius = 50
        
    def detect_ir_ball(self, img: np.ndarray, frame_number: int) -> Optional[BallDetectionResult]:
        """IR 기반 볼 검출 (실제로는 일반 이미지 검출)"""
        
        if img is None:
            return None
        
        try:
            # 1. 그레이스케일 변환
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # 2. 이미지 전처리 - 밝은 영역 강조
            # 볼은 일반적으로 밝은 색이므로 높은 값을 가짐
            _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
            # 3. 가우시안 블러로 노이즈 제거
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # 4. HoughCircles로 원형 객체 검출
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=50,
                param2=30,
                minRadius=self.min_radius,
                maxRadius=self.max_radius
            )
            
            best_circle = None
            best_score = 0
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # 경계 체크
                    if x < r or y < r or x >= img.shape[1] - r or y >= img.shape[0] - r:
                        continue
                    
                    # 원형 영역의 밝기 점수 계산
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, -1)
                    
                    # 원형 영역 내부의 평균 밝기
                    mean_intensity = cv2.mean(gray, mask=mask)[0]
                    
                    # 원형 영역 경계의 선명도 (라플라시안)
                    roi = gray[y-r:y+r, x-r:x+r]
                    if roi.size > 0:
                        edge_variance = cv2.Laplacian(roi, cv2.CV_64F).var()
                    else:
                        edge_variance = 0
                    
                    # 종합 점수 (밝기 + 경계 선명도)
                    score = (mean_intensity / 255.0) * 0.7 + min(edge_variance / 1000.0, 1.0) * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_circle = (x, y, r, score)
            
            # 5. 대안: 밝은 영역 기반 검출
            if best_circle is None:
                # 가장 밝은 연결된 영역 찾기
                contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 50 < area < 2000:  # 적절한 크기
                        # 외접원 계산
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        
                        if self.min_radius < radius < self.max_radius:
                            # 원형도 체크
                            perimeter = cv2.arcLength(contour, True)
                            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                            
                            if circularity > 0.4:  # 충분히 원형
                                score = circularity * 0.8 + (area / 2000.0) * 0.2
                                
                                if score > best_score:
                                    best_score = score
                                    best_circle = (int(x), int(y), int(radius), score)
            
            if best_circle is None:
                return None
            
            x, y, r, score = best_circle
            
            # 6. 모션 상태 판단
            motion_state = self._determine_motion_state(x, y, frame_number)
            
            # 7. 신뢰도 계산 (0.3-0.9 범위)
            confidence = max(0.3, min(0.9, score))
            
            return BallDetectionResult(
                center_x=float(x),
                center_y=float(y),
                radius=float(r),
                confidence=confidence,
                motion_state=motion_state
            )
            
        except Exception as e:
            print(f"Ball detection error: {e}")
            return None
    
    def _determine_motion_state(self, x: float, y: float, frame_number: int) -> str:
        """움직임 상태 판단"""
        
        current_pos = (x, y)
        self.previous_positions.append((current_pos, frame_number))
        
        # 최근 5프레임만 유지
        if len(self.previous_positions) > 5:
            self.previous_positions.pop(0)
        
        if len(self.previous_positions) < 3:
            return "static"
        
        # 최근 위치들 간의 거리 계산
        recent_distances = []
        for i in range(len(self.previous_positions) - 1):
            pos1, _ = self.previous_positions[i]
            pos2, _ = self.previous_positions[i + 1]
            
            distance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            recent_distances.append(distance)
        
        avg_movement = np.mean(recent_distances) if recent_distances else 0
        max_movement = max(recent_distances) if recent_distances else 0
        
        # 상태 결정
        if avg_movement < 2.0:
            return "static"
        elif avg_movement < 5.0:
            return "ready"
        elif max_movement > 15.0:
            return "launching"
        elif avg_movement > 10.0:
            return "launched"
        else:
            return "flying"
    
    def reset_tracking(self):
        """추적 초기화"""
        self.previous_positions = []

class SimplePhysicsCalculator:
    """간단한 물리량 계산기"""
    
    @staticmethod
    def calculate_3d_speed(positions: List[Tuple[float, float, float]], 
                          timestamps: List[float]) -> Dict[str, float]:
        """3D 속도 계산"""
        if len(positions) < 2:
            return {'speed_ms': 0.0, 'speed_mph': 0.0}
        
        # 간단한 2점 방식 (더 안정적)
        start_pos = positions[0]
        end_pos = positions[-1]
        dt = timestamps[-1] - timestamps[0]
        
        if dt <= 0:
            return {'speed_ms': 0.0, 'speed_mph': 0.0}
        
        # 3D 거리 계산 (mm 단위)
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1] 
        dz = end_pos[2] - start_pos[2]
        
        distance_mm = math.sqrt(dx**2 + dy**2 + dz**2)
        distance_m = distance_mm / 1000.0
        
        speed_ms = distance_m / dt
        speed_mph = speed_ms * 2.237  # m/s to mph
        
        return {
            'speed_ms': speed_ms,
            'speed_mph': speed_mph
        }
    
    @staticmethod
    def calculate_launch_angle(positions: List[Tuple[float, float, float]]) -> float:
        """발사각 계산"""
        if len(positions) < 2:
            return 0.0
        
        start_pos = positions[0]
        end_pos = positions[-1]
        
        dx = end_pos[0] - start_pos[0]  # 수평 거리
        dy = end_pos[1] - start_pos[1]  # 수직 거리 (Y축)
        
        if abs(dx) < 1.0:  # 거의 수직
            return 90.0 if dy > 0 else -90.0
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # -30도 ~ +60도 범위로 제한
        return max(-30.0, min(60.0, angle_deg))
    
    @staticmethod
    def calculate_direction_angle(positions: List[Tuple[float, float, float]]) -> float:
        """방향각 계산 (좌우)"""
        if len(positions) < 2:
            return 0.0
        
        start_pos = positions[0]
        end_pos = positions[-1]
        
        dx = end_pos[0] - start_pos[0]  # X축 이동
        dz = end_pos[2] - start_pos[2]  # Z축 이동 (깊이)
        
        if abs(dz) < 1.0:
            return 0.0
        
        angle_rad = math.atan2(dx, dz)
        angle_deg = math.degrees(angle_rad)
        
        # -45도 ~ +45도 범위로 제한
        return max(-45.0, min(45.0, angle_deg))

class SimpleSpinCalculator:
    """간단한 스핀 계산기"""
    
    @staticmethod
    def calculate_spin_from_trajectory(positions: List[Tuple[float, float, float]], 
                                     timestamps: List[float]) -> Dict[str, float]:
        """궤적 기반 스핀 추정"""
        if len(positions) < 3:
            return {
                'backspin_rpm': 0.0,
                'sidespin_rpm': 0.0,
                'spin_axis_deg': 0.0
            }
        
        # 궤적의 곡률을 이용한 스핀 추정
        mid_idx = len(positions) // 2
        start_pos = positions[0]
        mid_pos = positions[mid_idx]
        end_pos = positions[-1]
        
        # 백스핀 추정 (Y축 궤적 곡률)
        expected_y = start_pos[1] + (end_pos[1] - start_pos[1]) * 0.5
        actual_y = mid_pos[1]
        y_deviation = actual_y - expected_y
        
        # 추정 백스핀 (높은 궤적 = 높은 백스핀)
        backspin = abs(y_deviation) * 50.0  # 경험적 계수
        backspin = max(0.0, min(12000.0, backspin))  # 0-12K rpm 제한
        
        # 사이드스핀 추정 (X축 궤적 곡률)
        expected_x = start_pos[0] + (end_pos[0] - start_pos[0]) * 0.5
        actual_x = mid_pos[0]
        x_deviation = actual_x - expected_x
        
        sidespin = x_deviation * 30.0  # 경험적 계수
        sidespin = max(-3000.0, min(3000.0, sidespin))  # ±3K rpm 제한
        
        # 스핀축 계산
        if abs(backspin) > 100 or abs(sidespin) > 100:
            spin_axis = math.degrees(math.atan2(sidespin, backspin))
        else:
            spin_axis = 0.0
        
        return {
            'backspin_rpm': backspin,
            'sidespin_rpm': sidespin,
            'spin_axis_deg': spin_axis
        }

class SimpleClubCalculator:
    """간단한 클럽 물리량 계산기"""
    
    @staticmethod
    def calculate_club_speed(positions: List[Tuple[float, float, float]], 
                           timestamps: List[float]) -> Dict[str, float]:
        """클럽 스피드 계산"""
        if len(positions) < 2:
            return {'speed_ms': 0.0, 'speed_mph': 0.0}
        
        # 볼 스피드와 유사한 방식으로 계산
        start_pos = positions[0]
        end_pos = positions[-1]
        dt = timestamps[-1] - timestamps[0]
        
        if dt <= 0:
            return {'speed_ms': 0.0, 'speed_mph': 0.0}
        
        distance_mm = math.sqrt(
            (end_pos[0] - start_pos[0])**2 + 
            (end_pos[1] - start_pos[1])**2 + 
            (end_pos[2] - start_pos[2])**2
        )
        
        speed_ms = (distance_mm / 1000.0) / dt
        speed_mph = speed_ms * 2.237
        
        return {
            'speed_ms': speed_ms,
            'speed_mph': speed_mph
        }
    
    @staticmethod
    def calculate_attack_angle(positions: List[Tuple[float, float, float]], 
                             timestamps: List[float]) -> float:
        """어택 앵글 계산"""
        if len(positions) < 2:
            return 0.0
        
        start_pos = positions[0]
        end_pos = positions[-1]
        
        dx = abs(end_pos[0] - start_pos[0])
        dy = end_pos[1] - start_pos[1]  # Y축 변화
        
        if dx < 1.0:
            return 0.0
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # -15도 ~ +10도 범위로 제한
        return max(-15.0, min(10.0, angle_deg))
    
    @staticmethod
    def calculate_club_path(positions: List[Tuple[float, float, float]]) -> float:
        """클럽 패스 계산"""
        if len(positions) < 2:
            return 0.0
        
        start_pos = positions[0]
        end_pos = positions[-1]
        
        dx = end_pos[0] - start_pos[0]  # 좌우 이동
        dz = end_pos[2] - start_pos[2]  # 전후 이동
        
        if abs(dz) < 1.0:
            return 0.0
        
        angle_rad = math.atan2(dx, dz)
        angle_deg = math.degrees(angle_rad)
        
        # -30도 ~ +30도 범위로 제한
        return max(-30.0, min(30.0, angle_deg))
    
    @staticmethod
    def calculate_face_angle(club_path_deg: float, direction_angle_deg: float) -> float:
        """페이스 앵글 계산"""
        # 볼의 방향각과 클럽 패스의 차이로 추정
        face_angle = direction_angle_deg - club_path_deg * 0.3
        
        # -20도 ~ +20도 범위로 제한
        return max(-20.0, min(20.0, face_angle))
    
    @staticmethod
    def calculate_smash_factor(ball_speed_mph: float, club_speed_mph: float) -> float:
        """스매쉬 팩터 계산"""
        if club_speed_mph < 10.0:  # 너무 낮은 클럽 스피드
            return 0.0
        
        smash_factor = ball_speed_mph / club_speed_mph
        
        # 0.8 ~ 1.8 범위로 제한 (일반적인 골프 범위)
        return max(0.8, min(1.8, smash_factor))

# 3D 위치 계산을 위한 간단한 스테레오 비전
class SimpleStereoVision:
    """간단한 스테레오 비전"""
    
    def __init__(self):
        # 카메라 파라미터 (config.json 기반)
        self.baseline = 500.0  # mm
        self.fx = 1000.0  # 추정 초점거리
        self.fy = 1000.0
        self.height_diff = 500.0  # 상하 카메라 높이차
    
    def calculate_3d_position(self, top_point: Tuple[float, float], 
                            bottom_point: Tuple[float, float]) -> Tuple[float, float, float]:
        """3D 위치 계산"""
        
        x_top, y_top = top_point
        x_bottom, y_bottom = bottom_point
        
        # Y축 시차 계산 (수직 스테레오)
        disparity_y = abs(y_top - y_bottom)
        
        if disparity_y < 1.0:
            disparity_y = 1.0  # 최소값으로 설정
        
        # 깊이 계산 (Z축)
        Z = (self.fy * self.baseline) / disparity_y
        Z = max(500.0, min(5000.0, Z))  # 0.5m ~ 5m 제한
        
        # X축 계산
        X = (x_top + x_bottom) / 2.0 - 720  # 이미지 중심에서 상대적 위치
        X = X * Z / self.fx  # 실제 거리로 변환
        
        # Y축 계산 
        Y = (y_top + y_bottom) / 2.0 - 150  # 이미지 중심에서 상대적 위치
        Y = Y * Z / self.fy  # 실제 거리로 변환
        
        return (float(X), float(Y), float(Z))

class SimplePhysicsValidator:
    """간단한 물리 검증기"""
    
    @staticmethod
    def validate_ball_physics(ball_data: Dict) -> Dict[str, bool]:
        """볼 물리량 검증"""
        validations = {}
        
        # 볼 스피드 검증 (20-200 mph)
        ball_speed = ball_data.get('ball_speed_mph', 0)
        validations['ball_speed_valid'] = 20 <= ball_speed <= 200
        
        # 발사각 검증 (-15도 ~ 60도)
        launch_angle = ball_data.get('launch_angle_deg', 0)
        validations['launch_angle_valid'] = -15 <= launch_angle <= 60
        
        # 방향각 검증 (-45도 ~ 45도)
        direction_angle = ball_data.get('direction_angle_deg', 0)
        validations['direction_angle_valid'] = -45 <= direction_angle <= 45
        
        # 백스핀 검증 (0 ~ 12000 rpm)
        backspin = ball_data.get('backspin_rpm', 0)
        validations['backspin_valid'] = 0 <= backspin <= 12000
        
        # 사이드스핀 검증 (-5000 ~ 5000 rpm)
        sidespin = ball_data.get('sidespin_rpm', 0)
        validations['sidespin_valid'] = -5000 <= sidespin <= 5000
        
        return validations
    
    @staticmethod
    def validate_club_physics(club_data: Dict) -> Dict[str, bool]:
        """클럽 물리량 검증"""
        validations = {}
        
        # 클럽 스피드 검증 (30-150 mph)
        club_speed = club_data.get('club_speed_mph', 0)
        validations['club_speed_valid'] = 30 <= club_speed <= 150
        
        # 어택 앵글 검증 (-15도 ~ 15도)
        attack_angle = club_data.get('attack_angle_deg', 0)
        validations['attack_angle_valid'] = -15 <= attack_angle <= 15
        
        # 페이스 앵글 검증 (-30도 ~ 30도)
        face_angle = club_data.get('face_angle_deg', 0)
        validations['face_angle_valid'] = -30 <= face_angle <= 30
        
        # 클럽 패스 검증 (-30도 ~ 30도)
        club_path = club_data.get('club_path_deg', 0)
        validations['club_path_valid'] = -30 <= club_path <= 30
        
        # 스매쉬 팩터 검증 (0.8 ~ 1.8)
        smash_factor = club_data.get('smash_factor', 0)
        validations['smash_factor_valid'] = 0.8 <= smash_factor <= 1.8
        
        return validations

class SimpleTrajectoryPredictor:
    """간단한 궤적 예측기"""
    
    @staticmethod
    def predict_carry_distance(ball_speed_ms: float, launch_angle_deg: float, 
                             backspin_rpm: float) -> Dict[str, float]:
        """비거리 예측"""
        
        if ball_speed_ms < 1.0:
            return {
                'carry_m': 0.0,
                'max_height_m': 0.0,
                'flight_time_s': 0.0
            }
        
        # 간단한 탄도학 계산
        launch_angle_rad = math.radians(launch_angle_deg)
        g = 9.81  # 중력 가속도
        
        # 초기 속도 성분
        v0_x = ball_speed_ms * math.cos(launch_angle_rad)
        v0_y = ball_speed_ms * math.sin(launch_angle_rad)
        
        # 비행시간 (공기 저항 무시한 간단한 공식)
        flight_time = 2 * v0_y / g if v0_y > 0 else 0
        
        # 비거리
        carry_distance = v0_x * flight_time
        
        # 최고점 높이
        max_height = (v0_y ** 2) / (2 * g) if v0_y > 0 else 0
        
        # 스핀 효과 보정 (간단한 보정)
        spin_factor = 1.0 + (backspin_rpm / 10000.0) * 0.1
        carry_distance *= spin_factor
        max_height *= spin_factor
        
        return {
            'carry_m': max(0.0, carry_distance),
            'max_height_m': max(0.0, max_height), 
            'flight_time_s': max(0.0, flight_time)
        }