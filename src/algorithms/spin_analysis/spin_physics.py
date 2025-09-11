#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golf Physics Formulas Library v1.0
골프 물리학 산출공식 라이브러리
- 정확한 물리학적 공식 기반
- 820fps 고속 촬영 최적화
- 실제 골프장 조건 반영
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.interpolate import interp1d

@dataclass
class PhysicsConstants:
    """물리학 상수"""
    GRAVITY = 9.81  # m/s²
    AIR_DENSITY = 1.225  # kg/m³ (해수면 기준)
    BALL_DIAMETER = 0.04267  # m (42.67mm)
    BALL_MASS = 0.04593  # kg (45.93g)
    DRAG_COEFFICIENT = 0.47  # 구형 물체
    MAGNUS_COEFFICIENT = 0.25  # 마그누스 효과


class BallSpeedCalculator:
    """볼 스피드 계산기"""
    
    @staticmethod
    def calculate_3d_speed(positions: List[Tuple[float, float, float]], 
                          timestamps: List[float]) -> Dict[str, float]:
        """3D 속도 계산 (3점 미분법)"""
        if len(positions) < 3 or len(timestamps) < 3:
            return {'speed_ms': 0.0, 'speed_mph': 0.0}
        
        # 3점 미분법으로 더 정확한 속도 계산
        velocities = []
        
        for i in range(1, len(positions) - 1):
            dt = timestamps[i+1] - timestamps[i-1]
            if dt > 0:
                # 3점 미분: v = (p[i+1] - p[i-1]) / (2*dt)
                vx = (positions[i+1][0] - positions[i-1][0]) / dt
                vy = (positions[i+1][1] - positions[i-1][1]) / dt
                vz = (positions[i+1][2] - positions[i-1][2]) / dt
                
                speed = math.sqrt(vx**2 + vy**2 + vz**2)
                velocities.append(speed)
        
        if velocities:
            # 발사 직후 속도 (첫 번째 측정값)
            initial_speed = velocities[0] / 1000.0  # mm/s -> m/s
            speed_mph = initial_speed * 2.237  # m/s -> mph
            
            return {
                'speed_ms': initial_speed,
                'speed_mph': speed_mph,
                'velocity_components': {
                    'vx': (positions[1][0] - positions[0][0]) / (timestamps[1] - timestamps[0]) / 1000.0,
                    'vy': (positions[1][1] - positions[0][1]) / (timestamps[1] - timestamps[0]) / 1000.0,
                    'vz': (positions[1][2] - positions[0][2]) / (timestamps[1] - timestamps[0]) / 1000.0
                }
            }
        
        return {'speed_ms': 0.0, 'speed_mph': 0.0}
    
    @staticmethod
    def calculate_launch_angle(positions: List[Tuple[float, float, float]]) -> float:
        """발사각 계산 (탄도각)"""
        if len(positions) < 3:
            return 0.0
        
        # 발사 직후 2-3 프레임 사용
        start_pos = positions[0]
        end_pos = positions[2] if len(positions) > 2 else positions[1]
        
        # 수평 거리와 수직 거리
        horizontal_dist = math.sqrt(
            (end_pos[0] - start_pos[0])**2 + 
            (end_pos[2] - start_pos[2])**2
        )
        vertical_dist = end_pos[1] - start_pos[1]
        
        if horizontal_dist > 1.0:  # 최소 거리 임계값
            launch_angle = math.degrees(math.atan2(vertical_dist, horizontal_dist))
            return np.clip(launch_angle, -20.0, 45.0)
        
        return 0.0
    
    @staticmethod
    def calculate_direction_angle(positions: List[Tuple[float, float, float]]) -> float:
        """좌우 방향각 계산"""
        if len(positions) < 3:
            return 0.0
        
        start_pos = positions[0]
        end_pos = positions[2] if len(positions) > 2 else positions[1]
        
        # X-Z 평면에서의 방향각 (Z축을 전진방향으로 가정)
        dx = end_pos[0] - start_pos[0]
        dz = end_pos[2] - start_pos[2]
        
        if abs(dz) > 1.0:
            direction_angle = math.degrees(math.atan2(dx, dz))
            return np.clip(direction_angle, -30.0, 30.0)
        
        return 0.0


class SpinCalculator820fps:
    """820fps 기반 스핀 계산기"""
    
    def __init__(self):
        self.fps = 820.0
        self.frame_interval = 1.0 / self.fps  # 1.22ms
        
    def calculate_spin_from_trajectory(self, positions: List[Tuple[float, float, float]], 
                                     timestamps: List[float]) -> Dict[str, float]:
        """궤적 분석을 통한 스핀 계산"""
        if len(positions) < 7:  # 최소 7개 포인트 필요
            return self._default_spin_data()
        
        # 궤적 곡률 분석
        curvature_analysis = self._analyze_trajectory_curvature(positions)
        
        # 마그누스 효과를 이용한 스핀 추정
        spin_data = self._calculate_spin_from_magnus_effect(
            positions, timestamps, curvature_analysis
        )
        
        return spin_data
    
    def _analyze_trajectory_curvature(self, positions: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """궤적 곡률 분석"""
        # 2차 다항식 피팅을 통한 곡률 계산
        x_coords = np.array([p[0] for p in positions])
        y_coords = np.array([p[1] for p in positions])
        z_coords = np.array([p[2] for p in positions])
        
        # 정규화된 거리 계산
        distances = []
        for i in range(len(positions)):
            dist = math.sqrt(x_coords[i]**2 + y_coords[i]**2 + z_coords[i]**2)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # 수직면 곡률 (Y-Z 관계)
        try:
            # Z 좌표를 독립변수로 Y 좌표를 종속변수로 2차 피팅
            z_normalized = z_coords - z_coords[0]
            y_normalized = y_coords - y_coords[0]
            
            if len(z_normalized) >= 3 and np.std(z_normalized) > 10:
                vertical_coeff = np.polyfit(z_normalized, y_normalized, 2)
                vertical_curvature = 2 * vertical_coeff[0]
            else:
                vertical_curvature = 0.0
        except:
            vertical_curvature = 0.0
        
        # 수평면 곡률 (X-Z 관계)
        try:
            z_normalized = z_coords - z_coords[0]
            x_normalized = x_coords - x_coords[0]
            
            if len(z_normalized) >= 3 and np.std(z_normalized) > 10:
                horizontal_coeff = np.polyfit(z_normalized, x_normalized, 2)
                horizontal_curvature = 2 * horizontal_coeff[0]
            else:
                horizontal_curvature = 0.0
        except:
            horizontal_curvature = 0.0
        
        return {
            'vertical_curvature': vertical_curvature,
            'horizontal_curvature': horizontal_curvature
        }
    
    def _calculate_spin_from_magnus_effect(self, positions: List[Tuple[float, float, float]], 
                                         timestamps: List[float], 
                                         curvature_data: Dict[str, float]) -> Dict[str, float]:
        """마그누스 효과를 이용한 스핀 계산"""
        
        # 속도 계산
        speed_data = BallSpeedCalculator.calculate_3d_speed(positions, timestamps)
        ball_speed = speed_data.get('speed_ms', 0.0)
        
        if ball_speed < 1.0:  # 최소 속도 임계값
            return self._default_spin_data()
        
        # 물리 상수
        const = PhysicsConstants()
        
        # 마그누스 힘 계산을 통한 스핀 추정
        # F_magnus = 0.5 * rho * v^2 * A * C_magnus * (omega × v)
        
        # 백스핀 계산 (수직 곡률 기반)
        vertical_curvature = curvature_data['vertical_curvature']
        
        # 마그누스 힘에서 스핀율 역산
        # F_magnus = C_magnus * rho * v * omega * A
        magnus_force_vertical = abs(vertical_curvature) * const.BALL_MASS * const.GRAVITY
        
        if ball_speed > 0.1:
            # 백스핀 계산 (rpm)
            ball_area = math.pi * (const.BALL_DIAMETER / 2)**2
            omega_backspin = magnus_force_vertical / (
                const.MAGNUS_COEFFICIENT * const.AIR_DENSITY * ball_speed * ball_area
            )
            backspin_rpm = (omega_backspin * 60) / (2 * math.pi)
            backspin_rpm = abs(backspin_rpm)
        else:
            backspin_rpm = 0.0
        
        # 사이드스핀 계산 (수평 곡률 기반)
        horizontal_curvature = curvature_data['horizontal_curvature']
        magnus_force_horizontal = horizontal_curvature * const.BALL_MASS * const.GRAVITY
        
        if ball_speed > 0.1:
            omega_sidespin = magnus_force_horizontal / (
                const.MAGNUS_COEFFICIENT * const.AIR_DENSITY * ball_speed * ball_area
            )
            sidespin_rpm = (omega_sidespin * 60) / (2 * math.pi)
        else:
            sidespin_rpm = 0.0
        
        # 물리적 제약 조건 적용
        backspin_rpm = np.clip(backspin_rpm, 0.0, 12000.0)
        sidespin_rpm = np.clip(sidespin_rpm, -3000.0, 3000.0)
        
        # 스핀축 계산
        if backspin_rpm > 100:
            spin_axis_deg = math.degrees(math.atan2(sidespin_rpm, backspin_rpm))
        else:
            spin_axis_deg = 0.0
        
        # 총 스핀
        total_spin_rpm = math.sqrt(backspin_rpm**2 + sidespin_rpm**2)
        
        return {
            'backspin_rpm': backspin_rpm,
            'sidespin_rpm': sidespin_rpm,
            'spin_axis_deg': spin_axis_deg,
            'total_spin_rpm': total_spin_rpm
        }
    
    def _default_spin_data(self) -> Dict[str, float]:
        """기본 스핀 데이터"""
        return {
            'backspin_rpm': 0.0,
            'sidespin_rpm': 0.0,
            'spin_axis_deg': 0.0,
            'total_spin_rpm': 0.0
        }


class ClubPhysicsCalculator:
    """클럽 물리학 계산기"""
    
    @staticmethod
    def calculate_club_speed(positions: List[Tuple[float, float, float]], 
                           timestamps: List[float]) -> Dict[str, float]:
        """클럽 스피드 계산"""
        if len(positions) < 3:
            return {'speed_ms': 0.0, 'speed_mph': 0.0}
        
        # 임팩트 직전 최대 속도 찾기
        max_speed = 0.0
        max_speed_frame = 0
        
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                dz = positions[i][2] - positions[i-1][2]
                
                speed = math.sqrt(dx**2 + dy**2 + dz**2) / dt  # mm/s
                speed_ms = speed / 1000.0  # m/s
                
                if speed_ms > max_speed:
                    max_speed = speed_ms
                    max_speed_frame = i
        
        speed_mph = max_speed * 2.237
        
        return {
            'speed_ms': max_speed,
            'speed_mph': speed_mph,
            'impact_frame': max_speed_frame
        }
    
    @staticmethod
    def calculate_attack_angle(positions: List[Tuple[float, float, float]], 
                             timestamps: List[float]) -> float:
        """어택 앵글 계산"""
        if len(positions) < 3:
            return 0.0
        
        # 임팩트 구간 (마지막 3-4 프레임)
        impact_positions = positions[-4:] if len(positions) >= 4 else positions
        
        # 클럽헤드 궤적의 수직 성분 분석
        start_pos = impact_positions[0]
        end_pos = impact_positions[-1]
        
        # 수평 이동과 수직 이동
        horizontal_movement = math.sqrt(
            (end_pos[0] - start_pos[0])**2 + 
            (end_pos[2] - start_pos[2])**2
        )
        vertical_movement = end_pos[1] - start_pos[1]
        
        if horizontal_movement > 1.0:
            attack_angle = math.degrees(math.atan2(vertical_movement, horizontal_movement))
            return np.clip(attack_angle, -10.0, 15.0)
        
        return 0.0
    
    @staticmethod
    def calculate_club_path(positions: List[Tuple[float, float, float]]) -> float:
        """클럽 패스 계산"""
        if len(positions) < 3:
            return 0.0
        
        # 임팩트 구간 분석
        start_pos = positions[-3] if len(positions) >= 3 else positions[0]
        end_pos = positions[-1]
        
        # X-Z 평면에서의 이동 방향
        dx = end_pos[0] - start_pos[0]
        dz = end_pos[2] - start_pos[2]
        
        if abs(dz) > 1.0:
            club_path = math.degrees(math.atan2(dx, dz))
            return np.clip(club_path, -15.0, 15.0)
        
        return 0.0
    
    @staticmethod
    def calculate_face_angle(club_path: float, ball_direction: float) -> float:
        """페이스 앵글 계산 (클럽패스와 볼 방향각 기반)"""
        # 페이스 앵글은 볼의 시작 방향과 클럽 패스의 관계로 추정
        face_angle_estimate = ball_direction - (club_path * 0.7)  # 70% 영향
        return np.clip(face_angle_estimate, -15.0, 15.0)
    
    @staticmethod
    def calculate_smash_factor(ball_speed: float, club_speed: float) -> float:
        """스매쉬팩터 계산"""
        if club_speed > 0:
            smash_factor = ball_speed / club_speed
            # 물리적으로 가능한 범위 (1.0 ~ 1.5)
            return np.clip(smash_factor, 0.8, 1.6)
        return 0.0


class TrajectoryPredictor:
    """탄도 예측기"""
    
    @staticmethod
    def predict_carry_distance(ball_speed: float, launch_angle: float, 
                             backspin: float, altitude: float = 0.0) -> Dict[str, float]:
        """캐리 거리 예측"""
        if ball_speed < 1.0:
            return {'carry_m': 0.0, 'max_height_m': 0.0, 'flight_time_s': 0.0}
        
        const = PhysicsConstants()
        
        # 초기 속도 성분
        v0 = ball_speed
        angle_rad = math.radians(launch_angle)
        v0x = v0 * math.cos(angle_rad)
        v0y = v0 * math.sin(angle_rad)
        
        # 공기저항 계수 (레이놀즈 수 기반 보정)
        reynolds_number = const.AIR_DENSITY * ball_speed * const.BALL_DIAMETER / 1.8e-5
        cd = const.DRAG_COEFFICIENT * (1 + 0.1 * (reynolds_number / 100000))
        
        # 스핀 효과 (백스핀에 의한 양력)
        spin_rps = backspin / 60.0  # rpm to rps
        spin_parameter = (spin_rps * const.BALL_DIAMETER) / v0
        lift_coefficient = const.MAGNUS_COEFFICIENT * spin_parameter
        
        # 수치적분을 통한 탄도 계산
        dt = 0.01  # 0.01초 간격
        t = 0.0
        x, y = 0.0, 0.0
        vx, vy = v0x, v0y
        max_height = 0.0
        
        while y >= 0 and t < 30:  # 최대 30초 또는 지면 도달까지
            # 현재 속도
            v_current = math.sqrt(vx**2 + vy**2)
            
            if v_current > 0.1:
                # 공기저항
                drag_force = 0.5 * const.AIR_DENSITY * v_current**2 * \
                           (math.pi * (const.BALL_DIAMETER/2)**2) * cd
                drag_ax = -drag_force * vx / (v_current * const.BALL_MASS)
                drag_ay = -drag_force * vy / (v_current * const.BALL_MASS)
                
                # 마그누스 힘 (백스핀에 의한 양력)
                magnus_force = 0.5 * const.AIR_DENSITY * v_current**2 * \
                              (math.pi * (const.BALL_DIAMETER/2)**2) * lift_coefficient
                magnus_ay = magnus_force / const.BALL_MASS
                
                # 가속도 적용
                ax = drag_ax
                ay = -const.GRAVITY + drag_ay + magnus_ay
            else:
                ax, ay = 0, -const.GRAVITY
            
            # 위치와 속도 업데이트
            x += vx * dt
            y += vy * dt
            vx += ax * dt
            vy += ay * dt
            
            max_height = max(max_height, y)
            t += dt
        
        return {
            'carry_m': x,
            'max_height_m': max_height,
            'flight_time_s': t
        }


class GolfPhysicsValidator:
    """골프 물리학 검증기"""
    
    @staticmethod
    def validate_ball_physics(ball_data: Dict[str, float]) -> Dict[str, bool]:
        """볼 물리량 검증"""
        validations = {}
        
        # 볼 스피드 검증
        ball_speed_mph = ball_data.get('ball_speed_mph', 0)
        validations['ball_speed_valid'] = 50 <= ball_speed_mph <= 200
        
        # 발사각 검증
        launch_angle = ball_data.get('launch_angle_deg', 0)
        validations['launch_angle_valid'] = -20 <= launch_angle <= 45
        
        # 방향각 검증
        direction_angle = ball_data.get('direction_angle_deg', 0)
        validations['direction_valid'] = -30 <= direction_angle <= 30
        
        # 백스핀 검증
        backspin = ball_data.get('backspin_rpm', 0)
        validations['backspin_valid'] = 0 <= backspin <= 12000
        
        # 사이드스핀 검증
        sidespin = ball_data.get('sidespin_rpm', 0)
        validations['sidespin_valid'] = -3000 <= sidespin <= 3000
        
        return validations
    
    @staticmethod
    def validate_club_physics(club_data: Dict[str, float]) -> Dict[str, bool]:
        """클럽 물리량 검증"""
        validations = {}
        
        # 클럽 스피드 검증
        club_speed_mph = club_data.get('club_speed_mph', 0)
        validations['club_speed_valid'] = 60 <= club_speed_mph <= 150
        
        # 어택 앵글 검증
        attack_angle = club_data.get('attack_angle_deg', 0)
        validations['attack_angle_valid'] = -10 <= attack_angle <= 15
        
        # 페이스 앵글 검증
        face_angle = club_data.get('face_angle_deg', 0)
        validations['face_angle_valid'] = -15 <= face_angle <= 15
        
        # 클럽 패스 검증
        club_path = club_data.get('club_path_deg', 0)
        validations['club_path_valid'] = -15 <= club_path <= 15
        
        # 스매쉬팩터 검증
        smash_factor = club_data.get('smash_factor', 0)
        validations['smash_factor_valid'] = 0.8 <= smash_factor <= 1.6
        
        return validations


def main():
    """물리 공식 테스트"""
    print("=== Golf Physics Formulas Library v1.0 ===")
    
    # 테스트 데이터
    test_positions = [
        (0.0, 0.0, 1000.0),      # 시작점
        (15.0, 8.0, 1080.0),     # 1프레임 후
        (35.0, 18.0, 1180.0),    # 2프레임 후
        (60.0, 25.0, 1300.0),    # 3프레임 후
        (90.0, 30.0, 1440.0),    # 4프레임 후
        (125.0, 32.0, 1600.0),   # 5프레임 후
        (165.0, 33.0, 1780.0),   # 6프레임 후
    ]
    
    test_timestamps = [i * 0.00122 for i in range(len(test_positions))]  # 820fps
    
    # 볼 물리량 계산
    print("\n=== Ball Physics Calculation ===")
    
    # 스피드 계산
    speed_data = BallSpeedCalculator.calculate_3d_speed(test_positions, test_timestamps)
    print(f"Ball Speed: {speed_data['speed_mph']:.1f} mph ({speed_data['speed_ms']:.1f} m/s)")
    
    # 발사각 계산
    launch_angle = BallSpeedCalculator.calculate_launch_angle(test_positions)
    print(f"Launch Angle: {launch_angle:.1f}°")
    
    # 방향각 계산
    direction_angle = BallSpeedCalculator.calculate_direction_angle(test_positions)
    print(f"Direction Angle: {direction_angle:.1f}°")
    
    # 스핀 계산
    spin_calculator = SpinCalculator820fps()
    spin_data = spin_calculator.calculate_spin_from_trajectory(test_positions, test_timestamps)
    print(f"Backspin: {spin_data['backspin_rpm']:.0f} rpm")
    print(f"Sidespin: {spin_data['sidespin_rpm']:.0f} rpm")
    print(f"Spin Axis: {spin_data['spin_axis_deg']:.1f}°")
    
    # 탄도 예측
    print("\n=== Trajectory Prediction ===")
    trajectory = TrajectoryPredictor.predict_carry_distance(
        speed_data['speed_ms'], launch_angle, spin_data['backspin_rpm']
    )
    print(f"Carry Distance: {trajectory['carry_m']:.1f} m")
    print(f"Max Height: {trajectory['max_height_m']:.1f} m")
    print(f"Flight Time: {trajectory['flight_time_s']:.1f} s")
    
    # 검증
    print("\n=== Physics Validation ===")
    ball_physics = {
        'ball_speed_mph': speed_data['speed_mph'],
        'launch_angle_deg': launch_angle,
        'direction_angle_deg': direction_angle,
        'backspin_rpm': spin_data['backspin_rpm'],
        'sidespin_rpm': spin_data['sidespin_rpm']
    }
    
    validations = GolfPhysicsValidator.validate_ball_physics(ball_physics)
    for param, is_valid in validations.items():
        status = "✓" if is_valid else "✗"
        print(f"{param}: {status}")
    
    print("\nPhysics formulas loaded successfully!")


if __name__ == "__main__":
    main()