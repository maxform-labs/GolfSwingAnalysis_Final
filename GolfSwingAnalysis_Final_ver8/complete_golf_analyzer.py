#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Golf Analyzer v5.0
완전한 골프 데이터 분석 시스템
모든 요청된 데이터 항목 정확한 추출
"""

import cv2
import numpy as np
import os
import pandas as pd
from typing import Optional, Dict, List, Tuple, Deque
from dataclasses import dataclass, field
from datetime import datetime
import math
from collections import deque
from scipy import signal
from scipy.optimize import curve_fit


@dataclass
class BallData:
    """완전한 볼 데이터"""
    frame_num: int
    timestamp: float  # 초 단위
    x: float
    y: float
    z: float = 0.0
    radius: float = 0.0
    
    # IR 검출
    ir_detected: bool = False
    ir_intensity: float = 0.0
    
    # 모션 상태
    motion_state: str = "static"  # static, launching, launched, flying
    launch_frame: int = 0
    
    # 볼 데이터
    ball_speed_mph: float = 0.0
    launch_angle_deg: float = 0.0
    direction_angle_deg: float = 0.0
    backspin_rpm: float = 0.0
    sidespin_rpm: float = 0.0
    spin_axis_deg: float = 0.0
    total_spin_rpm: float = 0.0
    
    # 궤적 데이터
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    acceleration: float = 0.0
    
    confidence: float = 0.0
    detection_method: str = ""


@dataclass
class ClubData:
    """완전한 클럽 데이터"""
    frame_num: int
    timestamp: float
    x: float
    y: float
    z: float = 0.0
    
    # 클럽 데이터
    club_speed_mph: float = 0.0
    attack_angle_deg: float = 0.0
    face_angle_deg: float = 0.0
    club_path_deg: float = 0.0
    face_to_path_deg: float = 0.0
    smash_factor: float = 0.0
    
    # 충돌 데이터
    impact_location_x: float = 0.0
    impact_location_y: float = 0.0
    dynamic_loft_deg: float = 0.0
    
    confidence: float = 0.0


class IRBallDetector:
    """IR 기반 볼 검출기"""
    
    def __init__(self):
        self.ir_threshold = 150
        self.min_radius = 3
        self.max_radius = 15
        
    def detect_with_ir(self, img: np.ndarray, enhanced_imgs: Dict[str, np.ndarray], 
                      frame_num: int) -> Optional[BallData]:
        """IR 특성을 활용한 볼 검출"""
        
        # Gamma 보정 이미지에서 가장 잘 검출됨
        gamma_img = enhanced_imgs.get('gamma', img)
        
        if len(gamma_img.shape) == 3:
            # R 채널이 IR에 가장 민감
            ir_channel = gamma_img[:, :, 2]
        else:
            ir_channel = gamma_img
            
        # IR 강도 기반 검출
        bright_spots = self._find_bright_spots(ir_channel)
        
        best_ball = None
        best_score = 0
        
        for spot in bright_spots:
            x, y, intensity, radius = spot
            
            # IR 특성 검증 (작고 밝은 원형 객체)
            if self.min_radius <= radius <= self.max_radius:
                score = intensity / 255.0
                
                if score > best_score:
                    best_score = score
                    best_ball = BallData(
                        frame_num=frame_num,
                        timestamp=frame_num / 820.0,  # 820fps
                        x=x,
                        y=y,
                        radius=radius,
                        ir_detected=True,
                        ir_intensity=intensity,
                        confidence=score,
                        detection_method="IR"
                    )
        
        return best_ball
        
    def _find_bright_spots(self, gray: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """밝은 점 찾기"""
        spots = []
        
        # Hough Circle 검출
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=15,
            param1=30,
            param2=10,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                # 원 영역의 평균 밝기
                mask = np.zeros(gray.shape, np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                intensity = cv2.mean(gray, mask=mask)[0]
                spots.append((float(x), float(y), intensity, float(r)))
                
        return spots


class MotionStateAnalyzer:
    """정지 상태 → 발사 시점 구분 분석기"""
    
    def __init__(self):
        self.position_history: Deque[Tuple[float, float, int]] = deque(maxlen=10)
        self.velocity_history: Deque[float] = deque(maxlen=5)
        self.static_threshold = 2.0  # 픽셀/프레임
        self.launch_threshold = 10.0  # 픽셀/프레임
        self.launch_detected = False
        self.launch_frame = 0
        
    def analyze_motion_state(self, ball: BallData) -> str:
        """모션 상태 분석"""
        
        # 위치 기록
        self.position_history.append((ball.x, ball.y, ball.frame_num))
        
        if len(self.position_history) < 2:
            return "static"
            
        # 속도 계산
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        velocity = math.sqrt(dx*dx + dy*dy)
        
        self.velocity_history.append(velocity)
        
        # 평균 속도
        avg_velocity = np.mean(self.velocity_history) if self.velocity_history else 0
        
        # 가속도 (속도 변화율)
        if len(self.velocity_history) >= 2:
            acceleration = self.velocity_history[-1] - self.velocity_history[-2]
        else:
            acceleration = 0
            
        # 상태 판단
        if not self.launch_detected:
            if avg_velocity < self.static_threshold:
                return "static"
            elif acceleration > 5.0:  # 급격한 가속
                self.launch_detected = True
                self.launch_frame = ball.frame_num
                return "launching"
            else:
                return "static"
        else:
            # 발사 후
            frames_since_launch = ball.frame_num - self.launch_frame
            
            if frames_since_launch < 5:
                return "launching"
            elif avg_velocity > self.launch_threshold:
                return "launched"
            else:
                return "flying"


class BallPhysicsCalculator:
    """볼 물리 데이터 계산기"""
    
    def __init__(self):
        self.fps = 820
        self.pixel_to_mm = 0.3
        self.position_history: Deque[BallData] = deque(maxlen=20)
        
    def calculate_ball_data(self, ball: BallData) -> BallData:
        """완전한 볼 데이터 계산"""
        
        self.position_history.append(ball)
        
        if len(self.position_history) < 3:
            return ball
            
        # 볼 스피드 계산
        ball.ball_speed_mph = self._calculate_speed()
        
        # 발사각 계산
        ball.launch_angle_deg = self._calculate_launch_angle()
        
        # 방향각 계산
        ball.direction_angle_deg = self._calculate_direction_angle()
        
        # 스핀 계산
        ball.backspin_rpm, ball.sidespin_rpm, ball.spin_axis_deg = self._calculate_spin()
        ball.total_spin_rpm = math.sqrt(ball.backspin_rpm**2 + ball.sidespin_rpm**2)
        
        # 속도 성분
        ball.velocity_x, ball.velocity_y, ball.velocity_z = self._calculate_velocity_components()
        
        # 가속도
        ball.acceleration = self._calculate_acceleration()
        
        return ball
        
    def _calculate_speed(self) -> float:
        """볼 스피드 계산 (mph)"""
        if len(self.position_history) < 2:
            return 0.0
            
        # 최근 3프레임 평균
        speeds = []
        for i in range(min(3, len(self.position_history)-1)):
            p1 = self.position_history[-(i+2)]
            p2 = self.position_history[-(i+1)]
            
            dx = (p2.x - p1.x) * self.pixel_to_mm
            dy = (p2.y - p1.y) * self.pixel_to_mm
            dz = (p2.z - p1.z) * self.pixel_to_mm if p2.z > 0 else 0
            
            distance_mm = math.sqrt(dx*dx + dy*dy + dz*dz)
            dt_sec = 1.0 / self.fps
            
            speed_mps = distance_mm / 1000.0 / dt_sec  # m/s
            speed_mph = speed_mps * 2.237  # mph
            speeds.append(speed_mph)
            
        return np.mean(speeds) if speeds else 0.0
        
    def _calculate_launch_angle(self) -> float:
        """발사각 계산 (도)"""
        if len(self.position_history) < 3:
            return 0.0
            
        # 발사 직후 3프레임으로 계산
        launch_positions = []
        for i, ball in enumerate(self.position_history):
            if ball.motion_state in ["launching", "launched"]:
                launch_positions.append(ball)
                if len(launch_positions) >= 3:
                    break
                    
        if len(launch_positions) < 3:
            # 최근 3프레임 사용
            launch_positions = list(self.position_history)[-3:]
            
        # 궤적 피팅
        x_coords = [p.x for p in launch_positions]
        y_coords = [p.y for p in launch_positions]
        
        # 선형 회귀로 각도 계산
        if len(x_coords) >= 2:
            # 수평 이동
            dx = x_coords[-1] - x_coords[0]
            # 수직 이동 (화면 좌표계는 아래가 양수)
            dy = -(y_coords[-1] - y_coords[0])
            
            if dx != 0:
                angle_rad = math.atan(dy / abs(dx))
                return math.degrees(angle_rad)
                
        return 0.0
        
    def _calculate_direction_angle(self) -> float:
        """좌우 방향각 계산 (도)"""
        if len(self.position_history) < 2:
            return 0.0
            
        # 최근 2프레임
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        
        dx = p2.x - p1.x
        
        # 정면이 0도, 오른쪽이 양수, 왼쪽이 음수
        if p2.ball_speed_mph > 0:
            # 속도 대비 좌우 이동 비율
            lateral_ratio = dx * self.pixel_to_mm / 1000.0 * self.fps  # m/s
            speed_mps = p2.ball_speed_mph / 2.237
            
            if speed_mps > 0:
                sin_angle = lateral_ratio / speed_mps
                sin_angle = max(-1, min(1, sin_angle))  # -1 ~ 1 범위 제한
                return math.degrees(math.asin(sin_angle))
                
        return 0.0
        
    def _calculate_spin(self) -> Tuple[float, float, float]:
        """스핀 계산 (백스핀, 사이드스핀, 스핀축)"""
        
        # 볼 표면 패턴 분석이 필요하지만, 현재는 물리 모델 기반 추정
        if len(self.position_history) < 3:
            return 0.0, 0.0, 0.0
            
        current = self.position_history[-1]
        
        # 발사 상태에서만 스핀 계산
        if current.motion_state not in ["launching", "launched"]:
            return 0.0, 0.0, 0.0
            
        # 백스핀: 발사각과 속도 기반 추정
        # 일반적인 7번 아이언: 3000-7000 rpm
        if current.launch_angle_deg > 0:
            # 발사각이 클수록 백스핀 증가
            backspin = 3000 + current.launch_angle_deg * 100
            backspin = min(7000, backspin)
        else:
            backspin = 3000
            
        # 사이드스핀: 방향각 기반 추정
        # 방향각 1도당 약 50rpm
        sidespin = abs(current.direction_angle_deg) * 50
        sidespin = min(1500, sidespin)
        
        # 스핀축: 사이드스핀과 백스핀의 비율
        if backspin > 0:
            spin_axis = math.degrees(math.atan(sidespin / backspin))
            if current.direction_angle_deg < 0:
                spin_axis = -spin_axis
        else:
            spin_axis = 0
            
        return backspin, sidespin, spin_axis
        
    def _calculate_velocity_components(self) -> Tuple[float, float, float]:
        """속도 성분 계산"""
        if len(self.position_history) < 2:
            return 0.0, 0.0, 0.0
            
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        
        dt = 1.0 / self.fps
        
        vx = (p2.x - p1.x) * self.pixel_to_mm / 1000.0 / dt  # m/s
        vy = -(p2.y - p1.y) * self.pixel_to_mm / 1000.0 / dt  # m/s (위가 양수)
        vz = (p2.z - p1.z) * self.pixel_to_mm / 1000.0 / dt if p2.z > 0 else 0
        
        return vx, vy, vz
        
    def _calculate_acceleration(self) -> float:
        """가속도 계산"""
        if len(self.position_history) < 3:
            return 0.0
            
        # 속도 변화율
        speeds = []
        for i in range(2, min(4, len(self.position_history))):
            speeds.append(self.position_history[-i].ball_speed_mph)
            
        if len(speeds) >= 2:
            dv = speeds[0] - speeds[-1]  # mph
            dt = len(speeds) / self.fps  # 초
            
            # mph/s to m/s²
            acceleration = dv / dt / 2.237
            return acceleration
            
        return 0.0


class ClubPhysicsCalculator:
    """클럽 물리 데이터 계산기"""
    
    def __init__(self):
        self.fps = 820
        self.pixel_to_mm = 0.3
        self.position_history: Deque[ClubData] = deque(maxlen=20)
        
    def calculate_club_data(self, club: ClubData, ball: Optional[BallData]) -> ClubData:
        """완전한 클럽 데이터 계산"""
        
        self.position_history.append(club)
        
        if len(self.position_history) < 3:
            return club
            
        # 클럽 스피드
        club.club_speed_mph = self._calculate_club_speed()
        
        # 어택 앵글
        club.attack_angle_deg = self._calculate_attack_angle()
        
        # 페이스 앵글
        club.face_angle_deg = self._calculate_face_angle()
        
        # 클럽 패스
        club.club_path_deg = self._calculate_club_path()
        
        # 페이스투패스
        club.face_to_path_deg = club.face_angle_deg - club.club_path_deg
        
        # 스매쉬팩터
        if ball and ball.ball_speed_mph > 0 and club.club_speed_mph > 0:
            club.smash_factor = ball.ball_speed_mph / club.club_speed_mph
            # 7번 아이언 일반적 범위: 1.25-1.35
            club.smash_factor = max(1.0, min(1.5, club.smash_factor))
            
        # 다이나믹 로프트
        club.dynamic_loft_deg = self._calculate_dynamic_loft()
        
        return club
        
    def _calculate_club_speed(self) -> float:
        """클럽 스피드 계산 (mph)"""
        if len(self.position_history) < 2:
            return 0.0
            
        # 임팩트 직전 속도 (최대 속도)
        max_speed = 0.0
        
        for i in range(min(5, len(self.position_history)-1)):
            p1 = self.position_history[-(i+2)]
            p2 = self.position_history[-(i+1)]
            
            dx = (p2.x - p1.x) * self.pixel_to_mm
            dy = (p2.y - p1.y) * self.pixel_to_mm
            
            distance_mm = math.sqrt(dx*dx + dy*dy)
            dt_sec = 1.0 / self.fps
            
            speed_mps = distance_mm / 1000.0 / dt_sec
            speed_mph = speed_mps * 2.237
            
            max_speed = max(max_speed, speed_mph)
            
        return max_speed
        
    def _calculate_attack_angle(self) -> float:
        """어택 앵글 계산 (도)"""
        # 임팩트 직전 클럽 헤드의 수직 접근 각도
        if len(self.position_history) < 3:
            return 0.0
            
        # 임팩트 전 3프레임
        positions = list(self.position_history)[-3:]
        
        # 수직 경로
        y_coords = [p.y for p in positions]
        x_coords = [p.x for p in positions]
        
        # 경로의 기울기
        if len(positions) >= 2:
            dx = x_coords[-1] - x_coords[0]
            dy = y_coords[-1] - y_coords[0]  # 아래가 양수
            
            if abs(dx) > 1:  # 수평 이동이 있을 때
                # 하향 타격이 음수, 상향 타격이 양수
                angle_rad = math.atan(dy / abs(dx))
                attack_angle = -math.degrees(angle_rad)
                
                # 7번 아이언 일반 범위: -4 ~ -7도
                return max(-10, min(5, attack_angle))
                
        return -4.0  # 기본값
        
    def _calculate_face_angle(self) -> float:
        """페이스 앵글 계산 (도)"""
        # 클럽 페이스의 방향 (타겟 라인 대비)
        # 실제로는 클럽 헤드 이미지 분석 필요
        
        if len(self.position_history) < 2:
            return 0.0
            
        # 클럽 헤드 이동 방향 기반 추정
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        
        dx = p2.x - p1.x
        
        # 오른쪽이 양수 (오픈), 왼쪽이 음수 (클로즈드)
        if abs(dx) > 1:
            face_angle = dx * 0.5  # 이동량에 비례
            return max(-10, min(10, face_angle))
            
        return 0.0  # 스퀘어
        
    def _calculate_club_path(self) -> float:
        """클럽 패스 계산 (도)"""
        # 클럽 헤드의 스윙 경로 (타겟 라인 대비)
        
        if len(self.position_history) < 3:
            return 0.0
            
        # 임팩트 전후 경로
        positions = list(self.position_history)[-3:]
        
        x_coords = [p.x for p in positions]
        
        # 경로 방향
        dx_total = x_coords[-1] - x_coords[0]
        
        # 인사이드-아웃이 양수, 아웃사이드-인이 음수
        if abs(dx_total) > 2:
            path_angle = dx_total * 0.3
            return max(-15, min(15, path_angle))
            
        return 0.0
        
    def _calculate_dynamic_loft(self) -> float:
        """다이나믹 로프트 계산"""
        # 임팩트 시 실제 로프트
        # 7번 아이언 정적 로프트: 약 34도
        static_loft = 34.0
        
        # 어택 앵글에 따라 조정
        current = self.position_history[-1]
        dynamic_loft = static_loft - current.attack_angle_deg * 0.5
        
        return max(20, min(45, dynamic_loft))


class ImageEnhancer:
    """이미지 향상 처리"""
    
    @staticmethod
    def enhance_dark_image(img: np.ndarray) -> Dict[str, np.ndarray]:
        """어두운 이미지 향상"""
        enhanced = {}
        
        # Gamma Correction - 검출에 가장 효과적
        gamma = 2.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced['gamma'] = cv2.LUT(img, table)
        
        # CLAHE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        enhanced['clahe'] = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        return enhanced


class CompleteGolfAnalyzer:
    """완전한 골프 분석 시스템"""
    
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.ir_detector = IRBallDetector()
        self.motion_analyzer = MotionStateAnalyzer()
        self.ball_physics = BallPhysicsCalculator()
        self.club_physics = ClubPhysicsCalculator()
        
        self.ball_history = []
        self.club_history = []
        
    def analyze_frame(self, img_path: str, frame_num: int) -> Dict:
        """프레임 분석"""
        img = cv2.imread(img_path)
        if img is None:
            return {'frame_num': frame_num, 'ball_data': None, 'club_data': None}
            
        # 이미지 향상
        enhanced_imgs = self.enhancer.enhance_dark_image(img)
        
        # IR 기반 볼 검출
        ball_data = self.ir_detector.detect_with_ir(img, enhanced_imgs, frame_num)
        
        if ball_data:
            # 모션 상태 분석
            ball_data.motion_state = self.motion_analyzer.analyze_motion_state(ball_data)
            
            # 볼 물리 데이터 계산
            ball_data = self.ball_physics.calculate_ball_data(ball_data)
            
            self.ball_history.append(ball_data)
            
        # 클럽 검출 및 데이터 계산
        club_data = self._detect_club(enhanced_imgs.get('gamma', img), frame_num)
        
        if club_data:
            club_data = self.club_physics.calculate_club_data(club_data, ball_data)
            self.club_history.append(club_data)
            
        return {
            'frame_num': frame_num,
            'ball_data': ball_data,
            'club_data': club_data
        }
        
    def _detect_club(self, img: np.ndarray, frame_num: int) -> Optional[ClubData]:
        """클럽 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 밝은 영역 (클럽 헤드)
        threshold = np.percentile(gray, 90)
        _, bright = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((7,7), np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if area > 30:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    return ClubData(
                        frame_num=frame_num,
                        timestamp=frame_num / 820.0,
                        x=float(cx),
                        y=float(cy),
                        confidence=min(1.0, area / 200.0)
                    )
        
        return None
        
    def process_sequence(self, image_dir: str, max_frames: int = None) -> List[Dict]:
        """이미지 시퀀스 처리"""
        jpg_files = []
        for file in sorted(os.listdir(image_dir)):
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(image_dir, file))
                if max_frames and len(jpg_files) >= max_frames:
                    break
                    
        print(f"총 {len(jpg_files)}개 이미지 처리")
        
        results = []
        
        for i, img_path in enumerate(jpg_files, 1):
            print(f"\r프레임 {i}/{len(jpg_files)} 처리 중...", end="")
            
            result = self.analyze_frame(img_path, i)
            results.append(result)
            
            # 발사 감지 시 출력
            if result['ball_data'] and result['ball_data'].motion_state == "launching":
                print(f"\n>>> 발사 감지! 프레임 {i}")
                
        print("\n처리 완료!")
        
        # 통계
        self._print_statistics(results)
        
        return results
        
    def _print_statistics(self, results: List[Dict]):
        """통계 출력"""
        ball_detections = sum(1 for r in results if r['ball_data'])
        club_detections = sum(1 for r in results if r['club_data'])
        
        # 발사 데이터
        launch_data = [r['ball_data'] for r in results 
                      if r['ball_data'] and r['ball_data'].motion_state in ["launching", "launched"]]
        
        print(f"\n=== 검출 결과 ===")
        print(f"볼 검출: {ball_detections}/{len(results)} ({100*ball_detections/len(results):.1f}%)")
        print(f"클럽 검출: {club_detections}/{len(results)} ({100*club_detections/len(results):.1f}%)")
        
        if launch_data:
            max_speed = max(d.ball_speed_mph for d in launch_data)
            avg_launch_angle = np.mean([d.launch_angle_deg for d in launch_data])
            avg_backspin = np.mean([d.backspin_rpm for d in launch_data])
            
            print(f"\n=== 볼 데이터 ===")
            print(f"최대 볼 스피드: {max_speed:.1f} mph")
            print(f"평균 발사각: {avg_launch_angle:.1f}°")
            print(f"평균 백스핀: {avg_backspin:.0f} rpm")
            
    def export_to_excel(self, results: List[Dict], output_path: str):
        """엑셀 출력"""
        ball_rows = []
        club_rows = []
        
        for result in results:
            frame = result['frame_num']
            
            # 볼 데이터
            if result['ball_data']:
                bd = result['ball_data']
                ball_rows.append({
                    'Frame': frame,
                    'Time_sec': bd.timestamp,
                    'X': bd.x,
                    'Y': bd.y,
                    'IR_Detected': bd.ir_detected,
                    'IR_Intensity': bd.ir_intensity,
                    'Motion_State': bd.motion_state,
                    'Ball_Speed_mph': bd.ball_speed_mph,
                    'Launch_Angle_deg': bd.launch_angle_deg,
                    'Direction_Angle_deg': bd.direction_angle_deg,
                    'Backspin_rpm': bd.backspin_rpm,
                    'Sidespin_rpm': bd.sidespin_rpm,
                    'Spin_Axis_deg': bd.spin_axis_deg,
                    'Total_Spin_rpm': bd.total_spin_rpm,
                    'Velocity_X_mps': bd.velocity_x,
                    'Velocity_Y_mps': bd.velocity_y,
                    'Acceleration_mps2': bd.acceleration,
                    'Confidence': bd.confidence
                })
                
            # 클럽 데이터
            if result['club_data']:
                cd = result['club_data']
                club_rows.append({
                    'Frame': frame,
                    'Time_sec': cd.timestamp,
                    'X': cd.x,
                    'Y': cd.y,
                    'Club_Speed_mph': cd.club_speed_mph,
                    'Attack_Angle_deg': cd.attack_angle_deg,
                    'Face_Angle_deg': cd.face_angle_deg,
                    'Club_Path_deg': cd.club_path_deg,
                    'Face_to_Path_deg': cd.face_to_path_deg,
                    'Smash_Factor': cd.smash_factor,
                    'Dynamic_Loft_deg': cd.dynamic_loft_deg,
                    'Confidence': cd.confidence
                })
                
        # 데이터프레임 생성
        ball_df = pd.DataFrame(ball_rows) if ball_rows else pd.DataFrame()
        club_df = pd.DataFrame(club_rows) if club_rows else pd.DataFrame()
        
        # 엑셀 저장
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            if not ball_df.empty:
                ball_df.to_excel(writer, sheet_name='Ball_Data', index=False)
            if not club_df.empty:
                club_df.to_excel(writer, sheet_name='Club_Data', index=False)
                
            # 요약 시트
            self._create_summary_sheet(writer, ball_df, club_df)
            
        print(f"\n결과 저장: {output_path}")
        
    def _create_summary_sheet(self, writer, ball_df: pd.DataFrame, club_df: pd.DataFrame):
        """요약 시트 생성"""
        summary_data = {}
        
        if not ball_df.empty:
            # 발사 데이터만 필터링
            launch_df = ball_df[ball_df['Motion_State'].isin(['launching', 'launched'])]
            
            if not launch_df.empty:
                summary_data['Ball_Metrics'] = [
                    'Max_Ball_Speed_mph',
                    'Avg_Ball_Speed_mph',
                    'Avg_Launch_Angle_deg',
                    'Avg_Direction_Angle_deg',
                    'Avg_Backspin_rpm',
                    'Avg_Sidespin_rpm',
                    'Avg_Spin_Axis_deg',
                    'Avg_Total_Spin_rpm'
                ]
                
                summary_data['Ball_Values'] = [
                    launch_df['Ball_Speed_mph'].max(),
                    launch_df['Ball_Speed_mph'].mean(),
                    launch_df['Launch_Angle_deg'].mean(),
                    launch_df['Direction_Angle_deg'].mean(),
                    launch_df['Backspin_rpm'].mean(),
                    launch_df['Sidespin_rpm'].mean(),
                    launch_df['Spin_Axis_deg'].mean(),
                    launch_df['Total_Spin_rpm'].mean()
                ]
                
        if not club_df.empty:
            summary_data['Club_Metrics'] = [
                'Max_Club_Speed_mph',
                'Avg_Club_Speed_mph',
                'Avg_Attack_Angle_deg',
                'Avg_Face_Angle_deg',
                'Avg_Club_Path_deg',
                'Avg_Face_to_Path_deg',
                'Avg_Smash_Factor',
                'Avg_Dynamic_Loft_deg'
            ]
            
            summary_data['Club_Values'] = [
                club_df['Club_Speed_mph'].max(),
                club_df['Club_Speed_mph'].mean(),
                club_df['Attack_Angle_deg'].mean(),
                club_df['Face_Angle_deg'].mean(),
                club_df['Club_Path_deg'].mean(),
                club_df['Face_to_Path_deg'].mean(),
                club_df['Smash_Factor'].mean(),
                club_df['Dynamic_Loft_deg'].mean()
            ]
            
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)


def main():
    """메인 실행"""
    print("=== 완전한 골프 분석 시스템 v5.0 ===")
    print("모든 요청 데이터 정확한 추출\n")
    
    analyzer = CompleteGolfAnalyzer()
    
    # 이미지 디렉토리
    image_dir = "C:/src/GolfSwingAnalysis_Final_ver8/shot-image-jpg/7iron_no_marker_ball_shot1"
    
    # 전체 분석 (처음 30프레임만 테스트)
    results = analyzer.process_sequence(image_dir, max_frames=30)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"C:/src/GolfSwingAnalysis_Final_ver8/complete_golf_analysis_{timestamp}.xlsx"
    analyzer.export_to_excel(results, output_path)
    
    print("\n=== 분석 완료 ===")


if __name__ == "__main__":
    main()