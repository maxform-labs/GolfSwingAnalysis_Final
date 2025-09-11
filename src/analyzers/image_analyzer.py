"""
실제 골프 영상 이미지 분석기
Author: Maxform 개발팀
Description: 실제 캡처된 골프 이미지에서 볼과 클럽 데이터를 추출하는 시스템
- 1440x300 해상도 1~23 프레임 시퀀스 분석
- 일반렌즈/Gamma렌즈 지원
- 7번 아이언/드라이버 분석
"""

import cv2
import numpy as np
import os
import json
import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
import time
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# 스테레오 비전 시스템 임포트
from stereo_vision_vertical import VerticalStereoVision, VerticalStereoConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GolfBallData:
    """골프볼 데이터 클래스"""
    frame_number: int
    timestamp_ms: float
    center_2d: Tuple[int, int]  # 2D 중심점 (픽셀)
    center_3d: Tuple[float, float, float]  # 3D 좌표 (mm)
    radius: float  # 볼 반지름 (픽셀)
    confidence: float  # 검출 신뢰도
    velocity_3d: Optional[Tuple[float, float, float]] = None  # 3D 속도 (mm/s)

@dataclass
class ClubData:
    """클럽 데이터 클래스"""
    frame_number: int
    timestamp_ms: float
    club_head_2d: Tuple[int, int]  # 클럽헤드 2D 좌표 (픽셀)
    club_head_3d: Tuple[float, float, float]  # 클럽헤드 3D 좌표 (mm)
    club_shaft_angle: float  # 클럽 샤프트 각도 (degree)
    face_angle: float  # 페이스 각도 (degree)
    confidence: float  # 검출 신뢰도
    velocity_3d: Optional[Tuple[float, float, float]] = None  # 3D 속도 (mm/s)

@dataclass
class SwingAnalysisResult:
    """스윙 분석 결과"""
    shot_type: str  # "7iron" or "driver"
    ball_type: str  # "logo_marker_free", "logo", "marker"
    lens_type: str  # "normal" or "gamma"
    shot_number: int
    
    # 볼 데이터
    ball_data: List[GolfBallData]
    ball_speed: float  # m/s
    launch_angle: float  # degrees
    side_angle: float  # degrees
    spin_rate: float  # rpm
    
    # 클럽 데이터
    club_data: List[ClubData]
    club_speed: float  # m/s
    attack_angle: float  # degrees
    club_path: float  # degrees
    face_angle_impact: float  # degrees
    
    # 임팩트 분석
    impact_frame: int
    impact_time_ms: float
    carry_distance: float  # meters

class GolfImageAnalyzer:
    """골프 이미지 분석기"""
    
    def __init__(self, config_file: str = None):
        """
        골프 이미지 분석기 초기화
        
        Args:
            config_file: 설정 파일 경로
        """
        # 설정 로드
        if config_file:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()
        
        # 스테레오 비전 시스템 초기화
        stereo_config = VerticalStereoConfig(
            vertical_baseline=self.config['stereo_settings']['vertical_baseline'],
            camera1_height=self.config['stereo_settings']['camera1_height'],
            camera2_height=self.config['stereo_settings']['camera2_height'],
            camera1_angle=self.config['stereo_settings']['camera1_angle'],
            camera2_angle=self.config['stereo_settings']['camera2_angle'],
            tee_distance=self.config['stereo_settings']['tee_distance']
        )
        
        self.stereo_vision = VerticalStereoVision(stereo_config)
        
        # 볼 검출기 초기화
        self.ball_detector = GolfBallDetector()
        
        # 클럽 검출기 초기화
        self.club_detector = ClubDetector()
        
        # 성능 모니터링
        self.analysis_times = []
        
        logger.info("골프 이미지 분석기 초기화 완료")
    
    def _default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "camera_settings": {
                "resolution": [1440, 300],
                "fps": 820
            },
            "stereo_settings": {
                "vertical_baseline": 500.0,
                "camera1_height": 400.0,
                "camera2_height": 900.0,
                "camera1_angle": 0.0,
                "camera2_angle": 12.0,
                "tee_distance": 500.0
            },
            "detection_settings": {
                "ball_min_radius": 8,
                "ball_max_radius": 25,
                "ball_dp": 1,
                "ball_param1": 50,
                "ball_param2": 30,
                "club_edge_threshold": 100,
                "club_line_threshold": 150
            }
        }
    
    def analyze_shot_sequence(self, image_dir: str) -> SwingAnalysisResult:
        """
        골프 샷 시퀀스 분석
        
        Args:
            image_dir: 이미지 디렉토리 경로 (1_1.bmp ~ 2_23.bmp 포함)
            
        Returns:
            스윙 분석 결과
        """
        logger.info(f"샷 시퀀스 분석 시작: {image_dir}")
        start_time = time.time()
        
        # 이미지 파일 로드
        image_pairs = self._load_image_pairs(image_dir)
        if not image_pairs:
            raise ValueError(f"이미지를 찾을 수 없습니다: {image_dir}")
        
        # 샷 정보 파싱
        shot_info = self._parse_shot_info(image_dir)
        
        # 프레임별 분석
        ball_data_list = []
        club_data_list = []
        
        for frame_num, (upper_img, lower_img) in enumerate(image_pairs, 1):
            timestamp_ms = (frame_num - 1) * (1000.0 / self.config['camera_settings']['fps'])
            
            # 볼 검출 및 추적
            ball_data = self._analyze_ball_frame(upper_img, lower_img, frame_num, timestamp_ms)
            if ball_data:
                ball_data_list.append(ball_data)
            
            # 클럽 검출 및 추적
            club_data = self._analyze_club_frame(upper_img, lower_img, frame_num, timestamp_ms)
            if club_data:
                club_data_list.append(club_data)
        
        # 임팩트 프레임 검출
        impact_frame = self._detect_impact_frame(ball_data_list, club_data_list)
        
        # 물리 분석
        swing_analysis = self._calculate_swing_physics(
            ball_data_list, club_data_list, impact_frame, shot_info
        )
        
        analysis_time = time.time() - start_time
        self.analysis_times.append(analysis_time)
        
        logger.info(f"샷 분석 완료: {analysis_time:.2f}초, 임팩트 프레임: {impact_frame}")
        
        return swing_analysis
    
    def _load_image_pairs(self, image_dir: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """이미지 쌍 로드"""
        image_pairs = []
        
        # 일반렌즈 또는 Gamma렌즈 확인
        files = os.listdir(image_dir)
        is_gamma = any(f.startswith('Gamma_') for f in files)
        
        for i in range(1, 24):  # 1~23 프레임
            if is_gamma:
                upper_file = f"Gamma_1_{i}.bmp"
                lower_file = f"Gamma_2_{i}.bmp"
            else:
                upper_file = f"1_{i}.bmp"
                lower_file = f"2_{i}.bmp"
            
            upper_path = os.path.join(image_dir, upper_file)
            lower_path = os.path.join(image_dir, lower_file)
            
            if os.path.exists(upper_path) and os.path.exists(lower_path):
                upper_img = cv2.imread(upper_path)
                lower_img = cv2.imread(lower_path)
                
                if upper_img is not None and lower_img is not None:
                    image_pairs.append((upper_img, lower_img))
        
        return image_pairs
    
    def _parse_shot_info(self, image_dir: str) -> Dict[str, str]:
        """샷 정보 파싱"""
        path_parts = Path(image_dir).parts
        
        shot_info = {
            "shot_type": "7iron" if "7번 아이언" in str(image_dir) else "driver",
            "ball_type": "unknown",
            "lens_type": "gamma" if "Gamma_" in os.listdir(image_dir)[0] else "normal",
            "shot_number": 1
        }
        
        # 볼 타입 결정
        if "로고, 마커없는" in str(image_dir):
            shot_info["ball_type"] = "logo_marker_free"
        elif "로고볼" in str(image_dir):
            shot_info["ball_type"] = "logo"
        elif "마커볼" in str(image_dir):
            shot_info["ball_type"] = "marker"
        
        # 샷 번호
        if "-2" in str(image_dir):
            shot_info["shot_number"] = 2
        
        return shot_info
    
    def _analyze_ball_frame(self, upper_img: np.ndarray, lower_img: np.ndarray, 
                          frame_num: int, timestamp_ms: float) -> Optional[GolfBallData]:
        """프레임별 볼 분석"""
        # 상단 카메라에서 볼 검출
        ball_upper = self.ball_detector.detect_ball(upper_img)
        
        # 하단 카메라에서 볼 검출
        ball_lower = self.ball_detector.detect_ball(lower_img)
        
        if ball_upper and ball_lower:
            # 스테레오 매칭으로 3D 좌표 계산
            disparity_map = self.stereo_vision.calculate_vertical_disparity(upper_img, lower_img)
            points_3d = self.stereo_vision.calculate_3d_coordinates(
                disparity_map, [ball_upper['center']]
            )
            
            if points_3d and points_3d[0] != (0.0, 0.0, 0.0):
                # Tee 중심 좌표계로 변환
                tee_coords = self.stereo_vision.convert_to_tee_coordinates(points_3d)
                
                return GolfBallData(
                    frame_number=frame_num,
                    timestamp_ms=timestamp_ms,
                    center_2d=ball_upper['center'],
                    center_3d=tee_coords[0],
                    radius=ball_upper['radius'],
                    confidence=min(ball_upper['confidence'], ball_lower['confidence'])
                )
        
        return None
    
    def _analyze_club_frame(self, upper_img: np.ndarray, lower_img: np.ndarray,
                          frame_num: int, timestamp_ms: float) -> Optional[ClubData]:
        """프레임별 클럽 분석"""
        # 상단 카메라에서 클럽 검출
        club_upper = self.club_detector.detect_club(upper_img)
        
        # 하단 카메라에서 클럽 검출
        club_lower = self.club_detector.detect_club(lower_img)
        
        if club_upper and club_lower:
            # 스테레오 매칭으로 3D 좌표 계산
            disparity_map = self.stereo_vision.calculate_vertical_disparity(upper_img, lower_img)
            points_3d = self.stereo_vision.calculate_3d_coordinates(
                disparity_map, [club_upper['head_center']]
            )
            
            if points_3d and points_3d[0] != (0.0, 0.0, 0.0):
                # Tee 중심 좌표계로 변환
                tee_coords = self.stereo_vision.convert_to_tee_coordinates(points_3d)
                
                return ClubData(
                    frame_number=frame_num,
                    timestamp_ms=timestamp_ms,
                    club_head_2d=club_upper['head_center'],
                    club_head_3d=tee_coords[0],
                    club_shaft_angle=club_upper['shaft_angle'],
                    face_angle=club_upper['face_angle'],
                    confidence=min(club_upper['confidence'], club_lower['confidence'])
                )
        
        return None
    
    def _detect_impact_frame(self, ball_data: List[GolfBallData], 
                           club_data: List[ClubData]) -> int:
        """임팩트 프레임 검출"""
        if not ball_data or not club_data:
            return 1
        
        min_distance = float('inf')
        impact_frame = 1
        
        for ball in ball_data:
            for club in club_data:
                if abs(ball.frame_number - club.frame_number) <= 1:  # 같은 프레임 또는 인접 프레임
                    # 3D 거리 계산
                    distance = np.sqrt(
                        (ball.center_3d[0] - club.club_head_3d[0])**2 +
                        (ball.center_3d[1] - club.club_head_3d[1])**2 +
                        (ball.center_3d[2] - club.club_head_3d[2])**2
                    )
                    
                    if distance < min_distance and distance < 50.0:  # 50mm 이내
                        min_distance = distance
                        impact_frame = ball.frame_number
        
        return impact_frame
    
    def _calculate_swing_physics(self, ball_data: List[GolfBallData], 
                               club_data: List[ClubData], impact_frame: int, 
                               shot_info: Dict[str, str]) -> SwingAnalysisResult:
        """스윙 물리 계산"""
        # 볼 속도 계산
        ball_speed = self._calculate_ball_speed(ball_data, impact_frame)
        launch_angle, side_angle = self._calculate_ball_angles(ball_data, impact_frame)
        spin_rate = self._estimate_spin_rate(ball_data, impact_frame)
        
        # 클럽 속도 계산
        club_speed = self._calculate_club_speed(club_data, impact_frame)
        attack_angle = self._calculate_attack_angle(club_data, impact_frame)
        club_path = self._calculate_club_path(club_data, impact_frame)
        face_angle = self._get_face_angle_at_impact(club_data, impact_frame)
        
        # 캐리 거리 추정
        carry_distance = self._estimate_carry_distance(ball_speed, launch_angle)
        
        return SwingAnalysisResult(
            shot_type=shot_info["shot_type"],
            ball_type=shot_info["ball_type"],
            lens_type=shot_info["lens_type"],
            shot_number=shot_info["shot_number"],
            ball_data=ball_data,
            ball_speed=ball_speed,
            launch_angle=launch_angle,
            side_angle=side_angle,
            spin_rate=spin_rate,
            club_data=club_data,
            club_speed=club_speed,
            attack_angle=attack_angle,
            club_path=club_path,
            face_angle_impact=face_angle,
            impact_frame=impact_frame,
            impact_time_ms=impact_frame * (1000.0 / 820),
            carry_distance=carry_distance
        )
    
    def _calculate_ball_speed(self, ball_data: List[GolfBallData], impact_frame: int) -> float:
        """볼 속도 계산 (m/s)"""
        if len(ball_data) < 2:
            return 0.0
        
        # 임팩트 후 2-3 프레임의 속도를 계산
        post_impact_balls = [b for b in ball_data if b.frame_number > impact_frame and b.frame_number <= impact_frame + 3]
        
        if len(post_impact_balls) < 2:
            return 0.0
        
        # 연속 프레임 간 속도 계산
        velocities = []
        for i in range(len(post_impact_balls) - 1):
            ball1 = post_impact_balls[i]
            ball2 = post_impact_balls[i + 1]
            
            # 3D 거리 (mm)
            distance = np.sqrt(
                (ball2.center_3d[0] - ball1.center_3d[0])**2 +
                (ball2.center_3d[1] - ball1.center_3d[1])**2 +
                (ball2.center_3d[2] - ball1.center_3d[2])**2
            )
            
            # 시간 차이 (s)
            time_diff = (ball2.timestamp_ms - ball1.timestamp_ms) / 1000.0
            
            if time_diff > 0:
                velocity = (distance / 1000.0) / time_diff  # m/s
                velocities.append(velocity)
        
        return np.mean(velocities) if velocities else 0.0
    
    def _calculate_ball_angles(self, ball_data: List[GolfBallData], impact_frame: int) -> Tuple[float, float]:
        """볼 발사각 및 방향각 계산"""
        post_impact_balls = [b for b in ball_data if b.frame_number > impact_frame and b.frame_number <= impact_frame + 5]
        
        if len(post_impact_balls) < 2:
            return 0.0, 0.0
        
        # 첫 번째와 마지막 점으로 각도 계산
        start_ball = post_impact_balls[0]
        end_ball = post_impact_balls[-1]
        
        dx = end_ball.center_3d[0] - start_ball.center_3d[0]
        dy = end_ball.center_3d[1] - start_ball.center_3d[1]
        dz = end_ball.center_3d[2] - start_ball.center_3d[2]
        
        # 발사각 (수직면에서의 각도)
        horizontal_distance = np.sqrt(dx**2 + dz**2)
        launch_angle = np.degrees(np.arctan2(dy, horizontal_distance)) if horizontal_distance > 0 else 0.0
        
        # 방향각 (수평면에서의 각도)
        side_angle = np.degrees(np.arctan2(dx, dz)) if dz != 0 else 0.0
        
        return launch_angle, side_angle
    
    def _estimate_spin_rate(self, ball_data: List[GolfBallData], impact_frame: int) -> float:
        """스핀율 추정 (rpm)"""
        # 간단한 추정 - 실제로는 볼 표면 마커 추적이 필요
        return 2500.0  # 일반적인 7번 아이언 스핀율
    
    def _calculate_club_speed(self, club_data: List[ClubData], impact_frame: int) -> float:
        """클럽 속도 계산 (m/s)"""
        pre_impact_clubs = [c for c in club_data if c.frame_number < impact_frame and c.frame_number >= impact_frame - 3]
        
        if len(pre_impact_clubs) < 2:
            return 0.0
        
        # 임팩트 직전 속도 계산
        velocities = []
        for i in range(len(pre_impact_clubs) - 1):
            club1 = pre_impact_clubs[i]
            club2 = pre_impact_clubs[i + 1]
            
            distance = np.sqrt(
                (club2.club_head_3d[0] - club1.club_head_3d[0])**2 +
                (club2.club_head_3d[1] - club1.club_head_3d[1])**2 +
                (club2.club_head_3d[2] - club1.club_head_3d[2])**2
            )
            
            time_diff = (club2.timestamp_ms - club1.timestamp_ms) / 1000.0
            
            if time_diff > 0:
                velocity = (distance / 1000.0) / time_diff
                velocities.append(velocity)
        
        return np.mean(velocities) if velocities else 0.0
    
    def _calculate_attack_angle(self, club_data: List[ClubData], impact_frame: int) -> float:
        """어택 앵글 계산"""
        pre_impact_clubs = [c for c in club_data if c.frame_number <= impact_frame and c.frame_number >= impact_frame - 3]
        
        if len(pre_impact_clubs) < 2:
            return 0.0
        
        # 클럽헤드 궤도의 수직 각도
        start_club = pre_impact_clubs[0]
        end_club = pre_impact_clubs[-1]
        
        dy = end_club.club_head_3d[1] - start_club.club_head_3d[1]
        dz = end_club.club_head_3d[2] - start_club.club_head_3d[2]
        
        return np.degrees(np.arctan2(dy, dz)) if dz != 0 else 0.0
    
    def _calculate_club_path(self, club_data: List[ClubData], impact_frame: int) -> float:
        """클럽 패스 계산"""
        pre_impact_clubs = [c for c in club_data if c.frame_number <= impact_frame and c.frame_number >= impact_frame - 3]
        
        if len(pre_impact_clubs) < 2:
            return 0.0
        
        # 클럽헤드 궤도의 수평 각도
        start_club = pre_impact_clubs[0]
        end_club = pre_impact_clubs[-1]
        
        dx = end_club.club_head_3d[0] - start_club.club_head_3d[0]
        dz = end_club.club_head_3d[2] - start_club.club_head_3d[2]
        
        return np.degrees(np.arctan2(dx, dz)) if dz != 0 else 0.0
    
    def _get_face_angle_at_impact(self, club_data: List[ClubData], impact_frame: int) -> float:
        """임팩트 시 페이스 앵글"""
        impact_club = next((c for c in club_data if c.frame_number == impact_frame), None)
        return impact_club.face_angle if impact_club else 0.0
    
    def _estimate_carry_distance(self, ball_speed: float, launch_angle: float) -> float:
        """캐리 거리 추정 (미터)"""
        if ball_speed == 0:
            return 0.0
        
        # 간단한 탄도 계산 (공기저항 무시)
        g = 9.81
        launch_rad = np.radians(launch_angle)
        
        carry = (ball_speed**2 * np.sin(2 * launch_rad)) / g
        return carry
    
    def save_analysis_result(self, result: SwingAnalysisResult, output_file: str):
        """분석 결과 저장"""
        result_dict = asdict(result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"분석 결과 저장: {output_file}")


class GolfBallDetector:
    """골프볼 검출기"""
    
    def __init__(self):
        self.min_radius = 8
        self.max_radius = 25
        self.dp = 1
        self.param1 = 50
        self.param2 = 30
    
    def detect_ball(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """이미지에서 골프볼 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 원형 검출
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=30,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # 가장 확실한 원 선택 (중앙에 가장 가까운 것)
            center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
            best_circle = None
            min_distance = float('inf')
            
            for (x, y, r) in circles:
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    best_circle = (x, y, r)
            
            if best_circle:
                x, y, r = best_circle
                confidence = 1.0 - min_distance / (image.shape[0] + image.shape[1])
                
                return {
                    'center': (int(x), int(y)),
                    'radius': float(r),
                    'confidence': max(0.1, min(1.0, confidence))
                }
        
        return None


class ClubDetector:
    """클럽 검출기"""
    
    def __init__(self):
        self.edge_threshold = 100
        self.line_threshold = 150
    
    def detect_club(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """이미지에서 클럽 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 엣지 검출
        edges = cv2.Canny(gray, 50, self.edge_threshold)
        
        # 직선 검출 (클럽 샤프트)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.line_threshold,
            minLineLength=50,
            maxLineGap=10
        )
        
        if lines is not None and len(lines) > 0:
            # 가장 긴 직선을 클럽 샤프트로 가정
            longest_line = max(lines, key=lambda line: 
                np.sqrt((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2))
            
            x1, y1, x2, y2 = longest_line[0]
            
            # 클럽헤드 위치 추정 (직선의 끝점)
            head_x = x2 if y2 < y1 else x1  # 더 위쪽 점을 헤드로 가정
            head_y = y2 if y2 < y1 else y1
            
            # 샤프트 각도 계산
            shaft_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            return {
                'head_center': (int(head_x), int(head_y)),
                'shaft_angle': float(shaft_angle),
                'face_angle': float(shaft_angle + 90),  # 간단한 추정
                'confidence': 0.7  # 기본 신뢰도
            }
        
        return None


def main():
    """메인 함수 - 테스트 실행"""
    config_file = "config.json"
    analyzer = GolfImageAnalyzer(config_file)
    
    # 7번 아이언 샷 분석
    shot_dirs = [
        r"C:\src\GolfSwingAnalysis_Final_ver8\shot-image\7번 아이언\로고, 마커없는 볼-1",
        r"C:\src\GolfSwingAnalysis_Final_ver8\shot-image\7번 아이언\로고, 마커없는 볼-2"
    ]
    
    results = []
    for shot_dir in shot_dirs:
        if os.path.exists(shot_dir):
            try:
                result = analyzer.analyze_shot_sequence(shot_dir)
                results.append(result)
                
                # 결과 저장
                output_file = shot_dir.replace("shot-image", "analysis_results") + "_result.json"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                analyzer.save_analysis_result(result, output_file)
                
                logger.info(f"분석 완료: {shot_dir}")
                logger.info(f"볼 스피드: {result.ball_speed:.1f} m/s")
                logger.info(f"런치 앵글: {result.launch_angle:.1f}°")
                logger.info(f"클럽 스피드: {result.club_speed:.1f} m/s")
                
            except Exception as e:
                logger.error(f"분석 실패 {shot_dir}: {e}")
    
    return results


if __name__ == "__main__":
    results = main()