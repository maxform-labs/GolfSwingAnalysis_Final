"""
골프공 및 클럽 추적 시스템
Author: Maxform 개발팀
Description: 고정밀 골프공/클럽 검출 및 추적 시스템
- IR 조명 기반 배경 차감
- 칼만 필터 기반 안정적 추적
- 샷 이벤트 자동 감지
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.optimize import least_squares
import math

@dataclass
class TrackingPoint:
    """추적 점 정보"""
    x: float
    y: float
    z: float
    timestamp: float
    confidence: float = 1.0

@dataclass
class BallData:
    """골프공 데이터"""
    speed: float  # m/s
    launch_angle: float  # degrees
    direction_angle: float  # degrees (방향각으로 명칭 변경)
    backspin: float  # rpm
    sidespin: float  # rpm
    spin_axis: float  # degrees
    spin_rate: float  # rpm (총 스핀율)
    is_valid: bool = True  # 데이터 유효성

@dataclass
class ClubData:
    """클럽 데이터"""
    speed: float  # m/s
    attack_angle: float  # degrees
    club_path: float  # degrees
    face_angle: float  # degrees
    face_to_path: float  # degrees
    is_valid: bool = True  # 데이터 유효성

class ShotDetector:
    """샷 감지 클래스 - IR 조명 기반 개선"""
    
    def __init__(self, threshold: int = 30, min_area: int = 100, 
                 detection_frames: int = 3, time_window: float = 0.1):
        """
        샷 감지기 초기화
        
        Args:
            threshold: 움직임 감지 임계값
            min_area: 최소 객체 면적
            detection_frames: 연속 감지 프레임 수
            time_window: 감지 시간 윈도우
        """
        self.threshold = threshold
        self.min_area = min_area
        self.detection_frames = detection_frames
        self.time_window = time_window
        
        # 배경 차감기 (IR 조명 최적화)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=50)
        
        # 상태 변수
        self.previous_frame = None
        self.motion_history = []
        self.shot_detected = False
        self.shot_start_time = None
        
        # IR 조명 기반 개선된 파라미터
        self.ir_threshold_low = 200  # IR 조명에서 골프공 밝기 하한
        self.ir_threshold_high = 255  # IR 조명에서 골프공 밝기 상한
        
    def detect_shot_event(self, frame: np.ndarray, timestamp: float) -> bool:
        """
        샷 이벤트 감지 (IR 조명 기반 개선)
        
        Args:
            frame: 입력 프레임
            timestamp: 타임스탬프
            
        Returns:
            샷 감지 여부
        """
        # IR 조명 기반 이진화
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # IR 조명에서 골프공은 매우 밝게 나타남
        ir_binary = cv2.inRange(gray, self.ir_threshold_low, self.ir_threshold_high)
        
        # 배경 차감
        fg_mask = self.bg_subtractor.apply(frame)
        
        # IR 이진화와 배경 차감 결합
        combined_mask = cv2.bitwise_and(ir_binary, fg_mask)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 골프공 크기 필터링
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < 2000:  # 골프공 예상 면적 범위
                # 원형도 검사
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:  # 원형에 가까운 객체만
                        valid_contours.append(contour)
        
        # 급격한 움직임 감지
        motion_detected = len(valid_contours) > 0
        
        # 움직임 이력 업데이트
        self.motion_history.append((timestamp, motion_detected))
        
        # 시간 윈도우 밖의 이력 제거
        self.motion_history = [(t, m) for t, m in self.motion_history 
                              if timestamp - t <= self.time_window]
        
        # 연속 감지 확인
        recent_detections = [m for t, m in self.motion_history if m]
        
        if len(recent_detections) >= self.detection_frames and not self.shot_detected:
            self.shot_detected = True
            self.shot_start_time = timestamp
            return True
        
        # 샷 종료 감지 (움직임이 멈춤)
        if self.shot_detected and len(recent_detections) == 0:
            self.shot_detected = False
            self.shot_start_time = None
        
        return False
    
    def reset(self):
        """감지기 리셋"""
        self.motion_history.clear()
        self.shot_detected = False
        self.shot_start_time = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=50)

class BallTracker:
    """골프공 추적 클래스 - 고정밀 추적"""
    
    def __init__(self):
        """골프공 추적기 초기화"""
        # 칼만 필터 설정 (3D 위치 + 속도)
        self.kalman = cv2.KalmanFilter(6, 3)
        
        # 상태 전이 행렬
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 측정 행렬
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # 노이즈 공분산
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
        
        # 추적 상태
        self.tracking_points = []
        self.is_tracking = False
        self.last_position = None
        
        # ORB 특징점 검출기 (스핀 측정용)
        self.orb = cv2.ORB_create(nfeatures=100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def detect_golf_ball(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        골프공 검출 (IR 조명 최적화)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            검출된 골프공 바운딩 박스 리스트 (x, y, w, h)
        """
        # IR 조명 기반 이진화
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        
        # 원형 객체 검출 (HoughCircles)
        circles = cv2.HoughCircles(
            binary, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=25
        )
        
        ball_candidates = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # 골프공 크기 필터링
                if 5 <= r <= 25:
                    # 바운딩 박스 계산
                    bbox = (x - r, y - r, 2 * r, 2 * r)
                    ball_candidates.append(bbox)
        
        return ball_candidates
    
    def update_tracking(self, position_3d: Tuple[float, float, float], 
                       timestamp: float) -> TrackingPoint:
        """
        3D 위치로 추적 업데이트
        
        Args:
            position_3d: 3D 위치 (x, y, z)
            timestamp: 타임스탬프
            
        Returns:
            업데이트된 추적 점
        """
        measurement = np.array(position_3d, dtype=np.float32).reshape(-1, 1)
        
        if not self.is_tracking:
            # 초기 상태 설정
            self.kalman.statePre = np.array([
                position_3d[0], position_3d[1], position_3d[2],
                0, 0, 0  # 초기 속도는 0
            ], dtype=np.float32).reshape(-1, 1)
            
            self.kalman.statePost = self.kalman.statePre.copy()
            self.is_tracking = True
        
        # 예측 단계
        predicted = self.kalman.predict()
        
        # 업데이트 단계
        corrected = self.kalman.correct(measurement)
        
        # 추적 점 생성
        tracking_point = TrackingPoint(
            x=corrected[0, 0],
            y=corrected[1, 0],
            z=corrected[2, 0],
            timestamp=timestamp,
            confidence=1.0
        )
        
        self.tracking_points.append(tracking_point)
        self.last_position = (tracking_point.x, tracking_point.y, tracking_point.z)
        
        return tracking_point
    
    def calculate_spin_rate(self, ball_images: List[np.ndarray], 
                          timestamps: List[float]) -> Tuple[float, Tuple[float, float, float]]:
        """
        스핀율 계산 (영상 기반 패턴 추적)
        
        Args:
            ball_images: 골프공 ROI 이미지 시퀀스
            timestamps: 타임스탬프 리스트
            
        Returns:
            (스핀율 RPM, 스핀축 벡터)
        """
        if len(ball_images) < 2:
            return 0.0, (0.0, 1.0, 0.0)
        
        spin_vectors = []
        
        for i in range(1, len(ball_images)):
            # 특징점 검출 (ORB 알고리즘)
            kp1, des1 = self.orb.detectAndCompute(ball_images[i-1], None)
            kp2, des2 = self.orb.detectAndCompute(ball_images[i], None)
            
            if des1 is not None and des2 is not None and len(des1) > 5 and len(des2) > 5:
                # 특징점 매칭
                matches = self.bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                if len(matches) > 5:
                    # 회전 벡터 계산
                    rotation_vector = self._calculate_rotation_from_matches(kp1, kp2, matches)
                    if rotation_vector is not None:
                        spin_vectors.append(rotation_vector)
        
        if spin_vectors:
            # 평균 스핀율 계산
            avg_rotation = np.mean(spin_vectors, axis=0)
            spin_rate = np.linalg.norm(avg_rotation) * 60 / (2 * np.pi)  # RPM 변환
            
            # 스핀축 정규화
            if np.linalg.norm(avg_rotation) > 0:
                spin_axis = avg_rotation / np.linalg.norm(avg_rotation)
            else:
                spin_axis = np.array([0.0, 1.0, 0.0])  # 기본 백스핀 축
            
            return spin_rate, tuple(spin_axis)
        
        return 0.0, (0.0, 1.0, 0.0)
    
    def _calculate_rotation_from_matches(self, kp1, kp2, matches) -> Optional[np.ndarray]:
        """매칭된 특징점으로부터 회전 벡터 계산"""
        if len(matches) < 5:
            return None
        
        # 매칭된 점들 추출
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
        
        # 호모그래피 계산
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                # 회전 성분 추출 (간단한 근사)
                rotation_angle = np.arctan2(H[1, 0], H[0, 0])
                rotation_vector = np.array([0, 0, rotation_angle])  # Z축 회전으로 근사
                
                return rotation_vector
        except:
            pass
        
        return None
    
    def get_trajectory(self) -> List[TrackingPoint]:
        """추적된 궤적 반환"""
        return self.tracking_points.copy()
    
    def reset(self):
        """추적기 리셋"""
        self.tracking_points.clear()
        self.is_tracking = False
        self.last_position = None

class ClubTracker:
    """클럽 추적 클래스 - ROI 기반 정밀 추적"""
    
    def __init__(self):
        """클럽 추적기 초기화"""
        # 칼만 필터 설정
        self.kalman = cv2.KalmanFilter(6, 3)
        
        # 상태 전이 행렬
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 측정 행렬
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # 노이즈 공분산
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.2
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.2
        
        # 추적 상태
        self.tracking_points = []
        self.is_tracking = False
        self.club_template = None
        self.last_position = None
        
    def detect_club_head(self, frame: np.ndarray, 
                        previous_position: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int, int, int]]:
        """
        클럽 헤드 검출 (다단계 검출 알고리즘)
        
        Args:
            frame: 입력 프레임
            previous_position: 이전 위치 (x, y, w, h)
            
        Returns:
            검출된 클럽 헤드 바운딩 박스 리스트
        """
        # 1단계: 에지 검출
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 2단계: 컨투어 검출
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 클럽 헤드 후보 필터링
        club_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # 클럽 헤드 예상 면적 범위
                # 종횡비 검사
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 3.0:
                    club_candidates.append((x, y, w, h))
        
        # 3단계: 이전 위치 기반 필터링
        if previous_position and club_candidates:
            prev_x, prev_y, prev_w, prev_h = previous_position
            prev_center = (prev_x + prev_w // 2, prev_y + prev_h // 2)
            
            # 가장 가까운 후보 선택
            best_candidate = None
            min_distance = float('inf')
            
            for candidate in club_candidates:
                x, y, w, h = candidate
                center = (x + w // 2, y + h // 2)
                distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_candidate = candidate
            
            return [best_candidate] if best_candidate else []
        
        return club_candidates
    
    def track_club_head_template(self, frame: np.ndarray, 
                               template: np.ndarray, 
                               previous_position: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        템플릿 매칭 기반 클럽 헤드 추적
        
        Args:
            frame: 현재 프레임
            template: 클럽 헤드 템플릿
            previous_position: 이전 위치
            
        Returns:
            새로운 클럽 헤드 위치
        """
        # 이전 위치 주변 ROI 설정
        roi_margin = 50
        x, y, w, h = previous_position
        
        roi_x1 = max(0, x - roi_margin)
        roi_y1 = max(0, y - roi_margin)
        roi_x2 = min(frame.shape[1], x + w + roi_margin)
        roi_y2 = min(frame.shape[0], y + h + roi_margin)
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size == 0:
            return None
        
        # 템플릿 매칭
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.7:  # 임계값 이상인 경우에만 유효
            # 전체 프레임 좌표로 변환
            global_x = roi_x1 + max_loc[0]
            global_y = roi_y1 + max_loc[1]
            
            return (global_x, global_y, template.shape[1], template.shape[0])
        
        return None
    
    def update_tracking(self, position_3d: Tuple[float, float, float], 
                       timestamp: float) -> TrackingPoint:
        """
        3D 위치로 추적 업데이트
        
        Args:
            position_3d: 3D 위치 (x, y, z)
            timestamp: 타임스탬프
            
        Returns:
            업데이트된 추적 점
        """
        measurement = np.array(position_3d, dtype=np.float32).reshape(-1, 1)
        
        if not self.is_tracking:
            # 초기 상태 설정
            self.kalman.statePre = np.array([
                position_3d[0], position_3d[1], position_3d[2],
                0, 0, 0  # 초기 속도는 0
            ], dtype=np.float32).reshape(-1, 1)
            
            self.kalman.statePost = self.kalman.statePre.copy()
            self.is_tracking = True
        
        # 예측 및 업데이트
        predicted = self.kalman.predict()
        corrected = self.kalman.correct(measurement)
        
        # 추적 점 생성
        tracking_point = TrackingPoint(
            x=corrected[0, 0],
            y=corrected[1, 0],
            z=corrected[2, 0],
            timestamp=timestamp,
            confidence=1.0
        )
        
        self.tracking_points.append(tracking_point)
        self.last_position = (tracking_point.x, tracking_point.y, tracking_point.z)
        
        return tracking_point
    
    def calculate_face_angle(self, club_head_roi: np.ndarray) -> float:
        """
        ROI 기반 클럽 페이스 각도 계산
        
        Args:
            club_head_roi: 클럽 헤드 ROI 이미지
            
        Returns:
            페이스 각도 (degrees)
        """
        if club_head_roi.size == 0:
            return 0.0
        
        # 클럽 페이스 영역 추출 (앞쪽 1/3 영역)
        h, w = club_head_roi.shape[:2]
        face_roi = club_head_roi[:, :w//3]
        
        # 적응형 밝기 보정
        face_roi = self._adaptive_brightness_correction(face_roi)
        
        # 에지 검출
        edges = cv2.Canny(face_roi, 30, 100)
        
        # 허프 변환으로 직선 검출
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None:
            # 가장 긴 직선을 페이스 라인으로 선택
            dominant_line = self._find_dominant_line(lines)
            if dominant_line is not None:
                rho, theta = dominant_line
                # 각도 계산 (타겟 라인 기준)
                face_angle = (theta - np.pi/2) * 180 / np.pi
                return face_angle
        
        return 0.0
    
    def _adaptive_brightness_correction(self, roi: np.ndarray) -> np.ndarray:
        """적응형 밝기 보정"""
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 히스토그램 평활화
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        corrected = clahe.apply(roi)
        
        # 가우시안 블러로 노이즈 제거
        smoothed = cv2.GaussianBlur(corrected, (3, 3), 0)
        
        return smoothed
    
    def _find_dominant_line(self, lines: np.ndarray) -> Optional[Tuple[float, float]]:
        """가장 지배적인 직선 찾기"""
        if lines is None or len(lines) == 0:
            return None
        
        # 첫 번째 직선을 반환 (실제로는 더 정교한 선택 알고리즘 필요)
        return lines[0][0]
    
    def detect_impact_moment(self, ball_positions: List[TrackingPoint]) -> int:
        """
        임팩트 순간 감지
        
        Args:
            ball_positions: 골프공 위치 리스트
            
        Returns:
            임팩트 프레임 인덱스
        """
        if not ball_positions or not self.tracking_points:
            return -1
        
        min_distance = float('inf')
        impact_frame = -1
        
        # 시간 동기화된 위치들 비교
        for i, club_point in enumerate(self.tracking_points):
            # 가장 가까운 시간의 볼 위치 찾기
            closest_ball = min(ball_positions, 
                             key=lambda bp: abs(bp.timestamp - club_point.timestamp))
            
            # 거리 계산
            distance = np.sqrt(
                (closest_ball.x - club_point.x)**2 +
                (closest_ball.y - club_point.y)**2 +
                (closest_ball.z - club_point.z)**2
            )
            
            if distance < min_distance:
                min_distance = distance
                impact_frame = i
        
        return impact_frame
    
    def get_trajectory(self) -> List[TrackingPoint]:
        """추적된 궤적 반환"""
        return self.tracking_points.copy()
    
    def reset(self):
        """추적기 리셋"""
        self.tracking_points.clear()
        self.is_tracking = False
        self.club_template = None
        self.last_position = None

