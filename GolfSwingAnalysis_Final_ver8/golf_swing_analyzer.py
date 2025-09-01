"""
골프 스윙 분석 메인 시스템
Author: Maxform 개발팀
Description: 수직 스테레오 비전 기반 실시간 골프 스윙 분석 시스템
- 240fps 실시간 처리
- 볼/클럽 데이터 정밀 분석
- 멀티스레딩 최적화
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from stereo_vision_vertical import VerticalStereoVision, VerticalStereoConfig, KalmanTracker3D
from object_tracker import ShotDetector, BallTracker, ClubTracker, BallData, ClubData
from ir_synchronization import IRLightController

@dataclass
class AnalysisResult:
    """분석 결과"""
    timestamp: str
    ball_data: Optional[BallData]
    club_data: Optional[ClubData]
    shot_detected: bool
    processing_time: float
    frame_count: int
    accuracy_confidence: Dict[str, float]  # 각 측정값의 신뢰도

@dataclass
class SystemConfig:
    """시스템 설정"""
    camera_fps: int = 240
    roi_ball: Tuple[int, int, int, int] = (200, 150, 400, 300)  # x, y, w, h
    roi_club: Tuple[int, int, int, int] = (100, 100, 600, 400)
    shot_detection_threshold: int = 30
    min_tracking_points: int = 10
    max_tracking_time: float = 2.0  # seconds
    calibration_file: str = "stereo_calibration.json"
    processing_threads: int = 4
    target_accuracy: Dict[str, float] = None  # 목표 정확도
    
    def __post_init__(self):
        if self.target_accuracy is None:
            self.target_accuracy = {
                'ball_speed': 4.0,      # ±4%
                'launch_angle': 3.0,    # ±3%
                'direction_angle': 4.0, # ±4%
                'club_speed': 4.0,      # ±4%
                'attack_angle': 5.0,    # ±5%
                'club_path': 4.0,       # ±4%
                'face_angle': 7.0       # ±7%
            }

class RealTimeProcessor:
    """실시간 처리 파이프라인"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.frame_queue = queue.Queue(maxsize=10)
        self.detection_queue = queue.Queue(maxsize=10)
        self.analysis_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
        self.executor = ThreadPoolExecutor(max_workers=config.processing_threads)
        self.running = False
        
        # 성능 모니터링
        self.performance_monitor = PerformanceMonitor()
        
    def start_processing(self):
        """처리 파이프라인 시작"""
        self.running = True
        
        # 각 처리 단계를 별도 스레드로 실행
        self.executor.submit(self.frame_capture_thread)
        self.executor.submit(self.object_detection_thread)
        self.executor.submit(self.stereo_analysis_thread)
        self.executor.submit(self.result_output_thread)
    
    def stop_processing(self):
        """처리 파이프라인 중지"""
        self.running = False
        self.executor.shutdown(wait=True)
    
    def frame_capture_thread(self):
        """프레임 캡처 스레드 (1ms 목표)"""
        while self.running:
            start_time = self.performance_monitor.start_timing('frame_capture')
            
            frame_pair = self.capture_synchronized_frames()
            if frame_pair and not self.frame_queue.full():
                self.frame_queue.put(frame_pair)
            
            self.performance_monitor.end_timing('frame_capture', start_time)
    
    def object_detection_thread(self):
        """객체 검출 스레드 (5ms 목표)"""
        while self.running:
            if not self.frame_queue.empty():
                start_time = self.performance_monitor.start_timing('object_detection')
                
                frame_pair = self.frame_queue.get()
                detections = self.detect_objects(frame_pair)
                if detections and not self.detection_queue.full():
                    self.detection_queue.put(detections)
                
                self.performance_monitor.end_timing('object_detection', start_time)
    
    def stereo_analysis_thread(self):
        """스테레오 분석 스레드 (8ms 목표)"""
        while self.running:
            if not self.detection_queue.empty():
                start_time = self.performance_monitor.start_timing('stereo_analysis')
                
                detections = self.detection_queue.get()
                analysis_result = self.perform_3d_analysis(detections)
                if analysis_result and not self.analysis_queue.full():
                    self.analysis_queue.put(analysis_result)
                
                self.performance_monitor.end_timing('stereo_analysis', start_time)
    
    def result_output_thread(self):
        """결과 출력 스레드 (2ms 목표)"""
        while self.running:
            if not self.analysis_queue.empty():
                start_time = self.performance_monitor.start_timing('result_output')
                
                result = self.analysis_queue.get()
                self.output_result(result)
                
                self.performance_monitor.end_timing('result_output', start_time)
    
    def capture_synchronized_frames(self):
        """동기화된 프레임 캡처 with IR 동기화"""
        try:
            start_time = time.perf_counter()
            
            # IR 신호 대기 (동기화)
            ir_signal = self.wait_for_ir_sync(timeout_ms=1)
            if not ir_signal:
                return None
            
            # 동시에 두 카메라에서 프레임 캡처
            upper_frame = self.upper_camera.get_frame()
            lower_frame = self.lower_camera.get_frame()
            
            # 타이밍 검증 (1ms 이내)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > 1.0:
                logger.warning(f"Frame capture exceeded 1ms: {elapsed_ms:.2f}ms")
            
            return (upper_frame, lower_frame, time.perf_counter())
            
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None
    
    def detect_objects_gpu(self, frame_pair):
        """GPU 가속 객체 검출 (5ms 제한)"""
        try:
            start_time = time.perf_counter()
            upper_frame, lower_frame, timestamp = frame_pair
            
            # GPU로 병렬 검출
            ball_roi_upper = self.ball_detector.detect_gpu(upper_frame)
            ball_roi_lower = self.ball_detector.detect_gpu(lower_frame)
            club_roi_upper = self.club_detector.detect_gpu(upper_frame)
            club_roi_lower = self.club_detector.detect_gpu(lower_frame)
            
            detections = {
                'ball': {'upper': ball_roi_upper, 'lower': ball_roi_lower},
                'club': {'upper': club_roi_upper, 'lower': club_roi_lower},
                'timestamp': timestamp
            }
            
            # 타이밍 검증 (5ms 이내)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > 5.0:
                raise TimeoutError(f"Object detection exceeded 5ms: {elapsed_ms:.2f}ms")
            
            return detections
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return None
    
    def perform_3d_analysis_gpu(self, detections):
        """GPU 가속 3D 분석 (8ms 제한)"""
        try:
            start_time = time.perf_counter()
            
            # 스테레오 비전으로 3D 좌표 계산
            ball_3d = self.calculate_ball_3d_coordinates(detections['ball'])
            club_3d = self.calculate_club_3d_coordinates(detections['club'])
            
            # 베이지안 앙상블로 추정 개선
            if ball_3d and len(self.ball_trajectory) > 0:
                ensemble = BayesianEnsemble()
                refined_ball, confidence = ensemble.estimate_with_ensemble(
                    self.ball_trajectory + [ball_3d]
                )
                if refined_ball is not None:
                    ball_3d = refined_ball
            
            # 물리 검증 적용
            if ball_3d:
                ball_3d = self.apply_physics_constraints(ball_3d)
            
            result = {
                'ball_3d': ball_3d,
                'club_3d': club_3d,
                'timestamp': detections['timestamp'],
                'confidence': confidence if 'confidence' in locals() else 0.8
            }
            
            # 타이밍 검증 (8ms 이내)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > 8.0:
                raise TimeoutError(f"3D analysis exceeded 8ms: {elapsed_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"3D analysis failed: {e}")
            return None
    
    def output_result(self, result):
        """결과 출력 with 2ms 제한"""
        try:
            start_time = time.perf_counter()
            
            # 결과 포맷팅
            formatted_result = self.format_swing_data(result)
            
            # 실시간 스트리밍 출력
            self.result_queue.put(formatted_result)
            
            # UI 업데이트 (비동기)
            if self.ui_callback:
                self.ui_callback(formatted_result)
            
            # 타이밍 검증 (2ms 이내)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > 2.0:
                logger.warning(f"Result output exceeded 2ms: {elapsed_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Result output failed: {e}")
    
    def wait_for_ir_sync(self, timeout_ms=1):
        """IR 동기화 신호 대기"""
        try:
            # IR 센서에서 신호 읽기 (1ms 이내)
            return self.ir_sensor.wait_for_signal(timeout_ms)
        except:
            return True  # IR이 없으면 즉시 진행
    
    def calculate_ball_3d_coordinates(self, ball_detections):
        """볼의 3D 좌표 계산"""
        if not ball_detections['upper'] or not ball_detections['lower']:
            return None
        
        # 스테레오 매칭으로 3D 좌표 계산
        return self.stereo_vision.calculate_3d_point(
            ball_detections['upper']['center'],
            ball_detections['lower']['center']
        )
    
    def calculate_club_3d_coordinates(self, club_detections):
        """클럽의 3D 좌표 계산"""
        if not club_detections['upper'] or not club_detections['lower']:
            return None
        
        return self.stereo_vision.calculate_3d_point(
            club_detections['upper']['center'],
            club_detections['lower']['center']
        )
    
    def apply_physics_constraints(self, position_3d):
        """물리적 제약 조건 적용"""
        x, y, z = position_3d
        
        # 물리적 범위 제한
        x = np.clip(x, -5000, 5000)  # ±5m
        y = np.clip(y, -1000, 3000)  # -1m ~ +3m  
        z = np.clip(z, 500, 50000)   # 0.5m ~ 50m
        
        return (x, y, z)
    
    def format_swing_data(self, result):
        """스윙 데이터 포맷팅"""
        return {
            'timestamp': result['timestamp'],
            'ball_position': result['ball_3d'],
            'club_position': result['club_3d'],
            'confidence': result['confidence'],
            'frame_id': getattr(self, 'frame_counter', 0)
        }

class GolfSwingAnalyzer:
    """골프 스윙 분석 메인 클래스"""
    
    def __init__(self, config: SystemConfig):
        """
        골프 스윙 분석기 초기화
        
        Args:
            config: 시스템 설정
        """
        self.config = config
        self.stereo_vision = VerticalStereoVision(VerticalStereoConfig())
        self.shot_detector = ShotDetector()
        self.ball_tracker = BallTracker()
        self.club_tracker = ClubTracker()
        self.ir_controller = IRLightController()
        
        # 칼만 필터 추적기
        self.ball_kalman = KalmanTracker3D()
        self.club_kalman = KalmanTracker3D()
        
        # 실시간 처리기
        self.real_time_processor = RealTimeProcessor(config)
        
        # 상태 변수
        self.is_analyzing = False
        self.current_shot_data = None
        self.analysis_results = []
        
        # 성능 모니터링
        self.performance_monitor = PerformanceMonitor()
        
        # 캘리브레이션 로드
        if os.path.exists(config.calibration_file):
            self.stereo_vision.load_calibration(config.calibration_file)
    
    def start_analysis(self):
        """분석 시작"""
        if not self.stereo_vision.is_calibrated:
            raise ValueError("스테레오 카메라가 캘리브레이션되지 않았습니다")
        
        self.is_analyzing = True
        self.ir_controller.turn_on()
        self.real_time_processor.start_processing()
        
        print("골프 스윙 분석 시작")
    
    def stop_analysis(self):
        """분석 중지"""
        self.is_analyzing = False
        self.ir_controller.turn_off()
        self.real_time_processor.stop_processing()
        
        print("골프 스윙 분석 중지")
    
    def analyze_ball_data(self, ball_positions: List[Tuple[float, float, float]], 
                         timestamps: List[float]) -> BallData:
        """
        볼 데이터 분석
        
        Args:
            ball_positions: 3D 볼 위치 리스트
            timestamps: 타임스탬프 리스트
            
        Returns:
            분석된 볼 데이터
        """
        if len(ball_positions) < 3:
            return None
        
        # 볼 스피드 계산 (칼만 필터 적용)
        ball_speed = self._calculate_ball_speed(ball_positions, timestamps)
        
        # 발사각 계산
        launch_angle = self._calculate_launch_angle(ball_positions)
        
        # 방향각 계산
        direction_angle = self._calculate_direction_angle(ball_positions)
        
        # 스핀 측정 (영상 기반)
        spin_rate, spin_axis = self._calculate_spin_rate(ball_positions)
        
        # 데이터 검증
        is_valid, validation_message = self._validate_ball_data(
            ball_speed, launch_angle, direction_angle, spin_rate)
        
        if not is_valid:
            print(f"볼 데이터 검증 실패: {validation_message}")
        
        return BallData(
            speed=ball_speed,
            launch_angle=launch_angle,
            direction_angle=direction_angle,
            spin_rate=spin_rate,
            spin_axis=spin_axis,
            is_valid=is_valid
        )
    
    def analyze_club_data(self, club_positions: List[Tuple[float, float, float]], 
                         timestamps: List[float], impact_frame: int) -> ClubData:
        """
        클럽 데이터 분석
        
        Args:
            club_positions: 3D 클럽 위치 리스트
            timestamps: 타임스탬프 리스트
            impact_frame: 임팩트 프레임 인덱스
            
        Returns:
            분석된 클럽 데이터
        """
        if len(club_positions) < 3 or impact_frame < 0:
            return None
        
        # 클럽 스피드 계산 (임팩트 순간)
        club_speed = self._calculate_club_speed(club_positions, timestamps, impact_frame)
        
        # 어택 앵글 계산
        attack_angle = self._calculate_attack_angle(club_positions, impact_frame)
        
        # 클럽 패스 계산
        club_path = self._calculate_club_path(club_positions, impact_frame)
        
        # 페이스 앵글 계산 (ROI 기반)
        face_angle = self._calculate_face_angle(club_positions, impact_frame)
        
        # 데이터 검증
        is_valid, validation_message = self._validate_club_data(
            club_speed, attack_angle, club_path, face_angle)
        
        if not is_valid:
            print(f"클럽 데이터 검증 실패: {validation_message}")
        
        return ClubData(
            speed=club_speed,
            attack_angle=attack_angle,
            club_path=club_path,
            face_angle=face_angle,
            is_valid=is_valid
        )
    
    def _calculate_ball_speed(self, positions: List[Tuple[float, float, float]], 
                            timestamps: List[float]) -> float:
        """볼 스피드 계산 (칼만 필터 적용)"""
        speeds = []
        for i in range(1, len(positions)):
            # 3D 거리 계산
            distance = np.sqrt(
                (positions[i][0] - positions[i-1][0])**2 +
                (positions[i][1] - positions[i-1][1])**2 +
                (positions[i][2] - positions[i-1][2])**2
            )
            
            # 시간 간격
            time_delta = timestamps[i] - timestamps[i-1]
            
            if time_delta > 0:
                speed = distance / time_delta  # m/s
                speeds.append(speed)
        
        # 칼만 필터 적용으로 노이즈 제거
        if speeds:
            return self._apply_kalman_filter(speeds)
        return 0.0
    
    def _calculate_launch_angle(self, positions: List[Tuple[float, float, float]]) -> float:
        """발사각 계산"""
        if len(positions) < 3:
            return 0.0
        
        # 초기 3개 점을 사용한 궤적 벡터 계산
        p1, p2, p3 = positions[:3]
        
        # 수평 거리와 수직 거리 계산
        horizontal_distance = np.sqrt((p3[0] - p1[0])**2 + (p3[2] - p1[2])**2)
        vertical_distance = p3[1] - p1[1]
        
        if horizontal_distance > 0:
            launch_angle = np.arctan2(vertical_distance, horizontal_distance) * 180 / np.pi
            return launch_angle
        return 0.0
    
    def _calculate_direction_angle(self, positions: List[Tuple[float, float, float]]) -> float:
        """방향각 계산"""
        if len(positions) < 3:
            return 0.0
        
        # 초기 궤적에서 좌우 방향 벡터 계산
        p1, p2, p3 = positions[:3]
        
        # 전진 방향 (X축)과 좌우 방향 (Z축) 계산
        forward_distance = p3[0] - p1[0]
        lateral_distance = p3[2] - p1[2]
        
        if forward_distance > 0:
            direction_angle = np.arctan2(lateral_distance, forward_distance) * 180 / np.pi
            return direction_angle
        return 0.0
    
    def _calculate_spin_rate(self, positions: List[Tuple[float, float, float]]) -> Tuple[float, Tuple[float, float, float]]:
        """스핀율 계산 (영상 기반 패턴 추적)"""
        # 영상 기반 스핀 측정은 복잡한 알고리즘이 필요
        # 여기서는 기본 구현만 제공
        spin_rate = 0.0  # RPM
        spin_axis = (0.0, 1.0, 0.0)  # 기본 백스핀 축
        
        # TODO: 실제 영상 기반 스핀 측정 알고리즘 구현
        # - 골프공 로고/패턴 추적
        # - 광학 흐름 분석
        # - 회전 벡터 계산
        
        return spin_rate, spin_axis
    
    def _calculate_club_speed(self, positions: List[Tuple[float, float, float]], 
                            timestamps: List[float], impact_frame: int) -> float:
        """클럽 스피드 계산 (임팩트 순간)"""
        if impact_frame < 1 or impact_frame >= len(positions):
            return 0.0
        
        # 임팩트 직전 속도 계산
        p1 = positions[impact_frame - 1]
        p2 = positions[impact_frame]
        
        distance = np.sqrt(
            (p2[0] - p1[0])**2 +
            (p2[1] - p1[1])**2 +
            (p2[2] - p1[2])**2
        )
        
        time_delta = timestamps[impact_frame] - timestamps[impact_frame - 1]
        
        if time_delta > 0:
            return distance / time_delta  # m/s
        return 0.0
    
    def _calculate_attack_angle(self, positions: List[Tuple[float, float, float]], 
                              impact_frame: int) -> float:
        """어택 앵글 계산"""
        if impact_frame < 2 or impact_frame >= len(positions):
            return 0.0
        
        # 임팩트 직전 3개 점 사용
        pre_impact_points = positions[impact_frame-2:impact_frame+1]
        
        if len(pre_impact_points) < 3:
            return 0.0
        
        p1, p2, p3 = pre_impact_points
        
        # 수평 거리와 수직 거리 계산
        horizontal_distance = np.sqrt((p3[0] - p1[0])**2 + (p3[2] - p1[2])**2)
        vertical_distance = p3[1] - p1[1]
        
        if horizontal_distance > 0:
            attack_angle = -np.arctan2(vertical_distance, horizontal_distance) * 180 / np.pi
            return attack_angle
        return 0.0
    
    def _calculate_club_path(self, positions: List[Tuple[float, float, float]], 
                           impact_frame: int) -> float:
        """클럽 패스 계산"""
        if impact_frame < 2 or impact_frame >= len(positions):
            return 0.0
        
        # 임팩트 직전 궤적 벡터 계산
        pre_impact_points = positions[impact_frame-2:impact_frame+1]
        
        if len(pre_impact_points) < 3:
            return 0.0
        
        p1, p2, p3 = pre_impact_points
        
        # 클럽 패스 벡터 (수평면 투영)
        path_vector_x = p3[0] - p1[0]
        path_vector_z = p3[2] - p1[2]
        
        # 타겟 라인과의 각도 계산 (타겟 라인은 X축 방향으로 가정)
        if path_vector_x > 0:
            club_path_angle = np.arctan2(path_vector_z, path_vector_x) * 180 / np.pi
            return club_path_angle
        return 0.0
    
    def _calculate_face_angle(self, positions: List[Tuple[float, float, float]], 
                            impact_frame: int) -> float:
        """페이스 앵글 계산 (ROI 기반)"""
        # 페이스 앵글은 영상 기반 ROI 분석이 필요
        # 여기서는 기본 구현만 제공
        face_angle = 0.0
        
        # TODO: 실제 ROI 기반 페이스 앵글 측정 알고리즘 구현
        # - 클럽 헤드 ROI 추출
        # - 에지 검출 및 직선 인식
        # - 페이스 라인 각도 계산
        
        return face_angle
    
    def _apply_kalman_filter(self, values: List[float]) -> float:
        """칼만 필터 적용"""
        if not values:
            return 0.0
        
        # 간단한 이동 평균 필터 (실제로는 칼만 필터 구현 필요)
        return sum(values) / len(values)
    
    def _validate_ball_data(self, speed: float, launch_angle: float, 
                          direction_angle: float, spin_rate: float) -> Tuple[bool, str]:
        """볼 데이터 검증"""
        # 물리적 제약 조건 검증
        if speed < 5 or speed > 80:  # m/s
            return False, f"비현실적인 볼 스피드: {speed:.1f} m/s"
        
        if abs(launch_angle) > 45:  # degrees
            return False, f"비현실적인 발사각: {launch_angle:.1f}°"
        
        if abs(direction_angle) > 30:  # degrees
            return False, f"비현실적인 방향각: {direction_angle:.1f}°"
        
        if spin_rate > speed * 100:  # RPM
            return False, f"비현실적인 스핀율: {spin_rate:.0f} RPM"
        
        return True, "검증 통과"
    
    def _validate_club_data(self, speed: float, attack_angle: float, 
                          club_path: float, face_angle: float) -> Tuple[bool, str]:
        """클럽 데이터 검증"""
        # 클럽 스피드 범위 검증
        if speed < 10 or speed > 60:  # m/s
            return False, f"비현실적인 클럽 스피드: {speed:.1f} m/s"
        
        # 어택 앵글 범위 검증
        if abs(attack_angle) > 15:  # degrees
            return False, f"비현실적인 어택 앵글: {attack_angle:.1f}°"
        
        # 클럽 패스 범위 검증
        if abs(club_path) > 20:  # degrees
            return False, f"비현실적인 클럽 패스: {club_path:.1f}°"
        
        # 페이스 앵글 범위 검증
        if abs(face_angle) > 30:  # degrees
            return False, f"비현실적인 페이스 앵글: {face_angle:.1f}°"
        
        return True, "검증 통과"
    
    def get_performance_report(self) -> Dict:
        """성능 리포트 반환"""
        return {
            'real_time_processor': self.real_time_processor.performance_monitor.get_performance_report(),
            'stereo_vision': self.stereo_vision.performance_monitor.get_performance_report()
        }
    
    def save_analysis_result(self, result: AnalysisResult, filename: str = None):
        """분석 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"golf_analysis_{timestamp}.json"
        
        result_dict = asdict(result)
        
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        print(f"분석 결과 저장: {filename}")

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        from collections import deque
        self.timing_data = {
            'frame_capture': deque(maxlen=100),
            'object_detection': deque(maxlen=100),
            'stereo_matching': deque(maxlen=100),
            'analysis': deque(maxlen=100),
            'total': deque(maxlen=100)
        }
    
    def start_timing(self, operation: str) -> float:
        """타이밍 시작"""
        return time.perf_counter()
    
    def end_timing(self, operation: str, start_time: float) -> float:
        """타이밍 종료"""
        elapsed = time.perf_counter() - start_time
        self.timing_data[operation].append(elapsed * 1000)  # ms 단위
        return elapsed
    
    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """성능 리포트 반환"""
        report = {}
        for operation, timings in self.timing_data.items():
            if timings:
                report[operation] = {
                    'average_ms': sum(timings) / len(timings),
                    'max_ms': max(timings),
                    'min_ms': min(timings),
                    'fps': 1000 / (sum(timings) / len(timings)) if timings else 0
                }
        return report

# 사용 예제
if __name__ == "__main__":
    # 시스템 설정
    config = SystemConfig(
        camera_fps=240,
        processing_threads=4
    )
    
    # 분석기 초기화
    analyzer = GolfSwingAnalyzer(config)
    
    try:
        # 분석 시작
        analyzer.start_analysis()
        
        # 분석 실행 (실제 구현에서는 카메라 입력 처리)
        print("골프 스윙 분석 시스템이 실행 중입니다...")
        print("Ctrl+C를 눌러 종료하세요.")
        
        while True:
            time.sleep(1)
            
            # 성능 리포트 출력 (선택사항)
            performance = analyzer.get_performance_report()
            # print(f"성능: {performance}")
            
    except KeyboardInterrupt:
        print("\n분석을 중지합니다...")
    finally:
        analyzer.stop_analysis()
        print("골프 스윙 분석 시스템이 종료되었습니다.")

