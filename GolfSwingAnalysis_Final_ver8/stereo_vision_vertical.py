"""
수직형 스테레오 비전 시스템
Author: Maxform 개발팀
Description: 키오스크형 수직 배치 스테레오 카메라를 위한 3D 좌표 계산 시스템
- Y축 시차 기반 깊이 계산
- 12도 내향 각도 조정
- 실시간 240fps 처리 최적화
"""

import cv2
import numpy as np
import time
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from scipy.optimize import least_squares
import math
import threading
from concurrent.futures import ThreadPoolExecutor

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CameraParameters:
    """카메라 파라미터 클래스"""
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    image_size: Tuple[int, int]
    skew_parameter: float = 0.0  # 스큐 파라미터 추가

@dataclass
class VerticalStereoConfig:
    """수직형 스테레오 구성 설정"""
    vertical_baseline: float = 400.0  # mm, 수직 간격 (최적화)
    inward_angle: float = 12.0  # degrees, 내향 각도 (최적화)
    installation_height: float = 450.0  # mm, 설치 높이
    detection_zone_size: Tuple[float, float] = (400.0, 400.0)  # mm, 볼 인식 영역
    measurement_zone_size: Tuple[float, float] = (800.0, 800.0)  # mm, 볼 측정 영역
    max_distance_from_tee: float = 500.0  # mm, 티로부터 최대 거리
    target_fps: int = 240  # 목표 FPS
    processing_threads: int = 4  # 처리 스레드 수

class KalmanTracker3D:
    """3D 칼만 필터 추적기"""
    
    def __init__(self):
        self.kalman = cv2.KalmanFilter(6, 3)  # 6 상태, 3 측정값
        
        # 상태 전이 행렬 (위치 + 속도)
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
        
        # 노이즈 공분산 설정
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
        
    def predict(self):
        """상태 예측"""
        return self.kalman.predict()
    
    def update(self, measurement):
        """측정값으로 상태 업데이트"""
        self.kalman.correct(measurement.reshape(-1, 1))
        return self.kalman.statePost[:3].flatten()

class VerticalStereoVision:
    """수직형 스테레오 비전 시스템"""
    
    def __init__(self, config: VerticalStereoConfig):
        """
        수직형 스테레오 비전 시스템 초기화
        
        Args:
            config: 수직형 스테레오 구성 설정
        """
        self.config = config
        self.upper_camera_params = None
        self.lower_camera_params = None
        self.stereo_params = None
        self.is_calibrated = False
        
        # GPU 가속 지원 확인
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.gpu_available:
            logger.info("GPU 가속 사용 가능")
            self._init_gpu_resources()
        
        # 성능 모니터링
        self.performance_monitor = PerformanceMonitor()
        
        # 메모리 풀
        self.memory_pool = MemoryPool()
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=config.processing_threads)
        
    def _init_gpu_resources(self):
        """GPU 리소스 초기화"""
        if self.gpu_available:
            self.gpu_frame_upper = cv2.cuda_GpuMat()
            self.gpu_frame_lower = cv2.cuda_GpuMat()
            self.gpu_disparity = cv2.cuda_GpuMat()
            self.stereo_matcher_gpu = cv2.cuda.createStereoBM(numDisparities=64, blockSize=15)
    
    def calibrate_cameras(self, calibration_images: List[Tuple[np.ndarray, np.ndarray]], 
                         pattern_size: Tuple[int, int] = (9, 6)) -> bool:
        """
        수직 배치 스테레오 카메라 캘리브레이션
        
        Args:
            calibration_images: (상단 카메라, 하단 카메라) 이미지 쌍 리스트
            pattern_size: 체스보드 패턴 크기
            
        Returns:
            캘리브레이션 성공 여부
        """
        logger.info("수직 스테레오 카메라 캘리브레이션 시작")
        
        # 체스보드 3D 점 생성
        pattern_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        pattern_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        pattern_points *= 25.0  # 25mm 간격
        
        # 이미지 점과 객체 점 수집
        object_points = []
        upper_image_points = []
        lower_image_points = []
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        for upper_img, lower_img in calibration_images:
            # 그레이스케일 변환
            upper_gray = cv2.cvtColor(upper_img, cv2.COLOR_BGR2GRAY)
            lower_gray = cv2.cvtColor(lower_img, cv2.COLOR_BGR2GRAY)
            
            # 체스보드 코너 검출
            ret_upper, corners_upper = cv2.findChessboardCorners(upper_gray, pattern_size)
            ret_lower, corners_lower = cv2.findChessboardCorners(lower_gray, pattern_size)
            
            if ret_upper and ret_lower:
                # 서브픽셀 정확도로 코너 위치 개선
                corners_upper = cv2.cornerSubPix(upper_gray, corners_upper, (11, 11), (-1, -1), criteria)
                corners_lower = cv2.cornerSubPix(lower_gray, corners_lower, (11, 11), (-1, -1), criteria)
                
                object_points.append(pattern_points)
                upper_image_points.append(corners_upper)
                lower_image_points.append(corners_lower)
        
        if len(object_points) < 10:
            logger.error("캘리브레이션을 위한 충분한 이미지가 없습니다")
            return False
        
        # 개별 카메라 캘리브레이션
        image_size = calibration_images[0][0].shape[:2][::-1]
        
        # 상단 카메라 캘리브레이션
        ret_upper, mtx_upper, dist_upper, rvecs_upper, tvecs_upper = cv2.calibrateCamera(
            object_points, upper_image_points, image_size, None, None)
        
        # 하단 카메라 캘리브레이션
        ret_lower, mtx_lower, dist_lower, rvecs_lower, tvecs_lower = cv2.calibrateCamera(
            object_points, lower_image_points, image_size, None, None)
        
        # 스테레오 캘리브레이션
        ret_stereo, mtx_upper, dist_upper, mtx_lower, dist_lower, R, T, E, F = cv2.stereoCalibrate(
            object_points, upper_image_points, lower_image_points,
            mtx_upper, dist_upper, mtx_lower, dist_lower,
            image_size, flags=cv2.CALIB_FIX_INTRINSIC)
        
        # 재투영 오차 계산
        reprojection_error = self._calculate_reprojection_error(
            object_points, upper_image_points, mtx_upper, dist_upper, rvecs_upper, tvecs_upper)
        
        logger.info(f"재투영 오차: {reprojection_error:.3f} 픽셀")
        
        if reprojection_error > 0.5:
            logger.warning("재투영 오차가 높습니다. 캘리브레이션을 다시 수행하세요.")
        
        # 카메라 파라미터 저장
        self.upper_camera_params = CameraParameters(
            camera_matrix=mtx_upper,
            distortion_coeffs=dist_upper,
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            image_size=image_size
        )
        
        self.lower_camera_params = CameraParameters(
            camera_matrix=mtx_lower,
            distortion_coeffs=dist_lower,
            rotation_matrix=R,
            translation_vector=T,
            image_size=image_size
        )
        
        self.stereo_params = {
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'baseline': np.linalg.norm(T),
            'reprojection_error': reprojection_error
        }
        
        self.is_calibrated = True
        logger.info("수직 스테레오 카메라 캘리브레이션 완료")
        
        return True
    
    def _calculate_reprojection_error(self, object_points, image_points, 
                                    camera_matrix, dist_coeffs, rvecs, tvecs):
        """재투영 오차 계산"""
        total_error = 0
        total_points = 0
        
        for i in range(len(object_points)):
            projected_points, _ = cv2.projectPoints(
                object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            
            error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
            total_error += error
            total_points += len(object_points[i])
        
        return total_error / len(object_points)
    
    def calculate_vertical_disparity(self, upper_image: np.ndarray, 
                                   lower_image: np.ndarray) -> np.ndarray:
        """
        Y축 기반 시차 계산
        
        Args:
            upper_image: 상단 카메라 이미지
            lower_image: 하단 카메라 이미지
            
        Returns:
            Y축 시차 맵
        """
        start_time = self.performance_monitor.start_timing('stereo_matching')
        
        if self.gpu_available:
            disparity = self._calculate_disparity_gpu(upper_image, lower_image)
        else:
            disparity = self._calculate_disparity_cpu(upper_image, lower_image)
        
        self.performance_monitor.end_timing('stereo_matching', start_time)
        
        return disparity
    
    def _calculate_disparity_gpu(self, upper_image: np.ndarray, 
                               lower_image: np.ndarray) -> np.ndarray:
        """GPU를 사용한 시차 계산"""
        # 이미지를 GPU로 업로드
        self.gpu_frame_upper.upload(upper_image)
        self.gpu_frame_lower.upload(lower_image)
        
        # Y축 방향 시차 계산을 위한 이미지 회전
        gpu_upper_rotated = cv2.cuda.rotate(self.gpu_frame_upper, cv2.ROTATE_90_CLOCKWISE)
        gpu_lower_rotated = cv2.cuda.rotate(self.gpu_frame_lower, cv2.ROTATE_90_CLOCKWISE)
        
        # 스테레오 매칭
        self.stereo_matcher_gpu.compute(gpu_upper_rotated, gpu_lower_rotated, self.gpu_disparity)
        
        # 원래 방향으로 회전
        gpu_disparity_rotated = cv2.cuda.rotate(self.gpu_disparity, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # CPU로 다운로드
        disparity = gpu_disparity_rotated.download()
        
        return disparity.astype(np.float32) / 16.0
    
    def _calculate_disparity_cpu(self, upper_image: np.ndarray, 
                                lower_image: np.ndarray) -> np.ndarray:
        """CPU를 사용한 시차 계산"""
        # 스테레오 매칭 파라미터 설정
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        
        # Y축 방향 시차 계산을 위한 이미지 회전
        upper_rotated = cv2.rotate(upper_image, cv2.ROTATE_90_CLOCKWISE)
        lower_rotated = cv2.rotate(lower_image, cv2.ROTATE_90_CLOCKWISE)
        
        # 시차 맵 계산
        disparity = stereo.compute(upper_rotated, lower_rotated)
        
        # 원래 방향으로 회전
        disparity = cv2.rotate(disparity, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return disparity.astype(np.float32) / 16.0
    
    def calculate_3d_coordinates(self, disparity: np.ndarray, 
                               points_2d: List[Tuple[int, int]]) -> List[Tuple[float, float, float]]:
        """
        Y축 시차 기반 3D 좌표 계산
        
        Args:
            disparity: Y축 시차 맵
            points_2d: 2D 점 좌표 리스트
            
        Returns:
            3D 좌표 리스트
        """
        if not self.is_calibrated:
            raise ValueError("카메라가 캘리브레이션되지 않았습니다")
        
        points_3d = []
        
        fx = self.upper_camera_params.camera_matrix[0, 0]
        fy = self.upper_camera_params.camera_matrix[1, 1]
        cx = self.upper_camera_params.camera_matrix[0, 2]
        cy = self.upper_camera_params.camera_matrix[1, 2]
        baseline = self.config.vertical_baseline
        
        for x, y in points_2d:
            if 0 <= y < disparity.shape[0] and 0 <= x < disparity.shape[1]:
                d = disparity[y, x]
                
                if d > 0:  # 유효한 시차값
                    # Y축 시차 기반 깊이 계산 (수정된 공식)
                    Z = self.calculate_vertical_disparity_depth(x, y, d, fy, baseline)
                    
                    # X, Y 좌표
                    X = (x - cx) * Z / fx
                    Y = (y - cy) * Z / fy
                    
                    # 내향 각도 보정
                    X_corrected, Y_corrected, Z_corrected = self._correct_inward_angle(X, Y, Z)
                    
                    points_3d.append((X_corrected, Y_corrected, Z_corrected))
                else:
                    points_3d.append((0.0, 0.0, 0.0))
            else:
                points_3d.append((0.0, 0.0, 0.0))
        
        return points_3d
    
    def calculate_vertical_disparity_depth(self, x: float, y: float, disparity_y: float, 
                                         fy: float, baseline: float) -> float:
        """
        Y축 시차를 이용한 깊이 계산 (PDF 명세 준수)
        공식: Z = (fy × baseline) / (y_top - y_bottom)
        
        Args:
            x: X 좌표 (참조용)
            y: Y 좌표 (참조용)  
            disparity_y: Y축 시차값
            fy: Y방향 초점거리
            baseline: 수직 기준선 거리
            
        Returns:
            계산된 깊이 (mm 단위)
        """
        # 최소 시차값으로 0으로 나누기 방지
        if abs(disparity_y) < 0.1:
            return float('inf')
        
        # Y축 시차 기반 깊이 계산
        depth = (fy * baseline) / disparity_y
        
        # 물리적 제약 조건 적용 (0.5m ~ 50m)
        depth = np.clip(depth, 500.0, 50000.0)  # mm 단위
        
        return depth
    
    def _correct_inward_angle(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """내향 각도 보정"""
        angle_rad = np.radians(self.config.inward_angle)
        
        # 회전 행렬 적용
        x_corrected = x * np.cos(angle_rad) + z * np.sin(angle_rad)
        y_corrected = y
        z_corrected = -x * np.sin(angle_rad) + z * np.cos(angle_rad)
        
        return x_corrected, y_corrected, z_corrected
    
    def save_calibration(self, filename: str):
        """캘리브레이션 결과 저장"""
        if not self.is_calibrated:
            raise ValueError("캘리브레이션되지 않은 상태입니다")
        
        calibration_data = {
            'upper_camera': {
                'camera_matrix': self.upper_camera_params.camera_matrix.tolist(),
                'distortion_coeffs': self.upper_camera_params.distortion_coeffs.tolist(),
                'image_size': self.upper_camera_params.image_size
            },
            'lower_camera': {
                'camera_matrix': self.lower_camera_params.camera_matrix.tolist(),
                'distortion_coeffs': self.lower_camera_params.distortion_coeffs.tolist(),
                'rotation_matrix': self.lower_camera_params.rotation_matrix.tolist(),
                'translation_vector': self.lower_camera_params.translation_vector.tolist(),
                'image_size': self.lower_camera_params.image_size
            },
            'stereo_params': {
                'R': self.stereo_params['R'].tolist(),
                'T': self.stereo_params['T'].tolist(),
                'E': self.stereo_params['E'].tolist(),
                'F': self.stereo_params['F'].tolist(),
                'baseline': self.stereo_params['baseline'],
                'reprojection_error': self.stereo_params['reprojection_error']
            },
            'config': asdict(self.config)
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"캘리브레이션 데이터 저장: {filename}")
    
    def load_calibration(self, filename: str):
        """캘리브레이션 결과 로드"""
        with open(filename, 'r') as f:
            calibration_data = json.load(f)
        
        # 상단 카메라 파라미터
        upper_data = calibration_data['upper_camera']
        self.upper_camera_params = CameraParameters(
            camera_matrix=np.array(upper_data['camera_matrix']),
            distortion_coeffs=np.array(upper_data['distortion_coeffs']),
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            image_size=tuple(upper_data['image_size'])
        )
        
        # 하단 카메라 파라미터
        lower_data = calibration_data['lower_camera']
        self.lower_camera_params = CameraParameters(
            camera_matrix=np.array(lower_data['camera_matrix']),
            distortion_coeffs=np.array(lower_data['distortion_coeffs']),
            rotation_matrix=np.array(lower_data['rotation_matrix']),
            translation_vector=np.array(lower_data['translation_vector']),
            image_size=tuple(lower_data['image_size'])
        )
        
        # 스테레오 파라미터
        stereo_data = calibration_data['stereo_params']
        self.stereo_params = {
            'R': np.array(stereo_data['R']),
            'T': np.array(stereo_data['T']),
            'E': np.array(stereo_data['E']),
            'F': np.array(stereo_data['F']),
            'baseline': stereo_data['baseline'],
            'reprojection_error': stereo_data['reprojection_error']
        }
        
        self.is_calibrated = True
        logger.info(f"캘리브레이션 데이터 로드: {filename}")

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
    
    def get_average_timing(self, operation: str) -> float:
        """평균 타이밍 반환"""
        if self.timing_data[operation]:
            return sum(self.timing_data[operation]) / len(self.timing_data[operation])
        return 0
    
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

class MemoryPool:
    """메모리 풀 관리 클래스"""
    
    def __init__(self, pool_size: int = 20):
        import queue
        self.pool_size = pool_size
        self.available_buffers = queue.Queue()
        self.used_buffers = set()
        
        # 미리 메모리 버퍼 할당
        for _ in range(pool_size):
            buffer = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.available_buffers.put(buffer)
    
    def get_buffer(self) -> np.ndarray:
        """버퍼 가져오기"""
        if not self.available_buffers.empty():
            buffer = self.available_buffers.get()
            self.used_buffers.add(id(buffer))
            return buffer
        else:
            # 풀이 비어있으면 새로 할당
            return np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    def return_buffer(self, buffer: np.ndarray):
        """버퍼 반환"""
        if id(buffer) in self.used_buffers:
            self.used_buffers.remove(id(buffer))
            self.available_buffers.put(buffer)

