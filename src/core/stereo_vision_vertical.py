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
    """수직형 스테레오 구성 설정 - 실제 현장 설치 사양"""
    vertical_baseline: float = 500.0  # mm, 실제 카메라 간 수직 간격 (900-400)
    camera1_height: float = 400.0  # mm, 카메라1 바닥에서 높이
    camera2_height: float = 900.0  # mm, 카메라2 바닥에서 높이
    camera1_angle: float = 0.0  # degrees, 카메라1 각도 (수직)
    camera2_angle: float = 12.0  # degrees, 카메라2 각도 (후면으로 젖혀짐)
    tee_distance: float = 500.0  # mm, 카메라에서 Tee까지 거리
    calibration_area_x: Tuple[float, float] = (-200.0, 200.0)  # mm, 캘리브레이션 X 범위
    calibration_area_y: Tuple[float, float] = (-200.0, 200.0)  # mm, 캘리브레이션 Y 범위
    target_fps: int = 820  # 목표 FPS (발주사 요구사항)
    processing_threads: int = 4  # 처리 스레드 수
    # 1440x300 해상도 특화 파라미터
    image_resolution: Tuple[int, int] = (1440, 300)  # 발주사 gotkde해상도
    gpu_memory_limit: float = 6.0  # GB (GTX 3050 8GB 중 6GB 사용)

class KalmanTracker3D_820fps:
    """820fps 1440x300 최적화 3D 칼만 필터 추적기"""
    
    def __init__(self):
        self.fps = 820
        self.dt = 1.0 / 820  # 1.22ms per frame
        
        self.kalman = cv2.KalmanFilter(6, 3)  # 6 상태, 3 측정값
        
        # 820fps에 최적화된 상태 전이 행렬 (위치 + 속도)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, self.dt, 0,       0      ],
            [0, 1, 0, 0,       self.dt, 0      ],
            [0, 0, 1, 0,       0,       self.dt],
            [0, 0, 0, 1,       0,       0      ],
            [0, 0, 0, 0,       1,       0      ],
            [0, 0, 0, 0,       0,       1      ]
        ], dtype=np.float32)
        
        # 측정 행렬
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # 820fps 고속 촬영에 최적화된 노이즈 공분산 설정
        # 높은 프레임 레이트로 인한 정밀한 측정을 반영
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.005  # 더 낮은 프로세스 노이즈
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.02  # 더 낮은 측정 노이즈
        
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
        """GTX 3050 최적화 GPU 리소스 초기화"""
        if self.gpu_available:
            # GTX 3050 메모리 관리를 위한 작은 버퍼 사용
            self.gpu_frame_upper = cv2.cuda_GpuMat()
            self.gpu_frame_lower = cv2.cuda_GpuMat()
            self.gpu_disparity = cv2.cuda_GpuMat()
            
            # 1440x300 해상도에 최적화된 스테레오 매처
            # GTX 3050은 메모리가 제한적이므로 파라미터 조정
            self.stereo_matcher_gpu = cv2.cuda.createStereoBM(
                numDisparities=48,  # 64 -> 48로 축소 (메모리 절약)
                blockSize=11        # 15 -> 11로 축소 (속도 향상)
            )
            
            # GPU 메모리 사용량 모니터링
            self.gpu_memory_monitor = GPUMemoryMonitor()
            
            logger.info(f"GTX 3050 최적화 GPU 리소스 초기화 완료")
    
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
        
        # 1440x300 해상도와 GTX 3050에 최적화된 시차 계산 선택
        if self.gpu_available:
            disparity = self._calculate_disparity_gpu_1440x300_gtx3050(upper_image, lower_image)
        else:
            disparity = self._calculate_disparity_cpu_1440x300(upper_image, lower_image)
        
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
                    # 1440x300 최적화 Y축 시차 기반 깊이 계산
                    Z = self.calculate_vertical_disparity_depth_1440x300(x, y, d, fy, baseline)
                    
                    # X, Y 좌표
                    X = (x - cx) * Z / fx
                    Y = (y - cy) * Z / fy
                    
                    # 실제 현장 카메라 각도 보정
                    X_corrected, Y_corrected, Z_corrected = self._correct_camera_angles(X, Y, Z)
                    
                    points_3d.append((X_corrected, Y_corrected, Z_corrected))
                else:
                    points_3d.append((0.0, 0.0, 0.0))
            else:
                points_3d.append((0.0, 0.0, 0.0))
        
        return points_3d
    
    def calculate_vertical_disparity_depth_1440x300(self, x: float, y: float, disparity_y: float, 
                                                   fy: float, baseline: float) -> float:
        """
        1440x300 해상도 전용 Y축 시차를 이용한 깊이 계산 (PDF 명세 준수)
        공식: Z = (fy × baseline) / (y_top - y_bottom)
        
        1440x300 해상도 특성:
        - 세로 해상도 300픽셀로 제한되어 있음
        - 하지만 수직 스테레오에는 충분한 해상도
        - 시차 정밀도는 유지하면서 처리 속도 극대화
        
        Args:
            x: X 좌표 (1440픽셀 기준)
            y: Y 좌표 (300픽셀 기준)  
            disparity_y: Y축 시차값 (300픽셀 내에서)
            fy: Y방향 초점거리 (300픽셀 해상도 기준)
            baseline: 수직 기준선 거리
            
        Returns:
            계산된 깊이 (mm 단위)
        """
        # 300픽셀 해상도에 맞춘 최소 시차값 조정
        # 낮은 해상도를 고려하여 임계값 증가
        min_disparity_threshold = 0.2  # 0.1 -> 0.2로 증가
        
        if abs(disparity_y) < min_disparity_threshold:
            return float('inf')
        
        # Y축 시차 기반 깊이 계산 (1440x300 최적화)
        depth = (fy * baseline) / abs(disparity_y)
        
        # 1440x300 해상도 특성을 고려한 물리적 제약 조건
        # 세로 해상도가 낮아 원거리 정확도가 제한됨
        depth = np.clip(depth, 500.0, 20000.0)  # 50m -> 20m로 제한
        
        # 1440x300 해상도 보정 계수 적용
        resolution_correction = 300 / 1080  # 기준 해상도 대비 보정
        depth *= (1.0 + 0.1 * (1 - resolution_correction))  # 미세 조정
        
        return depth
    
    def _correct_camera_angles(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """실제 현장 설치 카메라 각도 보정
        
        카메라1: 바닥에서 400mm, 각도 0° (수직)
        카메라2: 바닥에서 900mm, 각도 12° (후면으로 젖혀짐)
        """
        # 카메라2의 12° 후면 각도 보정
        angle_rad = np.radians(self.config.camera2_angle)
        
        # Y축 중심 회전 (후면으로 젖혀진 각도)
        x_corrected = x * np.cos(angle_rad) - z * np.sin(angle_rad)
        y_corrected = y
        z_corrected = x * np.sin(angle_rad) + z * np.cos(angle_rad)
        
        # 높이 차이 보정 (500mm 수직 간격)
        height_offset = self.config.camera2_height - self.config.camera1_height  # 500mm
        y_corrected += height_offset * np.sin(angle_rad)
        
        return x_corrected, y_corrected, z_corrected
    
    def convert_to_tee_coordinates(self, points_3d: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """카메라 좌표계를 Tee 중심 좌표계로 변환
        
        실제 현장 설치 사양:
        - Tee는 카메라에서 정면으로 500mm 거리
        - 캘리브레이션 영역: Tee 중심 ±200mm (정사각형)
        
        Args:
            points_3d: 카메라 좌표계의 3D 점들
            
        Returns:
            Tee 중심 좌표계의 3D 점들
        """
        tee_centered_points = []
        
        for x_cam, y_cam, z_cam in points_3d:
            if x_cam == 0.0 and y_cam == 0.0 and z_cam == 0.0:
                tee_centered_points.append((0.0, 0.0, 0.0))
                continue
                
            # Tee 중심으로 좌표 이동
            # Z축: 카메라에서 Tee까지 500mm 거리를 기준점으로 설정
            x_tee = x_cam
            y_tee = y_cam - self.config.camera1_height  # 바닥 기준으로 변환
            z_tee = z_cam - self.config.tee_distance  # Tee를 기준점(0)으로 설정
            
            tee_centered_points.append((x_tee, y_tee, z_tee))
        
        return tee_centered_points
    
    def is_in_calibration_area(self, x_tee: float, y_tee: float) -> bool:
        """점이 캘리브레이션 영역 내부에 있는지 확인
        
        캘리브레이션 영역: Tee 중심 X축 ±200mm, Y축 ±200mm (정사각형)
        
        Args:
            x_tee, y_tee: Tee 중심 좌표계의 X, Y 좌표
            
        Returns:
            캘리브레이션 영역 내부 여부
        """
        x_min, x_max = self.config.calibration_area_x
        y_min, y_max = self.config.calibration_area_y
        
        return (x_min <= x_tee <= x_max) and (y_min <= y_tee <= y_max)
    
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
        
        # 1440x300 해상도에 맞춘 메모리 버퍼 할당
        for _ in range(pool_size):
            buffer = np.zeros((300, 1440, 3), dtype=np.uint8)  # 1440x300 해상도
            self.available_buffers.put(buffer)

class GPUMemoryMonitor:
    """GTX 3050 GPU 메모리 모니터링 클래스"""
    
    def __init__(self):
        self.max_memory_gb = 8.0  # GTX 3050 8GB VRAM
        self.safe_limit_gb = 6.0  # 안전 한계 6GB
        self.warning_threshold = 0.8  # 80% 경고 임계점
        
    def get_memory_usage(self) -> float:
        """현재 GPU 메모리 사용량 반환 (GB)"""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                free_memory = cv2.cuda.DeviceInfo().freeMemory() / (1024**3)  # GB
                used_memory = self.max_memory_gb - free_memory
                return used_memory
            return 0.0
        except:
            return 0.0
    
    def get_memory_usage_ratio(self) -> float:
        """메모리 사용률 반환 (0.0 ~ 1.0)"""
        used = self.get_memory_usage()
        return used / self.max_memory_gb
    
    def is_memory_warning(self) -> bool:
        """메모리 경고 상태 확인"""
        return self.get_memory_usage_ratio() > self.warning_threshold
    
    def cleanup_gpu_memory(self):
        """GPU 메모리 정리"""
        try:
            # OpenCV CUDA 메모리 정리
            cv2.cuda.setDevice(0)
            cv2.cuda.resetDevice()
            logger.info("GPU 메모리 정리 완료")
        except Exception as e:
            logger.warning(f"GPU 메모리 정리 실패: {e}")

class StereoVision1440x300Optimizer:
    """1440x300 해상도 전용 스테레오 비전 최적화 클래스"""
    
    def __init__(self, config: VerticalStereoConfig):
        self.config = config
        self.width = config.image_resolution[0]   # 1440
        self.height = config.image_resolution[1]  # 300
        
        # 1440x300 해상도 특성 분석
        self.aspect_ratio = self.width / self.height  # 4.8:1
        self.horizontal_precision = self.width / 1920   # 0.75 (75% 정밀도)
        self.vertical_precision = self.height / 1080    # 0.278 (28% 정밀도)
        
        # GPU 메모리 모니터
        self.gpu_monitor = GPUMemoryMonitor()
        
        logger.info(f"1440x300 해상도 최적화 초기화: 종횡비 {self.aspect_ratio:.1f}:1")
    
    def optimize_detection_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        1440x300 해상도에 최적화된 골프볼 검출 ROI 설정
        
        전략:
        - 수평 1440픽셀을 최대 활용
        - 수직 300픽셀에서 효율적인 중앙 영역 사용
        - 골프볼이 주로 나타나는 중앙 영역에 집중
        
        Returns:
            (x1, y1, x2, y2) ROI 좌표
        """
        # ROI 크기 설정 (1440x300 해상도 최적화)
        roi_width = min(600, self.width)    # 수평 600픽셀 (충분한 영역)
        roi_height = min(200, self.height)  # 수직 200픽셀 (300픽셀 중 중앙)
        
        # 중앙 위치 계산
        center_x = self.width // 2
        center_y = self.height // 2
        
        # ROI 좌표 계산
        x1 = max(0, center_x - roi_width // 2)
        x2 = min(self.width, center_x + roi_width // 2)
        y1 = max(0, center_y - roi_height // 2)
        y2 = min(self.height, center_y + roi_height // 2)
        
        return x1, y1, x2, y2
    
    def get_buffer(self) -> np.ndarray:
        """버퍼 가져오기"""
        if not self.available_buffers.empty():
            buffer = self.available_buffers.get()
            self.used_buffers.add(id(buffer))
            return buffer
        else:
            # 풀이 비어있으면 새로 할당 (1440x300)
            return np.zeros((300, 1440, 3), dtype=np.uint8)
    
    def return_buffer(self, buffer: np.ndarray):
        """버퍼 반환"""
        if id(buffer) in self.used_buffers:
            self.used_buffers.remove(id(buffer))
            self.available_buffers.put(buffer)

