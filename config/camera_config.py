"""
카메라 설정 - 820fps 수직 스테레오 비전
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

@dataclass
class CameraConfig:
    """카메라 설정 (820fps 수직 스테레오)"""
    
    # 기본 카메라 설정
    fps: int = 820
    resolution: Tuple[int, int] = (1440, 300)
    exposure_time_us: int = 1000  # 1ms (820fps용)
    gain: float = 2.0
    
    # 수직 스테레오 설정
    vertical_baseline: float = 500.0  # mm
    camera_separation: float = 500.0  # mm
    
    # 카메라 위치 (mm)
    camera1_position: Tuple[float, float, float] = (0, 0, 400)    # 하단
    camera2_position: Tuple[float, float, float] = (0, 0, 900)    # 상단
    
    # 카메라 회전 (degrees) - 내향 각도
    camera1_rotation: Tuple[float, float, float] = (0, 0, 12)     # 상향 12도
    camera2_rotation: Tuple[float, float, float] = (0, 0, -12)    # 하향 12도
    
    # 내부 파라미터 (1440x300 해상도)
    fx: float = 1200.0  # focal length x
    fy: float = 1200.0  # focal length y  
    cx: float = 720.0   # principal point x (1440/2)
    cy: float = 150.0   # principal point y (300/2)
    
    # 왜곡 계수
    k1: float = -0.1
    k2: float = 0.05
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    
    def get_camera_matrix(self) -> np.ndarray:
        """카메라 내부 파라미터 행렬"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy], 
            [0, 0, 1]
        ])
    
    def get_distortion_coeffs(self) -> np.ndarray:
        """왜곡 계수"""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])
    
    def get_rotation_matrix(self, camera_id: int) -> np.ndarray:
        """회전 행렬 (카메라 ID: 1=하단, 2=상단)"""
        if camera_id == 1:
            angles = self.camera1_rotation
        else:
            angles = self.camera2_rotation
        
        # Euler 각도를 회전 행렬로 변환
        rx, ry, rz = np.radians(angles)
        
        # Z-Y-X 회전 순서
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        return Rz @ Ry @ Rx
    
    def get_translation_vector(self, camera_id: int) -> np.ndarray:
        """변환 벡터 (카메라 ID: 1=하단, 2=상단)"""
        if camera_id == 1:
            return np.array(self.camera1_position)
        else:
            return np.array(self.camera2_position)
    
    def calculate_vertical_disparity_depth(self, y1: float, y2: float) -> float:
        """Y축 시차를 통한 깊이 계산 (1440x300 최적화)"""
        if abs(y1 - y2) < 1e-6:
            return float('inf')
        
        # Z = (fy × baseline) / (y_top - y_bottom)  
        depth = (self.fy * self.vertical_baseline) / (y1 - y2)
        return max(depth, 0)  # 음수 깊이 방지