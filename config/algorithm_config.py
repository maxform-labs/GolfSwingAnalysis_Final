"""
알고리즘 설정 - 고급 분석 알고리즘 파라미터
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np

@dataclass
class AlgorithmConfig:
    """알고리즘 설정 (820fps 최적화)"""
    
    # 칼만 필터 설정 (820fps 최적화)
    kalman_dt: float = 1/820  # 1.22ms
    process_noise: float = 0.005  # 고속 촬영용 저노이즈
    measurement_noise: float = 0.02  # 정밀 측정용 저노이즈
    
    # ROI 시스템 설정
    roi_stages: List[str] = ["FULL_SCREEN", "IMPACT_ZONE", "TRACKING", "FLIGHT_TRACKING"]
    roi_full_screen: Tuple[int, int, int, int] = (0, 0, 1440, 300)
    roi_impact_zone: Tuple[int, int, int, int] = (600, 100, 300, 150)
    roi_tracking: Tuple[int, int, int, int] = (500, 80, 400, 200)
    roi_flight_tracking: Tuple[int, int, int, int] = (300, 50, 800, 250)
    
    # 스핀 분석 설정 (820fps)
    spin_analysis_frames: int = 20  # 24.4ms 분량 (20/820)
    spin_pattern_matching_threshold: float = 0.7
    spin_rotation_detection_sensitivity: float = 0.5
    
    # 물리 검증 설정
    energy_conservation_tolerance: float = 0.05  # 5%
    trajectory_physics_tolerance: float = 0.03   # 3%
    spin_physics_tolerance: float = 0.08         # 8%
    
    # ML 보정 시스템
    ml_model_update_interval: int = 100  # frames
    confidence_threshold: float = 0.8
    outlier_detection_threshold: float = 3.0  # 3-sigma
    
    # 베이지안 앙상블 가중치
    estimator_weights: Dict[str, float] = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.estimator_weights is None:
            self.estimator_weights = {
                'AdvancedKalmanFilter': 0.25,
                'BayesianEstimator': 0.25,
                'KalmanEstimator': 0.20,
                'ParticleFilterEstimator': 0.15,
                'LeastSquaresEstimator': 0.15
            }
    
    # 정확도 목표 (95% 달성)
    accuracy_targets: Dict[str, float] = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.estimator_weights is None:
            self.estimator_weights = {
                'AdvancedKalmanFilter': 0.25,
                'BayesianEstimator': 0.25,
                'KalmanEstimator': 0.20,
                'ParticleFilterEstimator': 0.15,
                'LeastSquaresEstimator': 0.15
            }
        
        if self.accuracy_targets is None:
            self.accuracy_targets = {
                'ball_speed': 0.03,      # ±3.0%
                'launch_angle': 0.025,   # ±2.5%
                'direction_angle': 0.035, # ±3.5%
                'backspin': 0.08,        # ±8.0%
                'sidespin': 0.10,        # ±10.0%
                'spin_axis': 0.06,       # ±6.0%
                'club_speed': 0.035,     # ±3.5%
                'attack_angle': 0.045,   # ±4.5%
                'club_path': 0.035,      # ±3.5%
                'face_angle': 0.05       # ±5.0%
            }
    
    def get_kalman_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """칼만 필터 행렬 생성"""
        # 상태 전이 행렬 F (6x6: x,y,z,vx,vy,vz)
        F = np.eye(6)
        F[0, 3] = self.kalman_dt  # x += vx * dt
        F[1, 4] = self.kalman_dt  # y += vy * dt  
        F[2, 5] = self.kalman_dt  # z += vz * dt
        
        # 프로세스 노이즈 행렬 Q
        Q = np.eye(6) * self.process_noise
        
        # 측정 노이즈 행렬 R (3x3: x,y,z 측정)
        R = np.eye(3) * self.measurement_noise
        
        return F, Q, R
    
    def get_roi_by_stage(self, stage: str) -> Tuple[int, int, int, int]:
        """단계별 ROI 반환"""
        roi_map = {
            "FULL_SCREEN": self.roi_full_screen,
            "IMPACT_ZONE": self.roi_impact_zone,
            "TRACKING": self.roi_tracking,
            "FLIGHT_TRACKING": self.roi_flight_tracking
        }
        return roi_map.get(stage, self.roi_full_screen)
    
    def validate_accuracy(self, measurement_type: str, error_rate: float) -> bool:
        """정확도 목표 달성 여부 확인"""
        target = self.accuracy_targets.get(measurement_type, 0.1)
        return error_rate <= target