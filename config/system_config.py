"""
시스템 전체 설정
골프 스윙 분석 시스템 v4.4 - 통합 시스템 최적화
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import os

@dataclass
class SystemConfig:
    """시스템 설정 (820fps 대응)"""
    
    # 애플리케이션 정보
    app_name: str = "GolfSwingAnalysis_1440x300_GTX3050"
    app_version: str = "4.4.0"
    
    # 카메라 설정
    camera_fps: int = 820
    resolution: Tuple[int, int] = (1440, 300)
    vertical_baseline: float = 500.0  # mm
    camera1_height: float = 400.0     # mm (하단)
    camera2_height: float = 900.0     # mm (상단)
    inward_angle: float = 12.0        # degrees
    
    # 성능 설정
    max_cpu_usage: float = 70.0       # %
    gpu_acceleration: bool = True
    gpu_memory_limit_gb: float = 6.0  # GTX 3050 한계
    target_frame_time_ms: float = 1.22  # 820fps 목표
    
    # 정확도 목표
    accuracy_target: float = 0.95     # 95% 목표
    
    # ROI 설정
    roi_ball: Tuple[int, int, int, int] = (200, 150, 400, 300)  # x, y, w, h
    roi_club: Tuple[int, int, int, int] = (100, 100, 600, 400)
    
    # 추적 설정
    shot_detection_threshold: int = 30
    min_tracking_points: int = 10
    max_tracking_time: float = 2.0    # seconds
    
    # 멀티스레딩 설정
    worker_threads: int = 4
    frame_buffer_size: int = 15       # 18ms 분량
    
    # 데이터 경로 (통합 시스템 구조)
    data_root: str = "data"
    results_path: str = "data/results"
    images_path: str = "data/images"
    models_path: str = "data/models"
    
    # 통합 시스템 설정
    unified_analyzer_enabled: bool = True
    unified_processor_enabled: bool = True
    measurement_parameters: int = 13  # 6 볼 + 7 클럽 파라미터
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "camera_settings": {
                "fps": self.camera_fps,
                "resolution": list(self.resolution),
                "vertical_baseline": self.vertical_baseline,
                "camera1_height": self.camera1_height,
                "camera2_height": self.camera2_height,
                "inward_angle": self.inward_angle
            },
            "performance_settings": {
                "max_cpu_usage": self.max_cpu_usage,
                "gpu_acceleration": self.gpu_acceleration,
                "gpu_memory_limit_gb": self.gpu_memory_limit_gb,
                "target_frame_time_ms": self.target_frame_time_ms
            },
            "validation_settings": {
                "accuracy_target": self.accuracy_target
            }
        }
    
    @classmethod
    def create_directories(cls) -> None:
        """필요한 디렉토리 생성"""
        config = cls()
        directories = [
            config.data_root,
            config.results_path, 
            config.images_path,
            config.models_path,
            os.path.join(config.images_path, "shot-image"),
            os.path.join(config.images_path, "shot-image-jpg"),
            os.path.join(config.images_path, "shot-image-treated"),
            os.path.join(config.results_path, "analysis_results"),
            os.path.join(config.results_path, "debug_images"),
            os.path.join(config.models_path, "calibration")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)