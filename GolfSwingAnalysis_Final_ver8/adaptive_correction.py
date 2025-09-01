"""
적응형 보정 시스템 (Adaptive Correction System)
골퍼의 스킬 레벨과 측정 품질에 따른 동적 보정 구현
"""

import numpy as np
from typing import Dict, Tuple, Any, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SkillLevel(Enum):
    """골퍼 스킬 레벨"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"


@dataclass
class SkillProfile:
    """스킬 기반 보정 프로필"""
    speed_factor: float
    accuracy_threshold: float
    noise_tolerance: float
    measurement_confidence: float
    expected_consistency: float


class AdaptiveCorrector:
    """적응형 보정기 - 스킬 레벨 기반 측정값 보정"""
    
    def __init__(self):
        """적응형 보정기 초기화"""
        self.skill_profiles = self._initialize_skill_profiles()
        self.measurement_history = []
        self.current_skill_level = SkillLevel.INTERMEDIATE
        self.adaptive_noise_params = self._initialize_noise_params()
    
    def _initialize_skill_profiles(self) -> Dict[SkillLevel, SkillProfile]:
        """스킬 프로필 초기화"""
        return {
            SkillLevel.BEGINNER: SkillProfile(
                speed_factor=0.8,
                accuracy_threshold=0.15,
                noise_tolerance=0.3,
                measurement_confidence=0.6,
                expected_consistency=0.4
            ),
            SkillLevel.INTERMEDIATE: SkillProfile(
                speed_factor=0.9,
                accuracy_threshold=0.10,
                noise_tolerance=0.2,
                measurement_confidence=0.75,
                expected_consistency=0.65
            ),
            SkillLevel.ADVANCED: SkillProfile(
                speed_factor=1.0,
                accuracy_threshold=0.05,
                noise_tolerance=0.1,
                measurement_confidence=0.85,
                expected_consistency=0.8
            ),
            SkillLevel.PROFESSIONAL: SkillProfile(
                speed_factor=1.1,
                accuracy_threshold=0.03,
                noise_tolerance=0.05,
                measurement_confidence=0.95,
                expected_consistency=0.9
            )
        }
    
    def _initialize_noise_params(self) -> Dict[str, Dict[str, float]]:
        """적응형 노이즈 파라미터 초기화"""
        return {
            'kalman': {
                'process_noise_base': 0.01,
                'measurement_noise_base': 0.1,
                'adaptation_rate': 0.1
            },
            'measurement_quality': {
                'high_quality_threshold': 0.9,
                'low_quality_threshold': 0.7,
                'noise_scale_factor': 2.0
            }
        }
    
    def set_skill_level(self, skill_level: SkillLevel):
        """스킬 레벨 설정"""
        self.current_skill_level = skill_level
        logger.info(f"스킬 레벨 설정: {skill_level.value}")
    
    def apply_skill_correction(self, measurement: np.ndarray, 
                             measurement_type: str = 'general') -> np.ndarray:
        """
        스킬 레벨 기반 측정값 보정
        
        Args:
            measurement: 원본 측정값 [x, y, z] 또는 스칼라
            measurement_type: 측정값 타입 ('speed', 'angle', 'position', 'general')
            
        Returns:
            보정된 측정값
        """
        profile = self.skill_profiles[self.current_skill_level]
        
        if isinstance(measurement, (int, float)):
            return self._correct_scalar(measurement, profile, measurement_type)
        else:
            return self._correct_vector(measurement, profile, measurement_type)
    
    def _correct_scalar(self, value: float, profile: SkillProfile, 
                       measurement_type: str) -> float:
        """스칼라 값 보정"""
        corrected = value * profile.speed_factor
        
        # 측정값 타입별 특수 보정
        if measurement_type == 'speed':
            # 초보자는 속도가 과대 추정되는 경향
            if self.current_skill_level == SkillLevel.BEGINNER:
                corrected *= 0.85
        elif measurement_type == 'angle':
            # 각도 측정의 불확실성 보정
            if abs(corrected - value) > profile.accuracy_threshold:
                corrected = value + 0.1 * (corrected - value)
        
        return corrected
    
    def _correct_vector(self, vector: np.ndarray, profile: SkillProfile,
                       measurement_type: str) -> np.ndarray:
        """벡터 값 보정"""
        corrected = vector * profile.speed_factor
        
        # 임계값 기반 점진적 보정
        diff_magnitude = np.linalg.norm(corrected - vector)
        if diff_magnitude > profile.accuracy_threshold:
            # 점진적 보정 (급격한 변화 방지)
            correction_rate = min(0.1, profile.accuracy_threshold / diff_magnitude)
            corrected = vector + correction_rate * (corrected - vector)
        
        return corrected
    
    def adaptive_noise_tuning(self, measurement_quality: float) -> Dict[str, float]:
        """
        측정 품질 기반 적응형 노이즈 튜닝
        
        Args:
            measurement_quality: 측정 품질 점수 (0.0 ~ 1.0)
            
        Returns:
            조정된 노이즈 파라미터
        """
        params = self.adaptive_noise_params
        profile = self.skill_profiles[self.current_skill_level]
        
        # 측정 품질에 따른 노이즈 조정
        if measurement_quality < params['measurement_quality']['low_quality_threshold']:
            # 품질이 낮으면 측정 노이즈 증가
            q_factor = params['kalman']['process_noise_base'] * 2.0
            r_factor = params['kalman']['measurement_noise_base'] * params['measurement_quality']['noise_scale_factor']
        elif measurement_quality > params['measurement_quality']['high_quality_threshold']:
            # 품질이 높으면 측정 노이즈 감소
            q_factor = params['kalman']['process_noise_base'] * 0.5
            r_factor = params['kalman']['measurement_noise_base'] * 0.5
        else:
            # 일반적인 품질
            q_factor = params['kalman']['process_noise_base']
            r_factor = params['kalman']['measurement_noise_base']
        
        # 스킬 레벨 기반 추가 조정
        q_factor *= (1.0 - profile.measurement_confidence * 0.3)
        r_factor *= (1.0 - profile.measurement_confidence * 0.5)
        
        return {
            'process_noise': q_factor,
            'measurement_noise': r_factor,
            'measurement_quality': measurement_quality
        }
    
    def update_measurement_history(self, measurement: np.ndarray, 
                                 measurement_quality: float):
        """측정값 히스토리 업데이트"""
        self.measurement_history.append({
            'measurement': measurement,
            'quality': measurement_quality,
            'timestamp': np.datetime64('now')
        })
        
        # 히스토리 크기 제한 (최근 100개)
        if len(self.measurement_history) > 100:
            self.measurement_history.pop(0)
    
    def detect_skill_level_automatically(self) -> SkillLevel:
        """
        측정값 히스토리 기반 자동 스킬 레벨 감지
        
        Returns:
            감지된 스킬 레벨
        """
        if len(self.measurement_history) < 10:
            return SkillLevel.INTERMEDIATE
        
        # 최근 측정값들 분석
        recent_measurements = self.measurement_history[-20:]
        
        # 일관성 계산 (표준편차 기반)
        if len(recent_measurements) > 0:
            qualities = [m['quality'] for m in recent_measurements]
            measurements = [m['measurement'] for m in recent_measurements]
            
            avg_quality = np.mean(qualities)
            
            # 측정값 변동성 계산 (벡터인 경우 크기 사용)
            if len(measurements) > 1:
                if isinstance(measurements[0], np.ndarray):
                    magnitudes = [np.linalg.norm(m) for m in measurements]
                else:
                    magnitudes = measurements
                
                consistency = 1.0 - (np.std(magnitudes) / np.mean(magnitudes))
            else:
                consistency = 0.5
            
            # 스킬 레벨 결정
            combined_score = (avg_quality + consistency) / 2
            
            if combined_score > 0.85:
                return SkillLevel.PROFESSIONAL
            elif combined_score > 0.7:
                return SkillLevel.ADVANCED
            elif combined_score > 0.5:
                return SkillLevel.INTERMEDIATE
            else:
                return SkillLevel.BEGINNER
        
        return SkillLevel.INTERMEDIATE
    
    def apply_physical_constraints(self, measurement: np.ndarray, 
                                 constraint_type: str = 'position') -> np.ndarray:
        """
        물리적 제약 조건 적용
        
        Args:
            measurement: 측정값
            constraint_type: 제약 타입 ('position', 'velocity', 'acceleration')
            
        Returns:
            제약이 적용된 측정값
        """
        if constraint_type == 'position':
            # 위치 제약 (골프 스윙 범위 내)
            if len(measurement) >= 3:
                x, y, z = measurement[:3]
                x = np.clip(x, -5000, 5000)  # ±5m
                y = np.clip(y, -1000, 3000)  # -1m ~ +3m
                z = np.clip(z, 500, 50000)   # 0.5m ~ 50m
                
                result = measurement.copy()
                result[:3] = [x, y, z]
                return result
        
        elif constraint_type == 'velocity':
            # 속도 제약
            if isinstance(measurement, (int, float)):
                # 스칼라 속도
                return np.clip(measurement, 0, 80)  # 0 ~ 80 m/s
            else:
                # 벡터 속도
                speed = np.linalg.norm(measurement)
                if speed > 80:  # 최대 속도 제한
                    return measurement * (80 / speed)
                return measurement
        
        elif constraint_type == 'acceleration':
            # 가속도 제약
            if isinstance(measurement, (int, float)):
                return np.clip(measurement, -500, 500)  # ±500 m/s²
            else:
                accel_magnitude = np.linalg.norm(measurement)
                if accel_magnitude > 500:
                    return measurement * (500 / accel_magnitude)
                return measurement
        
        return measurement
    
    def get_correction_confidence(self, original: np.ndarray, 
                                corrected: np.ndarray) -> float:
        """
        보정 신뢰도 계산
        
        Args:
            original: 원본 측정값
            corrected: 보정된 측정값
            
        Returns:
            보정 신뢰도 (0.0 ~ 1.0)
        """
        profile = self.skill_profiles[self.current_skill_level]
        
        if isinstance(original, (int, float)):
            change_ratio = abs(corrected - original) / (abs(original) + 1e-6)
        else:
            change_ratio = np.linalg.norm(corrected - original) / (np.linalg.norm(original) + 1e-6)
        
        # 변화량이 클수록 신뢰도 감소
        confidence = max(0.0, 1.0 - change_ratio / profile.accuracy_threshold)
        
        # 스킬 레벨 기반 가중치 적용
        confidence *= profile.measurement_confidence
        
        return min(1.0, confidence)
    
    def get_adaptive_parameters(self) -> Dict[str, Any]:
        """현재 적응형 파라미터 반환"""
        profile = self.skill_profiles[self.current_skill_level]
        
        return {
            'skill_level': self.current_skill_level.value,
            'speed_factor': profile.speed_factor,
            'accuracy_threshold': profile.accuracy_threshold,
            'noise_tolerance': profile.noise_tolerance,
            'measurement_confidence': profile.measurement_confidence,
            'expected_consistency': profile.expected_consistency,
            'measurement_history_size': len(self.measurement_history)
        }


class MeasurementQualityEstimator:
    """측정 품질 평가기"""
    
    def __init__(self):
        """측정 품질 평가기 초기화"""
        self.quality_history = []
    
    def estimate_quality(self, measurement: np.ndarray, 
                        stereo_confidence: float = 0.8,
                        detection_confidence: float = 0.8,
                        tracking_consistency: float = 0.8) -> float:
        """
        종합적인 측정 품질 평가
        
        Args:
            measurement: 측정값
            stereo_confidence: 스테레오 매칭 신뢰도
            detection_confidence: 객체 검출 신뢰도  
            tracking_consistency: 추적 일관성
            
        Returns:
            측정 품질 점수 (0.0 ~ 1.0)
        """
        # 기본 품질 점수 (가중 평균)
        base_quality = (
            stereo_confidence * 0.4 +
            detection_confidence * 0.3 +
            tracking_consistency * 0.3
        )
        
        # 측정값 자체의 품질 평가
        measurement_quality = self._evaluate_measurement_quality(measurement)
        
        # 히스토리 기반 일관성 평가
        consistency_quality = self._evaluate_consistency()
        
        # 최종 품질 점수
        final_quality = (
            base_quality * 0.5 +
            measurement_quality * 0.3 +
            consistency_quality * 0.2
        )
        
        # 품질 히스토리 업데이트
        self.quality_history.append(final_quality)
        if len(self.quality_history) > 50:
            self.quality_history.pop(0)
        
        return np.clip(final_quality, 0.0, 1.0)
    
    def _evaluate_measurement_quality(self, measurement: np.ndarray) -> float:
        """측정값 자체의 품질 평가"""
        if isinstance(measurement, (int, float)):
            # 스칼라 값의 품질 평가
            if np.isnan(measurement) or np.isinf(measurement):
                return 0.0
            
            # 합리적인 범위 내인지 확인
            if 0 <= measurement <= 100:  # 적절한 범위
                return 1.0
            else:
                return 0.5
        
        else:
            # 벡터 값의 품질 평가
            if np.any(np.isnan(measurement)) or np.any(np.isinf(measurement)):
                return 0.0
            
            # 벡터 크기 확인
            magnitude = np.linalg.norm(measurement)
            if 0.1 <= magnitude <= 1000:  # 적절한 범위
                return 1.0
            else:
                return 0.5
    
    def _evaluate_consistency(self) -> float:
        """히스토리 기반 일관성 평가"""
        if len(self.quality_history) < 3:
            return 0.5
        
        recent_qualities = self.quality_history[-5:]
        consistency = 1.0 - np.std(recent_qualities)
        
        return np.clip(consistency, 0.0, 1.0)


# 사용 예제
if __name__ == "__main__":
    # 적응형 보정기 초기화
    corrector = AdaptiveCorrector()
    quality_estimator = MeasurementQualityEstimator()
    
    # 스킬 레벨 설정
    corrector.set_skill_level(SkillLevel.INTERMEDIATE)
    
    # 측정값 보정 예제
    original_measurement = np.array([1.5, 0.3, 5.2])
    
    # 측정 품질 평가
    quality = quality_estimator.estimate_quality(
        original_measurement,
        stereo_confidence=0.85,
        detection_confidence=0.9,
        tracking_consistency=0.8
    )
    
    # 적응형 노이즈 튜닝
    noise_params = corrector.adaptive_noise_tuning(quality)
    print(f"노이즈 파라미터: {noise_params}")
    
    # 스킬 기반 보정 적용
    corrected_measurement = corrector.apply_skill_correction(
        original_measurement, 'position'
    )
    
    # 물리적 제약 조건 적용
    final_measurement = corrector.apply_physical_constraints(
        corrected_measurement, 'position'
    )
    
    # 보정 신뢰도 계산
    confidence = corrector.get_correction_confidence(
        original_measurement, final_measurement
    )
    
    print(f"원본 측정값: {original_measurement}")
    print(f"보정된 측정값: {final_measurement}")
    print(f"측정 품질: {quality:.3f}")
    print(f"보정 신뢰도: {confidence:.3f}")
    
    # 현재 적응형 파라미터 출력
    params = corrector.get_adaptive_parameters()
    print(f"적응형 파라미터: {params}")