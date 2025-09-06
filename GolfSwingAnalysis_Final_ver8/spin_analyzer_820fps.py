#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
820fps 기반 볼 스핀 분석 모듈

이 모듈은 820fps 고속 촬영으로 획득한 연속 프레임에서
백스핀, 사이드스핀, 스핀축을 정밀 분석합니다.

목표 정확도:
- 백스핀: ±8% (기존 ±12% 대비 개선)
- 사이드스핀: ±10% (기존 ±15% 대비 개선)  
- 스핀축: ±6% (기존 ±10% 대비 개선)
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import threading
from concurrent.futures import ThreadPoolExecutor


@dataclass
class SpinData:
    """스핀 데이터 구조체"""
    backspin: float  # RPM
    sidespin: float  # RPM (양수=시계방향, 음수=반시계방향)
    spin_axis: float  # 도(°)
    total_spin: float  # RPM
    confidence: float  # 신뢰도 (0-1)
    processing_time: float  # 처리 시간 (ms)


class BallSpinDetector820fps:
    """820fps 환경에서의 볼 스핀 검출기"""
    
    def __init__(self, config_path: str = None):
        self.fps = 820
        self.frame_interval = 1.0 / 820  # 1.22ms
        
        # ORB 특징점 검출기 (820fps 최적화)
        self.pattern_detector = cv2.ORB_create(
            nfeatures=300,  # 특징점 수 최적화
            scaleFactor=1.1,
            nlevels=4,  # 레벨 축소로 속도 향상
            edgeThreshold=15,  # 경계 임계값 조정
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE
        )
        
        # 매처 설정
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 스핀 계산 매개변수
        self.spin_params = {
            'min_matches': 8,  # 최소 매칭점 수
            'max_matches': 50,  # 최대 매칭점 수
            'ransac_threshold': 2.0,
            'confidence_threshold': 0.7,
            'backspin_correction': 1.05,  # 820fps 보정 계수
            'sidespin_sensitivity': 1.08,  # 사이드스핀 민감도
            'spin_axis_precision': 1.02   # 스핀축 정밀도
        }
        
        # GPU 가속화 설정
        self.gpu_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.gpu_enabled:
            self.gpu_detector = cv2.cuda_ORB.create(nfeatures=300)
        
        # 이전 프레임 데이터 캐싱
        self.frame_cache = []
        self.max_cache_size = 10
        
        # 성능 메트릭
        self.performance_metrics = {
            'total_frames': 0,
            'successful_detections': 0,
            'average_processing_time': 0.0,
            'accuracy_scores': []
        }

    def detect_spin_pattern(self, ball_frames: List[np.ndarray]) -> Optional[SpinData]:
        """
        연속된 볼 프레임에서 회전 패턴을 감지하여 스핀 데이터 계산
        
        Args:
            ball_frames: 820fps로 촬영된 연속 볼 이미지 리스트
            
        Returns:
            SpinData: 계산된 스핀 데이터 또는 None
        """
        start_time = time.perf_counter()
        
        if len(ball_frames) < 3:
            return None
            
        try:
            # 회전 벡터 계산
            rotation_vectors = self._calculate_rotation_vectors(ball_frames)
            
            if not rotation_vectors:
                return None
            
            # 평균 회전 벡터 계산
            avg_rotation_vector = np.mean(rotation_vectors, axis=0)
            
            # 스핀 데이터 변환
            spin_data = self._convert_to_spin_data(avg_rotation_vector)
            
            # 처리 시간 기록
            processing_time = (time.perf_counter() - start_time) * 1000
            spin_data.processing_time = processing_time
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics(spin_data, processing_time)
            
            return spin_data
            
        except Exception as e:
            print(f"스핀 패턴 감지 오류: {e}")
            return None

    def _calculate_rotation_vectors(self, ball_frames: List[np.ndarray]) -> List[np.ndarray]:
        """연속 프레임에서 회전 벡터 계산"""
        rotation_vectors = []
        
        for i in range(len(ball_frames) - 1):
            current_frame = ball_frames[i]
            next_frame = ball_frames[i + 1]
            
            # 특징점 검출 및 매칭
            matches_data = self._detect_and_match_features(current_frame, next_frame)
            
            if matches_data is None:
                continue
            
            # 회전 벡터 계산
            rotation_vector = self._compute_rotation_vector(matches_data)
            
            if rotation_vector is not None:
                rotation_vectors.append(rotation_vector)
        
        return rotation_vectors

    def _detect_and_match_features(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[Dict]:
        """특징점 검출 및 매칭"""
        if self.gpu_enabled:
            return self._detect_features_gpu(frame1, frame2)
        else:
            return self._detect_features_cpu(frame1, frame2)

    def _detect_features_cpu(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[Dict]:
        """CPU 기반 특징점 검출"""
        # 그레이스케일 변환
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
        
        # 특징점 검출
        kp1, des1 = self.pattern_detector.detectAndCompute(gray1, None)
        kp2, des2 = self.pattern_detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < self.spin_params['min_matches']:
            return None
        
        # 특징점 매칭
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 양질의 매칭점만 선택
        good_matches = matches[:min(self.spin_params['max_matches'], len(matches))]
        
        if len(good_matches) < self.spin_params['min_matches']:
            return None
        
        # 매칭점 좌표 추출
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return {
            'points1': pts1,
            'points2': pts2,
            'matches': good_matches,
            'keypoints1': kp1,
            'keypoints2': kp2
        }

    def _detect_features_gpu(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[Dict]:
        """GPU 기반 특징점 검출 (성능 최적화)"""
        try:
            # GPU 메모리로 업로드
            gpu_frame1 = cv2.cuda_GpuMat()
            gpu_frame2 = cv2.cuda_GpuMat()
            gpu_frame1.upload(frame1)
            gpu_frame2.upload(frame2)
            
            # 그레이스케일 변환
            if len(frame1.shape) == 3:
                gpu_gray1 = cv2.cuda.cvtColor(gpu_frame1, cv2.COLOR_BGR2GRAY)
                gpu_gray2 = cv2.cuda.cvtColor(gpu_frame2, cv2.COLOR_BGR2GRAY)
            else:
                gpu_gray1 = gpu_frame1
                gpu_gray2 = gpu_frame2
            
            # GPU에서 특징점 검출
            kp1, des1 = self.gpu_detector.detectAndComputeAsync(gpu_gray1, cv2.cuda_GpuMat())
            kp2, des2 = self.gpu_detector.detectAndComputeAsync(gpu_gray2, cv2.cuda_GpuMat())
            
            # CPU로 결과 다운로드
            if des1 is None or des2 is None:
                return None
            
            # 매칭 수행
            matches = self.matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            good_matches = matches[:min(self.spin_params['max_matches'], len(matches))]
            
            if len(good_matches) < self.spin_params['min_matches']:
                return None
            
            # 매칭점 좌표 추출
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            return {
                'points1': pts1,
                'points2': pts2,
                'matches': good_matches,
                'keypoints1': kp1,
                'keypoints2': kp2
            }
            
        except Exception as e:
            print(f"GPU 특징점 검출 실패, CPU로 폴백: {e}")
            return self._detect_features_cpu(frame1, frame2)

    def _compute_rotation_vector(self, matches_data: Dict) -> Optional[np.ndarray]:
        """매칭된 특징점으로부터 회전 벡터 계산"""
        pts1 = matches_data['points1']
        pts2 = matches_data['points2']
        
        if len(pts1) < self.spin_params['min_matches']:
            return None
        
        try:
            # 기본 변환 행렬 계산 (Affine Transform)
            transform_matrix, mask = cv2.estimateAffinePartial2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.spin_params['ransac_threshold'],
                confidence=self.spin_params['confidence_threshold']
            )
            
            if transform_matrix is None:
                return None
            
            # 회전 각도 추출
            rotation_angle = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
            
            # 스케일 추출 (크기 변화)
            scale = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[1, 0]**2)
            
            # 변위 벡터 계산
            translation = np.array([transform_matrix[0, 2], transform_matrix[1, 2]])
            
            # 3D 회전 벡터 구성 (X: pitch, Y: yaw, Z: roll)
            rotation_vector = np.array([
                translation[1] * scale,  # X축 회전 (상하 움직임)
                rotation_angle,          # Y축 회전 (좌우 회전 - 백스핀)
                translation[0] * scale   # Z축 회전 (좌우 움직임 - 사이드스핀)
            ])
            
            return rotation_vector
            
        except Exception as e:
            print(f"회전 벡터 계산 오류: {e}")
            return None

    def _convert_to_spin_data(self, rotation_vector: np.ndarray) -> SpinData:
        """회전 벡터를 스핀 데이터로 변환"""
        # 백스핀 계산 (Y축 회전)
        backspin_rpm = self._calculate_backspin_820fps(rotation_vector)
        
        # 사이드스핀 계산 (Z축 회전)
        sidespin_rpm = self._calculate_sidespin_820fps(rotation_vector)
        
        # 스핀축 계산
        spin_axis_degrees = self._calculate_spin_axis_820fps(rotation_vector)
        
        # 총 스핀 계산
        total_spin = np.sqrt(backspin_rpm**2 + sidespin_rpm**2)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(rotation_vector)
        
        return SpinData(
            backspin=backspin_rpm,
            sidespin=sidespin_rpm,
            spin_axis=spin_axis_degrees,
            total_spin=total_spin,
            confidence=confidence,
            processing_time=0.0  # 나중에 설정됨
        )

    def _calculate_backspin_820fps(self, rotation_vector: np.ndarray) -> float:
        """
        820fps 기반 백스핀 계산
        
        회전 벡터의 Y축 성분을 이용하여 백스핀 계산
        높은 프레임 레이트로 더 정확한 미세 회전 감지 가능
        """
        # Y축 회전 성분 추출 (백스핀)
        y_rotation = rotation_vector[1]
        
        # 820fps에서의 각속도 계산
        angular_velocity = y_rotation / self.frame_interval
        
        # RPM으로 변환
        backspin_rpm = abs(angular_velocity) * 60 / (2 * np.pi)
        
        # 유효 범위 제한
        backspin_rpm = np.clip(backspin_rpm, 0, 15000)
        
        # 820fps 보정 계수 적용
        correction_factor = self.spin_params['backspin_correction']
        backspin_rpm *= correction_factor
        
        return backspin_rpm

    def _calculate_sidespin_820fps(self, rotation_vector: np.ndarray) -> float:
        """
        820fps 기반 사이드스핀 계산
        
        회전 벡터의 Z축 성분을 이용하여 사이드스핀 계산
        """
        # Z축 회전 성분 추출 (사이드스핀)
        z_rotation = rotation_vector[2]
        
        # 820fps에서의 각속도 계산
        angular_velocity = z_rotation / self.frame_interval
        
        # RPM으로 변환 (방향 고려)
        sidespin_rpm = angular_velocity * 60 / (2 * np.pi)
        
        # 유효 범위 제한
        sidespin_rpm = np.clip(sidespin_rpm, -4000, 4000)
        
        # 820fps 민감도 향상 보정
        sensitivity_factor = self.spin_params['sidespin_sensitivity']
        sidespin_rpm *= sensitivity_factor
        
        return sidespin_rpm

    def _calculate_spin_axis_820fps(self, rotation_vector: np.ndarray) -> float:
        """
        820fps 기반 스핀축 계산
        
        3차원 회전 벡터로부터 스핀축 각도 계산
        """
        # 회전 벡터 정규화
        magnitude = np.linalg.norm(rotation_vector)
        if magnitude < 1e-6:
            return 0.0
        
        normalized_vector = rotation_vector / magnitude
        
        # 스핀축 각도 계산 (Y-Z 평면에서의 각도)
        spin_axis_angle = np.arctan2(normalized_vector[2], normalized_vector[1])
        
        # 라디안을 도로 변환
        spin_axis_degrees = np.degrees(spin_axis_angle)
        
        # 유효 범위 제한
        spin_axis_degrees = np.clip(spin_axis_degrees, -45, 45)
        
        # 820fps 정밀도 향상
        precision_factor = self.spin_params['spin_axis_precision']
        spin_axis_degrees *= precision_factor
        
        return spin_axis_degrees

    def _calculate_confidence(self, rotation_vector: np.ndarray) -> float:
        """회전 벡터 기반 신뢰도 계산"""
        # 벡터 크기가 클수록 높은 신뢰도
        magnitude = np.linalg.norm(rotation_vector)
        
        # 정규화된 신뢰도 계산
        confidence = min(1.0, magnitude / 10.0)
        
        # 최소 신뢰도 보장
        confidence = max(0.1, confidence)
        
        return confidence

    def _update_performance_metrics(self, spin_data: SpinData, processing_time: float):
        """성능 메트릭 업데이트"""
        self.performance_metrics['total_frames'] += 1
        
        if spin_data.confidence > 0.5:
            self.performance_metrics['successful_detections'] += 1
        
        # 평균 처리 시간 업데이트
        n = self.performance_metrics['total_frames']
        old_avg = self.performance_metrics['average_processing_time']
        self.performance_metrics['average_processing_time'] = (old_avg * (n-1) + processing_time) / n

    def get_performance_report(self) -> Dict:
        """성능 리포트 생성"""
        metrics = self.performance_metrics.copy()
        
        if metrics['total_frames'] > 0:
            metrics['success_rate'] = metrics['successful_detections'] / metrics['total_frames']
        else:
            metrics['success_rate'] = 0.0
        
        metrics['target_processing_time'] = 1.22  # ms
        metrics['performance_ratio'] = 1.22 / max(metrics['average_processing_time'], 0.01)
        
        return metrics


class SpinAnalysisManager:
    """스핀 분석 매니저 - 다중 프레임 처리 및 결과 통합"""
    
    def __init__(self, config_path: str = None):
        self.detector = BallSpinDetector820fps(config_path)
        self.frame_buffer = []
        self.max_buffer_size = 20  # 820fps에서 약 24ms 분량
        
        # 결과 필터링
        self.result_filter = SpinResultFilter()
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def add_frame(self, ball_roi: np.ndarray, timestamp: float) -> Optional[SpinData]:
        """
        새로운 볼 프레임 추가 및 스핀 분석
        
        Args:
            ball_roi: 볼 영역 이미지
            timestamp: 타임스탬프
            
        Returns:
            SpinData: 분석된 스핀 데이터 (충분한 프레임이 쌓인 경우)
        """
        # 프레임 버퍼에 추가
        self.frame_buffer.append({
            'image': ball_roi,
            'timestamp': timestamp
        })
        
        # 버퍼 크기 제한
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        # 충분한 프레임이 쌓이면 분석 수행
        if len(self.frame_buffer) >= 5:  # 최소 5프레임
            frames = [frame['image'] for frame in self.frame_buffer[-10:]]  # 최근 10프레임 사용
            
            # 비동기 스핀 분석
            future = self.executor.submit(self.detector.detect_spin_pattern, frames)
            
            try:
                spin_data = future.result(timeout=0.001)  # 1ms 타임아웃
                
                if spin_data:
                    # 결과 필터링 및 후처리
                    filtered_result = self.result_filter.filter_result(spin_data)
                    return filtered_result
                    
            except Exception as e:
                print(f"스핀 분석 타임아웃 또는 오류: {e}")
        
        return None


class SpinResultFilter:
    """스핀 분석 결과 필터링 및 후처리"""
    
    def __init__(self):
        self.history = []
        self.max_history = 10
        
    def filter_result(self, spin_data: SpinData) -> SpinData:
        """결과 필터링 및 스무딩"""
        # 이전 결과와 비교하여 급격한 변화 제한
        if self.history:
            last_result = self.history[-1]
            
            # 백스핀 스무딩
            if abs(spin_data.backspin - last_result.backspin) > 1000:
                spin_data.backspin = last_result.backspin * 0.7 + spin_data.backspin * 0.3
            
            # 사이드스핀 스무딩
            if abs(spin_data.sidespin - last_result.sidespin) > 500:
                spin_data.sidespin = last_result.sidespin * 0.7 + spin_data.sidespin * 0.3
        
        # 히스토리 업데이트
        self.history.append(spin_data)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return spin_data


# 테스트 및 검증 함수
def test_spin_detector():
    """스핀 검출기 테스트"""
    detector = BallSpinDetector820fps()
    
    print("820fps 스핀 검출기 테스트 시작...")
    
    # 테스트 이미지 생성 (시뮬레이션)
    test_frames = []
    for i in range(10):
        # 가상 볼 이미지 생성
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(frame, (50 + i, 50), 30, (255, 255, 255), -1)
        test_frames.append(frame)
    
    # 스핀 검출 테스트
    result = detector.detect_spin_pattern(test_frames)
    
    if result:
        print(f"백스핀: {result.backspin:.1f} RPM")
        print(f"사이드스핀: {result.sidespin:.1f} RPM")
        print(f"스핀축: {result.spin_axis:.1f}°")
        print(f"신뢰도: {result.confidence:.2f}")
        print(f"처리시간: {result.processing_time:.2f}ms")
    else:
        print("스핀 검출 실패")
    
    # 성능 리포트
    report = detector.get_performance_report()
    print(f"성능 리포트: {report}")


if __name__ == "__main__":
    test_spin_detector()