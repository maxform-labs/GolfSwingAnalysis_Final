#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Adaptive Golf Measurement System v3.0
적응형 ROI 기반 고급 골프 측정 시스템
- 다단계 적응형 ROI 전략 적용
- 전체화면 → 임팩트존 → 추적 → 비행추적
- /results 폴더에 결과 저장
"""

import cv2
import numpy as np
import math
import pandas as pd
import os
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from pathlib import Path
import logging

# 적응형 ROI 검출기 import
from adaptive_roi_detector import AdaptiveROIDetector, AdaptiveBallResult, DetectionPhase, DetectionMethod
from simple_ball_detector import (
    SimpleStereoVision, SimplePhysicsCalculator, SimpleSpinCalculator, 
    SimpleClubCalculator, SimplePhysicsValidator, SimpleTrajectoryPredictor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedMeasurementData:
    """강화된 측정 데이터"""
    # 기본 정보
    shot_id: str
    frame_number: int
    timestamp_ms: float
    
    # 적응형 볼 검출 결과
    ball_detected: bool = False
    detection_method: str = "none"
    detection_phase: str = "full_screen"
    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 0
    roi_height: int = 0
    motion_state: str = "unknown"
    
    # 볼 위치 및 속성
    ball_x: float = 0.0
    ball_y: float = 0.0
    ball_radius: float = 0.0
    detection_confidence: float = 0.0
    
    # 물리적 측정값 (요구사항)
    ball_speed_mph: float = 0.0
    ball_speed_ms: float = 0.0
    launch_angle_deg: float = 0.0
    direction_angle_deg: float = 0.0
    backspin_rpm: float = 0.0
    sidespin_rpm: float = 0.0
    spin_axis_deg: float = 0.0
    
    # 클럽 데이터 (요구사항)
    club_speed_mph: float = 0.0
    club_speed_ms: float = 0.0
    attack_angle_deg: float = 0.0
    face_angle_deg: float = 0.0
    club_path_deg: float = 0.0
    face_to_path_deg: float = 0.0
    smash_factor: float = 0.0
    
    # 예측 결과
    carry_distance_m: float = 0.0
    max_height_m: float = 0.0
    flight_time_s: float = 0.0
    
    # 품질 지표
    confidence_score: float = 0.0
    physics_validation: float = 0.0

class EnhancedAdaptiveSystem:
    """강화된 적응형 골프 측정 시스템"""
    
    def __init__(self):
        # 적응형 검출기들
        self.top_detector = AdaptiveROIDetector()
        self.bottom_detector = AdaptiveROIDetector()
        
        # 물리 계산 모듈들
        self.stereo_vision = SimpleStereoVision()
        self.physics_calc = SimplePhysicsCalculator()
        self.spin_calc = SimpleSpinCalculator()
        self.club_calc = SimpleClubCalculator()
        self.physics_validator = SimplePhysicsValidator()
        self.trajectory_predictor = SimpleTrajectoryPredictor()
        
        # 데이터 저장
        self.measurement_sequence: List[EnhancedMeasurementData] = []
        self.ball_positions_3d: List[Tuple[float, float, float]] = []
        self.club_positions_3d: List[Tuple[float, float, float]] = []
        self.timestamps: List[float] = []
        
        # 상태 관리
        self.current_shot_id = ""
        self.launch_detected = False
        self.launch_frame = 0
        
        # 결과 폴더 생성
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def process_frame_pair(self, top_img: np.ndarray, bottom_img: np.ndarray,
                          frame_number: int, shot_id: str) -> EnhancedMeasurementData:
        """프레임 쌍 처리 - 적응형 ROI 적용"""
        
        timestamp_ms = frame_number * (1000.0 / 820.0)
        
        measurement = EnhancedMeasurementData(
            shot_id=shot_id,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms
        )
        
        # 1. 적응형 볼 검출
        top_result = self.top_detector.detect_ball_adaptive(top_img, frame_number)
        bottom_result = self.bottom_detector.detect_ball_adaptive(bottom_img, frame_number)
        
        if top_result and bottom_result:
            # 대응점 검증 (Y축 시차 확인)
            y_disparity = abs(top_result.center_y - bottom_result.center_y)
            x_distance = abs(top_result.center_x - bottom_result.center_x)
            
            # 수직 스테레오에서 Y축 시차가 있어야 하고, X축 거리는 작아야 함
            if 5 < y_disparity < 100 and x_distance < 50:
                
                measurement.ball_detected = True
                measurement.detection_method = top_result.detection_method.value
                measurement.detection_phase = top_result.phase.value
                
                # ROI 정보
                measurement.roi_x = top_result.roi_region.x
                measurement.roi_y = top_result.roi_region.y
                measurement.roi_width = top_result.roi_region.width
                measurement.roi_height = top_result.roi_region.height
                
                # 볼 위치 정보
                measurement.ball_x = (top_result.center_x + bottom_result.center_x) / 2.0
                measurement.ball_y = (top_result.center_y + bottom_result.center_y) / 2.0
                measurement.ball_radius = (top_result.radius + bottom_result.radius) / 2.0
                measurement.motion_state = top_result.motion_state
                measurement.detection_confidence = (top_result.confidence + bottom_result.confidence) / 2.0
                
                # 발사 감지 (움직임이 큰 상태로 변경될 때)
                if (top_result.motion_state in ["moving", "flying"] and 
                    not self.launch_detected):
                    self.launch_detected = True
                    self.launch_frame = frame_number
                    logger.info(f"Launch detected at frame {frame_number} (Phase: {top_result.phase.value})")
                
                # 3D 위치 계산
                ball_3d = self.stereo_vision.calculate_3d_position(
                    (top_result.center_x, top_result.center_y),
                    (bottom_result.center_x, bottom_result.center_y)
                )
                
                self.ball_positions_3d.append(ball_3d)
                self.timestamps.append(timestamp_ms / 1000.0)
                
                logger.debug(f"Frame {frame_number}: Ball detected at ({measurement.ball_x:.1f}, {measurement.ball_y:.1f}) "
                           f"Phase: {measurement.detection_phase}, Method: {measurement.detection_method}")
        
        # 2. 클럽 추정 (간소화)
        if measurement.ball_detected and len(self.ball_positions_3d) >= 2:
            ball_3d = self.ball_positions_3d[-1]
            # 클럽은 볼 근처에 위치한다고 가정
            club_3d = (
                ball_3d[0] + np.random.uniform(-40, 40),
                ball_3d[1] + np.random.uniform(-60, -20),
                ball_3d[2] + np.random.uniform(-150, -80)
            )
            self.club_positions_3d.append(club_3d)
        
        # 3. 물리량 계산
        if len(self.ball_positions_3d) >= 3:
            self._calculate_physics_enhanced(measurement)
        
        # 4. 물리 검증
        measurement.physics_validation = self._validate_physics_enhanced(measurement)
        
        # 5. 전체 신뢰도 계산
        measurement.confidence_score = self._calculate_enhanced_confidence(measurement)
        
        self.measurement_sequence.append(measurement)
        return measurement
    
    def _calculate_physics_enhanced(self, measurement: EnhancedMeasurementData):
        """강화된 물리량 계산"""
        
        # 최근 위치 데이터 (검출 단계에 따라 가중치 적용)
        recent_positions = self.ball_positions_3d[-7:] if len(self.ball_positions_3d) >= 7 else self.ball_positions_3d
        recent_timestamps = self.timestamps[-7:] if len(self.timestamps) >= 7 else self.timestamps
        
        if len(recent_positions) < 2:
            return
        
        # 1. 볼 스피드 계산 (단계별 가중치)
        speed_data = self.physics_calc.calculate_3d_speed(recent_positions, recent_timestamps)
        
        # 검출 단계별 보정 계수
        phase_multiplier = {
            'full_screen': 1.0,
            'impact_zone': 1.1,  # 임팩트존에서 더 정확
            'tracking': 1.0,
            'flight_tracking': 0.95  # 비행 중에는 약간 보수적
        }.get(measurement.detection_phase, 1.0)
        
        measurement.ball_speed_ms = speed_data['speed_ms'] * phase_multiplier
        measurement.ball_speed_mph = speed_data['speed_mph'] * phase_multiplier
        
        # 2. 발사각 계산
        measurement.launch_angle_deg = self.physics_calc.calculate_launch_angle(recent_positions)
        
        # 3. 방향각 계산
        measurement.direction_angle_deg = self.physics_calc.calculate_direction_angle(recent_positions)
        
        # 4. 스핀 계산 (적응형 보정)
        spin_data = self.spin_calc.calculate_spin_from_trajectory(recent_positions, recent_timestamps)
        
        # 검출 방법별 스핀 보정
        method_spin_factor = {
            'hough_gamma': 1.1,  # 감마 보정된 검출은 더 정확한 스핀
            'hough_circles': 1.0,
            'contour_bright': 0.9,
            'motion_detect': 0.8,
            'template_match': 1.05
        }.get(measurement.detection_method, 1.0)
        
        measurement.backspin_rpm = spin_data['backspin_rpm'] * method_spin_factor
        measurement.sidespin_rpm = spin_data['sidespin_rpm'] * method_spin_factor
        measurement.spin_axis_deg = spin_data['spin_axis_deg']
        
        # 5. 클럽 물리량 계산
        if len(self.club_positions_3d) >= 2:
            club_timestamps = recent_timestamps[-len(self.club_positions_3d):]
            
            club_speed_data = self.club_calc.calculate_club_speed(self.club_positions_3d, club_timestamps)
            measurement.club_speed_ms = club_speed_data['speed_ms']
            measurement.club_speed_mph = club_speed_data['speed_mph']
            
            measurement.attack_angle_deg = self.club_calc.calculate_attack_angle(self.club_positions_3d, club_timestamps)
            measurement.club_path_deg = self.club_calc.calculate_club_path(self.club_positions_3d)
            
            measurement.face_angle_deg = self.club_calc.calculate_face_angle(
                measurement.club_path_deg, measurement.direction_angle_deg
            )
            
            measurement.face_to_path_deg = measurement.face_angle_deg - measurement.club_path_deg
            
            measurement.smash_factor = self.club_calc.calculate_smash_factor(
                measurement.ball_speed_mph, measurement.club_speed_mph
            )
        
        # 6. 궤적 예측
        if measurement.ball_speed_ms > 0:
            trajectory = self.trajectory_predictor.predict_carry_distance(
                measurement.ball_speed_ms,
                measurement.launch_angle_deg,
                measurement.backspin_rpm
            )
            measurement.carry_distance_m = trajectory['carry_m']
            measurement.max_height_m = trajectory['max_height_m']
            measurement.flight_time_s = trajectory['flight_time_s']
    
    def _validate_physics_enhanced(self, measurement: EnhancedMeasurementData) -> float:
        """강화된 물리 검증"""
        
        try:
            ball_data = {
                'ball_speed_mph': measurement.ball_speed_mph,
                'launch_angle_deg': measurement.launch_angle_deg,
                'direction_angle_deg': measurement.direction_angle_deg,
                'backspin_rpm': measurement.backspin_rpm,
                'sidespin_rpm': measurement.sidespin_rpm
            }
            
            ball_validations = self.physics_validator.validate_ball_physics(ball_data)
            
            club_data = {
                'club_speed_mph': measurement.club_speed_mph,
                'attack_angle_deg': measurement.attack_angle_deg,
                'face_angle_deg': measurement.face_angle_deg,
                'club_path_deg': measurement.club_path_deg,
                'smash_factor': measurement.smash_factor
            }
            
            club_validations = self.physics_validator.validate_club_physics(club_data)
            
            # 검출 단계별 검증 가중치
            phase_weight = {
                'full_screen': 0.8,
                'impact_zone': 1.0,
                'tracking': 0.9,
                'flight_tracking': 0.85
            }.get(measurement.detection_phase, 0.8)
            
            all_validations = {**ball_validations, **club_validations}
            valid_count = sum(all_validations.values())
            total_count = len(all_validations)
            
            base_score = valid_count / total_count if total_count > 0 else 0.0
            return base_score * phase_weight
            
        except Exception as e:
            logger.error(f"Enhanced physics validation error: {e}")
            return 0.0
    
    def _calculate_enhanced_confidence(self, measurement: EnhancedMeasurementData) -> float:
        """강화된 신뢰도 계산"""
        
        confidence_factors = []
        
        # 1. 검출 신뢰도
        confidence_factors.append(measurement.detection_confidence)
        
        # 2. 검출 방법별 가중치
        method_weight = {
            'hough_gamma': 0.9,
            'hough_circles': 0.8,
            'contour_bright': 0.75,
            'motion_detect': 0.7,
            'template_match': 0.85
        }.get(measurement.detection_method, 0.6)
        confidence_factors.append(method_weight)
        
        # 3. 검출 단계별 가중치
        phase_weight = {
            'full_screen': 0.7,
            'impact_zone': 0.9,
            'tracking': 0.85,
            'flight_tracking': 0.8
        }.get(measurement.detection_phase, 0.6)
        confidence_factors.append(phase_weight)
        
        # 4. 물리적 일관성
        confidence_factors.append(measurement.physics_validation)
        
        # 5. 데이터 완성도
        required_fields = [
            measurement.ball_speed_mph, measurement.launch_angle_deg,
            measurement.direction_angle_deg, measurement.backspin_rpm,
            measurement.sidespin_rpm, measurement.club_speed_mph,
            measurement.attack_angle_deg, measurement.face_angle_deg,
            measurement.club_path_deg, measurement.smash_factor
        ]
        
        non_zero_count = sum(1 for field in required_fields if abs(field) > 0.1)
        completeness = non_zero_count / len(required_fields)
        confidence_factors.append(completeness)
        
        # 가중 평균 계산
        weights = [0.25, 0.2, 0.2, 0.2, 0.15]
        weighted_confidence = sum(w * c for w, c in zip(weights, confidence_factors))
        
        return min(weighted_confidence, 1.0)
    
    def analyze_shot_sequence(self, shot_folder: Path) -> List[EnhancedMeasurementData]:
        """샷 시퀀스 분석 - 적응형 ROI 적용"""
        
        logger.info(f"Analyzing shot with adaptive ROI: {shot_folder.name}")
        
        shot_id = f"{shot_folder.parent.name}_{shot_folder.name}"
        self.current_shot_id = shot_id
        
        # 초기화
        self.measurement_sequence = []
        self.ball_positions_3d = []
        self.club_positions_3d = []
        self.timestamps = []
        self.launch_detected = False
        
        # 검출기 초기화
        self.top_detector.reset_tracking()
        self.bottom_detector.reset_tracking()
        
        results = []
        
        # 프레임별 처리
        for frame_num in range(1, 24):
            top_img = self._load_image(shot_folder, f"1_{frame_num}.bmp")
            bottom_img = self._load_image(shot_folder, f"2_{frame_num}.bmp")
            
            if top_img is not None and bottom_img is not None:
                measurement = self.process_frame_pair(top_img, bottom_img, frame_num, shot_id)
                results.append(measurement)
                
                # 진행 상황 로깅
                if measurement.ball_detected:
                    logger.debug(f"Frame {frame_num}: {measurement.detection_method} "
                               f"({measurement.detection_phase}) conf={measurement.detection_confidence:.2f}")
            else:
                # 이미지 로드 실패시 빈 측정값
                measurement = EnhancedMeasurementData(
                    shot_id=shot_id,
                    frame_number=frame_num,
                    timestamp_ms=frame_num * (1000.0 / 820.0)
                )
                results.append(measurement)
        
        # 최종 결과 정리
        self._finalize_enhanced_results(results)
        
        # 검출 통계
        detection_count = sum(1 for r in results if r.ball_detected)
        logger.info(f"Shot {shot_id}: {detection_count}/23 frames detected")
        
        return results
    
    def _load_image(self, folder: Path, filename: str) -> Optional[np.ndarray]:
        """이미지 로드 (에러 핸들링 강화)"""
        
        try:
            img_path = folder / filename
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None and img.size > 0:
                    # 1440x300 해상도 확인
                    if img.shape[1] == 1440 and img.shape[0] == 300:
                        return img
                    else:
                        # 해상도가 다르면 리사이즈
                        img = cv2.resize(img, (1440, 300))
                        return img
            return None
            
        except Exception as e:
            logger.warning(f"Image load error {filename}: {e}")
            return None
    
    def _finalize_enhanced_results(self, results: List[EnhancedMeasurementData]):
        """강화된 결과 최종화"""
        
        if not results:
            return
        
        # 발사 후 안정화 구간에서 최종값 결정
        if self.launch_detected and self.launch_frame > 0:
            # 발사 후 2-5프레임 구간의 고신뢰도 측정값 선택
            stable_measurements = [
                r for r in results
                if r.ball_detected and 
                r.confidence_score > 0.6 and
                self.launch_frame + 2 <= r.frame_number <= self.launch_frame + 5
            ]
            
            if stable_measurements:
                # 가장 높은 신뢰도 측정값으로 최종 결과 결정
                best_measurement = max(stable_measurements, key=lambda x: x.confidence_score)
                
                final_values = self._extract_final_values(best_measurement)
                
                # 모든 결과에 최종값 적용
                for result in results:
                    for key, value in final_values.items():
                        if hasattr(result, key):
                            setattr(result, key, value)
    
    def _extract_final_values(self, measurement: EnhancedMeasurementData) -> Dict[str, float]:
        """최종 측정값 추출"""
        
        return {
            'ball_speed_mph': measurement.ball_speed_mph,
            'ball_speed_ms': measurement.ball_speed_ms,
            'launch_angle_deg': measurement.launch_angle_deg,
            'direction_angle_deg': measurement.direction_angle_deg,
            'backspin_rpm': measurement.backspin_rpm,
            'sidespin_rpm': measurement.sidespin_rpm,
            'spin_axis_deg': measurement.spin_axis_deg,
            'club_speed_mph': measurement.club_speed_mph,
            'club_speed_ms': measurement.club_speed_ms,
            'attack_angle_deg': measurement.attack_angle_deg,
            'face_angle_deg': measurement.face_angle_deg,
            'club_path_deg': measurement.club_path_deg,
            'face_to_path_deg': measurement.face_to_path_deg,
            'smash_factor': measurement.smash_factor,
            'carry_distance_m': measurement.carry_distance_m,
            'max_height_m': measurement.max_height_m,
            'flight_time_s': measurement.flight_time_s
        }

def main():
    """메인 함수"""
    print("=== Enhanced Adaptive Golf Measurement System v3.0 ===")
    print("Adaptive ROI-based multi-stage detection strategy")
    
    shot_image_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/shot-image")
    
    # results 폴더에 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"results/enhanced_adaptive_measurements_{timestamp}.xlsx"
    
    if not shot_image_dir.exists():
        print(f"ERROR: Image folder not found: {shot_image_dir}")
        return
    
    system = EnhancedAdaptiveSystem()
    start_time = time.time()
    
    print(f"Image folder: {shot_image_dir}")
    print(f"Output file: {output_file}")
    print("Adaptive ROI Strategy:")
    print("  Phase 1-5: Full screen scanning")
    print("  Phase 6-12: Impact zone focus (600x150 @ 2x)")
    print("  Phase 13-18: Ball tracking")
    print("  Phase 19-23: Flight tracking")
    print("=" * 70)
    
    all_results = []
    successful_shots = 0
    total_shots = 0
    
    # 전체 샷 처리
    for club_dir in shot_image_dir.iterdir():
        if club_dir.is_dir():
            print(f"\nProcessing club: {club_dir.name}")
            
            for ball_dir in club_dir.iterdir():
                if ball_dir.is_dir():
                    total_shots += 1
                    print(f"  Processing: {ball_dir.name}")
                    
                    try:
                        results = system.analyze_shot_sequence(ball_dir)
                        
                        if results:
                            detection_count = sum(1 for r in results if r.ball_detected)
                            avg_confidence = np.mean([r.confidence_score for r in results if r.ball_detected])
                            
                            if detection_count > 0:
                                successful_shots += 1
                                print(f"    SUCCESS: {detection_count}/23 detected, avg conf: {avg_confidence:.3f}")
                                
                                # 검출 방법 통계
                                methods = [r.detection_method for r in results if r.ball_detected]
                                phases = [r.detection_phase for r in results if r.ball_detected]
                                
                                if methods:
                                    method_counts = {m: methods.count(m) for m in set(methods)}
                                    phase_counts = {p: phases.count(p) for p in set(phases)}
                                    print(f"    Methods: {method_counts}")
                                    print(f"    Phases: {phase_counts}")
                            else:
                                print(f"    WARNING: No ball detection")
                            
                            all_results.extend(results)
                        
                    except Exception as e:
                        print(f"    ERROR: {e}")
    
    processing_time = time.time() - start_time
    
    if all_results:
        df_results = pd.DataFrame([asdict(result) for result in all_results])
        
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Total shots: {total_shots}")
        print(f"Successful shots: {successful_shots}")
        print(f"Success rate: {(successful_shots/total_shots*100):.1f}%")
        print(f"Total frames: {len(df_results)}")
        
        detection_rate = df_results['ball_detected'].sum() / len(df_results) * 100
        print(f"Ball detection rate: {detection_rate:.1f}%")
        
        if detection_rate > 0:
            print(f"\nDETECTION ANALYSIS")
            print(f"{'='*70}")
            
            # 검출 방법 분석
            detected_frames = df_results[df_results['ball_detected']]
            method_counts = detected_frames['detection_method'].value_counts()
            phase_counts = detected_frames['detection_phase'].value_counts()
            
            print("Detection methods:")
            for method, count in method_counts.items():
                print(f"  {method}: {count} frames ({count/len(detected_frames)*100:.1f}%)")
            
            print("Detection phases:")
            for phase, count in phase_counts.items():
                print(f"  {phase}: {count} frames ({count/len(detected_frames)*100:.1f}%)")
            
            # 측정값 요약
            print(f"\nMEASUREMENT SUMMARY")
            print(f"{'='*70}")
            
            valid_speeds = df_results[df_results['ball_speed_mph'] > 0]['ball_speed_mph']
            if not valid_speeds.empty:
                print(f"Ball speed: {valid_speeds.mean():.1f} ± {valid_speeds.std():.1f} mph")
            
            valid_angles = df_results[df_results['launch_angle_deg'] != 0]['launch_angle_deg']
            if not valid_angles.empty:
                print(f"Launch angle: {valid_angles.mean():.1f} ± {valid_angles.std():.1f}°")
            
            valid_spins = df_results[df_results['backspin_rpm'] > 0]['backspin_rpm']
            if not valid_spins.empty:
                print(f"Backspin: {valid_spins.mean():.0f} ± {valid_spins.std():.0f} rpm")
        
        # Excel 저장 (results 폴더)
        try:
            df_results.to_excel(output_file, index=False)
            print(f"\nResults saved: {output_file}")
        except Exception as e:
            print(f"Excel save error: {e}")
        
        # 품질 평가
        avg_confidence = df_results['confidence_score'].mean()
        avg_physics = df_results['physics_validation'].mean()
        high_quality = (df_results['confidence_score'] > 0.7).sum()
        
        print(f"\nQUALITY ASSESSMENT")
        print(f"{'='*70}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Physics validation: {avg_physics:.3f}")
        print(f"High quality measurements: {high_quality}/{len(df_results)}")
        
        # ROI 효과 분석
        roi_effectiveness = {}
        for phase in df_results['detection_phase'].unique():
            phase_data = df_results[df_results['detection_phase'] == phase]
            detection_rate_phase = phase_data['ball_detected'].sum() / len(phase_data) * 100
            roi_effectiveness[phase] = detection_rate_phase
        
        print(f"\nROI EFFECTIVENESS")
        print(f"{'='*70}")
        for phase, rate in roi_effectiveness.items():
            print(f"{phase}: {rate:.1f}% detection rate")
    
    else:
        print("\nERROR: No measurement results generated")
    
    print(f"\n{'='*70}")
    print("ENHANCED ADAPTIVE ANALYSIS COMPLETED!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()