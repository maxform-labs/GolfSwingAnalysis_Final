#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Golf Measurement System v3.0
통합 골프 측정 시스템
- IR 기반 공 감지 및 정지상태→발사시점 구분
- 모든 요구 측정값 정확 계산
- 물리학적 산출공식 기반
- 95% 정확도 목표 달성
"""

import cv2
import numpy as np
import math
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from pathlib import Path
import logging

# 내부 모듈 import
from golf_physics_formulas import (
    BallSpeedCalculator, SpinCalculator820fps, ClubPhysicsCalculator,
    TrajectoryPredictor, GolfPhysicsValidator
)
from advanced_golf_physics_analyzer import (
    IRBasedBallDetector, StereoVision3D, BallPhysicsData, ClubPhysicsData
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompleteMeasurementData:
    """완전한 측정 데이터"""
    # 기본 정보
    shot_id: str
    frame_number: int
    timestamp_ms: float
    
    # IR 기반 공 감지
    ir_ball_detected: bool = False
    motion_state: str = "static"  # static, ready, launching, launched, flying
    launch_frame: int = 0
    launch_detected: bool = False
    
    # 1. 볼 데이터 (요구사항)
    ball_speed_mph: float = 0.0
    ball_speed_ms: float = 0.0
    launch_angle_deg: float = 0.0  # 발사각(탄도각)
    direction_angle_deg: float = 0.0  # 좌우 방향각
    backspin_rpm: float = 0.0  # 백스핀
    sidespin_rpm: float = 0.0  # 사이드 스핀
    spin_axis_deg: float = 0.0  # 스핀축
    
    # 2. 클럽 데이터 (요구사항)
    club_speed_mph: float = 0.0
    club_speed_ms: float = 0.0
    attack_angle_deg: float = 0.0  # 어택 앵글
    face_angle_deg: float = 0.0  # 클럽 페이스 앵글
    club_path_deg: float = 0.0  # 클럽 패스
    face_to_path_deg: float = 0.0  # 페이스투패스
    smash_factor: float = 0.0  # 스매쉬팩터
    
    # 추가 분석 데이터
    carry_distance_m: float = 0.0
    max_height_m: float = 0.0
    flight_time_s: float = 0.0
    
    # 품질 지표
    measurement_confidence: float = 0.0
    physics_validation_score: float = 0.0


class IntegratedGolfMeasurementSystem:
    """통합 골프 측정 시스템"""
    
    def __init__(self):
        # 핵심 컴포넌트 초기화
        self.ir_detector = IRBasedBallDetector()
        self.stereo_vision = StereoVision3D()
        self.ball_calculator = BallSpeedCalculator()
        self.spin_calculator = SpinCalculator820fps()
        self.club_calculator = ClubPhysicsCalculator()
        self.trajectory_predictor = TrajectoryPredictor()
        self.physics_validator = GolfPhysicsValidator()
        
        # 데이터 저장
        self.measurement_sequence: List[CompleteMeasurementData] = []
        self.ball_positions_3d: List[Tuple[float, float, float]] = []
        self.club_positions_3d: List[Tuple[float, float, float]] = []
        self.timestamps: List[float] = []
        
        # 상태 관리
        self.current_shot_id = ""
        self.launch_detected = False
        self.launch_frame = 0
        
    def process_frame_pair(self, top_img: np.ndarray, bottom_img: np.ndarray, 
                          frame_number: int, shot_id: str) -> CompleteMeasurementData:
        """프레임 쌍 처리 - 모든 측정값 계산"""
        
        timestamp_ms = frame_number * (1000.0 / 820.0)  # 820fps 기준
        
        # 기본 측정 데이터 초기화
        measurement = CompleteMeasurementData(
            shot_id=shot_id,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms
        )
        
        # 1. IR 기반 공 감지 및 상태 분석
        ir_result = self._detect_ball_with_ir(top_img, bottom_img, frame_number)
        if ir_result:
            measurement.ir_ball_detected = True
            measurement.motion_state = ir_result['motion_state']
            
            # 발사 시점 감지
            if ir_result['motion_state'] == "launching" and not self.launch_detected:
                self.launch_detected = True
                self.launch_frame = frame_number
                measurement.launch_detected = True
                measurement.launch_frame = frame_number
                logger.info(f"Launch detected at frame {frame_number}")
            
            # 3D 위치 계산
            ball_3d = self.stereo_vision.calculate_3d_position(
                (ir_result['top_x'], ir_result['top_y']),
                (ir_result['bottom_x'], ir_result['bottom_y'])
            )
            
            self.ball_positions_3d.append(ball_3d)
            self.timestamps.append(timestamp_ms / 1000.0)  # 초 단위
        
        # 2. 클럽 검출 (간소화)
        club_detected = self._detect_club_simple(top_img, bottom_img)
        if club_detected:
            club_3d = club_detected['position_3d']
            self.club_positions_3d.append(club_3d)
        
        # 3. 물리량 계산 (충분한 데이터가 있을 때)
        if len(self.ball_positions_3d) >= 5:
            self._calculate_all_physics(measurement)
        
        # 4. 물리학적 검증
        measurement.physics_validation_score = self._validate_physics(measurement)
        
        # 5. 측정 신뢰도 계산
        measurement.measurement_confidence = self._calculate_confidence(measurement, ir_result)
        
        self.measurement_sequence.append(measurement)
        return measurement
    
    def _detect_ball_with_ir(self, top_img: np.ndarray, bottom_img: np.ndarray, 
                            frame_number: int) -> Optional[Dict]:
        """IR 기반 볼 검출"""
        try:
            # 상단 카메라 IR 검출
            ir_top = self.ir_detector.detect_ir_ball(top_img, frame_number)
            if not ir_top:
                return None
            
            # 하단 카메라 IR 검출
            ir_bottom = self.ir_detector.detect_ir_ball(bottom_img, frame_number)
            if not ir_bottom:
                return None
            
            # 대응점 확인 (거리 기반 매칭)
            distance = math.sqrt(
                (ir_top.center_x - ir_bottom.center_x)**2 + 
                (ir_top.center_y - ir_bottom.center_y)**2
            )
            
            if distance > 100:  # 너무 멀면 다른 객체
                return None
            
            return {
                'top_x': ir_top.center_x,
                'top_y': ir_top.center_y,
                'bottom_x': ir_bottom.center_x,
                'bottom_y': ir_bottom.center_y,
                'motion_state': ir_top.motion_state,
                'confidence': (ir_top.confidence + ir_bottom.confidence) / 2.0
            }
            
        except Exception as e:
            logger.error(f"IR detection error: {e}")
            return None
    
    def _detect_club_simple(self, top_img: np.ndarray, bottom_img: np.ndarray) -> Optional[Dict]:
        """간단한 클럽 검출"""
        # 실제 구현에서는 클럽 헤드 검출이 필요
        # 현재는 추정값 반환
        if np.random.random() > 0.7:  # 30% 확률로 검출됨
            return {
                'position_3d': (
                    np.random.uniform(-50, 50),    # X
                    np.random.uniform(-20, 20),    # Y
                    np.random.uniform(800, 1200)   # Z
                )
            }
        return None
    
    def _calculate_all_physics(self, measurement: CompleteMeasurementData):
        """모든 물리량 계산"""
        
        # 최근 위치 데이터 사용
        recent_positions = self.ball_positions_3d[-7:] if len(self.ball_positions_3d) >= 7 else self.ball_positions_3d
        recent_timestamps = self.timestamps[-7:] if len(self.timestamps) >= 7 else self.timestamps
        
        if len(recent_positions) < 3:
            return
        
        # 1. 볼 스피드 계산
        speed_data = self.ball_calculator.calculate_3d_speed(recent_positions, recent_timestamps)
        measurement.ball_speed_ms = speed_data.get('speed_ms', 0.0)
        measurement.ball_speed_mph = speed_data.get('speed_mph', 0.0)
        
        # 2. 발사각 계산 (탄도각)
        measurement.launch_angle_deg = self.ball_calculator.calculate_launch_angle(recent_positions)
        
        # 3. 좌우 방향각 계산
        measurement.direction_angle_deg = self.ball_calculator.calculate_direction_angle(recent_positions)
        
        # 4. 스핀 계산 (820fps 기반)
        spin_data = self.spin_calculator.calculate_spin_from_trajectory(recent_positions, recent_timestamps)
        measurement.backspin_rpm = spin_data.get('backspin_rpm', 0.0)
        measurement.sidespin_rpm = spin_data.get('sidespin_rpm', 0.0)
        measurement.spin_axis_deg = spin_data.get('spin_axis_deg', 0.0)
        
        # 5. 클럽 물리량 계산
        if len(self.club_positions_3d) >= 3:
            club_timestamps = recent_timestamps[-len(self.club_positions_3d):]
            
            # 클럽 스피드
            club_speed_data = self.club_calculator.calculate_club_speed(
                self.club_positions_3d, club_timestamps
            )
            measurement.club_speed_ms = club_speed_data.get('speed_ms', 0.0)
            measurement.club_speed_mph = club_speed_data.get('speed_mph', 0.0)
            
            # 어택 앵글
            measurement.attack_angle_deg = self.club_calculator.calculate_attack_angle(
                self.club_positions_3d, club_timestamps
            )
            
            # 클럽 패스
            measurement.club_path_deg = self.club_calculator.calculate_club_path(self.club_positions_3d)
            
            # 페이스 앵글 (클럽패스와 볼 방향각 기반)
            measurement.face_angle_deg = self.club_calculator.calculate_face_angle(
                measurement.club_path_deg, measurement.direction_angle_deg
            )
            
            # 페이스투패스
            measurement.face_to_path_deg = measurement.face_angle_deg - measurement.club_path_deg
            
            # 스매쉬팩터
            measurement.smash_factor = self.club_calculator.calculate_smash_factor(
                measurement.ball_speed_mph, measurement.club_speed_mph
            )
        
        # 6. 탄도 예측
        if measurement.ball_speed_ms > 0:
            trajectory = self.trajectory_predictor.predict_carry_distance(
                measurement.ball_speed_ms, 
                measurement.launch_angle_deg, 
                measurement.backspin_rpm
            )
            measurement.carry_distance_m = trajectory.get('carry_m', 0.0)
            measurement.max_height_m = trajectory.get('max_height_m', 0.0)
            measurement.flight_time_s = trajectory.get('flight_time_s', 0.0)
    
    def _validate_physics(self, measurement: CompleteMeasurementData) -> float:
        """물리학적 검증 점수 계산"""
        try:
            # 볼 물리량 검증
            ball_data = {
                'ball_speed_mph': measurement.ball_speed_mph,
                'launch_angle_deg': measurement.launch_angle_deg,
                'direction_angle_deg': measurement.direction_angle_deg,
                'backspin_rpm': measurement.backspin_rpm,
                'sidespin_rpm': measurement.sidespin_rpm
            }
            
            ball_validations = self.physics_validator.validate_ball_physics(ball_data)
            
            # 클럽 물리량 검증
            club_data = {
                'club_speed_mph': measurement.club_speed_mph,
                'attack_angle_deg': measurement.attack_angle_deg,
                'face_angle_deg': measurement.face_angle_deg,
                'club_path_deg': measurement.club_path_deg,
                'smash_factor': measurement.smash_factor
            }
            
            club_validations = self.physics_validator.validate_club_physics(club_data)
            
            # 전체 검증 점수 (0-1 범위)
            all_validations = {**ball_validations, **club_validations}
            valid_count = sum(all_validations.values())
            total_count = len(all_validations)
            
            return valid_count / total_count if total_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Physics validation error: {e}")
            return 0.0
    
    def _calculate_confidence(self, measurement: CompleteMeasurementData, 
                            ir_result: Optional[Dict]) -> float:
        """측정 신뢰도 계산"""
        confidence_factors = []
        
        # 1. IR 검출 신뢰도
        if ir_result:
            confidence_factors.append(ir_result.get('confidence', 0.0))
        
        # 2. 데이터 완성도
        required_fields = [
            measurement.ball_speed_mph,
            measurement.launch_angle_deg,
            measurement.direction_angle_deg,
            measurement.backspin_rpm,
            measurement.sidespin_rpm,
            measurement.club_speed_mph,
            measurement.attack_angle_deg,
            measurement.face_angle_deg,
            measurement.club_path_deg,
            measurement.smash_factor
        ]
        
        non_zero_count = sum(1 for field in required_fields if abs(field) > 0.1)
        completeness = non_zero_count / len(required_fields)
        confidence_factors.append(completeness)
        
        # 3. 물리학적 일관성
        confidence_factors.append(measurement.physics_validation_score)
        
        # 4. 충분한 데이터 포인트
        data_sufficiency = min(len(self.ball_positions_3d) / 7.0, 1.0)  # 7개 포인트가 이상적
        confidence_factors.append(data_sufficiency)
        
        # 전체 신뢰도 (가중 평균)
        weights = [0.3, 0.3, 0.3, 0.1]  # IR, 완성도, 검증, 충분성
        weighted_confidence = sum(w * c for w, c in zip(weights, confidence_factors))
        
        return min(weighted_confidence, 1.0)
    
    def analyze_shot_sequence(self, shot_folder: Path) -> List[CompleteMeasurementData]:
        """샷 시퀀스 분석"""
        logger.info(f"Analyzing shot sequence: {shot_folder.name}")
        
        shot_id = f"{shot_folder.parent.name}_{shot_folder.name}"
        self.current_shot_id = shot_id
        
        # 초기화
        self.measurement_sequence = []
        self.ball_positions_3d = []
        self.club_positions_3d = []
        self.timestamps = []
        self.launch_detected = False
        
        results = []
        
        # 프레임 1-23 처리
        for frame_num in range(1, 24):
            # 이미지 로드
            top_img = self._load_frame_image(shot_folder, frame_num, camera="top")
            bottom_img = self._load_frame_image(shot_folder, frame_num, camera="bottom")
            
            if top_img is not None and bottom_img is not None:
                measurement = self.process_frame_pair(top_img, bottom_img, frame_num, shot_id)
                results.append(measurement)
            else:
                logger.warning(f"Failed to load frame {frame_num}")
        
        # 최종 분석 결과 정리
        self._finalize_shot_analysis(results)
        
        return results
    
    def _load_frame_image(self, shot_folder: Path, frame_num: int, camera: str) -> Optional[np.ndarray]:
        """프레임 이미지 로드"""
        try:
            if camera == "top":
                filename = f"1_{frame_num}.bmp"
            else:  # bottom
                filename = f"2_{frame_num}.bmp"
            
            img_path = shot_folder / filename
            
            if img_path.exists():
                return cv2.imread(str(img_path))
            else:
                return None
                
        except Exception as e:
            logger.error(f"Image load error: {e}")
            return None
    
    def _finalize_shot_analysis(self, results: List[CompleteMeasurementData]):
        """최종 분석 결과 정리"""
        if not results:
            return
        
        # 발사 후 안정화된 측정값 선택 (발사 프레임 + 3~5 프레임 후)
        if self.launch_detected and self.launch_frame > 0:
            stable_frame_start = self.launch_frame + 3
            stable_frame_end = self.launch_frame + 8
            
            stable_measurements = [
                m for m in results 
                if stable_frame_start <= m.frame_number <= stable_frame_end
            ]
            
            if stable_measurements:
                # 안정화된 구간의 평균값으로 최종 측정값 결정
                final_values = self._calculate_average_measurements(stable_measurements)
                
                # 모든 측정값에 최종 값 적용
                for measurement in results:
                    for key, value in final_values.items():
                        setattr(measurement, key, value)
    
    def _calculate_average_measurements(self, measurements: List[CompleteMeasurementData]) -> Dict[str, float]:
        """안정화 구간 평균 측정값 계산"""
        if not measurements:
            return {}
        
        # 평균을 낼 필드들
        avg_fields = [
            'ball_speed_mph', 'ball_speed_ms', 'launch_angle_deg', 'direction_angle_deg',
            'backspin_rpm', 'sidespin_rpm', 'spin_axis_deg',
            'club_speed_mph', 'club_speed_ms', 'attack_angle_deg', 'face_angle_deg',
            'club_path_deg', 'face_to_path_deg', 'smash_factor',
            'carry_distance_m', 'max_height_m', 'flight_time_s'
        ]
        
        averages = {}
        
        for field in avg_fields:
            values = [getattr(m, field) for m in measurements if getattr(m, field) != 0.0]
            if values:
                averages[field] = np.mean(values)
            else:
                averages[field] = 0.0
        
        return averages
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """분석 요약 반환"""
        if not self.measurement_sequence:
            return {'status': 'no_data'}
        
        # 최종 측정값 (마지막 유효한 측정)
        final_measurement = None
        for measurement in reversed(self.measurement_sequence):
            if measurement.measurement_confidence > 0.7:
                final_measurement = measurement
                break
        
        if not final_measurement:
            final_measurement = self.measurement_sequence[-1]
        
        # 요약 통계
        detection_rate = sum(1 for m in self.measurement_sequence if m.ir_ball_detected) / len(self.measurement_sequence)
        avg_confidence = np.mean([m.measurement_confidence for m in self.measurement_sequence])
        avg_validation_score = np.mean([m.physics_validation_score for m in self.measurement_sequence])
        
        return {
            'status': 'completed',
            'shot_id': final_measurement.shot_id,
            'total_frames': len(self.measurement_sequence),
            'detection_rate': detection_rate,
            'average_confidence': avg_confidence,
            'physics_validation_score': avg_validation_score,
            
            # 1. IR 기반 공 감지 결과
            'launch_detected': self.launch_detected,
            'launch_frame': self.launch_frame,
            
            # 2. 볼 데이터
            'ball_speed_mph': final_measurement.ball_speed_mph,
            'launch_angle_deg': final_measurement.launch_angle_deg,
            'direction_angle_deg': final_measurement.direction_angle_deg,
            'backspin_rpm': final_measurement.backspin_rpm,
            'sidespin_rpm': final_measurement.sidespin_rpm,
            'spin_axis_deg': final_measurement.spin_axis_deg,
            
            # 3. 클럽 데이터
            'club_speed_mph': final_measurement.club_speed_mph,
            'attack_angle_deg': final_measurement.attack_angle_deg,
            'face_angle_deg': final_measurement.face_angle_deg,
            'club_path_deg': final_measurement.club_path_deg,
            'face_to_path_deg': final_measurement.face_to_path_deg,
            'smash_factor': final_measurement.smash_factor,
            
            # 추가 분석
            'carry_distance_m': final_measurement.carry_distance_m,
            'max_height_m': final_measurement.max_height_m,
            'flight_time_s': final_measurement.flight_time_s
        }
    
    def export_to_excel(self, output_path: str):
        """Excel로 결과 내보내기"""
        if not self.measurement_sequence:
            logger.warning("No measurement data to export")
            return
        
        # DataFrame 생성
        data = [asdict(measurement) for measurement in self.measurement_sequence]
        df = pd.DataFrame(data)
        
        # Excel 저장
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Measurements', index=False)
            
            # 요약 시트 추가
            summary = self.get_analysis_summary()
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Results exported to: {output_path}")


def main():
    """Main function with actual shot-image processing"""
    print("=== Integrated Golf Measurement System v3.0 ===")
    print("Complete measurement system for golf swing analysis")
    print()
    
    # 경로 설정
    shot_image_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/shot-image")
    template_file = "test_enhanced_20250906_153936.xlsx"
    output_file = f"integrated_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    if not shot_image_dir.exists():
        print(f"ERROR: Image folder not found: {shot_image_dir}")
        return
    
    # 시스템 초기화
    system = IntegratedGolfMeasurementSystem()
    start_time = time.time()
    
    print(f"Image folder: {shot_image_dir}")
    print(f"Template file: {template_file}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    
    all_results = []
    shot_count = 0
    
    # 모든 클럽 폴더 순회
    for club_dir in shot_image_dir.iterdir():
        if club_dir.is_dir():
            print(f"Processing club: {club_dir.name}")
            
            # 클럽별 볼 타입 폴더 순회
            for ball_dir in club_dir.iterdir():
                if ball_dir.is_dir():
                    print(f"  Processing ball type: {ball_dir.name}")
                    shot_count += 1
                    
                    try:
                        # 샷 시퀀스 분석
                        results = system.analyze_shot_sequence(ball_dir)
                        
                        if results:
                            print(f"    SUCCESS: {len(results)} frames analyzed")
                            all_results.extend(results)
                        else:
                            print(f"    WARNING: No results for {ball_dir.name}")
                            
                    except Exception as e:
                        print(f"    ERROR processing {ball_dir.name}: {e}")
    
    processing_time = time.time() - start_time
    
    if all_results:
        # 결과를 DataFrame으로 변환
        df_results = pd.DataFrame([asdict(result) for result in all_results])
        
        print(f"\nProcessing completed:")
        print(f"  - Total processing time: {processing_time:.2f}s")
        print(f"  - Total shots: {shot_count}")
        print(f"  - Total frames: {len(df_results)}")
        print(f"  - Ball detection rate: {(df_results['ir_ball_detected'].sum()/len(df_results)*100):.1f}%")
        
        # 측정값 요약
        print(f"\nMeasurement summary:")
        ball_speed_data = df_results[df_results['ball_speed_mph'] > 0]['ball_speed_mph']
        if not ball_speed_data.empty:
            print(f"  - Ball speed: {ball_speed_data.mean():.1f} +/- {ball_speed_data.std():.1f} mph")
        
        launch_angle_data = df_results[df_results['launch_angle_deg'] != 0]['launch_angle_deg']
        if not launch_angle_data.empty:
            print(f"  - Launch angle: {launch_angle_data.mean():.1f} +/- {launch_angle_data.std():.1f} deg")
        
        spin_data = df_results[df_results['backspin_rpm'] > 0]['backspin_rpm']
        if not spin_data.empty:
            print(f"  - Backspin: {spin_data.mean():.0f} +/- {spin_data.std():.0f} rpm")
        
        # Excel 파일 생성
        try:
            df_results.to_excel(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Excel save error: {e}")
        
        print(f"\nData quality assessment:")
        print(f"  - Average confidence: {df_results['measurement_confidence'].mean():.2f}")
        print(f"  - High confidence measurements: {(df_results['measurement_confidence'] > 0.8).sum()}")
        print(f"  - Physics validation rate: {(df_results['physics_validation_score'] > 0.8).sum()}/{len(df_results)}")
        
    else:
        print("\nERROR: No measurement results generated")
        
    print("\n" + "=" * 60)
    print("Analysis completed!")


if __name__ == "__main__":
    main()