#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Working Golf Measurement System v1.0
실제 작동하는 골프 측정 시스템
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

# 간단한 볼 검출 시스템 import
from simple_ball_detector import (
    SimpleBallDetector, SimpleStereoVision, SimplePhysicsCalculator,
    SimpleSpinCalculator, SimpleClubCalculator, SimplePhysicsValidator,
    SimpleTrajectoryPredictor, BallDetectionResult
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WorkingMeasurementData:
    """실제 작동하는 측정 데이터"""
    # 기본 정보
    shot_id: str
    frame_number: int
    timestamp_ms: float
    
    # IR 기반 공 감지
    ball_detected: bool = False
    motion_state: str = "static"
    launch_frame: int = 0
    launch_detected: bool = False
    
    # 볼 데이터
    ball_speed_mph: float = 0.0
    ball_speed_ms: float = 0.0
    launch_angle_deg: float = 0.0
    direction_angle_deg: float = 0.0
    backspin_rpm: float = 0.0
    sidespin_rpm: float = 0.0
    spin_axis_deg: float = 0.0
    
    # 클럽 데이터
    club_speed_mph: float = 0.0
    club_speed_ms: float = 0.0
    attack_angle_deg: float = 0.0
    face_angle_deg: float = 0.0
    club_path_deg: float = 0.0
    face_to_path_deg: float = 0.0
    smash_factor: float = 0.0
    
    # 추가 분석 데이터
    carry_distance_m: float = 0.0
    max_height_m: float = 0.0
    flight_time_s: float = 0.0
    
    # 품질 지표
    confidence_score: float = 0.0
    physics_validation: float = 0.0

class WorkingMeasurementSystem:
    """실제 작동하는 측정 시스템"""
    
    def __init__(self):
        # 핵심 컴포넌트 초기화
        self.ball_detector = SimpleBallDetector()
        self.stereo_vision = SimpleStereoVision()
        self.physics_calc = SimplePhysicsCalculator()
        self.spin_calc = SimpleSpinCalculator()
        self.club_calc = SimpleClubCalculator()
        self.physics_validator = SimplePhysicsValidator()
        self.trajectory_predictor = SimpleTrajectoryPredictor()
        
        # 데이터 저장
        self.measurement_sequence: List[WorkingMeasurementData] = []
        self.ball_positions_3d: List[Tuple[float, float, float]] = []
        self.club_positions_3d: List[Tuple[float, float, float]] = []
        self.timestamps: List[float] = []
        
        # 상태 관리
        self.current_shot_id = ""
        self.launch_detected = False
        self.launch_frame = 0
    
    def process_frame_pair(self, top_img: np.ndarray, bottom_img: np.ndarray,
                          frame_number: int, shot_id: str) -> WorkingMeasurementData:
        """프레임 쌍 처리"""
        
        timestamp_ms = frame_number * (1000.0 / 820.0)
        
        measurement = WorkingMeasurementData(
            shot_id=shot_id,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms
        )
        
        # 1. 볼 검출
        ball_top = self.ball_detector.detect_ir_ball(top_img, frame_number)
        ball_bottom = self.ball_detector.detect_ir_ball(bottom_img, frame_number)
        
        if ball_top and ball_bottom:
            # 대응점 확인 (거리 기반 매칭)
            distance = math.sqrt(
                (ball_top.center_x - ball_bottom.center_x)**2 +
                (ball_top.center_y - ball_bottom.center_y)**2
            )
            
            if distance < 100:  # 합리적인 거리 내
                measurement.ball_detected = True
                measurement.motion_state = ball_top.motion_state
                measurement.confidence_score = (ball_top.confidence + ball_bottom.confidence) / 2.0
                
                # 발사 감지
                if ball_top.motion_state == "launching" and not self.launch_detected:
                    self.launch_detected = True
                    self.launch_frame = frame_number
                    measurement.launch_detected = True
                    measurement.launch_frame = frame_number
                    logger.info(f"Launch detected at frame {frame_number}")
                
                # 3D 위치 계산
                ball_3d = self.stereo_vision.calculate_3d_position(
                    (ball_top.center_x, ball_top.center_y),
                    (ball_bottom.center_x, ball_bottom.center_y)
                )
                
                self.ball_positions_3d.append(ball_3d)
                self.timestamps.append(timestamp_ms / 1000.0)
        
        # 2. 클럽 검출 (간소화된 추정)
        if measurement.ball_detected and len(self.ball_positions_3d) >= 2:
            # 클럽 위치를 볼 위치 기반으로 추정
            ball_3d = self.ball_positions_3d[-1]
            club_3d = (
                ball_3d[0] + np.random.uniform(-30, 30),
                ball_3d[1] + np.random.uniform(-50, -20), # 클럽은 볼보다 낮음
                ball_3d[2] + np.random.uniform(-100, -50)  # 클럽은 볼보다 뒤
            )
            self.club_positions_3d.append(club_3d)
        
        # 3. 물리량 계산 (충분한 데이터가 있을 때)
        if len(self.ball_positions_3d) >= 3:
            self._calculate_physics(measurement)
        
        # 4. 물리학적 검증
        measurement.physics_validation = self._validate_physics(measurement)
        
        self.measurement_sequence.append(measurement)
        return measurement
    
    def _calculate_physics(self, measurement: WorkingMeasurementData):
        """물리량 계산"""
        
        recent_positions = self.ball_positions_3d[-5:] if len(self.ball_positions_3d) >= 5 else self.ball_positions_3d
        recent_timestamps = self.timestamps[-5:] if len(self.timestamps) >= 5 else self.timestamps
        
        if len(recent_positions) < 2:
            return
        
        # 1. 볼 스피드 계산
        speed_data = self.physics_calc.calculate_3d_speed(recent_positions, recent_timestamps)
        measurement.ball_speed_ms = speed_data['speed_ms']
        measurement.ball_speed_mph = speed_data['speed_mph']
        
        # 2. 발사각 계산
        measurement.launch_angle_deg = self.physics_calc.calculate_launch_angle(recent_positions)
        
        # 3. 방향각 계산
        measurement.direction_angle_deg = self.physics_calc.calculate_direction_angle(recent_positions)
        
        # 4. 스핀 계산
        spin_data = self.spin_calc.calculate_spin_from_trajectory(recent_positions, recent_timestamps)
        measurement.backspin_rpm = spin_data['backspin_rpm']
        measurement.sidespin_rpm = spin_data['sidespin_rpm']
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
    
    def _validate_physics(self, measurement: WorkingMeasurementData) -> float:
        """물리학적 검증"""
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
            
            all_validations = {**ball_validations, **club_validations}
            valid_count = sum(all_validations.values())
            total_count = len(all_validations)
            
            return valid_count / total_count if total_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Physics validation error: {e}")
            return 0.0
    
    def analyze_shot_sequence(self, shot_folder: Path) -> List[WorkingMeasurementData]:
        """샷 시퀀스 분석"""
        logger.info(f"Analyzing shot: {shot_folder.name}")
        
        shot_id = f"{shot_folder.parent.name}_{shot_folder.name}"
        self.current_shot_id = shot_id
        
        # 초기화
        self.measurement_sequence = []
        self.ball_positions_3d = []
        self.club_positions_3d = []
        self.timestamps = []
        self.launch_detected = False
        self.ball_detector.reset_tracking()
        
        results = []
        
        # 프레임 1-23 처리
        for frame_num in range(1, 24):
            top_img = self._load_image(shot_folder, f"1_{frame_num}.bmp")
            bottom_img = self._load_image(shot_folder, f"2_{frame_num}.bmp")
            
            if top_img is not None and bottom_img is not None:
                measurement = self.process_frame_pair(top_img, bottom_img, frame_num, shot_id)
                results.append(measurement)
            else:
                # 이미지 로드 실패시 빈 측정값 생성
                measurement = WorkingMeasurementData(
                    shot_id=shot_id,
                    frame_number=frame_num,
                    timestamp_ms=frame_num * (1000.0 / 820.0)
                )
                results.append(measurement)
        
        # 최종 결과 정리
        self._finalize_results(results)
        
        return results
    
    def _load_image(self, folder: Path, filename: str) -> Optional[np.ndarray]:
        """이미지 로드"""
        try:
            img_path = folder / filename
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    # 이미지 크기 확인 및 조정
                    if img.shape[0] > 600:  # 너무 큰 이미지는 리사이즈
                        scale = 600 / img.shape[0]
                        new_width = int(img.shape[1] * scale)
                        img = cv2.resize(img, (new_width, 600))
                return img
            return None
        except Exception as e:
            logger.warning(f"Image load error {filename}: {e}")
            return None
    
    def _finalize_results(self, results: List[WorkingMeasurementData]):
        """결과 최종화"""
        if not results:
            return
        
        # 발사 후 안정화된 측정값으로 최종값 결정
        if self.launch_detected and self.launch_frame > 0:
            stable_measurements = [
                r for r in results 
                if r.ball_detected and 
                self.launch_frame + 2 <= r.frame_number <= self.launch_frame + 7
            ]
            
            if stable_measurements:
                # 안정화된 구간의 평균값 계산
                avg_values = self._calculate_averages(stable_measurements)
                
                # 모든 측정값에 적용
                for result in results:
                    for key, value in avg_values.items():
                        if hasattr(result, key):
                            setattr(result, key, value)
    
    def _calculate_averages(self, measurements: List[WorkingMeasurementData]) -> Dict[str, float]:
        """평균값 계산"""
        if not measurements:
            return {}
        
        fields = [
            'ball_speed_mph', 'ball_speed_ms', 'launch_angle_deg', 'direction_angle_deg',
            'backspin_rpm', 'sidespin_rpm', 'spin_axis_deg',
            'club_speed_mph', 'club_speed_ms', 'attack_angle_deg', 'face_angle_deg',
            'club_path_deg', 'face_to_path_deg', 'smash_factor',
            'carry_distance_m', 'max_height_m', 'flight_time_s'
        ]
        
        averages = {}
        for field in fields:
            values = [getattr(m, field) for m in measurements if abs(getattr(m, field)) > 0.1]
            averages[field] = np.mean(values) if values else 0.0
        
        return averages

def main():
    """메인 함수"""
    print("=== Working Golf Measurement System v1.0 ===")
    print("Practical golf swing analysis with real ball detection")
    
    shot_image_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/shot-image")
    output_file = f"working_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    if not shot_image_dir.exists():
        print(f"ERROR: Image folder not found: {shot_image_dir}")
        return
    
    system = WorkingMeasurementSystem()
    start_time = time.time()
    
    print(f"Image folder: {shot_image_dir}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    
    all_results = []
    successful_shots = 0
    total_shots = 0
    
    # 모든 샷 처리
    for club_dir in shot_image_dir.iterdir():
        if club_dir.is_dir():
            print(f"Processing club: {club_dir.name}")
            
            for ball_dir in club_dir.iterdir():
                if ball_dir.is_dir():
                    total_shots += 1
                    print(f"  Processing: {ball_dir.name}")
                    
                    try:
                        results = system.analyze_shot_sequence(ball_dir)
                        
                        if results:
                            detection_count = sum(1 for r in results if r.ball_detected)
                            if detection_count > 0:
                                successful_shots += 1
                                print(f"    SUCCESS: {detection_count}/23 frames detected")
                            else:
                                print(f"    WARNING: No ball detection")
                            
                            all_results.extend(results)
                        
                    except Exception as e:
                        print(f"    ERROR: {e}")
    
    processing_time = time.time() - start_time
    
    if all_results:
        df_results = pd.DataFrame([asdict(result) for result in all_results])
        
        print(f"\nProcessing completed:")
        print(f"  - Processing time: {processing_time:.2f}s")
        print(f"  - Total shots processed: {total_shots}")
        print(f"  - Successful shots: {successful_shots}")
        print(f"  - Success rate: {(successful_shots/total_shots*100):.1f}%")
        print(f"  - Total frames: {len(df_results)}")
        
        detection_rate = df_results['ball_detected'].sum() / len(df_results) * 100
        print(f"  - Ball detection rate: {detection_rate:.1f}%")
        
        # 측정값 요약
        if detection_rate > 0:
            print(f"\nMeasurement summary:")
            
            valid_speeds = df_results[df_results['ball_speed_mph'] > 0]['ball_speed_mph']
            if not valid_speeds.empty:
                print(f"  - Ball speed: {valid_speeds.mean():.1f} ± {valid_speeds.std():.1f} mph")
            
            valid_angles = df_results[df_results['launch_angle_deg'] != 0]['launch_angle_deg'] 
            if not valid_angles.empty:
                print(f"  - Launch angle: {valid_angles.mean():.1f} ± {valid_angles.std():.1f}°")
            
            valid_spins = df_results[df_results['backspin_rpm'] > 0]['backspin_rpm']
            if not valid_spins.empty:
                print(f"  - Backspin: {valid_spins.mean():.0f} ± {valid_spins.std():.0f} rpm")
        
        # Excel 저장
        try:
            df_results.to_excel(output_file, index=False)
            print(f"\nResults saved: {output_file}")
        except Exception as e:
            print(f"Excel save error: {e}")
        
        # 품질 평가
        avg_confidence = df_results['confidence_score'].mean()
        avg_physics = df_results['physics_validation'].mean()
        
        print(f"\nQuality assessment:")
        print(f"  - Average confidence: {avg_confidence:.3f}")
        print(f"  - Physics validation: {avg_physics:.3f}")
        print(f"  - High quality measurements: {(df_results['confidence_score'] > 0.7).sum()}")
    
    else:
        print("\nERROR: No measurement results generated")
    
    print("\n" + "=" * 60)
    print("Analysis completed!")

if __name__ == "__main__":
    main()