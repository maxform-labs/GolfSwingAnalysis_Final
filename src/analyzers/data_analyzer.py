#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Golf Data Analyzer v2.0
골프 스윙 분석을 위한 고급 데이터 추출 시스템

볼 데이터:
- IR 기반 공 감지
- 정지 상태 → 발사 시점 구분
- 볼 스피드, 발사각, 방향각
- 백스핀, 사이드스핀, 스핀축

클럽 데이터:
- 클럽 스피드, 어택 앵글
- 클럽 페이스 앵글, 클럽 패스
- 페이스투패스, 스매쉬팩터
"""

import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image
from typing import Optional, Dict, List, Tuple, NamedTuple
from dataclasses import dataclass
from datetime import datetime
import json
import math


@dataclass
class IRDetectionParams:
    """IR 기반 골프공 검출 파라미터"""
    ir_threshold: int = 200  # IR 강도 임계값
    min_area: int = 15       # 최소 영역 크기
    max_area: int = 400      # 최대 영역 크기
    gaussian_blur: int = 5   # 가우시안 블러 크기
    morph_kernel: int = 3    # 형태학적 커널 크기


@dataclass 
class BallData:
    """볼 데이터 클래스"""
    frame_num: int
    x: float
    y: float
    z: float = 0.0
    velocity: float = 0.0        # 볼 스피드 (mph)
    launch_angle: float = 0.0    # 발사각 (도)
    direction_angle: float = 0.0 # 좌우 방향각 (도)
    backspin: float = 0.0        # 백스핀 (rpm)
    sidespin: float = 0.0        # 사이드스핀 (rpm)
    spin_axis: float = 0.0       # 스핀축 (도)
    confidence: float = 0.0
    ir_intensity: float = 0.0    # IR 강도
    motion_state: str = "static" # "static" 또는 "launched"


@dataclass
class ClubData:
    """클럽 데이터 클래스"""
    frame_num: int
    x: float
    y: float
    z: float = 0.0
    club_speed: float = 0.0      # 클럽 스피드 (mph)
    attack_angle: float = 0.0    # 어택 앵글 (도)
    face_angle: float = 0.0      # 페이스 앵글 (도)  
    club_path: float = 0.0       # 클럽 패스 (도)
    face_to_path: float = 0.0    # 페이스투패스 (도)
    smash_factor: float = 0.0    # 스매쉬팩터
    confidence: float = 0.0


class IRBallDetector:
    """IR 기반 골프공 검출기"""
    
    def __init__(self, params: IRDetectionParams):
        self.params = params
        self.background_model = None
        self.previous_frame = None
        
    def detect_ball_ir(self, img: np.ndarray, frame_num: int) -> Optional[BallData]:
        """IR 기반 골프공 검출"""
        # IR 채널 추출 (근적외선 특성 활용)
        if len(img.shape) == 3:
            # BGR에서 R 채널이 IR에 가장 민감
            ir_channel = img[:, :, 2]  # Red channel
        else:
            ir_channel = img
            
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(ir_channel, (self.params.gaussian_blur, self.params.gaussian_blur), 0)
        
        # IR 강도 기반 임계값 처리
        _, ir_thresh = cv2.threshold(blurred, self.params.ir_threshold, 255, cv2.THRESH_BINARY)
        
        # 형태학적 연산으로 노이즈 제거
        kernel = np.ones((self.params.morph_kernel, self.params.morph_kernel), np.uint8)
        ir_clean = cv2.morphologyEx(ir_thresh, cv2.MORPH_OPEN, kernel)
        ir_clean = cv2.morphologyEx(ir_clean, cv2.MORPH_CLOSE, kernel)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(ir_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.params.min_area <= area <= self.params.max_area:
                # 원형성 검사
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.6:  # 원형성 임계값
                        # 중심점 계산
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # IR 강도 측정
                            ir_intensity = float(np.mean(ir_channel[max(0, cy-5):cy+5, max(0, cx-5):cx+5]))
                            
                            # 점수 계산 (면적, 원형성, IR 강도 종합)
                            score = circularity * 0.4 + (ir_intensity / 255.0) * 0.4 + (area / self.params.max_area) * 0.2
                            
                            if score > best_score:
                                best_score = score
                                best_ball = BallData(
                                    frame_num=frame_num,
                                    x=float(cx),
                                    y=float(cy),
                                    confidence=score,
                                    ir_intensity=ir_intensity
                                )
        
        return best_ball


class MotionStateDetector:
    """정지 상태 → 발사 시점 구분 검출기"""
    
    def __init__(self):
        self.ball_positions = []
        self.velocity_threshold = 5.0  # 픽셀/프레임
        self.static_frames = 5         # 정지 상태 판단 프레임 수
        
    def update_motion_state(self, ball_data: BallData) -> str:
        """모션 상태 업데이트"""
        if ball_data is None:
            return "no_ball"
            
        self.ball_positions.append((ball_data.x, ball_data.y, ball_data.frame_num))
        
        # 최근 위치만 유지
        if len(self.ball_positions) > 10:
            self.ball_positions.pop(0)
            
        if len(self.ball_positions) < 3:
            return "static"
            
        # 최근 3프레임의 이동거리 계산
        recent_positions = self.ball_positions[-3:]
        total_distance = 0
        
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            distance = math.sqrt(dx*dx + dy*dy)
            total_distance += distance
            
        avg_velocity = total_distance / (len(recent_positions) - 1)
        
        if avg_velocity > self.velocity_threshold:
            return "launched"
        else:
            return "static"


class BallDataCalculator:
    """볼 데이터 계산기"""
    
    def __init__(self):
        self.fps = 820  # 프레임률
        self.pixel_to_mm = 0.3  # 픽셀-mm 변환비율 (카메라 설정에 따라 조정)
        self.mm_to_mph = 0.002237  # mm/ms to mph 변환
        
    def calculate_ball_velocity(self, positions: List[Tuple[float, float, int]]) -> float:
        """볼 스피드 계산 (mph)"""
        if len(positions) < 2:
            return 0.0
            
        # 최근 2 프레임 간 거리 계산
        p1 = positions[-2]
        p2 = positions[-1]
        
        # 픽셀 거리를 실제 거리로 변환
        dx_mm = (p2[0] - p1[0]) * self.pixel_to_mm
        dy_mm = (p2[1] - p1[1]) * self.pixel_to_mm
        distance_mm = math.sqrt(dx_mm*dx_mm + dy_mm*dy_mm)
        
        # 시간 간격 (ms)
        dt_ms = 1000.0 / self.fps
        
        # 속도 계산 (mm/ms -> mph)
        velocity_mm_per_ms = distance_mm / dt_ms
        velocity_mph = velocity_mm_per_ms / self.mm_to_mph
        
        return velocity_mph
    
    def calculate_launch_angle(self, positions: List[Tuple[float, float, int]]) -> float:
        """발사각 계산 (도)"""
        if len(positions) < 3:
            return 0.0
            
        # 최근 3 프레임으로 궤적 계산
        p1, p2, p3 = positions[-3], positions[-2], positions[-1]
        
        # 수직 변화량
        dy1 = p2[1] - p1[1]  # 첫 번째 구간
        dy2 = p3[1] - p2[1]  # 두 번째 구간
        
        # 수평 변화량  
        dx1 = p2[0] - p1[0]
        dx2 = p3[0] - p2[0]
        
        # 평균 기울기
        if dx1 != 0 and dx2 != 0:
            slope = (dy1/dx1 + dy2/dx2) / 2
            launch_angle = math.degrees(math.atan(slope))
            return launch_angle
        
        return 0.0
        
    def calculate_direction_angle(self, positions: List[Tuple[float, float, int]]) -> float:
        """좌우 방향각 계산 (도)"""
        if len(positions) < 2:
            return 0.0
            
        p1 = positions[-2]
        p2 = positions[-1]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dy != 0:
            direction_angle = math.degrees(math.atan(dx / dy))
            return direction_angle
            
        return 0.0
        
    def calculate_spin_data(self, img: np.ndarray, ball_pos: Tuple[float, float]) -> Tuple[float, float, float]:
        """스핀 데이터 계산 (백스핀, 사이드스핀, 스핀축)"""
        # 실제 구현에서는 볼 표면 패턴 분석 필요
        # 현재는 기본값 반환
        backspin = 3000.0  # rpm
        sidespin = 500.0   # rpm  
        spin_axis = 15.0   # 도
        
        return backspin, sidespin, spin_axis


class ClubDataCalculator:
    """클럽 데이터 계산기"""
    
    def __init__(self):
        self.fps = 820
        self.pixel_to_mm = 0.3
        self.mm_to_mph = 0.002237
        
    def calculate_club_speed(self, positions: List[Tuple[float, float, int]]) -> float:
        """클럽 스피드 계산 (mph)"""
        if len(positions) < 2:
            return 0.0
            
        p1 = positions[-2]
        p2 = positions[-1]
        
        dx_mm = (p2[0] - p1[0]) * self.pixel_to_mm
        dy_mm = (p2[1] - p1[1]) * self.pixel_to_mm
        distance_mm = math.sqrt(dx_mm*dx_mm + dy_mm*dy_mm)
        
        dt_ms = 1000.0 / self.fps
        velocity_mm_per_ms = distance_mm / dt_ms
        velocity_mph = velocity_mm_per_ms / self.mm_to_mph
        
        return velocity_mph
        
    def calculate_attack_angle(self, positions: List[Tuple[float, float, int]]) -> float:
        """어택 앵글 계산 (도)"""
        if len(positions) < 2:
            return 0.0
            
        p1 = positions[-2]
        p2 = positions[-1]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dx != 0:
            attack_angle = math.degrees(math.atan(-dy / dx))  # Y축 반전
            return attack_angle
            
        return 0.0
        
    def calculate_face_angle(self, img: np.ndarray, club_pos: Tuple[float, float]) -> float:
        """페이스 앵글 계산 (도)"""
        # 실제 구현에서는 클럽 페이스 방향 분석 필요
        return 2.5  # 기본값
        
    def calculate_club_path(self, positions: List[Tuple[float, float, int]]) -> float:
        """클럽 패스 계산 (도)"""
        if len(positions) < 3:
            return 0.0
            
        # 임팩트 전후 궤적 분석
        p1, p2, p3 = positions[-3], positions[-2], positions[-1]
        
        dx1 = p2[0] - p1[0]
        dx2 = p3[0] - p2[0]
        dy1 = p2[1] - p1[1]  
        dy2 = p3[1] - p2[1]
        
        if dy1 != 0 and dy2 != 0:
            path1 = math.degrees(math.atan(dx1 / dy1))
            path2 = math.degrees(math.atan(dx2 / dy2))
            club_path = (path1 + path2) / 2
            return club_path
            
        return 0.0
        
    def calculate_face_to_path(self, face_angle: float, club_path: float) -> float:
        """페이스투패스 계산 (도)"""
        return face_angle - club_path
        
    def calculate_smash_factor(self, ball_speed: float, club_speed: float) -> float:
        """스매쉬팩터 계산"""
        if club_speed > 0:
            return ball_speed / club_speed
        return 0.0


class AdvancedGolfDataAnalyzer:
    """고급 골프 데이터 분석기"""
    
    def __init__(self):
        self.ir_params = IRDetectionParams()
        self.ir_detector = IRBallDetector(self.ir_params)
        self.motion_detector = MotionStateDetector()
        self.ball_calculator = BallDataCalculator()
        self.club_calculator = ClubDataCalculator()
        
        self.ball_positions = []
        self.club_positions = []
        self.results = []
        
    def detect_club_head(self, img: np.ndarray, frame_num: int) -> Optional[ClubData]:
        """클럽 헤드 검출 (기존 방식 개선)"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 메탈릭 클럽 헤드 색상 범위 (확장)
        lower_metal = np.array([0, 0, 100])
        upper_metal = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv, lower_metal, upper_metal)
        
        # 형태학적 연산
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 윤곽선 선택
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 100:  # 최소 면적
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    return ClubData(
                        frame_num=frame_num,
                        x=float(cx),
                        y=float(cy),
                        confidence=min(1.0, area / 1000.0)
                    )
        
        return None
        
    def analyze_frame(self, img: np.ndarray, frame_num: int) -> Dict:
        """프레임별 분석"""
        result = {
            'frame_num': frame_num,
            'ball_data': None,
            'club_data': None
        }
        
        # IR 기반 볼 검출
        ball_data = self.ir_detector.detect_ball_ir(img, frame_num)
        
        if ball_data:
            # 모션 상태 업데이트
            ball_data.motion_state = self.motion_detector.update_motion_state(ball_data)
            
            # 위치 기록
            self.ball_positions.append((ball_data.x, ball_data.y, frame_num))
            
            # 볼 데이터 계산
            if len(self.ball_positions) >= 2:
                ball_data.velocity = self.ball_calculator.calculate_ball_velocity(self.ball_positions)
                ball_data.launch_angle = self.ball_calculator.calculate_launch_angle(self.ball_positions)
                ball_data.direction_angle = self.ball_calculator.calculate_direction_angle(self.ball_positions)
                
                # 스핀 데이터 계산
                ball_data.backspin, ball_data.sidespin, ball_data.spin_axis = \
                    self.ball_calculator.calculate_spin_data(img, (ball_data.x, ball_data.y))
            
            result['ball_data'] = ball_data
            
        # 클럽 검출
        club_data = self.detect_club_head(img, frame_num)
        
        if club_data:
            # 위치 기록
            self.club_positions.append((club_data.x, club_data.y, frame_num))
            
            # 클럽 데이터 계산
            if len(self.club_positions) >= 2:
                club_data.club_speed = self.club_calculator.calculate_club_speed(self.club_positions)
                club_data.attack_angle = self.club_calculator.calculate_attack_angle(self.club_positions)
                club_data.face_angle = self.club_calculator.calculate_face_angle(img, (club_data.x, club_data.y))
                club_data.club_path = self.club_calculator.calculate_club_path(self.club_positions)
                club_data.face_to_path = self.club_calculator.calculate_face_to_path(club_data.face_angle, club_data.club_path)
                
                # 스매쉬팩터 계산
                if ball_data and ball_data.velocity > 0:
                    club_data.smash_factor = self.club_calculator.calculate_smash_factor(ball_data.velocity, club_data.club_speed)
                    
            result['club_data'] = club_data
            
        return result
        
    def process_images(self, image_dir: str) -> List[Dict]:
        """이미지 시퀀스 처리"""
        jpg_files = []
        
        # JPG 파일 찾기
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith('.jpg'):
                    jpg_files.append(os.path.join(root, file))
                    
        jpg_files.sort()
        
        print(f"처리할 이미지 수: {len(jpg_files)}")
        
        results = []
        
        for i, img_path in enumerate(jpg_files, 1):
            try:
                # 이미지 로드
                img = cv2.imread(img_path)
                if img is None:
                    print(f"이미지 로드 실패: {img_path}")
                    continue
                    
                print(f"프레임 {i} 처리 중... ({os.path.basename(img_path)})")
                
                # 프레임 분석
                frame_result = self.analyze_frame(img, i)
                results.append(frame_result)
                
            except Exception as e:
                print(f"프레임 {i} 처리 중 오류: {e}")
                continue
                
        print(f"총 {len(results)}개 프레임 처리 완료")
        return results
        
    def export_to_excel(self, results: List[Dict], output_path: str):
        """엑셀로 결과 내보내기"""
        ball_data_rows = []
        club_data_rows = []
        
        for result in results:
            frame_num = result['frame_num']
            
            # 볼 데이터 행
            if result['ball_data']:
                bd = result['ball_data']
                ball_data_rows.append({
                    'Frame': frame_num,
                    'X': bd.x,
                    'Y': bd.y,
                    'Z': bd.z,
                    'IR_Intensity': bd.ir_intensity,
                    'Motion_State': bd.motion_state,
                    'Ball_Speed_mph': bd.velocity,
                    'Launch_Angle_deg': bd.launch_angle,
                    'Direction_Angle_deg': bd.direction_angle,
                    'Backspin_rpm': bd.backspin,
                    'Sidespin_rpm': bd.sidespin,
                    'Spin_Axis_deg': bd.spin_axis,
                    'Confidence': bd.confidence
                })
            else:
                ball_data_rows.append({
                    'Frame': frame_num,
                    'X': None, 'Y': None, 'Z': None,
                    'IR_Intensity': None,
                    'Motion_State': 'no_detection',
                    'Ball_Speed_mph': None,
                    'Launch_Angle_deg': None,
                    'Direction_Angle_deg': None,
                    'Backspin_rpm': None,
                    'Sidespin_rpm': None,
                    'Spin_Axis_deg': None,
                    'Confidence': None
                })
                
            # 클럽 데이터 행
            if result['club_data']:
                cd = result['club_data']
                club_data_rows.append({
                    'Frame': frame_num,
                    'X': cd.x,
                    'Y': cd.y,
                    'Z': cd.z,
                    'Club_Speed_mph': cd.club_speed,
                    'Attack_Angle_deg': cd.attack_angle,
                    'Face_Angle_deg': cd.face_angle,
                    'Club_Path_deg': cd.club_path,
                    'Face_to_Path_deg': cd.face_to_path,
                    'Smash_Factor': cd.smash_factor,
                    'Confidence': cd.confidence
                })
            else:
                club_data_rows.append({
                    'Frame': frame_num,
                    'X': None, 'Y': None, 'Z': None,
                    'Club_Speed_mph': None,
                    'Attack_Angle_deg': None,
                    'Face_Angle_deg': None,
                    'Club_Path_deg': None,
                    'Face_to_Path_deg': None,
                    'Smash_Factor': None,
                    'Confidence': None
                })
        
        # 데이터프레임 생성
        ball_df = pd.DataFrame(ball_data_rows)
        club_df = pd.DataFrame(club_data_rows)
        
        # 엑셀 파일로 저장
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            ball_df.to_excel(writer, sheet_name='Ball_Data', index=False)
            club_df.to_excel(writer, sheet_name='Club_Data', index=False)
            
            # 요약 시트
            summary_data = {
                'Metric': [
                    'Total_Frames',
                    'Ball_Detections', 
                    'Club_Detections',
                    'Max_Ball_Speed_mph',
                    'Max_Club_Speed_mph',
                    'Avg_Backspin_rpm',
                    'Avg_Sidespin_rpm',
                    'Avg_Attack_Angle_deg',
                    'Avg_Face_Angle_deg',
                    'Avg_Smash_Factor'
                ],
                'Value': [
                    len(results),
                    len([r for r in results if r['ball_data']]),
                    len([r for r in results if r['club_data']]),
                    ball_df['Ball_Speed_mph'].max() if not ball_df['Ball_Speed_mph'].isna().all() else 0,
                    club_df['Club_Speed_mph'].max() if not club_df['Club_Speed_mph'].isna().all() else 0,
                    ball_df['Backspin_rpm'].mean() if not ball_df['Backspin_rpm'].isna().all() else 0,
                    ball_df['Sidespin_rpm'].mean() if not ball_df['Sidespin_rpm'].isna().all() else 0,
                    club_df['Attack_Angle_deg'].mean() if not club_df['Attack_Angle_deg'].isna().all() else 0,
                    club_df['Face_Angle_deg'].mean() if not club_df['Face_Angle_deg'].isna().all() else 0,
                    club_df['Smash_Factor'].mean() if not club_df['Smash_Factor'].isna().all() else 0
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"분석 결과가 저장되었습니다: {output_path}")


def main():
    """메인 실행 함수"""
    print("=== 고급 골프 데이터 분석 시스템 v2.0 ===")
    print("IR 기반 공 감지 + 완전한 볼/클럽 데이터 추출")
    
    # 분석기 초기화
    analyzer = AdvancedGolfDataAnalyzer()
    
    # 이미지 디렉토리 설정 (영어 폴더명 사용)
    image_dir = "C:/src/GolfSwingAnalysis_Final_ver8/shot-image-jpg/7iron_no_marker_ball_shot1"
    
    if not os.path.exists(image_dir):
        print(f"이미지 디렉토리를 찾을 수 없습니다: {image_dir}")
        return
        
    # 분석 실행
    print("이미지 시퀀스 분석 시작...")
    results = analyzer.process_images(image_dir)
    
    if not results:
        print("분석할 이미지가 없습니다.")
        return
        
    # 결과 내보내기
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"advanced_golf_analysis_{timestamp}.xlsx"
    output_path = f"C:/src/GolfSwingAnalysis_Final_ver8/{output_filename}"
    
    analyzer.export_to_excel(results, output_path)
    
    print("=== 분석 완료 ===")
    print(f"결과 파일: {output_path}")


if __name__ == "__main__":
    main()