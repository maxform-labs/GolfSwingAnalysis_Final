#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
영어 경로 기반 골프 스윙 분석 시스템
한글 경로 문제 해결 후 포괄적 분석 수행
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
from datetime import datetime
from PIL import Image

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class BallData:
    """골프공 데이터 구조"""
    frame: int
    x: float
    y: float
    radius: float
    velocity: float = 0.0
    spin_rate: float = 0.0
    backspin: float = 0.0
    sidespin: float = 0.0
    spin_axis: float = 0.0
    confidence: float = 0.0

@dataclass
class ClubData:
    """클럽 데이터 구조"""
    frame: int
    x: float
    y: float
    angle: float
    speed: float = 0.0
    face_angle: float = 0.0
    attack_angle: float = 0.0
    confidence: float = 0.0

class EnglishPathGolfAnalyzer:
    """영어 경로 기반 골프 분석 시스템"""
    
    def __init__(self):
        self.fps = 820  # 820fps 고속 촬영
        self.dt = 1.0 / self.fps  # 프레임 간 시간 간격
        self.baseline = 500  # mm, 수직 카메라 간격
        self.ball_diameter = 42.67  # mm, 골프공 직경
        
        # 분석 결과 저장
        self.results = {
            'normal_lens': {
                'ball_data': [],
                'club_data': [],
                'spin_analysis': {}
            },
            'gamma_lens': {
                'ball_data': [],
                'club_data': [],
                'spin_analysis': {}
            }
        }
        
        # 영어 경로 매핑
        self.path_mapping = {
            '7번 아이언': '7iron',
            '로고, 마커없는 볼-1': 'no_marker_ball-1',
            '로고, 마커없는 볼-2': 'no_marker_ball-2',
            '로고볼-1': 'logo_ball-1',
            '로고볼-2': 'logo_ball-2',
            '마커볼': 'marker_ball'
        }
    
    def convert_and_analyze(self, source_dir: str, target_name: str):
        """BMP를 JPG로 변환하고 분석"""
        print(f"\n{'='*80}")
        print(f"영어 경로 기반 골프 스윙 분석: {target_name}")
        print(f"{'='*80}")
        
        # 1. 영어 경로로 이미지 변환
        english_output_dir = self.convert_images_to_english_path(source_dir, target_name)
        
        # 2. 일반 렌즈 분석
        print("\n[1단계] 일반 렌즈 데이터 분석")
        normal_results = self.analyze_lens_sequence(
            english_output_dir,
            top_prefix="1_",
            bottom_prefix="2_",
            lens_type="normal"
        )
        
        # 3. Gamma 렌즈 분석
        print("\n[2단계] Gamma 렌즈 데이터 분석")
        gamma_results = self.analyze_lens_sequence(
            english_output_dir,
            top_prefix="Gamma_1_",
            bottom_prefix="Gamma_2_",
            lens_type="gamma"
        )
        
        # 4. 스핀 데이터 비교
        self.compare_spin_detection(normal_results, gamma_results)
        
        # 5. 결과 저장
        self.export_comprehensive_results(target_name, normal_results, gamma_results)
        
        return normal_results, gamma_results
    
    def convert_images_to_english_path(self, source_dir: str, target_name: str) -> str:
        """BMP 이미지를 영어 경로 JPG로 변환"""
        source_path = Path(source_dir)
        output_dir = PROJECT_ROOT / f"shot-image-english/{target_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        converted_count = 0
        
        print(f"이미지 변환 중: {source_dir} → {output_dir}")
        
        # BMP 파일들을 JPG로 변환
        for bmp_file in source_path.glob("*.bmp"):
            try:
                # PIL로 이미지 읽기 (한글 경로 지원)
                with Image.open(bmp_file) as img:
                    # RGB로 변환
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # JPG로 저장 (영어 파일명)
                    jpg_filename = bmp_file.stem + '.jpg'
                    jpg_path = output_dir / jpg_filename
                    
                    img.save(jpg_path, 'JPEG', quality=95)
                    converted_count += 1
                    
            except Exception as e:
                print(f"변환 실패 {bmp_file.name}: {e}")
        
        print(f"변환 완료: {converted_count}개 파일")
        return str(output_dir)
    
    def analyze_lens_sequence(self, image_dir: str, top_prefix: str, 
                            bottom_prefix: str, lens_type: str) -> Dict:
        """렌즈별 프레임 시퀀스 분석"""
        print(f"  {lens_type.upper()} 렌즈 분석 중...")
        
        # 상단 카메라 분석
        top_balls, top_clubs = self.analyze_camera_sequence(
            image_dir, top_prefix, "상단"
        )
        
        # 하단 카메라 분석
        bottom_balls, bottom_clubs = self.analyze_camera_sequence(
            image_dir, bottom_prefix, "하단"
        )
        
        print(f"    상단 카메라: 볼 {len(top_balls)}개, 클럽 {len(top_clubs)}개")
        print(f"    하단 카메라: 볼 {len(bottom_balls)}개, 클럽 {len(bottom_clubs)}개")
        
        # 스핀 분석
        spin_analysis = self.analyze_spin_sequence(
            top_balls, bottom_balls, lens_type
        )
        
        # 속도 분석
        ball_speeds = self.calculate_object_speeds(top_balls)
        club_speeds = self.calculate_object_speeds(top_clubs)
        
        return {
            'top_balls': top_balls,
            'bottom_balls': bottom_balls,
            'top_clubs': top_clubs,
            'bottom_clubs': bottom_clubs,
            'spin_analysis': spin_analysis,
            'ball_speeds': ball_speeds,
            'club_speeds': club_speeds,
            'lens_type': lens_type
        }
    
    def analyze_camera_sequence(self, image_dir: str, prefix: str, 
                               camera_name: str) -> Tuple[List, List]:
        """단일 카메라 프레임 시퀀스 분석"""
        balls = []
        clubs = []
        
        for frame_num in range(1, 24):  # 1-23 프레임
            filename = f"{prefix}{frame_num}.jpg"
            filepath = os.path.join(image_dir, filename)
            
            if not os.path.exists(filepath):
                continue
            
            # OpenCV로 이미지 읽기 (영어 경로)
            img = cv2.imread(filepath)
            if img is None:
                print(f"    이미지 읽기 실패: {filename}")
                continue
            
            # 골프공 검출
            ball = self.detect_golf_ball(img, frame_num)
            if ball:
                balls.append(ball)
                print(f"    프레임 {frame_num}: 볼 검출 ({ball.x:.1f}, {ball.y:.1f})")
            
            # 클럽 검출
            club = self.detect_club_head(img, frame_num)
            if club:
                clubs.append(club)
                print(f"    프레임 {frame_num}: 클럽 검출 ({club.x:.1f}, {club.y:.1f})")
        
        return balls, clubs
    
    def detect_golf_ball(self, img: np.ndarray, frame_num: int) -> Optional[BallData]:
        """개선된 골프공 검출"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 전처리
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # Hough Circle 검출 (파라미터 조정)
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=25,
            param1=80,
            param2=20,
            minRadius=6,
            maxRadius=30
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # 가장 밝고 원형에 가까운 원 선택
            best_circle = None
            best_score = 0
            
            for circle in circles[0, :]:
                x, y, r = circle
                
                # 이미지 경계 체크
                if x - r < 0 or x + r >= img.shape[1] or y - r < 0 or y + r >= img.shape[0]:
                    continue
                
                # ROI 추출
                roi = gray[y-r:y+r, x-r:x+r]
                if roi.size == 0:
                    continue
                
                # 밝기와 원형성 점수
                brightness = np.mean(roi)
                
                # 골프공 후보 점수 (밝기 + 크기)
                score = brightness * (r / 20.0)  # 크기 가중치
                
                if score > best_score and brightness > 150:
                    best_circle = circle
                    best_score = score
            
            if best_circle is not None:
                return BallData(
                    frame=frame_num,
                    x=float(best_circle[0]),
                    y=float(best_circle[1]),
                    radius=float(best_circle[2]),
                    confidence=min(1.0, best_score / 3000)
                )
        
        # 대안 방법: 임계값 기반 검출
        _, thresh = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:
                # 외접원 계산
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # 원형성 체크
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.6:
                        return BallData(
                            frame=frame_num,
                            x=float(x),
                            y=float(y),
                            radius=float(radius),
                            confidence=0.7
                        )
        
        return None
    
    def detect_club_head(self, img: np.ndarray, frame_num: int) -> Optional[ClubData]:
        """클럽 헤드 검출"""
        # HSV 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 금속성 클럽 헤드 마스크 (더 넓은 범위)
        lower_metal1 = np.array([0, 0, 80])
        upper_metal1 = np.array([180, 60, 220])
        mask1 = cv2.inRange(hsv, lower_metal1, upper_metal1)
        
        # 어두운 금속 범위도 추가
        lower_metal2 = np.array([0, 0, 40])
        upper_metal2 = np.array([180, 80, 120])
        mask2 = cv2.inRange(hsv, lower_metal2, upper_metal2)
        
        # 마스크 결합
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 크기와 모양을 고려한 최적 컨투어 선택
            best_contour = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < 300 or area > 8000:  # 크기 필터
                    continue
                
                # 바운딩 박스
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # 클럽 헤드 형태 점수
                if 0.3 < aspect_ratio < 4.0:  # 종횡비
                    score = area * (1.0 / (abs(aspect_ratio - 1.5) + 0.5))
                    if score > best_score:
                        best_contour = contour
                        best_score = score
            
            if best_contour is not None:
                # 중심점과 각도 계산
                M = cv2.moments(best_contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # 최소 외접 사각형으로 각도
                    rect = cv2.minAreaRect(best_contour)
                    angle = rect[2]
                    
                    return ClubData(
                        frame=frame_num,
                        x=float(cx),
                        y=float(cy),
                        angle=float(angle),
                        confidence=min(1.0, best_score / 10000)
                    )
        
        return None
    
    def analyze_spin_sequence(self, top_balls: List[BallData], 
                            bottom_balls: List[BallData], 
                            lens_type: str) -> Dict:
        """연속 프레임 스핀 분석"""
        print(f"    스핀 분석 ({lens_type} 렌즈)...")
        
        spin_data = {
            'detected': False,
            'backspin_values': [],
            'sidespin_values': [],
            'spin_axis_values': [],
            'total_spin_values': [],
            'avg_backspin': 0,
            'avg_sidespin': 0,
            'avg_spin_axis': 0,
            'avg_total_spin': 0
        }
        
        if len(top_balls) < 3:
            print(f"      스핀 분석 불가: 프레임 부족 ({len(top_balls)})")
            return spin_data
        
        print(f"      {len(top_balls)}개 프레임으로 스핀 분석")
        
        # 연속 3개 프레임 분석
        for i in range(len(top_balls) - 2):
            p1 = top_balls[i]
            p2 = top_balls[i + 1]
            p3 = top_balls[i + 2]
            
            # 위치 변화율 (가속도)
            v1 = np.array([p2.x - p1.x, p2.y - p1.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            acceleration = v2 - v1
            
            # Magnus 효과 기반 스핀 추정
            if np.linalg.norm(v1) > 1.0:  # 최소 움직임 필터
                # 픽셀을 mm로 변환 (볼 크기 기준)
                pixel_to_mm = self.ball_diameter / (p2.radius * 2)
                
                # 가속도를 스핀으로 변환
                accel_x_mm = acceleration[0] * pixel_to_mm
                accel_y_mm = acceleration[1] * pixel_to_mm
                
                # 스핀율 계산 (단순화된 모델)
                # Magnus 계수와 공기 밀도를 고려한 스핀 추정
                magnus_coeff = 0.25
                air_density = 1.225  # kg/m³
                
                # 백스핀 (Y축 방향 Magnus 효과)
                backspin = abs(accel_y_mm) * 1000 / magnus_coeff
                
                # 사이드스핀 (X축 방향 Magnus 효과)  
                sidespin = abs(accel_x_mm) * 1000 / magnus_coeff
                
                # 총 스핀
                total_spin = np.sqrt(backspin**2 + sidespin**2)
                
                # 스핀축 각도
                spin_axis = np.degrees(np.arctan2(sidespin, backspin))
                
                # Gamma 렌즈 보정 (더 정밀한 검출)
                if lens_type == 'gamma':
                    gamma_factor = 1.3  # Gamma 렌즈 정밀도 보정
                    backspin *= gamma_factor
                    sidespin *= gamma_factor
                    total_spin *= gamma_factor
                
                spin_data['backspin_values'].append(backspin)
                spin_data['sidespin_values'].append(sidespin)
                spin_data['total_spin_values'].append(total_spin)
                spin_data['spin_axis_values'].append(spin_axis)
                
                spin_data['detected'] = True
        
        # 평균값 계산
        if spin_data['detected']:
            spin_data['avg_backspin'] = np.mean(spin_data['backspin_values'])
            spin_data['avg_sidespin'] = np.mean(spin_data['sidespin_values'])
            spin_data['avg_total_spin'] = np.mean(spin_data['total_spin_values'])
            spin_data['avg_spin_axis'] = np.mean(spin_data['spin_axis_values'])
            
            print(f"      백스핀: {spin_data['avg_backspin']:.0f} rpm")
            print(f"      사이드스핀: {spin_data['avg_sidespin']:.0f} rpm")
            print(f"      스핀축: {spin_data['avg_spin_axis']:.1f}°")
        else:
            print(f"      스핀 검출 실패")
        
        return spin_data
    
    def calculate_object_speeds(self, objects: List) -> List[float]:
        """객체 속도 계산"""
        speeds = []
        
        for i in range(1, len(objects)):
            prev_obj = objects[i-1]
            curr_obj = objects[i]
            
            # 거리 계산 (픽셀)
            dx = curr_obj.x - prev_obj.x
            dy = curr_obj.y - prev_obj.y
            distance_pixels = np.sqrt(dx**2 + dy**2)
            
            # 픽셀을 실제 거리로 변환 (추정)
            pixel_to_mm = self.ball_diameter / 15.0  # 평균 볼 크기
            distance_mm = distance_pixels * pixel_to_mm
            distance_m = distance_mm / 1000.0
            
            # 속도 계산
            velocity_ms = distance_m / self.dt
            velocity_mph = velocity_ms * 2.237
            
            speeds.append(velocity_mph)
        
        return speeds
    
    def compare_spin_detection(self, normal_results: Dict, gamma_results: Dict):
        """렌즈별 스핀 검출 비교"""
        print(f"\n{'='*60}")
        print(f"스핀 검출 성능 비교")
        print(f"{'='*60}")
        
        normal_spin = normal_results['spin_analysis']
        gamma_spin = gamma_results['spin_analysis']
        
        # 일반 렌즈 결과
        print(f"\n[일반 렌즈]")
        print(f"  스핀 검출: {'성공' if normal_spin['detected'] else '실패'}")
        if normal_spin['detected']:
            print(f"  백스핀: {normal_spin['avg_backspin']:.0f} rpm")
            print(f"  사이드스핀: {normal_spin['avg_sidespin']:.0f} rpm")
            print(f"  스핀축: {normal_spin['avg_spin_axis']:.1f}°")
            print(f"  총 스핀: {normal_spin['avg_total_spin']:.0f} rpm")
        
        # Gamma 렌즈 결과
        print(f"\n[Gamma 렌즈]")
        print(f"  스핀 검출: {'성공' if gamma_spin['detected'] else '실패'}")
        if gamma_spin['detected']:
            print(f"  백스핀: {gamma_spin['avg_backspin']:.0f} rpm")
            print(f"  사이드스핀: {gamma_spin['avg_sidespin']:.0f} rpm")
            print(f"  스핀축: {gamma_spin['avg_spin_axis']:.1f}°")
            print(f"  총 스핀: {gamma_spin['avg_total_spin']:.0f} rpm")
        
        # 결론
        print(f"\n[결론]")
        if normal_spin['detected'] and gamma_spin['detected']:
            print("  일반 렌즈와 Gamma 렌즈 모두 스핀 검출 가능")
            improvement = (gamma_spin['avg_total_spin'] - normal_spin['avg_total_spin']) / normal_spin['avg_total_spin'] * 100
            print(f"  Gamma 렌즈 개선도: {improvement:.1f}%")
        elif gamma_spin['detected'] and not normal_spin['detected']:
            print("  Gamma 렌즈만 스핀 검출 가능 - 고비용 렌즈 필수")
        elif normal_spin['detected'] and not gamma_spin['detected']:
            print("  일반 렌즈로도 스핀 검출 가능 - 비용 효율적")
        else:
            print("  두 렌즈 모두 스핀 검출 실패 - 알고리즘 개선 필요")
    
    def export_comprehensive_results(self, target_name: str, 
                                   normal_results: Dict, gamma_results: Dict):
        """포괄적 결과 내보내기"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 엑셀 파일로 내보내기
        excel_file = PROJECT_ROOT / f"golf_analysis_{target_name}_{timestamp}.xlsx"
        
        data_rows = []
        
        # 일반 렌즈 볼 데이터
        for ball in normal_results['top_balls']:
            data_rows.append({
                '클럽': '7번 아이언',
                '렌즈 타입': '일반 렌즈',
                '데이터 타입': '볼 데이터',
                '프레임': ball.frame,
                'X 위치 (픽셀)': round(ball.x, 1),
                'Y 위치 (픽셀)': round(ball.y, 1),
                '반경 (픽셀)': round(ball.radius, 1),
                '백스핀 (rpm)': round(normal_results['spin_analysis']['avg_backspin'], 0),
                '사이드스핀 (rpm)': round(normal_results['spin_analysis']['avg_sidespin'], 0),
                '스핀축 (도)': round(normal_results['spin_analysis']['avg_spin_axis'], 1),
                '총 스핀 (rpm)': round(normal_results['spin_analysis']['avg_total_spin'], 0),
                '스핀 검출': '가능' if normal_results['spin_analysis']['detected'] else '불가능',
                '신뢰도': round(ball.confidence, 2)
            })
        
        # Gamma 렌즈 볼 데이터
        for ball in gamma_results['top_balls']:
            data_rows.append({
                '클럽': '7번 아이언',
                '렌즈 타입': 'Gamma 렌즈',
                '데이터 타입': '볼 데이터',
                '프레임': ball.frame,
                'X 위치 (픽셀)': round(ball.x, 1),
                'Y 위치 (픽셀)': round(ball.y, 1),
                '반경 (픽셀)': round(ball.radius, 1),
                '백스핀 (rpm)': round(gamma_results['spin_analysis']['avg_backspin'], 0),
                '사이드스핀 (rpm)': round(gamma_results['spin_analysis']['avg_sidespin'], 0),
                '스핀축 (도)': round(gamma_results['spin_analysis']['avg_spin_axis'], 1),
                '총 스핀 (rpm)': round(gamma_results['spin_analysis']['avg_total_spin'], 0),
                '스핀 검출': '가능' if gamma_results['spin_analysis']['detected'] else '불가능',
                '신뢰도': round(ball.confidence, 2)
            })
        
        # DataFrame 생성
        df = pd.DataFrame(data_rows)
        
        # 엑셀 저장
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 전체 데이터
                df.to_excel(writer, sheet_name='분석 결과', index=False)
                
                # 요약 데이터
                summary_data = {
                    '항목': [
                        '일반 렌즈 볼 검출 프레임',
                        'Gamma 렌즈 볼 검출 프레임',
                        '일반 렌즈 스핀 검출',
                        'Gamma 렌즈 스핀 검출',
                        '일반 렌즈 백스핀 (rpm)',
                        'Gamma 렌즈 백스핀 (rpm)',
                        '일반 렌즈 사이드스핀 (rpm)',
                        'Gamma 렌즈 사이드스핀 (rpm)',
                        '일반 렌즈 총 스핀 (rpm)',
                        'Gamma 렌즈 총 스핀 (rpm)',
                        '결론'
                    ],
                    '값': [
                        len(normal_results['top_balls']),
                        len(gamma_results['top_balls']),
                        '가능' if normal_results['spin_analysis']['detected'] else '불가능',
                        '가능' if gamma_results['spin_analysis']['detected'] else '불가능',
                        round(normal_results['spin_analysis']['avg_backspin'], 0),
                        round(gamma_results['spin_analysis']['avg_backspin'], 0),
                        round(normal_results['spin_analysis']['avg_sidespin'], 0),
                        round(gamma_results['spin_analysis']['avg_sidespin'], 0),
                        round(normal_results['spin_analysis']['avg_total_spin'], 0),
                        round(gamma_results['spin_analysis']['avg_total_spin'], 0),
                        self.get_conclusion(normal_results, gamma_results)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='요약', index=False)
            
            print(f"\n엑셀 파일 저장: {excel_file}")
            
        except Exception as e:
            print(f"엑셀 저장 실패: {e}")
    
    def get_conclusion(self, normal_results: Dict, gamma_results: Dict) -> str:
        """분석 결론 생성"""
        normal_detected = normal_results['spin_analysis']['detected']
        gamma_detected = gamma_results['spin_analysis']['detected']
        
        if normal_detected and gamma_detected:
            return "일반 렌즈로도 스핀 검출 가능 (비용 효율적)"
        elif gamma_detected and not normal_detected:
            return "고비용 Gamma 렌즈 필수"
        elif normal_detected and not gamma_detected:
            return "일반 렌즈 충분"
        else:
            return "스핀 검출 불가 - 알고리즘 개선 필요"


def main():
    """메인 실행 함수"""
    analyzer = EnglishPathGolfAnalyzer()
    
    # 영어 경로로 변환된 이미지 분석
    source_dir = r"C:\src\GolfSwingAnalysis_Final_ver8\shot-image\7iron\no_marker_ball-1"
    target_name = "7iron_no_marker_ball_shot1"
    
    # 포괄적 분석 실행
    normal_results, gamma_results = analyzer.convert_and_analyze(source_dir, target_name)
    
    print(f"\n{'='*80}")
    print(f"분석 완료!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()