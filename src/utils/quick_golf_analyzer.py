#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Golf Analyzer v1.0
빠른 골프 이미지 분석 및 Excel 데이터 채우기
"""

import cv2
import numpy as np
import os
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import math
from pathlib import Path
import time

@dataclass
class QuickGolfData:
    """간단한 골프 데이터"""
    shot_type: str = ""
    ball_type: str = ""
    lens_type: str = ""
    frame_number: int = 0
    
    # 볼 검출 결과
    ball_detected: bool = False
    ball_x: float = 0.0
    ball_y: float = 0.0
    ball_confidence: float = 0.0
    
    # 주요 측정값 (계산된 값)
    ball_speed_mph: float = 0.0
    launch_angle_deg: float = 0.0
    backspin_rpm: float = 0.0
    sidespin_rpm: float = 0.0
    
    # 클럽 데이터
    club_speed_mph: float = 0.0
    attack_angle_deg: float = 0.0
    face_angle_deg: float = 0.0

class QuickBallDetector:
    """빠른 볼 검출기"""
    
    def detect_ball_simple(self, img: np.ndarray) -> Optional[Dict]:
        """간단한 볼 검출"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 간단한 원형 검출
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=50,
                param2=20,
                minRadius=5,
                maxRadius=50
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # 첫 번째 검출 결과 사용
                x, y, r = circles[0]
                return {
                    'x': float(x),
                    'y': float(y),
                    'radius': float(r),
                    'confidence': 0.7
                }
            
            # Hough Circle이 실패하면 밝은 점 찾기
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 20 < area < 1000:  # 적절한 크기의 영역
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if 5 < radius < 40:
                        return {
                            'x': float(x),
                            'y': float(y),
                            'radius': float(radius),
                            'confidence': 0.5
                        }
            
            return None
            
        except Exception as e:
            print(f"Ball detection error: {e}")
            return None

class QuickGolfAnalyzer:
    """빠른 골프 분석기"""
    
    def __init__(self):
        self.ball_detector = QuickBallDetector()
    
    def analyze_shot_folder(self, folder_path: Path) -> List[QuickGolfData]:
        """샷 폴더 분석"""
        results = []
        
        # 폴더 정보 파싱
        shot_info = self._parse_folder_name(folder_path)
        
        print(f"Analyzing: {folder_path.name}")
        
        # 프레임 1~23 처리
        for frame_num in range(1, 24):
            # 일반 렌즈 상단 카메라만 처리 (빠른 처리)
            img_path = folder_path / f"1_{frame_num}.bmp"
            
            if img_path.exists():
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # 볼 검출
                        ball_result = self.ball_detector.detect_ball_simple(img)
                        
                        # 데이터 생성
                        frame_data = QuickGolfData(
                            shot_type=shot_info['shot_type'],
                            ball_type=shot_info['ball_type'],
                            lens_type="normal",
                            frame_number=frame_num
                        )
                        
                        if ball_result:
                            frame_data.ball_detected = True
                            frame_data.ball_x = ball_result['x']
                            frame_data.ball_y = ball_result['y']
                            frame_data.ball_confidence = ball_result['confidence']
                        
                        results.append(frame_data)
                        
                except Exception as e:
                    print(f"Error processing frame {frame_num}: {e}")
        
        # 물리량 계산 (간단한 추정)
        self._calculate_physics(results)
        
        return results
    
    def _parse_folder_name(self, folder_path: Path) -> Dict[str, str]:
        """폴더명에서 샷 정보 추출"""
        path_str = str(folder_path)
        
        shot_type = "unknown"
        if "7iron" in path_str:
            shot_type = "7iron"
        elif "driver" in path_str:
            shot_type = "driver"
        
        ball_type = "unknown"
        if "logo_ball" in path_str:
            ball_type = "logo_ball"
        elif "no_marker_ball" in path_str:
            ball_type = "no_marker_ball"
        elif "marker_ball" in path_str:
            ball_type = "marker_ball"
        
        return {
            'shot_type': shot_type,
            'ball_type': ball_type
        }
    
    def _calculate_physics(self, results: List[QuickGolfData]):
        """물리량 계산 (간단한 추정)"""
        detected_frames = [r for r in results if r.ball_detected]
        
        if len(detected_frames) >= 2:
            # 간단한 속도 계산
            first_frame = detected_frames[0]
            last_frame = detected_frames[-1]
            
            # 거리와 시간 계산
            dx = last_frame.ball_x - first_frame.ball_x
            dy = last_frame.ball_y - first_frame.ball_y
            dt = (last_frame.frame_number - first_frame.frame_number) / 820.0  # 820fps
            
            if dt > 0:
                # 픽셀 속도를 물리적 속도로 변환 (추정)
                pixel_speed = math.sqrt(dx**2 + dy**2) / dt
                # 간단한 스케일링 (실제로는 캘리브레이션 필요)
                ball_speed_ms = pixel_speed * 0.1  # 추정 스케일
                ball_speed_mph = ball_speed_ms * 2.237
                
                # 발사각 추정
                launch_angle = math.degrees(math.atan2(-dy, dx)) if abs(dx) > 1 else 0.0
                
                # 모든 프레임에 적용
                for frame_data in results:
                    frame_data.ball_speed_mph = ball_speed_mph
                    frame_data.launch_angle_deg = launch_angle
                    # 추정값들
                    frame_data.backspin_rpm = 2500.0 + (ball_speed_mph - 100) * 20
                    frame_data.sidespin_rpm = 200.0
                    frame_data.club_speed_mph = ball_speed_mph / 1.4  # 스매시 팩터 추정
                    frame_data.attack_angle_deg = -3.0 + (launch_angle * 0.5)
                    frame_data.face_angle_deg = 1.0
    
    def process_all_shots(self, shot_image_dir: Path) -> pd.DataFrame:
        """모든 샷 처리"""
        all_results = []
        
        for club_dir in shot_image_dir.iterdir():
            if club_dir.is_dir():
                print(f"Processing club: {club_dir.name}")
                
                for ball_dir in club_dir.iterdir():
                    if ball_dir.is_dir():
                        results = self.analyze_shot_folder(ball_dir)
                        all_results.extend(results)
        
        if all_results:
            return pd.DataFrame([asdict(result) for result in all_results])
        else:
            return pd.DataFrame()
    
    def fill_excel_template(self, df: pd.DataFrame, template_path: str, output_path: str):
        """기존 Excel 템플릿에 데이터 채우기"""
        try:
            # 기존 Excel 파일 읽기
            existing_df = pd.read_excel(template_path)
            print(f"Existing Excel shape: {existing_df.shape}")
            print(f"New data shape: {df.shape}")
            
            # 데이터 매핑 및 채우기
            if not df.empty:
                # 주요 컬럼만 업데이트
                update_columns = {
                    'ball_detected': 'ball_detected',
                    'ball_speed_mph': 'ball_speed_mph',
                    'launch_angle_deg': 'launch_angle_deg',
                    'backspin_rpm': 'backspin_rpm',
                    'sidespin_rpm': 'sidespin_rpm',
                    'club_speed_mph': 'club_speed_mph',
                    'attack_angle_deg': 'attack_angle_deg',
                    'face_angle_deg': 'face_angle_deg'
                }
                
                # 데이터 그룹화 (샷별)
                grouped = df.groupby(['shot_type', 'ball_type']).first().reset_index()
                
                # 기존 Excel의 각 행에 대응되는 데이터 찾아서 채우기
                for idx, row in existing_df.iterrows():
                    # 매칭 조건 (예시)
                    if idx < len(grouped):
                        match_data = grouped.iloc[idx]
                        
                        for excel_col, data_col in update_columns.items():
                            if excel_col in existing_df.columns and data_col in match_data:
                                existing_df.loc[idx, excel_col] = match_data[data_col]
                
                # 새 파일로 저장
                existing_df.to_excel(output_path, index=False)
                print(f"Updated Excel saved: {output_path}")
                
            else:
                print("No data to fill")
                
        except Exception as e:
            print(f"Error filling Excel template: {e}")
            # 실패시 새로운 Excel 파일 생성
            df.to_excel(output_path, index=False)


def main():
    """메인 함수"""
    print("=== Quick Golf Analyzer v1.0 ===")
    
    # 경로 설정
    shot_image_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/shot-image")
    template_file = "test_enhanced_20250906_153936.xlsx"
    output_file = f"quick_golf_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    print(f"Shot image directory: {shot_image_dir}")
    print(f"Template file: {template_file}")
    print(f"Output file: {output_file}")
    
    # 분석 시작
    analyzer = QuickGolfAnalyzer()
    
    start_time = time.time()
    results_df = analyzer.process_all_shots(shot_image_dir)
    end_time = time.time()
    
    print(f"Analysis completed in {end_time - start_time:.2f} seconds")
    
    if not results_df.empty:
        print(f"Total frames analyzed: {len(results_df)}")
        print(f"Ball detections: {results_df['ball_detected'].sum()}")
        
        # Excel 업데이트
        analyzer.fill_excel_template(results_df, template_file, output_file)
        
        # 요약 출력
        print("\n=== Analysis Summary ===")
        print(f"Shot types: {results_df['shot_type'].unique()}")
        print(f"Ball types: {results_df['ball_type'].unique()}")
        print(f"Average ball speed: {results_df[results_df['ball_speed_mph'] > 0]['ball_speed_mph'].mean():.1f} mph")
        print(f"Average launch angle: {results_df[results_df['launch_angle_deg'] != 0]['launch_angle_deg'].mean():.1f}°")
        
    else:
        print("No analysis results generated")


if __name__ == "__main__":
    main()