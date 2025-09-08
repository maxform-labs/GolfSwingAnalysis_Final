#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel 결과 파일 확인 도구
"""

import pandas as pd
import numpy as np

def analyze_excel_results(filepath):
    """Excel 결과 분석"""
    print(f"=== Analysis of {filepath} ===")
    
    try:
        # Excel 파일 읽기
        df = pd.read_excel(filepath)
        
        print(f"Total records: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"\nColumns: {list(df.columns)}")
        
        # 데이터 요약
        print("\n=== Data Summary ===")
        print(f"IR Ball Detection Rate: {(df['ir_ball_detected'].sum())/len(df)*100:.1f}%")
        
        # 핵심 측정값 통계
        core_measurements = [
            'ball_speed_mph', 'launch_angle_deg', 'direction_angle_deg',
            'backspin_rpm', 'sidespin_rpm', 'club_speed_mph', 
            'attack_angle_deg', 'face_angle_deg', 'smash_factor'
        ]
        
        print("\n=== Core Measurements Statistics ===")
        for measurement in core_measurements:
            if measurement in df.columns:
                non_zero_count = (df[measurement] != 0).sum()
                mean_val = df[df[measurement] != 0][measurement].mean() if non_zero_count > 0 else 0
                print(f"{measurement:20s}: {non_zero_count:3d}/115 ({non_zero_count/115*100:4.1f}%) | Mean: {mean_val:8.2f}")
        
        # 신뢰도 분석
        print(f"\n=== Quality Analysis ===")
        print(f"Average Confidence: {df['measurement_confidence'].mean():.3f}")
        print(f"High Confidence (>0.8): {(df['measurement_confidence'] > 0.8).sum()}/115")
        print(f"Physics Validation: {df['physics_validation_score'].mean():.3f}")
        print(f"Valid Physics (>0.8): {(df['physics_validation_score'] > 0.8).sum()}/115")
        
        # 샷별 통계
        print(f"\n=== Shot Analysis ===")
        shot_groups = df.groupby('shot_id')
        print(f"Number of shots: {len(shot_groups)}")
        
        for shot_id, group in shot_groups:
            detection_rate = group['ir_ball_detected'].sum() / len(group) * 100
            avg_confidence = group['measurement_confidence'].mean()
            print(f"{shot_id:25s}: {detection_rate:4.1f}% detection, {avg_confidence:.3f} confidence")
        
        # 샘플 데이터 표시
        print(f"\n=== Sample Data (First 3 rows) ===")
        sample_cols = ['shot_id', 'frame_number', 'ir_ball_detected', 'ball_speed_mph', 
                      'launch_angle_deg', 'measurement_confidence']
        if all(col in df.columns for col in sample_cols):
            print(df[sample_cols].head(3).to_string())
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")

if __name__ == "__main__":
    analyze_excel_results("integrated_measurements_20250908_102713.xlsx")