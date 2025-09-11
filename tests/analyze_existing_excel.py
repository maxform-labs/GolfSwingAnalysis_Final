#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 Excel 파일 분석 도구
"""

import pandas as pd
import numpy as np
import os

def analyze_existing_excel():
    """기존 Excel 파일 분석"""
    excel_file = "C:/src/GolfSwingAnalysis_Final_ver8/test_enhanced_20250906_154225.xlsx"
    
    if not os.path.exists(excel_file):
        print(f"File not found: {excel_file}")
        return
    
    try:
        df = pd.read_excel(excel_file)
        
        print("=== 기존 Excel 파일 분석 ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print("\n=== 데이터 샘플 (첫 5행) ===")
        print(df.head().to_string())
        
        print("\n=== 컬럼별 데이터 통계 ===")
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                non_zero = (df[col] != 0).sum()
                total = len(df[col])
                mean_val = df[col].mean()
                print(f"{col:25s}: {non_zero:3d}/{total} non-zero ({non_zero/total*100:5.1f}%) | Mean: {mean_val:8.3f}")
            else:
                unique_vals = df[col].nunique()
                print(f"{col:25s}: {unique_vals} unique values")
        
        print("\n=== ROI 및 검출 전략 분석 ===")
        
        # ROI 관련 컬럼 확인
        roi_columns = [col for col in df.columns if 'roi' in col.lower() or 'region' in col.lower()]
        if roi_columns:
            print(f"ROI 관련 컬럼: {roi_columns}")
            for col in roi_columns:
                print(f"  {col}: {df[col].describe()}")
        
        # 검출 전략 관련 컬럼 확인
        detection_columns = [col for col in df.columns if any(word in col.lower() for word in ['detect', 'track', 'stage', 'phase', 'mode'])]
        if detection_columns:
            print(f"검출 전략 관련 컬럼: {detection_columns}")
            for col in detection_columns:
                if df[col].dtype == 'object':
                    print(f"  {col}: {df[col].value_counts().to_dict()}")
                else:
                    print(f"  {col}: {df[col].describe()}")
        
        # 프레임별 분석
        if 'frame_number' in df.columns:
            print(f"\n=== 프레임별 분석 ===")
            print(f"Frame range: {df['frame_number'].min()} - {df['frame_number'].max()}")
            
            # 프레임별 검출 패턴 분석
            if 'ball_detected' in df.columns:
                frame_detection = df.groupby('frame_number')['ball_detected'].sum()
                print(f"Frames with ball detection: {(frame_detection > 0).sum()}")
        
        return df
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

if __name__ == "__main__":
    analyze_existing_excel()