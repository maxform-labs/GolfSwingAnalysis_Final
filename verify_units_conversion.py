#!/usr/bin/env python3
"""
단위 변환 검증 스크립트
- m/s를 mph로 변환하는 정확성 확인
- 실제 데이터와 분석 결과 비교
- 단위 통일 후 정확도 검증
"""

import pandas as pd
import json
from pathlib import Path

def verify_units_conversion():
    """단위 변환 정확성 검증"""
    
    print("=== 단위 변환 검증 ===\n")
    
    # 5번 아이언 실제 데이터 로드
    csv_file = "data/video_ballData_20250930/video_ballData_20250930/5Iron_0930/shotdata_20250930.csv"
    df = pd.read_csv(csv_file)
    
    # 분석 결과 로드
    analysis_file = "multi_club_analysis_results.json"
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    club_summary = analysis_data['summary']['club_summaries']['5Iron']
    
    print("1. 5번 아이언 실제 데이터 (CSV)")
    print("=" * 50)
    print(f"{'샷':<4} {'볼 스피드(m/s)':<15} {'발사각(°)':<10} {'방향각(°)':<10} {'총 스핀(rpm)':<12}")
    print("-" * 50)
    
    for i, row in df.iterrows():
        shot_num = i + 1
        ball_speed_ms = row['BallSpeed(m/s)']
        launch_angle = row['LaunchAngle(deg)']
        launch_direction = row['LaunchDirection(deg)']
        total_spin = row['TotalSpin(rpm)']
        
        print(f"{shot_num:<4} {ball_speed_ms:<15.1f} {launch_angle:<10.1f} {launch_direction:<10.1f} {total_spin:<12.0f}")
    
    print("\n2. 단위 변환 (m/s → mph)")
    print("=" * 50)
    print("변환 공식: mph = m/s × 2.237")
    print(f"{'샷':<4} {'볼 스피드(m/s)':<15} {'볼 스피드(mph)':<15} {'발사각(°)':<10}")
    print("-" * 50)
    
    for i, row in df.iterrows():
        shot_num = i + 1
        ball_speed_ms = row['BallSpeed(m/s)']
        ball_speed_mph = ball_speed_ms * 2.237
        launch_angle = row['LaunchAngle(deg)']
        
        print(f"{shot_num:<4} {ball_speed_ms:<15.1f} {ball_speed_mph:<15.1f} {launch_angle:<10.1f}")
    
    print("\n3. 분석 결과와 비교")
    print("=" * 80)
    print(f"{'샷':<4} {'실제(mph)':<12} {'분석(mph)':<12} {'오차(%)':<10} {'실제 발사각':<12} {'분석 발사각':<12} {'오차(%)':<10}")
    print("-" * 80)
    
    total_speed_error = 0
    total_angle_error = 0
    valid_comparisons = 0
    
    for i, row in df.iterrows():
        shot_num = i + 1
        actual_speed_ms = row['BallSpeed(m/s)']
        actual_speed_mph = actual_speed_ms * 2.237
        actual_angle = row['LaunchAngle(deg)']
        
        # 분석 결과에서 해당 샷 데이터 찾기
        shot_data = club_summary['shot_details'][str(shot_num)]
        analyzed_speed = shot_data['avg_ball_speed']
        analyzed_angle = shot_data['avg_launch_angle']
        
        # 오차 계산
        speed_error = abs(actual_speed_mph - analyzed_speed) / actual_speed_mph * 100 if actual_speed_mph > 0 else 0
        angle_error = abs(actual_angle - analyzed_angle) / actual_angle * 100 if actual_angle > 0 else 0
        
        total_speed_error += speed_error
        total_angle_error += angle_error
        valid_comparisons += 1
        
        print(f"{shot_num:<4} {actual_speed_mph:<12.1f} {analyzed_speed:<12.1f} {speed_error:<10.2f} {actual_angle:<12.1f} {analyzed_angle:<12.1f} {angle_error:<10.2f}")
    
    # 평균 오차 계산
    avg_speed_error = total_speed_error / valid_comparisons if valid_comparisons > 0 else 0
    avg_angle_error = total_angle_error / valid_comparisons if valid_comparisons > 0 else 0
    
    print("-" * 80)
    print(f"{'평균':<4} {'':<12} {'':<12} {avg_speed_error:<10.2f} {'':<12} {'':<12} {avg_angle_error:<10.2f}")
    
    print("\n4. 정확도 평가")
    print("=" * 50)
    print(f"볼 스피드 평균 오차: {avg_speed_error:.2f}%")
    print(f"발사각 평균 오차: {avg_angle_error:.2f}%")
    
    if avg_speed_error < 1.0 and avg_angle_error < 1.0:
        accuracy_level = "완벽한 정확도"
    elif avg_speed_error < 5.0 and avg_angle_error < 5.0:
        accuracy_level = "매우 높은 정확도"
    elif avg_speed_error < 10.0 and avg_angle_error < 10.0:
        accuracy_level = "높은 정확도"
    else:
        accuracy_level = "개선 필요"
    
    print(f"정확도 수준: {accuracy_level}")
    
    print("\n5. 단위 변환 검증")
    print("=" * 50)
    print("변환 공식 확인:")
    print("- 1 m/s = 2.237 mph")
    print("- 33.8 m/s × 2.237 = 75.6 mph")
    print("- 41.9 m/s × 2.237 = 93.7 mph")
    print("- 30.6 m/s × 2.237 = 68.5 mph")
    
    # 몇 개 샘플로 변환 검증
    test_values = [33.8, 41.9, 30.6, 44.2, 42.0]
    print("\n변환 검증:")
    for ms in test_values:
        mph = ms * 2.237
        print(f"{ms} m/s → {mph:.1f} mph")
    
    print("\n6. 결론")
    print("=" * 50)
    if avg_speed_error < 0.1 and avg_angle_error < 0.1:
        print("✅ 단위 변환이 정확하며, 분석 결과가 실제 데이터와 완벽하게 일치합니다.")
        print("✅ m/s에서 mph로의 변환이 올바르게 적용되었습니다.")
        print("✅ 이미지 분석을 통한 골프공 검출이 매우 정확합니다.")
    else:
        print("⚠️  일부 오차가 발견되었습니다. 추가 검토가 필요합니다.")
    
    return {
        'avg_speed_error': avg_speed_error,
        'avg_angle_error': avg_angle_error,
        'accuracy_level': accuracy_level,
        'total_shots': len(df)
    }

if __name__ == "__main__":
    result = verify_units_conversion()
