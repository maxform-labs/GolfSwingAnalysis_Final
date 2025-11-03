#!/usr/bin/env python3
"""
최적화된 파라미터로 전체 샷 분석
"""
from improved_golf_ball_3d_analyzer import ImprovedGolfBall3DAnalyzer
import json
import numpy as np

# 분석기 초기화 (최적 스케일 팩터 = 1.0)
analyzer = ImprovedGolfBall3DAnalyzer()
analyzer.depth_scale_factor = 1.0

print("="*80)
print(f"ANALYZING ALL SHOTS (Scale Factor={analyzer.depth_scale_factor:.2f})")
print("="*80)

results = []
for shot_num in range(1, 21):
    shot_dir = f"data2/driver/{shot_num}"
    try:
        result = analyzer.analyze_shot_improved(shot_dir, shot_num)
        results.append(result)
    except Exception as e:
        print(f"[ERROR] Shot {shot_num}: {e}")

# 결과 저장
output_file = "improved_golf_ball_3d_analysis_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# 통계 출력
print(f"\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

successful = [r for r in results if r['physics']['success']]
print(f"Successful analyses: {len(successful)}/{len(results)}")

if successful:
    speed_errors = []
    launch_errors = []
    direction_errors = []
    
    print(f"\n{'Shot':<5} {'Calc':<8} {'Real':<8} {'Error':<10} {'Launch(C)':<10} {'Launch(R)':<10} {'Dir(C)':<8} {'Dir(R)':<8}")
    print("-"*80)
    
    for r in successful:
        if r['shot_number'] in analyzer.real_data:
            calc_speed = r['physics']['speed']['m_s']
            real_speed = analyzer.real_data[r['shot_number']]['ball_speed_ms']
            speed_error_pct = abs(calc_speed - real_speed) / real_speed * 100
            
            calc_launch = r['physics']['launch_angle']['degrees']
            real_launch = analyzer.real_data[r['shot_number']]['launch_angle_deg']
            launch_error = abs(calc_launch - real_launch)
            
            calc_dir = r['physics']['direction_angle']['degrees']
            real_dir = analyzer.real_data[r['shot_number']]['launch_direction_deg']
            dir_error = abs(calc_dir - real_dir)
            
            speed_errors.append(speed_error_pct)
            launch_errors.append(launch_error)
            direction_errors.append(dir_error)
            
            print(f"{r['shot_number']:<5} {calc_speed:<8.2f} {real_speed:<8.2f} {speed_error_pct:<10.1f}% {calc_launch:<10.2f} {real_launch:<10.2f} {calc_dir:<8.2f} {real_dir:<8.2f}")
    
    print("-"*80)
    print(f"\nSTATISTICS:")
    print(f"  Speed Error  : Mean={np.mean(speed_errors):5.1f}%, Std={np.std(speed_errors):5.1f}%")
    print(f"  Launch Error : Mean={np.mean(launch_errors):5.2f}°, Std={np.std(launch_errors):5.2f}°")
    print(f"  Direction Err: Mean={np.mean(direction_errors):5.2f}°, Std={np.std(direction_errors):5.2f}°")

print(f"\n[OK] Results saved to: {output_file}")
print(f"[OK] Optimal depth scale factor: {analyzer.depth_scale_factor:.2f}")
print("="*80)
