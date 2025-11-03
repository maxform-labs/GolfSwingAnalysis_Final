#!/usr/bin/env python3
"""
개선된 전체 샷 분석 스크립트
- 방향각 계산 로직 개선
- Kalman 필터 튜닝
- 검출 파라미터 완화
"""
import json
import numpy as np
from improved_golf_ball_3d_analyzer import ImprovedGolfBall3DAnalyzer
import os

def main():
    print("\n" + "="*80)
    print("IMPROVED 20-SHOT ANALYSIS")
    print("="*80)
    
    analyzer = ImprovedGolfBall3DAnalyzer()
    analyzer.depth_scale_factor = 1.0
    
    all_results = []
    
    for shot_num in range(1, 21):
        shot_dir = f"data2/driver/{shot_num}"
        if not os.path.exists(shot_dir):
            print(f"[SKIP] Shot {shot_num}: directory not found")
            continue
        
        result = analyzer.analyze_shot_improved(shot_dir, shot_num)
        all_results.append(result)
    
    # 통계 계산
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    successful = [r for r in all_results if r['physics']['success']]
    print(f"Successful analyses: {len(successful)}/{len(all_results)}")
    
    # 상세 결과 테이블
    print("\nShot  Calc     Real     Error      Launch(C)  Launch(R)  Dir(C)     Dir(R)     Dir_M2     Dir_M3")
    print("-" * 120)
    
    speed_errors = []
    launch_errors = []
    direction_errors = []
    direction_m2_errors = []
    direction_m3_errors = []
    
    for result in successful:
        shot_num = result['shot_number']
        if shot_num not in analyzer.real_data:
            continue
        
        real = analyzer.real_data[shot_num]
        calc = result['physics']
        
        calc_speed = calc['speed']['m_s']
        real_speed = real['ball_speed_ms']
        speed_error_pct = abs(calc_speed - real_speed) / real_speed * 100
        
        calc_launch = calc['launch_angle']['degrees']
        real_launch = real['launch_angle_deg']
        launch_error = abs(calc_launch - real_launch)
        
        calc_dir = calc['direction_angle']['degrees']
        calc_dir_m2 = calc['direction_angle']['method2']
        calc_dir_m3 = calc['direction_angle']['method3']
        real_dir = real['launch_direction_deg']
        
        dir_error = abs(calc_dir - real_dir)
        dir_m2_error = abs(calc_dir_m2 - real_dir)
        dir_m3_error = abs(calc_dir_m3 - real_dir)
        
        speed_errors.append(speed_error_pct)
        launch_errors.append(launch_error)
        direction_errors.append(dir_error)
        direction_m2_errors.append(dir_m2_error)
        direction_m3_errors.append(dir_m3_error)
        
        print(f"{shot_num:2d}    {calc_speed:6.2f}   {real_speed:6.2f}   "
              f"{speed_error_pct:5.1f} %    {calc_launch:6.2f}     {real_launch:6.2f}     "
              f"{calc_dir:7.2f}    {real_dir:7.2f}    {calc_dir_m2:7.2f}    {calc_dir_m3:7.2f}")
    
    print("-" * 120)
    
    # 통계
    if speed_errors:
        print(f"\nSTATISTICS:")
        print(f"  Speed Error     : Mean= {np.mean(speed_errors):5.1f}%, Std= {np.std(speed_errors):5.1f}%")
        print(f"  Launch Error    : Mean= {np.mean(launch_errors):5.2f}°, Std= {np.std(launch_errors):5.2f}°")
        print(f"  Direction Error (Method1): Mean= {np.mean(direction_errors):6.2f}°, Std= {np.std(direction_errors):6.2f}°")
        print(f"  Direction Error (Method2): Mean= {np.mean(direction_m2_errors):6.2f}°, Std= {np.std(direction_m2_errors):6.2f}°")
        print(f"  Direction Error (Method3): Mean= {np.mean(direction_m3_errors):6.2f}°, Std= {np.std(direction_m3_errors):6.2f}°")
        
        # 최적 방향각 계산 방법 선택
        methods = [
            ('Method1: atan2(vx, vz)', np.mean(direction_errors)),
            ('Method2: atan2(vz, vx)', np.mean(direction_m2_errors)),
            ('Method3: atan2(-vx, vz)', np.mean(direction_m3_errors))
        ]
        methods.sort(key=lambda x: x[1])
        
        print(f"\n  BEST DIRECTION METHOD: {methods[0][0]} with {methods[0][1]:.2f}° average error")
    
    # 결과 저장
    output_file = "improved_golf_ball_3d_analysis_results_v2.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved to: {output_file}")
    print(f"[OK] Optimal depth scale factor: {analyzer.depth_scale_factor:.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
