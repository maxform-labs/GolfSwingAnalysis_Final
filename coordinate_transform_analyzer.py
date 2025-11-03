#!/usr/bin/env python3
"""
좌표계 변환 분석 및 구현
- 카메라 좌표계와 골프 좌표계의 관계 분석
- 실측 데이터 기반 회전 행렬 추정
- 변환된 좌표계로 방향각 재계산
"""
import numpy as np
import json
import os
from typing import Dict, List, Tuple
from improved_golf_ball_3d_analyzer import ImprovedGolfBall3DAnalyzer

class CoordinateTransformAnalyzer:
    def __init__(self):
        """좌표 변환 분석기 초기화"""
        self.analyzer = ImprovedGolfBall3DAnalyzer()
        self.analyzer.depth_scale_factor = 1.0
        
        print("="*80)
        print("COORDINATE TRANSFORM ANALYZER")
        print("="*80)
    
    def analyze_velocity_directions(self, num_shots=5):
        """
        속도 방향 분석 - 실측 데이터와 계산 데이터 비교
        
        Returns:
            분석 결과 딕셔너리
        """
        print("\n[STEP 1] Analyzing velocity directions from real shots...")
        
        results = []
        
        for shot_num in range(1, num_shots + 1):
            shot_dir = f"data2/driver/{shot_num}"
            if not os.path.exists(shot_dir):
                continue
            
            result = self.analyzer.analyze_shot_improved(shot_dir, shot_num)
            
            if result['physics']['success'] and shot_num in self.analyzer.real_data:
                real = self.analyzer.real_data[shot_num]
                calc = result['physics']
                
                vx = calc['velocity']['vx']
                vy = calc['velocity']['vy']
                vz = calc['velocity']['vz']
                
                real_direction = real['launch_direction_deg']
                calc_direction = calc['direction_angle']['degrees']
                
                results.append({
                    'shot': shot_num,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'real_direction': real_direction,
                    'calc_direction': calc_direction,
                    'direction_error': abs(calc_direction - real_direction)
                })
                
                print(f"  Shot {shot_num}: VX={vx:8.1f}, VY={vy:8.1f}, VZ={vz:8.1f} mm/s")
                print(f"           Real Dir={real_direction:6.2f}°, Calc Dir={calc_direction:7.2f}°, Error={abs(calc_direction - real_direction):6.2f}°")
        
        return results
    
    def estimate_coordinate_transform(self, velocity_data: List[Dict]):
        """
        좌표 변환 행렬 추정
        
        골프 좌표계 정의:
        - X_golf: 우측 (+) / 좌측 (-)
        - Y_golf: 위 (+) / 아래 (-)
        - Z_golf: 전방(타겟) (+) / 후방 (-)
        
        방향각: atan2(X_golf, Z_golf)
        - 0°: 정면
        - +각: 우측
        - -각: 좌측
        """
        print("\n[STEP 2] Estimating coordinate transform...")
        
        # 가설 1: Z축 반전 필요 (VZ가 음수 → 양수로)
        transform_hypotheses = [
            {
                'name': 'Identity (no transform)',
                'matrix': np.eye(3),
                'description': 'X_golf=X_cam, Y_golf=Y_cam, Z_golf=Z_cam'
            },
            {
                'name': 'Z-axis inversion',
                'matrix': np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                'description': 'X_golf=X_cam, Y_golf=Y_cam, Z_golf=-Z_cam'
            },
            {
                'name': 'X-Z swap',
                'matrix': np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
                'description': 'X_golf=Z_cam, Y_golf=Y_cam, Z_golf=X_cam'
            },
            {
                'name': 'X-Z swap + Z inversion',
                'matrix': np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
                'description': 'X_golf=-Z_cam, Y_golf=Y_cam, Z_golf=X_cam'
            },
            {
                'name': 'X-Z swap + X inversion',
                'matrix': np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
                'description': 'X_golf=Z_cam, Y_golf=Y_cam, Z_golf=-X_cam'
            },
            {
                'name': 'X inversion',
                'matrix': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                'description': 'X_golf=-X_cam, Y_golf=Y_cam, Z_golf=Z_cam'
            },
            {
                'name': 'X inversion + Z inversion',
                'matrix': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                'description': 'X_golf=-X_cam, Y_golf=Y_cam, Z_golf=-Z_cam'
            }
        ]
        
        # 각 변환 행렬 테스트
        best_transform = None
        best_error = float('inf')
        
        for hyp in transform_hypotheses:
            total_error = 0
            
            print(f"\n  Testing: {hyp['name']}")
            print(f"  {hyp['description']}")
            
            for data in velocity_data:
                v_cam = np.array([data['vx'], data['vy'], data['vz']])
                v_golf = hyp['matrix'] @ v_cam
                
                vx_golf, vy_golf, vz_golf = v_golf
                
                # 방향각 계산
                direction_golf = np.degrees(np.arctan2(vx_golf, vz_golf))
                real_direction = data['real_direction']
                
                error = abs(direction_golf - real_direction)
                # 180도 wrap 처리
                if error > 180:
                    error = 360 - error
                
                total_error += error
                
                print(f"    Shot {data['shot']}: Dir={direction_golf:7.2f}° (Real={real_direction:6.2f}°) Error={error:6.2f}°")
            
            avg_error = total_error / len(velocity_data)
            print(f"  Average Error: {avg_error:.2f}°")
            
            if avg_error < best_error:
                best_error = avg_error
                best_transform = hyp
        
        print(f"\n[RESULT] Best transform: {best_transform['name']}")
        print(f"         Average error: {best_error:.2f}°")
        print(f"         Transform matrix:")
        print(best_transform['matrix'])
        
        return best_transform
    
    def apply_transform_and_reanalyze(self, transform_matrix: np.ndarray, num_shots=20):
        """
        변환 적용하고 전체 재분석
        """
        print(f"\n[STEP 3] Applying transform to all {num_shots} shots...")
        
        results = []
        
        for shot_num in range(1, num_shots + 1):
            shot_dir = f"data2/driver/{shot_num}"
            if not os.path.exists(shot_dir):
                continue
            
            result = self.analyzer.analyze_shot_improved(shot_dir, shot_num)
            
            if result['physics']['success'] and shot_num in self.analyzer.real_data:
                real = self.analyzer.real_data[shot_num]
                calc = result['physics']
                
                # 카메라 좌표계 속도
                v_cam = np.array([calc['velocity']['vx'], calc['velocity']['vy'], calc['velocity']['vz']])
                
                # 골프 좌표계로 변환
                v_golf = transform_matrix @ v_cam
                vx_golf, vy_golf, vz_golf = v_golf
                
                # 속도 재계산
                speed_golf = np.linalg.norm(v_golf) / 1000.0  # mm/s -> m/s
                
                # 발사각 재계산
                horizontal_speed = np.sqrt(vx_golf**2 + vz_golf**2)
                if horizontal_speed > 0:
                    launch_angle_golf = np.degrees(np.arctan2(vy_golf, horizontal_speed))
                else:
                    launch_angle_golf = 90.0 if vy_golf > 0 else -90.0
                
                # 방향각 재계산 (변환된 좌표계)
                direction_angle_golf = np.degrees(np.arctan2(vx_golf, vz_golf))
                
                # 오차 계산
                speed_error = abs(speed_golf - real['ball_speed_ms'])
                speed_error_pct = (speed_error / real['ball_speed_ms']) * 100
                launch_error = abs(launch_angle_golf - real['launch_angle_deg'])
                direction_error = abs(direction_angle_golf - real['launch_direction_deg'])
                
                # 180도 wrap 처리
                if direction_error > 180:
                    direction_error = 360 - direction_error
                
                results.append({
                    'shot': shot_num,
                    'speed_calc': speed_golf,
                    'speed_real': real['ball_speed_ms'],
                    'speed_error_pct': speed_error_pct,
                    'launch_calc': launch_angle_golf,
                    'launch_real': real['launch_angle_deg'],
                    'launch_error': launch_error,
                    'direction_calc': direction_angle_golf,
                    'direction_real': real['launch_direction_deg'],
                    'direction_error': direction_error,
                    'vx_golf': vx_golf,
                    'vy_golf': vy_golf,
                    'vz_golf': vz_golf
                })
        
        return results
    
    def print_comparison_table(self, results: List[Dict]):
        """결과 비교 테이블 출력"""
        print("\n" + "="*120)
        print("TRANSFORMED COORDINATE SYSTEM RESULTS")
        print("="*120)
        print("Shot  Speed(C)  Speed(R)  Err%    Launch(C)  Launch(R)  Err    Dir(C)     Dir(R)     Err")
        print("-" * 120)
        
        speed_errors = []
        launch_errors = []
        direction_errors = []
        
        for r in results:
            print(f"{r['shot']:2d}    {r['speed_calc']:6.2f}    {r['speed_real']:6.2f}    "
                  f"{r['speed_error_pct']:5.1f}   {r['launch_calc']:7.2f}    {r['launch_real']:6.2f}    "
                  f"{r['launch_error']:5.2f}  {r['direction_calc']:8.2f}   {r['direction_real']:7.2f}    "
                  f"{r['direction_error']:6.2f}")
            
            speed_errors.append(r['speed_error_pct'])
            launch_errors.append(r['launch_error'])
            direction_errors.append(r['direction_error'])
        
        print("-" * 120)
        print(f"\nSTATISTICS:")
        print(f"  Speed Error     : Mean= {np.mean(speed_errors):5.1f}%, Std= {np.std(speed_errors):5.1f}%")
        print(f"  Launch Error    : Mean= {np.mean(launch_errors):5.2f}°, Std= {np.std(launch_errors):5.2f}°")
        print(f"  Direction Error : Mean= {np.mean(direction_errors):6.2f}°, Std= {np.std(direction_errors):6.2f}°")
        print("="*120)
        
        return {
            'speed_error_mean': np.mean(speed_errors),
            'launch_error_mean': np.mean(launch_errors),
            'direction_error_mean': np.mean(direction_errors)
        }

def main():
    """메인 함수"""
    analyzer = CoordinateTransformAnalyzer()
    
    # 1단계: 속도 방향 분석 (5개 샷)
    velocity_data = analyzer.analyze_velocity_directions(num_shots=5)
    
    # 2단계: 최적 변환 행렬 추정
    best_transform = analyzer.estimate_coordinate_transform(velocity_data)
    
    # 3단계: 전체 샷 재분석
    transformed_results = analyzer.apply_transform_and_reanalyze(
        best_transform['matrix'], 
        num_shots=20
    )
    
    # 4단계: 결과 출력
    stats = analyzer.print_comparison_table(transformed_results)
    
    # 5단계: 결과 저장
    output = {
        'best_transform': {
            'name': best_transform['name'],
            'description': best_transform['description'],
            'matrix': best_transform['matrix'].tolist()
        },
        'statistics': stats,
        'results': transformed_results
    }
    
    output_file = "coordinate_transform_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved to: {output_file}")
    
    # 개선 비율 계산
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    print(f"Direction Error: 94.55° → {stats['direction_error_mean']:.2f}°")
    improvement_pct = ((94.55 - stats['direction_error_mean']) / 94.55) * 100
    print(f"Improvement: {improvement_pct:.1f}%")
    print("="*80)

if __name__ == "__main__":
    main()
