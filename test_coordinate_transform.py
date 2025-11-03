#!/usr/bin/env python3
"""
좌표계 변환 간단 테스트
"""
import numpy as np
import json

def load_results(filename="improved_golf_ball_3d_analysis_results_v2.json"):
    """기존 분석 결과 로드"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_coordinate_transforms():
    """좌표 변환 행렬 테스트"""
    print("="*80)
    print("COORDINATE TRANSFORM TEST")
    print("="*80)
    
    results = load_results()
    
    # 데이터 추출 (처음 5개)
    velocity_data = []
    for r in results[:5]:
        if r['physics']['success']:
            velocity_data.append({
                'shot': r['shot_number'],
                'vx': r['physics']['velocity']['vx'],
                'vy': r['physics']['velocity']['vy'],
                'vz': r['physics']['velocity']['vz'],
                'real_direction': r['real_data']['launch_direction_deg']
            })
    
    print(f"\nLoaded {len(velocity_data)} shots for analysis")
    
    # 7가지 변환 행렬
    transforms = [
        ('Identity', np.eye(3)),
        ('Z-inversion', np.array([[1,0,0],[0,1,0],[0,0,-1]])),
        ('X-Z swap', np.array([[0,0,1],[0,1,0],[1,0,0]])),
        ('X-Z swap + Z-inv', np.array([[0,0,-1],[0,1,0],[1,0,0]])),
        ('X-Z swap + X-inv', np.array([[0,0,1],[0,1,0],[-1,0,0]])),
        ('X-inversion', np.array([[-1,0,0],[0,1,0],[0,0,1]])),
        ('X-inv + Z-inv', np.array([[-1,0,0],[0,1,0],[0,0,-1]]))
    ]
    
    best_name = None
    best_matrix = None
    best_error = float('inf')
    
    for name, matrix in transforms:
        errors = []
        
        print(f"\n{name}:")
        for data in velocity_data:
            v_cam = np.array([data['vx'], data['vy'], data['vz']])
            v_golf = matrix @ v_cam
            
            direction = np.degrees(np.arctan2(v_golf[0], v_golf[2]))
            real = data['real_direction']
            
            error = abs(direction - real)
            if error > 180:
                error = 360 - error
            
            errors.append(error)
            print(f"  Shot {data['shot']}: Dir={direction:7.2f}° Real={real:6.2f}° Error={error:5.2f}°")
        
        avg_error = np.mean(errors)
        print(f"  Average: {avg_error:.2f}°")
        
        if avg_error < best_error:
            best_error = avg_error
            best_name = name
            best_matrix = matrix
    
    print("\n" + "="*80)
    print(f"BEST: {best_name} with {best_error:.2f}° average error")
    print("="*80)
    print(best_matrix)
    
    # 전체 샷 재분석
    print("\n" + "="*80)
    print("APPLYING TO ALL SHOTS")
    print("="*80)
    
    all_errors = []
    print("\nShot  Dir(Calc)  Dir(Real)  Error")
    print("-" * 40)
    
    for r in results:
        if r['physics']['success']:
            v_cam = np.array([
                r['physics']['velocity']['vx'],
                r['physics']['velocity']['vy'],
                r['physics']['velocity']['vz']
            ])
            v_golf = best_matrix @ v_cam
            
            direction = np.degrees(np.arctan2(v_golf[0], v_golf[2]))
            real = r['real_data']['launch_direction_deg']
            
            error = abs(direction - real)
            if error > 180:
                error = 360 - error
            
            all_errors.append(error)
            
            print(f"{r['shot_number']:2d}    {direction:8.2f}   {real:8.2f}   {error:6.2f}")
    
    print("-" * 40)
    print(f"Average: {np.mean(all_errors):.2f}° (was 94.55°)")
    print(f"Improvement: {((94.55 - np.mean(all_errors)) / 94.55 * 100):.1f}%")
    
    # 저장
    output = {
        'best_transform': best_name,
        'matrix': best_matrix.tolist(),
        'direction_error_mean': float(np.mean(all_errors)),
        'direction_error_std': float(np.std(all_errors)),
        'improvement_percent': float((94.55 - np.mean(all_errors)) / 94.55 * 100)
    }
    
    with open('coordinate_transform_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n[OK] Results saved to coordinate_transform_results.json")
    print("="*80)

if __name__ == "__main__":
    test_coordinate_transforms()
