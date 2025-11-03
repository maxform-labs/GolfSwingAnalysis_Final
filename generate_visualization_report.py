#!/usr/bin/env python3
"""
골프공 3D 분석 시각화 리포트
- 속도/각도 비교 그래프
- 3D 궤적 시각화
- 오차 분포 차트
- 좌표계 진단
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def load_results(filename="improved_golf_ball_3d_analysis_results_v2.json"):
    """결과 파일 로드"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_statistics(results):
    """통계 추출"""
    data = {
        'shot_numbers': [],
        'calc_speed': [],
        'real_speed': [],
        'speed_errors': [],
        'calc_launch': [],
        'real_launch': [],
        'launch_errors': [],
        'calc_direction': [],
        'real_direction': [],
        'direction_errors': [],
        'vx': [],
        'vy': [],
        'vz': [],
        'tracked_frames': []
    }
    
    for result in results:
        if not result['physics']['success']:
            continue
        
        shot_num = result['shot_number']
        real = result['real_data']
        calc = result['physics']
        
        data['shot_numbers'].append(shot_num)
        
        # 속도
        calc_speed = calc['speed']['m_s']
        real_speed = real['ball_speed_ms']
        data['calc_speed'].append(calc_speed)
        data['real_speed'].append(real_speed)
        data['speed_errors'].append(abs(calc_speed - real_speed) / real_speed * 100)
        
        # 발사각
        calc_launch = calc['launch_angle']['degrees']
        real_launch = real['launch_angle_deg']
        data['calc_launch'].append(calc_launch)
        data['real_launch'].append(real_launch)
        data['launch_errors'].append(abs(calc_launch - real_launch))
        
        # 방향각
        calc_dir = calc['direction_angle']['degrees']
        real_dir = real['launch_direction_deg']
        data['calc_direction'].append(calc_dir)
        data['real_direction'].append(real_dir)
        data['direction_errors'].append(abs(calc_dir - real_dir))
        
        # 속도 벡터
        data['vx'].append(calc['velocity']['vx'])
        data['vy'].append(calc['velocity']['vy'])
        data['vz'].append(calc['velocity']['vz'])
        
        # 추적 프레임 수
        data['tracked_frames'].append(result['tracked_frames'])
    
    return data

def plot_speed_comparison(data):
    """속도 비교 그래프"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 계산 vs 실측
    ax1.scatter(data['real_speed'], data['calc_speed'], alpha=0.6, s=100)
    ax1.plot([40, 70], [40, 70], 'r--', label='Perfect match')
    ax1.set_xlabel('Real Speed (m/s)', fontsize=12)
    ax1.set_ylabel('Calculated Speed (m/s)', fontsize=12)
    ax1.set_title('Speed: Calculated vs Real', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 오차율
    ax2.bar(data['shot_numbers'], data['speed_errors'], alpha=0.7)
    ax2.axhline(y=np.mean(data['speed_errors']), color='r', linestyle='--', 
                label=f'Mean: {np.mean(data["speed_errors"]):.1f}%')
    ax2.set_xlabel('Shot Number', fontsize=12)
    ax2.set_ylabel('Speed Error (%)', fontsize=12)
    ax2.set_title('Speed Error by Shot', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('speed_analysis.png', dpi=150, bbox_inches='tight')
    print("[OK] speed_analysis.png saved")
    plt.close()

def plot_angle_comparison(data):
    """각도 비교 그래프"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 발사각 비교
    ax1.scatter(data['real_launch'], data['calc_launch'], alpha=0.6, s=100)
    ax1.plot([0, 15], [0, 15], 'r--', label='Perfect match')
    ax1.set_xlabel('Real Launch Angle (°)', fontsize=12)
    ax1.set_ylabel('Calculated Launch Angle (°)', fontsize=12)
    ax1.set_title('Launch Angle: Calculated vs Real', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 발사각 오차
    ax2.bar(data['shot_numbers'], data['launch_errors'], alpha=0.7)
    ax2.axhline(y=np.mean(data['launch_errors']), color='r', linestyle='--',
                label=f'Mean: {np.mean(data["launch_errors"]):.1f}°')
    ax2.set_xlabel('Shot Number', fontsize=12)
    ax2.set_ylabel('Launch Angle Error (°)', fontsize=12)
    ax2.set_title('Launch Angle Error by Shot', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 방향각 비교 (문제 영역)
    ax3.scatter(data['real_direction'], data['calc_direction'], alpha=0.6, s=100, c='orange')
    ax3.plot([-10, 10], [-10, 10], 'r--', label='Perfect match')
    ax3.set_xlabel('Real Direction Angle (°)', fontsize=12)
    ax3.set_ylabel('Calculated Direction Angle (°)', fontsize=12)
    ax3.set_title('Direction Angle: Calculated vs Real (ISSUE)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 방향각 오차
    ax4.bar(data['shot_numbers'], data['direction_errors'], alpha=0.7, color='orange')
    ax4.axhline(y=np.mean(data['direction_errors']), color='r', linestyle='--',
                label=f'Mean: {np.mean(data["direction_errors"]):.1f}°')
    ax4.set_xlabel('Shot Number', fontsize=12)
    ax4.set_ylabel('Direction Angle Error (°)', fontsize=12)
    ax4.set_title('Direction Angle Error by Shot', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('angle_analysis.png', dpi=150, bbox_inches='tight')
    print("[OK] angle_analysis.png saved")
    plt.close()

def plot_velocity_vectors(data):
    """속도 벡터 분석"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # VX 분포
    ax1.bar(data['shot_numbers'], data['vx'], alpha=0.7, color='blue')
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Shot Number', fontsize=12)
    ax1.set_ylabel('VX (mm/s)', fontsize=12)
    ax1.set_title('X Velocity Component (Left/Right)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # VY 분포
    ax2.bar(data['shot_numbers'], data['vy'], alpha=0.7, color='green')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Shot Number', fontsize=12)
    ax2.set_ylabel('VY (mm/s)', fontsize=12)
    ax2.set_title('Y Velocity Component (Up/Down)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # VZ 분포
    ax3.bar(data['shot_numbers'], data['vz'], alpha=0.7, color='red')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Shot Number', fontsize=12)
    ax3.set_ylabel('VZ (mm/s)', fontsize=12)
    ax3.set_title('Z Velocity Component (Forward/Backward)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # VX vs VZ (방향각 관련)
    ax4.scatter(data['vz'], data['vx'], alpha=0.6, s=100, c=data['real_direction'], cmap='coolwarm')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('VZ (mm/s) - Forward', fontsize=12)
    ax4.set_ylabel('VX (mm/s) - Right', fontsize=12)
    ax4.set_title('VX vs VZ (colored by real direction)', fontsize=14)
    cbar = plt.colorbar(ax4.scatter(data['vz'], data['vx'], c=data['real_direction'], cmap='coolwarm'), ax=ax4)
    cbar.set_label('Real Direction (°)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('velocity_vector_analysis.png', dpi=150, bbox_inches='tight')
    print("[OK] velocity_vector_analysis.png saved")
    plt.close()

def plot_3d_trajectory_sample(results):
    """샘플 3D 궤적"""
    # 샷 1, 4, 12, 20 (좋은 결과들) 시각화
    sample_shots = [1, 4, 12, 20]
    
    fig = plt.figure(figsize=(16, 12))
    
    for idx, shot_num in enumerate(sample_shots):
        result = [r for r in results if r['shot_number'] == shot_num]
        if not result or not result[0]['physics']['success']:
            continue
        
        result = result[0]
        detections = result['detections']
        
        # 3D 위치 추출
        positions_raw = np.array([d['pos_3d_raw'] for d in detections])
        positions_filtered = np.array([d['pos_3d_filtered'] for d in detections])
        
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        
        # Raw 위치
        ax.plot(positions_raw[:, 0], positions_raw[:, 2], positions_raw[:, 1], 
                'o-', alpha=0.5, label='Raw', markersize=4)
        
        # Filtered 위치
        ax.plot(positions_filtered[:, 0], positions_filtered[:, 2], positions_filtered[:, 1],
                's-', alpha=0.8, label='Kalman Filtered', markersize=6)
        
        # 시작점 표시
        ax.scatter([positions_raw[0, 0]], [positions_raw[0, 2]], [positions_raw[0, 1]],
                  c='green', s=200, marker='o', label='Start')
        
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Z (mm)', fontsize=10)
        ax.set_zlabel('Y (mm)', fontsize=10)
        ax.set_title(f'Shot {shot_num} - {result["tracked_frames"]} frames', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('3d_trajectory_samples.png', dpi=150, bbox_inches='tight')
    print("[OK] 3d_trajectory_samples.png saved")
    plt.close()

def plot_error_distribution(data):
    """오차 분포 히스토그램"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 속도 오차 분포
    ax1.hist(data['speed_errors'], bins=10, alpha=0.7, edgecolor='black')
    ax1.axvline(x=np.mean(data['speed_errors']), color='r', linestyle='--',
                label=f'Mean: {np.mean(data["speed_errors"]):.1f}%')
    ax1.set_xlabel('Speed Error (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Speed Error Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 발사각 오차 분포
    ax2.hist(data['launch_errors'], bins=10, alpha=0.7, edgecolor='black', color='green')
    ax2.axvline(x=np.mean(data['launch_errors']), color='r', linestyle='--',
                label=f'Mean: {np.mean(data["launch_errors"]):.1f}°')
    ax2.set_xlabel('Launch Angle Error (°)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Launch Angle Error Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 방향각 오차 분포
    ax3.hist(data['direction_errors'], bins=10, alpha=0.7, edgecolor='black', color='orange')
    ax3.axvline(x=np.mean(data['direction_errors']), color='r', linestyle='--',
                label=f'Mean: {np.mean(data["direction_errors"]):.1f}°')
    ax3.set_xlabel('Direction Angle Error (°)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Direction Angle Error Distribution (ISSUE)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=150, bbox_inches='tight')
    print("[OK] error_distribution.png saved")
    plt.close()

def plot_tracking_quality(data):
    """추적 품질 분석"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 추적 프레임 수
    ax1.bar(data['shot_numbers'], data['tracked_frames'], alpha=0.7)
    ax1.axhline(y=np.mean(data['tracked_frames']), color='r', linestyle='--',
                label=f'Mean: {np.mean(data["tracked_frames"]):.1f} frames')
    ax1.set_xlabel('Shot Number', fontsize=12)
    ax1.set_ylabel('Tracked Frames', fontsize=12)
    ax1.set_title('Detection Success Rate (Tracked Frames)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 추적 프레임 vs 속도 오차
    ax2.scatter(data['tracked_frames'], data['speed_errors'], alpha=0.6, s=100)
    ax2.set_xlabel('Tracked Frames', fontsize=12)
    ax2.set_ylabel('Speed Error (%)', fontsize=12)
    ax2.set_title('Tracking Quality vs Speed Error', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 상관관계 표시
    correlation = np.corrcoef(data['tracked_frames'], data['speed_errors'])[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=ax2.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('tracking_quality.png', dpi=150, bbox_inches='tight')
    print("[OK] tracking_quality.png saved")
    plt.close()

def diagnose_direction_issue(data):
    """방향각 문제 진단"""
    print("\n" + "="*80)
    print("DIRECTION ANGLE DIAGNOSTIC")
    print("="*80)
    
    # VZ 부호 분석
    vz_positive = sum(1 for vz in data['vz'] if vz > 0)
    vz_negative = sum(1 for vz in data['vz'] if vz < 0)
    
    print(f"\nVelocity Z Component Analysis:")
    print(f"  Positive VZ (forward): {vz_positive} shots")
    print(f"  Negative VZ (backward): {vz_negative} shots")
    print(f"  -> Expected: All positive (ball moves forward)")
    
    if vz_negative > vz_positive:
        print(f"  [WARNING] Most shots have negative VZ!")
        print(f"  [ISSUE] Z-axis may be inverted in coordinate system")
    
    # VX 부호 vs 실측 방향각
    print(f"\nVelocity X Component vs Real Direction:")
    for i in range(min(5, len(data['shot_numbers']))):
        vx = data['vx'][i]
        real_dir = data['real_direction'][i]
        calc_dir = data['calc_direction'][i]
        print(f"  Shot {data['shot_numbers'][i]}: VX={vx:7.1f} mm/s, "
              f"Real Dir={real_dir:6.2f}°, Calc Dir={calc_dir:7.2f}°")
    
    print(f"\n  -> Real direction near 0° means straight")
    print(f"  -> Calculated direction should also be near 0°")
    print(f"  -> Large discrepancy indicates coordinate transform issue")
    
    # 권장 사항
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("1. Check camera rotation matrix (R) in calibration file")
    print("2. Verify camera alignment with target line")
    print("3. Consider Z-axis inversion: use -vz instead of vz")
    print("4. May need coordinate transform: (X_golf, Y_golf, Z_golf) = R @ (X_cam, Y_cam, Z_cam)")
    print("="*80)

def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION REPORT")
    print("="*80)
    
    # 결과 로드
    results = load_results()
    data = extract_statistics(results)
    
    print(f"\nProcessing {len(data['shot_numbers'])} successful shots...")
    
    # 그래프 생성
    plot_speed_comparison(data)
    plot_angle_comparison(data)
    plot_velocity_vectors(data)
    plot_3d_trajectory_sample(results)
    plot_error_distribution(data)
    plot_tracking_quality(data)
    
    # 방향각 문제 진단
    diagnose_direction_issue(data)
    
    print("\n" + "="*80)
    print("VISUALIZATION REPORT COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - speed_analysis.png")
    print("  - angle_analysis.png")
    print("  - velocity_vector_analysis.png")
    print("  - 3d_trajectory_samples.png")
    print("  - error_distribution.png")
    print("  - tracking_quality.png")
    print("="*80)

if __name__ == "__main__":
    main()
