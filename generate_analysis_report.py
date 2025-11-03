#!/usr/bin/env python3
"""
골프공 3D 분석 결과 리포트 생성기
- 계산된 물리량과 실측 데이터 비교
- 오차 분석 및 시각화
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class AnalysisReportGenerator:
    def __init__(self, results_file="golf_ball_3d_analysis_results.json"):
        """리포트 생성기 초기화"""
        self.results_file = results_file
        self.load_results()
        self.load_real_data()
        
        print("=" * 80)
        print("ANALYSIS REPORT GENERATOR")
        print("=" * 80)
        print(f"Results file: {self.results_file}")
        print(f"Total shots: {len(self.results)}")
        print("=" * 80)
    
    def load_results(self):
        """분석 결과 로드"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
    
    def load_real_data(self):
        """실측 데이터 로드"""
        # 실제 측정된 데이터 (예시)
        # 실제 프로젝트에서는 별도 파일에서 로드
        self.real_data = {
            1: {'speed_m_s': 62.0, 'launch_angle': 12.0, 'direction_angle': 0.0},
            2: {'speed_m_s': 60.0, 'launch_angle': 11.0, 'direction_angle': 2.0},
            3: {'speed_m_s': 63.0, 'launch_angle': 13.0, 'direction_angle': -1.0},
            4: {'speed_m_s': 61.5, 'launch_angle': 12.5, 'direction_angle': 1.0},
            5: {'speed_m_s': 62.5, 'launch_angle': 11.5, 'direction_angle': -0.5}
        }
    
    def calculate_errors(self):
        """오차 계산"""
        errors = []
        
        for result in self.results:
            shot_num = result['shot_number']
            
            if not result['physics']['success']:
                continue
            
            if shot_num not in self.real_data:
                continue
            
            calc = result['physics']
            real = self.real_data[shot_num]
            
            # 속력 오차
            speed_calc = calc['speed']['m_s']
            speed_real = real['speed_m_s']
            speed_error = abs(speed_calc - speed_real)
            speed_error_percent = (speed_error / speed_real) * 100
            
            # 발사각 오차
            launch_calc = calc['launch_angle']['degrees']
            launch_real = real['launch_angle']
            launch_error = abs(launch_calc - launch_real)
            
            # 방향각 오차
            direction_calc = calc['direction_angle']['degrees']
            direction_real = real['direction_angle']
            direction_error = abs(direction_calc - direction_real)
            
            errors.append({
                'shot_number': shot_num,
                'speed': {
                    'calculated': speed_calc,
                    'real': speed_real,
                    'error': speed_error,
                    'error_percent': speed_error_percent
                },
                'launch_angle': {
                    'calculated': launch_calc,
                    'real': launch_real,
                    'error': launch_error
                },
                'direction_angle': {
                    'calculated': direction_calc,
                    'real': direction_real,
                    'error': direction_error
                }
            })
        
        return errors
    
    def generate_comparison_table(self, errors):
        """비교 테이블 생성"""
        print("\n" + "=" * 100)
        print("CALCULATED vs REAL DATA COMPARISON")
        print("=" * 100)
        print(f"{'Shot':<6} {'Speed(calc)':<12} {'Speed(real)':<12} {'Error':<10} "
              f"{'Launch(calc)':<14} {'Launch(real)':<14} {'Error':<10}")
        print("-" * 100)
        
        for error in errors:
            shot = error['shot_number']
            speed_calc = error['speed']['calculated']
            speed_real = error['speed']['real']
            speed_err_pct = error['speed']['error_percent']
            launch_calc = error['launch_angle']['calculated']
            launch_real = error['launch_angle']['real']
            launch_err = error['launch_angle']['error']
            
            print(f"{shot:<6} {speed_calc:<12.2f} {speed_real:<12.2f} {speed_err_pct:<10.2f}% "
                  f"{launch_calc:<14.2f} {launch_real:<14.2f} {launch_err:<10.2f}°")
        
        print("-" * 100)
        
        # 평균 오차
        avg_speed_error = np.mean([e['speed']['error_percent'] for e in errors])
        avg_launch_error = np.mean([e['launch_angle']['error'] for e in errors])
        
        print(f"\nAverage Errors:")
        print(f"  Speed: {avg_speed_error:.2f}%")
        print(f"  Launch Angle: {avg_launch_error:.2f}°")
        print("=" * 100)
    
    def generate_plots(self, errors, output_dir="analysis_plots"):
        """시각화 그래프 생성"""
        os.makedirs(output_dir, exist_ok=True)
        
        shot_numbers = [e['shot_number'] for e in errors]
        
        # 1. 속력 비교
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        speeds_calc = [e['speed']['calculated'] for e in errors]
        speeds_real = [e['speed']['real'] for e in errors]
        
        ax1.plot(shot_numbers, speeds_calc, 'o-', label='Calculated', markersize=8)
        ax1.plot(shot_numbers, speeds_real, 's-', label='Real', markersize=8)
        ax1.set_xlabel('Shot Number')
        ax1.set_ylabel('Speed (m/s)')
        ax1.set_title('Ball Speed Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        speed_errors_pct = [e['speed']['error_percent'] for e in errors]
        ax2.bar(shot_numbers, speed_errors_pct, color='orange', alpha=0.7)
        ax2.set_xlabel('Shot Number')
        ax2.set_ylabel('Error (%)')
        ax2.set_title('Speed Error')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speed_comparison.png'), dpi=150)
        plt.close()
        
        # 2. 발사각 비교
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        launch_calc = [e['launch_angle']['calculated'] for e in errors]
        launch_real = [e['launch_angle']['real'] for e in errors]
        
        ax1.plot(shot_numbers, launch_calc, 'o-', label='Calculated', markersize=8)
        ax1.plot(shot_numbers, launch_real, 's-', label='Real', markersize=8)
        ax1.set_xlabel('Shot Number')
        ax1.set_ylabel('Launch Angle (°)')
        ax1.set_title('Launch Angle Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        launch_errors = [e['launch_angle']['error'] for e in errors]
        ax2.bar(shot_numbers, launch_errors, color='green', alpha=0.7)
        ax2.set_xlabel('Shot Number')
        ax2.set_ylabel('Error (°)')
        ax2.set_title('Launch Angle Error')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'launch_angle_comparison.png'), dpi=150)
        plt.close()
        
        # 3. 3D 궤적 시각화
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.results)))
        
        for i, result in enumerate(self.results):
            if result['physics']['success'] and len(result['detections']) > 0:
                positions = [d['pos_3d'] for d in result['detections']]
                positions = np.array(positions)
                
                ax.plot(positions[:, 0], positions[:, 2], positions[:, 1], 
                       'o-', color=colors[i], label=f"Shot {result['shot_number']}", 
                       markersize=4, linewidth=1.5)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z - Depth (mm)')
        ax.set_zlabel('Y - Height (mm)')
        ax.set_title('3D Golf Ball Trajectories')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3d_trajectories.png'), dpi=150)
        plt.close()
        
        print(f"\nPlots saved to: {output_dir}/")
    
    def generate_report(self, output_file="analysis_report.md"):
        """마크다운 리포트 생성"""
        errors = self.calculate_errors()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 골프공 3D 물리량 분석 리포트\n\n")
            f.write("## 1. 시스템 개요\n\n")
            f.write("### 캘리브레이션 설정\n")
            f.write("- **이미지 크기**: 1440x1080 (캘리브레이션), 1440x300 (ROI 측정)\n")
            f.write("- **카메라 구성**: 수직 스테레오비전 (Vertical Stereo)\n")
            f.write("- **베이스라인**: 470mm (카메라 간 수직 거리)\n")
            f.write("- **프레임 레이트**: 820 fps\n")
            f.write("- **좌표계**: X=수평, Y=수직, Z=깊이\n\n")
            
            f.write("### 실제 거리 정보\n")
            f.write("- **카메라1 - 골프공**: 900-1000mm\n")
            f.write("- **카메라2 - 골프공**: 500-600mm\n\n")
            
            f.write("## 2. 분석 결과\n\n")
            f.write(f"- **전체 샷 수**: {len(self.results)}\n")
            
            successful = [r for r in self.results if r['physics']['success']]
            f.write(f"- **성공적으로 분석된 샷**: {len(successful)}\n")
            f.write(f"- **실측 데이터와 비교 가능한 샷**: {len(errors)}\n\n")
            
            f.write("## 3. 계산 결과 vs 실측 데이터 비교\n\n")
            f.write("| Shot | Speed (calc) | Speed (real) | Error | Launch (calc) | Launch (real) | Error |\n")
            f.write("|------|--------------|--------------|-------|---------------|---------------|-------|\n")
            
            for error in errors:
                shot = error['shot_number']
                speed_c = error['speed']['calculated']
                speed_r = error['speed']['real']
                speed_e = error['speed']['error_percent']
                launch_c = error['launch_angle']['calculated']
                launch_r = error['launch_angle']['real']
                launch_e = error['launch_angle']['error']
                
                f.write(f"| {shot} | {speed_c:.2f} m/s | {speed_r:.2f} m/s | {speed_e:.2f}% | "
                       f"{launch_c:.2f}° | {launch_r:.2f}° | {launch_e:.2f}° |\n")
            
            f.write("\n## 4. 오차 통계\n\n")
            
            if errors:
                avg_speed_error = np.mean([e['speed']['error_percent'] for e in errors])
                std_speed_error = np.std([e['speed']['error_percent'] for e in errors])
                avg_launch_error = np.mean([e['launch_angle']['error'] for e in errors])
                std_launch_error = np.std([e['launch_angle']['error'] for e in errors])
                
                f.write(f"### 속력 오차\n")
                f.write(f"- **평균**: {avg_speed_error:.2f}%\n")
                f.write(f"- **표준편차**: {std_speed_error:.2f}%\n\n")
                
                f.write(f"### 발사각 오차\n")
                f.write(f"- **평균**: {avg_launch_error:.2f}°\n")
                f.write(f"- **표준편차**: {std_launch_error:.2f}°\n\n")
            
            f.write("## 5. 개선 사항\n\n")
            f.write("### 적용된 보정\n")
            f.write("1. **ROI 좌표계 변환**: 1440x300 ROI를 1440x1080 캘리브레이션 좌표계로 정확히 매핑\n")
            f.write("2. **실제 거리 기반 깊이 보정**: 측정된 카메라-골프공 거리를 활용한 스케일 보정\n")
            f.write("3. **수직 스테레오비전 최적화**: Y축 시차를 고려한 깊이 계산\n\n")
            
            f.write("### 향후 개선 방향\n")
            f.write("1. 더 많은 캘리브레이션 이미지 수집으로 정확도 향상\n")
            f.write("2. 렌즈 왜곡 보정 파라미터 정밀화\n")
            f.write("3. 다중 프레임 데이터를 활용한 칼만 필터 적용\n")
            f.write("4. 골프공 검출 알고리즘 개선 (딥러닝 기반)\n\n")
            
            f.write("## 6. 시각화\n\n")
            f.write("![Speed Comparison](analysis_plots/speed_comparison.png)\n\n")
            f.write("![Launch Angle Comparison](analysis_plots/launch_angle_comparison.png)\n\n")
            f.write("![3D Trajectories](analysis_plots/3d_trajectories.png)\n\n")
        
        print(f"\nMarkdown report saved to: {output_file}")
        
        return errors
    
    def run(self):
        """전체 리포트 생성 프로세스 실행"""
        print("\nGenerating analysis report...")
        
        # 오차 계산 및 출력
        errors = self.calculate_errors()
        self.generate_comparison_table(errors)
        
        # 그래프 생성
        self.generate_plots(errors)
        
        # 마크다운 리포트 생성
        self.generate_report()
        
        print("\n" + "=" * 80)
        print("REPORT GENERATION COMPLETED")
        print("=" * 80)
        print("✓ Comparison table printed")
        print("✓ Plots saved to: analysis_plots/")
        print("✓ Markdown report saved to: analysis_report.md")
        print("=" * 80)

def main():
    """메인 함수"""
    results_file = "golf_ball_3d_analysis_results.json"
    
    if not os.path.exists(results_file):
        print(f"[ERROR] Results file not found: {results_file}")
        print("Please run golf_ball_3d_physics_analyzer.py first!")
        return
    
    generator = AnalysisReportGenerator(results_file=results_file)
    generator.run()

if __name__ == "__main__":
    main()
