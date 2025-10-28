#!/usr/bin/env python3
"""
골프 스윙 분석 결과 시각화 가이드
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GolfSwingVisualizer:
    def __init__(self):
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """플롯 스타일 설정"""
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def visualize_3d_trajectory(self, trajectory_data, title="Golf Ball 3D Trajectory"):
        """3D 궤적 시각화"""
        if not trajectory_data:
            print("❌ 궤적 데이터가 없습니다.")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 궤적 데이터 추출
        x_coords = [pos[0] for pos in trajectory_data]
        y_coords = [pos[1] for pos in trajectory_data]
        z_coords = [pos[2] for pos in trajectory_data]
        
        # 3D 궤적 그리기
        ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2, label='Ball Trajectory')
        ax.scatter(x_coords, y_coords, z_coords, c='red', s=50, label='Ball Positions')
        
        # 시작점과 끝점 강조
        ax.scatter(x_coords[0], y_coords[0], z_coords[0], c='green', s=100, label='Start')
        ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], c='red', s=100, label='End')
        
        # 축 설정
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        ax.legend()
        
        # 축 범위 설정
        ax.set_xlim(min(x_coords)-50, max(x_coords)+50)
        ax.set_ylim(min(y_coords)-50, max(y_coords)+50)
        ax.set_zlim(min(z_coords)-50, max(z_coords)+50)
        
        plt.tight_layout()
        plt.savefig('golf_ball_3d_trajectory.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_speed_analysis(self, speed_data, time_data):
        """속도 분석 시각화"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 속도 그래프
        ax1.plot(time_data, speed_data, 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Speed (mph)')
        ax1.set_title('Golf Ball Speed Over Time')
        ax1.grid(True, alpha=0.3)
        
        # 속도 히스토그램
        ax2.hist(speed_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Speed (mph)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Speed Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('golf_ball_speed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_angle_analysis(self, launch_angles, direction_angles):
        """각도 분석 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 발사각 분포
        ax1.hist(launch_angles, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax1.set_xlabel('Launch Angle (degrees)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Launch Angle Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 방향각 분포
        ax2.hist(direction_angles, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Direction Angle (degrees)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Direction Angle Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('golf_ball_angle_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_analysis_dashboard(self, analysis_results):
        """분석 대시보드 생성"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 속도 추이
        if 'speed_data' in analysis_results:
            ax1.plot(analysis_results['time_data'], analysis_results['speed_data'], 'b-', linewidth=2)
            ax1.set_title('Ball Speed Over Time')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Speed (mph)')
            ax1.grid(True, alpha=0.3)
        
        # 2. 발사각 추이
        if 'launch_angles' in analysis_results:
            ax2.plot(analysis_results['time_data'], analysis_results['launch_angles'], 'g-', linewidth=2)
            ax2.set_title('Launch Angle Over Time')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Launch Angle (degrees)')
            ax2.grid(True, alpha=0.3)
        
        # 3. 방향각 추이
        if 'direction_angles' in analysis_results:
            ax3.plot(analysis_results['time_data'], analysis_results['direction_angles'], 'r-', linewidth=2)
            ax3.set_title('Direction Angle Over Time')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Direction Angle (degrees)')
            ax3.grid(True, alpha=0.3)
        
        # 4. 성능 지표
        if 'performance_metrics' in analysis_results:
            metrics = analysis_results['performance_metrics']
            labels = list(metrics.keys())
            values = list(metrics.values())
            
            bars = ax4.bar(labels, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            ax4.set_title('Performance Metrics')
            ax4.set_ylabel('Value')
            
            # 값 표시
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('golf_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_chart(self, club_data):
        """클럽별 비교 차트"""
        if not club_data:
            print("❌ 클럽 데이터가 없습니다.")
            return
        
        clubs = list(club_data.keys())
        avg_speeds = [club_data[club]['avg_speed'] for club in clubs]
        avg_launch_angles = [club_data[club]['avg_launch_angle'] for club in clubs]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 평균 속도 비교
        bars1 = ax1.bar(clubs, avg_speeds, color='skyblue', alpha=0.7)
        ax1.set_title('Average Ball Speed by Club')
        ax1.set_ylabel('Speed (mph)')
        
        for bar, speed in zip(bars1, avg_speeds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{speed:.1f}', ha='center', va='bottom')
        
        # 평균 발사각 비교
        bars2 = ax2.bar(clubs, avg_launch_angles, color='lightgreen', alpha=0.7)
        ax2.set_title('Average Launch Angle by Club')
        ax2.set_ylabel('Launch Angle (degrees)')
        
        for bar, angle in zip(bars2, avg_launch_angles):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{angle:.1f}°', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('club_comparison_chart.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """시각화 예제"""
    print("🎨 골프 스윙 분석 시각화 가이드")
    
    # 시각화 도구 초기화
    visualizer = GolfSwingVisualizer()
    
    # 예제 데이터 생성
    trajectory_data = [
        [100, 500, 0],
        [150, 480, 20],
        [200, 450, 50],
        [250, 400, 80],
        [300, 350, 100]
    ]
    
    speed_data = [0, 45, 52, 48, 44]
    time_data = [0, 0.1, 0.2, 0.3, 0.4]
    
    launch_angles = [12, 14, 13, 15, 14]
    direction_angles = [2, 1, 0, -1, -2]
    
    # 시각화 실행
    print("📊 3D 궤적 시각화...")
    visualizer.visualize_3d_trajectory(trajectory_data)
    
    print("📈 속도 분석 시각화...")
    visualizer.visualize_speed_analysis(speed_data, time_data)
    
    print("📐 각도 분석 시각화...")
    visualizer.visualize_angle_analysis(launch_angles, direction_angles)
    
    # 분석 대시보드
    analysis_results = {
        'speed_data': speed_data,
        'time_data': time_data,
        'launch_angles': launch_angles,
        'direction_angles': direction_angles,
        'performance_metrics': {
            'Detection Rate': 95.0,
            'Processing Time': 45.0,
            'Accuracy': 97.5,
            'Success Rate': 98.0
        }
    }
    
    print("📋 분석 대시보드 생성...")
    visualizer.create_analysis_dashboard(analysis_results)
    
    # 클럽별 비교
    club_data = {
        'Driver': {'avg_speed': 145, 'avg_launch_angle': 12},
        '5-Iron': {'avg_speed': 125, 'avg_launch_angle': 18},
        '7-Iron': {'avg_speed': 115, 'avg_launch_angle': 22},
        'PW': {'avg_speed': 95, 'avg_launch_angle': 28}
    }
    
    print("🏌️ 클럽별 비교 차트...")
    visualizer.create_comparison_chart(club_data)
    
    print("✅ 모든 시각화 완료!")

if __name__ == "__main__":
    main()
