"""
골프 스윙 분석 시뮬레이션 환경 및 시각화 도구
Author: Maxform 개발팀
Description: 골프 스윙 분석 시스템 테스트를 위한 시뮬레이션 환경
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import json

from golf_swing_analyzer import GolfSwingAnalyzer, SystemConfig, AnalysisResult
from object_tracker import TrackingPoint, BallData, ClubData

@dataclass
class SimulationParameters:
    """시뮬레이션 파라미터"""
    ball_initial_speed: float = 45.0  # m/s
    ball_launch_angle: float = 12.0  # degrees
    ball_azimuth_angle: float = 2.0  # degrees
    ball_backspin: float = 2500.0  # rpm
    ball_sidespin: float = 300.0  # rpm
    
    club_speed: float = 35.0  # m/s
    club_attack_angle: float = -2.0  # degrees
    club_path: float = 1.0  # degrees
    
    gravity: float = 9.81  # m/s^2
    air_resistance: float = 0.01
    simulation_fps: int = 240
    simulation_duration: float = 3.0  # seconds

class GolfSwingSimulator:
    """골프 스윙 시뮬레이터"""
    
    def __init__(self, params: Optional[SimulationParameters] = None):
        self.params = params or SimulationParameters()
        self.ball_trajectory = []
        self.club_trajectory = []
        self.timestamps = []
        
    def simulate_ball_trajectory(self) -> List[TrackingPoint]:
        """골프공 궤적 시뮬레이션"""
        dt = 1.0 / self.params.simulation_fps
        total_frames = int(self.params.simulation_duration * self.params.simulation_fps)
        
        # 초기 조건
        v0 = self.params.ball_initial_speed
        launch_rad = math.radians(self.params.ball_launch_angle)
        azimuth_rad = math.radians(self.params.ball_azimuth_angle)
        
        # 초기 속도 성분
        vx0 = v0 * math.cos(launch_rad) * math.cos(azimuth_rad)
        vy0 = v0 * math.cos(launch_rad) * math.sin(azimuth_rad)
        vz0 = v0 * math.sin(launch_rad)
        
        # 스핀 효과 계산
        backspin_rad = self.params.ball_backspin * 2 * math.pi / 60  # rad/s
        sidespin_rad = self.params.ball_sidespin * 2 * math.pi / 60  # rad/s
        
        trajectory = []
        x, y, z = 0.0, 0.0, 0.0
        vx, vy, vz = vx0, vy0, vz0
        
        for frame in range(total_frames):
            t = frame * dt
            
            # 중력 효과
            vz -= self.params.gravity * dt
            
            # 공기 저항 효과
            v_total = math.sqrt(vx**2 + vy**2 + vz**2)
            if v_total > 0:
                drag_factor = self.params.air_resistance * v_total * dt
                vx *= (1 - drag_factor)
                vy *= (1 - drag_factor)
                vz *= (1 - drag_factor)
            
            # 스핀 효과 (Magnus force)
            magnus_factor = 0.0001  # 스핀 효과 계수
            if backspin_rad > 0:
                vz += magnus_factor * backspin_rad * vx * dt  # 백스핀은 상승력 제공
            if sidespin_rad > 0:
                vy += magnus_factor * sidespin_rad * vx * dt  # 사이드스핀은 좌우 편향
            
            # 위치 업데이트
            x += vx * dt
            y += vy * dt
            z += vz * dt
            
            # 지면에 닿으면 시뮬레이션 종료
            if z < 0 and frame > 10:
                break
            
            # 추적 점 생성
            point = TrackingPoint(
                x=x, y=y, z=z,
                timestamp=time.time() + t,
                confidence=1.0
            )
            trajectory.append(point)
        
        self.ball_trajectory = trajectory
        return trajectory
    
    def simulate_club_trajectory(self) -> List[TrackingPoint]:
        """클럽 궤적 시뮬레이션"""
        dt = 1.0 / self.params.simulation_fps
        swing_duration = 0.5  # 스윙 지속 시간 (초)
        total_frames = int(swing_duration * self.params.simulation_fps)
        
        trajectory = []
        
        for frame in range(total_frames):
            t = frame * dt
            
            # 스윙 궤적 (원호 운동)
            swing_progress = t / swing_duration
            
            # 각도 (백스윙에서 팔로우스루까지)
            angle = -math.pi/3 + swing_progress * (2*math.pi/3)  # -60도에서 +60도
            
            # 반지름 (클럽 길이)
            radius = 1.2  # 1.2m
            
            # 위치 계산
            x = radius * math.sin(angle) * swing_progress
            y = math.sin(swing_progress * math.pi) * 0.2  # 좌우 움직임
            z = -0.5 + radius * (1 - math.cos(angle)) * 0.5  # 수직 움직임
            
            # 어택 앵글 적용
            attack_rad = math.radians(self.params.club_attack_angle)
            z += x * math.tan(attack_rad)
            
            # 클럽 패스 적용
            path_rad = math.radians(self.params.club_path)
            y += x * math.tan(path_rad)
            
            # 추적 점 생성
            point = TrackingPoint(
                x=x, y=y, z=z,
                timestamp=time.time() + t,
                confidence=1.0
            )
            trajectory.append(point)
        
        self.club_trajectory = trajectory
        return trajectory
    
    def generate_synthetic_frames(self, image_size: Tuple[int, int] = (640, 480)) -> List[Tuple[np.ndarray, np.ndarray]]:
        """합성 프레임 생성"""
        frames = []
        ball_traj = self.simulate_ball_trajectory()
        club_traj = self.simulate_club_trajectory()
        
        # 카메라 파라미터 (가상)
        focal_length = 800
        cx, cy = image_size[0] // 2, image_size[1] // 2
        baseline = 0.06  # 6cm
        
        max_frames = max(len(ball_traj), len(club_traj))
        
        for i in range(max_frames):
            # 배경 이미지 생성
            left_frame = np.random.randint(50, 100, (*image_size[::-1], 3), dtype=np.uint8)
            right_frame = left_frame.copy()
            
            # 노이즈 추가
            noise = np.random.randint(-20, 20, left_frame.shape, dtype=np.int16)
            left_frame = np.clip(left_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            right_frame = np.clip(right_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 골프공 그리기
            if i < len(ball_traj):
                ball_point = ball_traj[i]
                if ball_point.z > 0:  # 지면 위에 있을 때만
                    # 3D → 2D 투영
                    left_u = int(focal_length * ball_point.x / ball_point.z + cx)
                    left_v = int(focal_length * ball_point.y / ball_point.z + cy)
                    
                    right_u = int(focal_length * (ball_point.x - baseline) / ball_point.z + cx)
                    right_v = left_v
                    
                    # 공 그리기 (흰색 원)
                    if 0 <= left_u < image_size[0] and 0 <= left_v < image_size[1]:
                        cv2.circle(left_frame, (left_u, left_v), 8, (255, 255, 255), -1)
                        cv2.circle(left_frame, (left_u, left_v), 10, (200, 200, 200), 2)
                    
                    if 0 <= right_u < image_size[0] and 0 <= right_v < image_size[1]:
                        cv2.circle(right_frame, (right_u, right_v), 8, (255, 255, 255), -1)
                        cv2.circle(right_frame, (right_u, right_v), 10, (200, 200, 200), 2)
            
            # 클럽 헤드 그리기
            if i < len(club_traj):
                club_point = club_traj[i]
                if club_point.z > -1.0:  # 지면 근처에서만
                    # 3D → 2D 투영
                    left_u = int(focal_length * club_point.x / (club_point.z + 2.0) + cx)
                    left_v = int(focal_length * club_point.y / (club_point.z + 2.0) + cy)
                    
                    right_u = int(focal_length * (club_point.x - baseline) / (club_point.z + 2.0) + cx)
                    right_v = left_v
                    
                    # 클럽 헤드 그리기 (회색 사각형)
                    if 0 <= left_u < image_size[0] and 0 <= left_v < image_size[1]:
                        cv2.rectangle(left_frame, 
                                    (left_u-15, left_v-8), (left_u+15, left_v+8), 
                                    (150, 150, 150), -1)
                        cv2.rectangle(left_frame, 
                                    (left_u-15, left_v-8), (left_u+15, left_v+8), 
                                    (100, 100, 100), 2)
                    
                    if 0 <= right_u < image_size[0] and 0 <= right_v < image_size[1]:
                        cv2.rectangle(right_frame, 
                                    (right_u-15, right_v-8), (right_u+15, right_v+8), 
                                    (150, 150, 150), -1)
                        cv2.rectangle(right_frame, 
                                    (right_u-15, right_v-8), (right_u+15, right_v+8), 
                                    (100, 100, 100), 2)
            
            frames.append((left_frame, right_frame))
        
        return frames

class VisualizationTool:
    """시각화 도구"""
    
    def __init__(self):
        self.fig = None
        self.axes = None
        
    def plot_trajectory_3d(self, ball_trajectory: List[TrackingPoint], 
                          club_trajectory: List[TrackingPoint] = None,
                          save_path: Optional[str] = None):
        """3D 궤적 플롯"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 골프공 궤적
        if ball_trajectory:
            ball_x = [p.x for p in ball_trajectory]
            ball_y = [p.y for p in ball_trajectory]
            ball_z = [p.z for p in ball_trajectory]
            
            ax.plot(ball_x, ball_y, ball_z, 'b-', linewidth=2, label='Ball Trajectory')
            ax.scatter(ball_x[0], ball_y[0], ball_z[0], c='green', s=100, label='Start')
            ax.scatter(ball_x[-1], ball_y[-1], ball_z[-1], c='red', s=100, label='End')
        
        # 클럽 궤적
        if club_trajectory:
            club_x = [p.x for p in club_trajectory]
            club_y = [p.y for p in club_trajectory]
            club_z = [p.z for p in club_trajectory]
            
            ax.plot(club_x, club_y, club_z, 'r--', linewidth=2, label='Club Trajectory')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Golf Swing 3D Trajectory')
        ax.legend()
        ax.grid(True)
        
        # 지면 표시
        xx, yy = np.meshgrid(np.linspace(-1, 5, 10), np.linspace(-2, 2, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='green')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def plot_analysis_results(self, results: List[AnalysisResult], 
                            save_path: Optional[str] = None):
        """분석 결과 플롯"""
        if not results:
            print("표시할 결과가 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Golf Swing Analysis Results', fontsize=16)
        
        # 데이터 추출
        ball_speeds = [r.ball_data.speed for r in results if r.ball_data]
        launch_angles = [r.ball_data.launch_angle for r in results if r.ball_data]
        backspins = [r.ball_data.backspin for r in results if r.ball_data]
        club_speeds = [r.club_data.speed for r in results if r.club_data]
        attack_angles = [r.club_data.attack_angle for r in results if r.club_data]
        processing_times = [r.processing_time for r in results]
        
        # 볼 스피드
        if ball_speeds:
            axes[0, 0].hist(ball_speeds, bins=10, alpha=0.7, color='blue')
            axes[0, 0].set_title('Ball Speed Distribution')
            axes[0, 0].set_xlabel('Speed (m/s)')
            axes[0, 0].set_ylabel('Frequency')
        
        # 발사각
        if launch_angles:
            axes[0, 1].hist(launch_angles, bins=10, alpha=0.7, color='green')
            axes[0, 1].set_title('Launch Angle Distribution')
            axes[0, 1].set_xlabel('Angle (degrees)')
            axes[0, 1].set_ylabel('Frequency')
        
        # 백스핀
        if backspins:
            axes[0, 2].hist(backspins, bins=10, alpha=0.7, color='red')
            axes[0, 2].set_title('Backspin Distribution')
            axes[0, 2].set_xlabel('Spin (rpm)')
            axes[0, 2].set_ylabel('Frequency')
        
        # 클럽 스피드
        if club_speeds:
            axes[1, 0].hist(club_speeds, bins=10, alpha=0.7, color='orange')
            axes[1, 0].set_title('Club Speed Distribution')
            axes[1, 0].set_xlabel('Speed (m/s)')
            axes[1, 0].set_ylabel('Frequency')
        
        # 어택 앵글
        if attack_angles:
            axes[1, 1].hist(attack_angles, bins=10, alpha=0.7, color='purple')
            axes[1, 1].set_title('Attack Angle Distribution')
            axes[1, 1].set_xlabel('Angle (degrees)')
            axes[1, 1].set_ylabel('Frequency')
        
        # 처리 시간
        axes[1, 2].hist(processing_times, bins=10, alpha=0.7, color='brown')
        axes[1, 2].set_title('Processing Time Distribution')
        axes[1, 2].set_xlabel('Time (seconds)')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def create_animation(self, ball_trajectory: List[TrackingPoint],
                        club_trajectory: List[TrackingPoint] = None,
                        save_path: Optional[str] = None):
        """궤적 애니메이션 생성"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 전체 궤적 범위 설정
        all_x = [p.x for p in ball_trajectory]
        all_y = [p.y for p in ball_trajectory]
        all_z = [p.z for p in ball_trajectory]
        
        if club_trajectory:
            all_x.extend([p.x for p in club_trajectory])
            all_y.extend([p.y for p in club_trajectory])
            all_z.extend([p.z for p in club_trajectory])
        
        ax.set_xlim(min(all_x)-1, max(all_x)+1)
        ax.set_ylim(min(all_y)-1, max(all_y)+1)
        ax.set_zlim(0, max(all_z)+1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Golf Swing Animation')
        
        # 애니메이션 함수
        def animate(frame):
            ax.clear()
            ax.set_xlim(min(all_x)-1, max(all_x)+1)
            ax.set_ylim(min(all_y)-1, max(all_y)+1)
            ax.set_zlim(0, max(all_z)+1)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Golf Swing Animation - Frame {frame}')
            
            # 지면
            xx, yy = np.meshgrid(np.linspace(min(all_x)-1, max(all_x)+1, 10), 
                               np.linspace(min(all_y)-1, max(all_y)+1, 10))
            zz = np.zeros_like(xx)
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='green')
            
            # 골프공 궤적 (현재 프레임까지)
            if frame < len(ball_trajectory):
                ball_x = [p.x for p in ball_trajectory[:frame+1]]
                ball_y = [p.y for p in ball_trajectory[:frame+1]]
                ball_z = [p.z for p in ball_trajectory[:frame+1]]
                
                if ball_x:
                    ax.plot(ball_x, ball_y, ball_z, 'b-', linewidth=2, label='Ball')
                    ax.scatter(ball_x[-1], ball_y[-1], ball_z[-1], 
                             c='blue', s=100, marker='o')
            
            # 클럽 궤적
            if club_trajectory and frame < len(club_trajectory):
                club_x = [p.x for p in club_trajectory[:frame+1]]
                club_y = [p.y for p in club_trajectory[:frame+1]]
                club_z = [p.z for p in club_trajectory[:frame+1]]
                
                if club_x:
                    ax.plot(club_x, club_y, club_z, 'r--', linewidth=2, label='Club')
                    ax.scatter(club_x[-1], club_y[-1], club_z[-1], 
                             c='red', s=100, marker='s')
            
            ax.legend()
        
        # 애니메이션 생성
        frames = max(len(ball_trajectory), len(club_trajectory) if club_trajectory else 0)
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                     interval=50, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
        
        plt.show()
        return anim

def run_simulation_demo():
    """시뮬레이션 데모 실행"""
    print("골프 스윙 분석 시뮬레이션 데모 시작...")
    
    # 시뮬레이션 파라미터 설정
    sim_params = SimulationParameters(
        ball_initial_speed=50.0,
        ball_launch_angle=15.0,
        ball_azimuth_angle=3.0,
        ball_backspin=3000.0,
        club_speed=40.0,
        club_attack_angle=-3.0
    )
    
    # 시뮬레이터 생성
    simulator = GolfSwingSimulator(sim_params)
    
    # 궤적 시뮬레이션
    print("궤적 시뮬레이션 중...")
    ball_trajectory = simulator.simulate_ball_trajectory()
    club_trajectory = simulator.simulate_club_trajectory()
    
    print(f"볼 궤적 점 수: {len(ball_trajectory)}")
    print(f"클럽 궤적 점 수: {len(club_trajectory)}")
    
    # 합성 프레임 생성
    print("합성 프레임 생성 중...")
    frames = simulator.generate_synthetic_frames()
    print(f"생성된 프레임 수: {len(frames)}")
    
    # 분석 시스템으로 테스트
    config = SystemConfig(min_tracking_points=5, max_tracking_time=1.0)
    analyzer = GolfSwingAnalyzer(config)
    
    print("시뮬레이션된 프레임으로 분석 테스트 중...")
    analyzer.start_analysis()
    
    # 실제 추적 데이터 주입 (시뮬레이션된 궤적 사용)
    for i, point in enumerate(ball_trajectory[:20]):  # 처음 20개 점만 사용
        analyzer.ball_tracker.update_tracking(
            np.array([point.x, point.y, point.z]), point.timestamp
        )
    
    for i, point in enumerate(club_trajectory[:15]):  # 처음 15개 점만 사용
        analyzer.club_tracker.update_tracking(
            np.array([point.x, point.y, point.z]), point.timestamp
        )
    
    analyzer.complete_analysis()
    
    # 결과 출력
    if analyzer.current_analysis:
        print("\n=== 시뮬레이션 분석 결과 ===")
        analyzer.print_analysis_result(analyzer.current_analysis)
    
    # 시각화
    print("시각화 생성 중...")
    visualizer = VisualizationTool()
    
    # 3D 궤적 플롯
    visualizer.plot_trajectory_3d(ball_trajectory, club_trajectory, 
                                 save_path="simulation_trajectory.png")
    
    # 분석 결과 플롯
    if analyzer.analysis_results:
        visualizer.plot_analysis_results(analyzer.analysis_results,
                                       save_path="simulation_analysis.png")
    
    print("시뮬레이션 데모 완료!")

if __name__ == "__main__":
    run_simulation_demo()

