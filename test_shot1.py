#!/usr/bin/env python3
"""간단한 테스트 스크립트"""
from golf_ball_3d_physics_analyzer import GolfBall3DPhysicsAnalyzer

analyzer = GolfBall3DPhysicsAnalyzer()
result = analyzer.analyze_shot('data2/driver/1', 1)
print(f"\nTracked: {result['tracked_frames']} frames")
print(f"Success: {result['physics']['success']}")
if result['physics']['success']:
    print(f"Speed: {result['physics']['speed']['m_s']:.2f} m/s")
    print(f"Launch angle: {result['physics']['launch_angle']['degrees']:.2f}°")
