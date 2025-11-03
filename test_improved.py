#!/usr/bin/env python3
"""개선된 시스템 테스트"""
from improved_golf_ball_3d_analyzer import ImprovedGolfBall3DAnalyzer

analyzer = ImprovedGolfBall3DAnalyzer()

# 샷 1 테스트
result = analyzer.analyze_shot_improved('data2/driver/1', 1)

print(f"\nTest completed!")
print(f"Tracked frames: {result['tracked_frames']}")
if result['physics']['success']:
    print(f"Speed: {result['physics']['speed']['m_s']:.2f} m/s")
    if result['real_data']:
        print(f"Real speed: {result['real_data']['ball_speed_ms']:.2f} m/s")
