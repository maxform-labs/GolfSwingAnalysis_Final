#!/usr/bin/env python3
"""
최적화된 성능 골프 스윙 분석기
- 100ms 내 처리 목표 달성을 위한 최적화
- 단순화된 알고리즘 사용
- 빠른 골프공 검출
"""

import cv2
import numpy as np
import time
import os
import glob
from pathlib import Path

class OptimizedPerformanceAnalyzer:
    def __init__(self):
        # 캘리브레이션 파라미터
        self.baseline = 470.0  # mm
        self.focal_length = 1440  # pixels
        self.image_size = (1440, 300)
        
        # 최적화된 골프공 검출 파라미터 (빠른 처리용)
        self.ball_params = {
            'min_radius': 10,
            'max_radius': 50,
            'param1': 50,
            'param2': 30,
            'min_dist': 30
        }
    
    def fast_preprocess_image(self, image):
        """빠른 이미지 전처리"""
        # 간단한 밝기 조정만 사용
        bright = cv2.convertScaleAbs(image, alpha=1.5, beta=20)
        return bright
    
    def fast_detect_golf_ball(self, image):
        """빠른 골프공 검출"""
        processed = self.fast_preprocess_image(image)
        
        # 단일 파라미터 세트만 사용
        circles = cv2.HoughCircles(
            processed,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.ball_params['min_dist'],
            param1=self.ball_params['param1'],
            param2=self.ball_params['param2'],
            minRadius=self.ball_params['min_radius'],
            maxRadius=self.ball_params['max_radius']
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # 첫 번째 검출된 원을 사용
            for circle in circles:
                x, y, r = circle
                if (r < x < image.shape[1] - r and r < y < image.shape[0] - r):
                    return circle
        
        return None
    
    def fast_calculate_3d_coordinates(self, point1, point2):
        """빠른 3D 좌표 계산"""
        if point1 is None or point2 is None:
            return None
        
        # Y축 시차 계산
        disparity = abs(point1[1] - point2[1])
        
        if disparity < 2:
            return None
        
        # 3D 좌표 계산
        z = (self.focal_length * self.baseline) / disparity
        
        if z < 200 or z > 8000:
            return None
        
        x = (point1[0] - self.image_size[0]/2) * z / self.focal_length
        y = (point1[1] - self.image_size[1]/2) * z / self.focal_length
        
        return np.array([x, y, z])
    
    def analyze_frame_pair_fast(self, img1, img2):
        """빠른 프레임 쌍 분석"""
        start_time = time.time()
        
        # 골프공 검출만 수행 (가장 중요한 부분)
        ball1 = self.fast_detect_golf_ball(img1)
        ball2 = self.fast_detect_golf_ball(img2)
        
        # 3D 좌표 계산
        ball_3d = None
        if ball1 is not None and ball2 is not None:
            ball_3d = self.fast_calculate_3d_coordinates(ball1[:2], ball2[:2])
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # ms
        
        result = {
            'processing_time_ms': processing_time,
            'ball_detected_cam1': ball1 is not None,
            'ball_detected_cam2': ball2 is not None,
            'ball_3d': ball_3d.tolist() if ball_3d is not None else None,
            'ball1': ball1.tolist() if ball1 is not None else None,
            'ball2': ball2.tolist() if ball2 is not None else None
        }
        
        return result
    
    def performance_test_optimized(self, test_images=20):
        """최적화된 성능 테스트"""
        print("=== 최적화된 성능 테스트 ===")
        print(f"목표: 100ms 내 처리")
        print(f"테스트 이미지 수: {test_images}")
        print()
        
        # 테스트 이미지 로드 (5번 아이언 샷 1 사용)
        test_path = Path("data/video_ballData_20250930/video_ballData_20250930/5Iron_0930/1")
        if not test_path.exists():
            print("테스트 이미지 경로를 찾을 수 없습니다.")
            return
        
        # 이미지 파일들 로드
        image_files = sorted(glob.glob(str(test_path / "*.bmp")))
        cam1_images = [f for f in image_files if "1_" in os.path.basename(f) and "Gamma" not in os.path.basename(f)]
        cam2_images = [f for f in image_files if "2_" in os.path.basename(f) and "Gamma" not in os.path.basename(f)]
        
        # 테스트할 이미지 수 제한
        test_count = min(test_images, len(cam1_images), len(cam2_images))
        
        results = []
        total_time = 0
        successful_detections = 0
        under_100ms_count = 0
        
        print("프레임별 처리 시간:")
        for i in range(test_count):
            # 이미지 로드
            img1 = cv2.imread(cam1_images[i], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(cam2_images[i], cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                continue
            
            # 빠른 분석
            result = self.analyze_frame_pair_fast(img1, img2)
            results.append(result)
            
            processing_time = result['processing_time_ms']
            total_time += processing_time
            
            if result['ball_detected_cam1'] or result['ball_detected_cam2']:
                successful_detections += 1
            
            if processing_time < 100:
                under_100ms_count += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"  프레임 {i+1:2d}: {processing_time:6.1f}ms {status}")
        
        # 결과 요약
        avg_time = total_time / len(results) if results else 0
        success_rate = (under_100ms_count / len(results) * 100) if results else 0
        detection_rate = (successful_detections / len(results) * 100) if results else 0
        
        print(f"\n=== 성능 테스트 결과 ===")
        print(f"총 테스트 수: {len(results)}")
        print(f"평균 처리 시간: {avg_time:.1f}ms")
        print(f"100ms 이하 비율: {success_rate:.1f}%")
        print(f"골프공 검출 성공률: {detection_rate:.1f}%")
        
        if success_rate >= 90:
            print("\n🎯 목표 달성: 90% 이상이 100ms 내에 완료되었습니다!")
        elif success_rate >= 50:
            print(f"\n⚠️  부분 달성: {success_rate:.1f}%가 100ms 내에 완료되었습니다.")
        else:
            print(f"\n❌ 목표 미달성: {success_rate:.1f}%만 100ms 내에 완료되었습니다.")
        
        return results, {
            'avg_processing_time': avg_time,
            'success_rate': success_rate,
            'detection_rate': detection_rate,
            'total_tests': len(results)
        }

def main():
    analyzer = OptimizedPerformanceAnalyzer()
    
    print("최적화된 골프 스윙 분석기 성능 테스트")
    print("=" * 50)
    
    # 성능 테스트 실행
    results, summary = analyzer.performance_test_optimized(test_images=20)
    
    print(f"\n최종 결과:")
    print(f"- 평균 처리 시간: {summary['avg_processing_time']:.1f}ms")
    print(f"- 100ms 이하 성공률: {summary['success_rate']:.1f}%")
    print(f"- 골프공 검출 성공률: {summary['detection_rate']:.1f}%")

if __name__ == "__main__":
    main()
