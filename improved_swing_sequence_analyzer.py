#!/usr/bin/env python3
"""
개선된 스윙 시퀀스 분석기
- 실제 이미지에 맞는 파라미터 조정
- Gamma 조정 vs 일반 사진 품질 비교
- 정확도 우선 (최대 150ms 처리 시간 허용)
"""

import cv2
import numpy as np
import time
import os
import glob
from pathlib import Path

class ImprovedSwingSequenceAnalyzer:
    def __init__(self):
        # 캘리브레이션 파라미터
        self.baseline = 470.0  # mm
        self.focal_length = 1440  # pixels
        self.image_size = (1440, 300)
        
        # 실제 이미지에 맞게 조정된 골프공 검출 파라미터
        self.ball_params = {
            'min_radius': 5,      # 더 작은 반지름 허용
            'max_radius': 80,     # 더 큰 반지름 허용
            'param1': 30,         # 더 낮은 임계값
            'param2': 15,         # 더 낮은 임계값
            'min_dist': 20        # 더 가까운 거리 허용
        }
    
    def preprocess_image(self, image, use_gamma=True):
        """이미지 전처리"""
        if use_gamma:
            # Gamma 보정 적용
            gamma = 1.5
            gamma_corrected = np.power(image / 255.0, gamma) * 255.0
            gamma_corrected = np.uint8(gamma_corrected)
        else:
            gamma_corrected = image
        
        # 노이즈 제거
        denoised = cv2.medianBlur(gamma_corrected, 3)
        
        # 밝기 조정
        bright = cv2.convertScaleAbs(denoised, alpha=1.5, beta=30)
        
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        bright = clahe.apply(bright)
        
        return bright
    
    def detect_golf_ball_improved(self, image, use_gamma=True):
        """개선된 골프공 검출"""
        processed = self.preprocess_image(image, use_gamma)
        
        # 다양한 파라미터 세트로 시도
        param_sets = [
            {'param1': 30, 'param2': 15},
            {'param1': 40, 'param2': 20},
            {'param1': 50, 'param2': 25},
            {'param1': 20, 'param2': 10},
            {'param1': 60, 'param2': 30}
        ]
        
        all_circles = []
        
        for params in param_sets:
            circles = cv2.HoughCircles(
                processed,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=self.ball_params['min_dist'],
                param1=params['param1'],
                param2=params['param2'],
                minRadius=self.ball_params['min_radius'],
                maxRadius=self.ball_params['max_radius']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for circle in circles:
                    x, y, r = circle
                    # 이미지 경계 내에 있는지 확인
                    if (r < x < image.shape[1] - r and 
                        r < y < image.shape[0] - r):
                        all_circles.append(circle)
        
        if not all_circles:
            return None, 0.0
        
        # 가장 좋은 원 선택
        best_circle = None
        best_score = 0
        
        for circle in all_circles:
            x, y, r = circle
            # 점수 계산: 반지름 + 위치 보너스
            score = r * 2 + (image.shape[0] - y) * 0.1
            if score > best_score:
                best_score = score
                best_circle = circle
        
        # 신뢰도 계산
        confidence = min(1.0, best_score / 80.0)
        
        return best_circle, confidence
    
    def calculate_3d_coordinates(self, point1, point2):
        """3D 좌표 계산"""
        if point1 is None or point2 is None:
            return None
        
        # Y축 시차 계산
        disparity = abs(point1[1] - point2[1])
        
        if disparity < 1:  # 더 낮은 임계값
            return None
        
        # 3D 좌표 계산
        z = (self.focal_length * self.baseline) / disparity
        
        if z < 100 or z > 10000:  # 더 넓은 범위
            return None
        
        x = (point1[0] - self.image_size[0]/2) * z / self.focal_length
        y = (point1[1] - self.image_size[1]/2) * z / self.focal_length
        
        return np.array([x, y, z])
    
    def analyze_swing_sequence(self, shot_path, use_gamma=True):
        """스윙 시퀀스 전체 분석"""
        print(f"스윙 시퀀스 분석 시작: {shot_path}")
        print(f"Gamma 조정 사용: {'예' if use_gamma else '아니오'}")
        
        # 이미지 파일들 로드
        image_files = sorted(glob.glob(str(shot_path / "*.bmp")))
        
        if use_gamma:
            cam1_images = [f for f in image_files if "Gamma_1_" in os.path.basename(f)]
            cam2_images = [f for f in image_files if "Gamma_2_" in os.path.basename(f)]
        else:
            cam1_images = [f for f in image_files if "1_" in os.path.basename(f) and "Gamma" not in os.path.basename(f)]
            cam2_images = [f for f in image_files if "2_" in os.path.basename(f) and "Gamma" not in os.path.basename(f)]
        
        print(f"카메라1 이미지: {len(cam1_images)}개")
        print(f"카메라2 이미지: {len(cam2_images)}개")
        
        # 프레임별 분석 결과
        frame_results = []
        valid_frames = []
        total_processing_time = 0
        
        for i in range(min(len(cam1_images), len(cam2_images))):
            # 이미지 로드
            img1 = cv2.imread(cam1_images[i], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(cam2_images[i], cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                continue
            
            start_time = time.time()
            
            # 골프공 검출
            ball1, conf1 = self.detect_golf_ball_improved(img1, use_gamma)
            ball2, conf2 = self.detect_golf_ball_improved(img2, use_gamma)
            
            # 3D 좌표 계산
            ball_3d = None
            if ball1 is not None and ball2 is not None:
                ball_3d = self.calculate_3d_coordinates(ball1[:2], ball2[:2])
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # ms
            total_processing_time += processing_time
            
            frame_result = {
                'frame_num': i + 1,
                'processing_time_ms': processing_time,
                'ball1': ball1.tolist() if ball1 is not None else None,
                'ball2': ball2.tolist() if ball2 is not None else None,
                'confidence1': conf1,
                'confidence2': conf2,
                'ball_3d': ball_3d.tolist() if ball_3d is not None else None,
                'valid': ball1 is not None and ball2 is not None and ball_3d is not None
            }
            
            frame_results.append(frame_result)
            
            if frame_result['valid']:
                valid_frames.append(frame_result)
            
            print(f"  프레임 {i+1:2d}: {processing_time:6.1f}ms | 검출: {'✅' if frame_result['valid'] else '❌'} | 신뢰도: {conf1:.2f}/{conf2:.2f}")
        
        # 품질 분석
        quality_analysis = self.analyze_sequence_quality(frame_results)
        
        return {
            'total_frames': len(frame_results),
            'valid_frames': len(valid_frames),
            'frame_results': frame_results,
            'valid_frame_results': valid_frames,
            'quality_analysis': quality_analysis,
            'use_gamma': use_gamma,
            'total_processing_time': total_processing_time,
            'avg_processing_time': total_processing_time / len(frame_results) if frame_results else 0
        }
    
    def analyze_sequence_quality(self, frame_results):
        """스윙 시퀀스 품질 분석"""
        valid_frames = [f for f in frame_results if f['valid']]
        
        if len(valid_frames) < 2:
            return {
                'quality_score': 0.0,
                'avg_confidence': 0.0,
                'movement_detected': False,
                'sequence_type': 'insufficient_data',
                'recommendations': ['더 많은 유효한 프레임이 필요합니다']
            }
        
        # 움직임 분석
        movements = []
        for i in range(1, len(valid_frames)):
            prev_3d = np.array(valid_frames[i-1]['ball_3d'])
            curr_3d = np.array(valid_frames[i]['ball_3d'])
            movement = np.linalg.norm(curr_3d - prev_3d)
            movements.append(movement)
        
        avg_movement = np.mean(movements) if movements else 0
        max_movement = np.max(movements) if movements else 0
        
        # 신뢰도 분석
        avg_confidence = np.mean([f['confidence1'] + f['confidence2'] for f in valid_frames]) / 2
        
        # 시퀀스 타입 결정
        if avg_movement < 5.0:
            sequence_type = 'stationary'
            quality_score = 0.3
        elif max_movement > 50:
            sequence_type = 'active_swing'
            quality_score = 0.9
        else:
            sequence_type = 'slow_movement'
            quality_score = 0.6
        
        # 최종 품질 점수
        final_quality_score = (quality_score * 0.6 + avg_confidence * 0.4)
        
        recommendations = []
        if final_quality_score < 0.5:
            recommendations.append('더 나은 조명 조건이 필요합니다')
        if avg_movement < 5.0:
            recommendations.append('골프공이 움직이지 않는 프레임입니다')
        if len(valid_frames) < 5:
            recommendations.append('더 많은 유효한 프레임이 필요합니다')
        
        return {
            'quality_score': final_quality_score,
            'avg_movement': avg_movement,
            'max_movement': max_movement,
            'avg_confidence': avg_confidence,
            'movement_detected': avg_movement > 5.0,
            'sequence_type': sequence_type,
            'recommendations': recommendations
        }
    
    def compare_gamma_vs_normal(self, shot_path):
        """Gamma 조정 vs 일반 사진 품질 비교"""
        print(f"\n=== Gamma 조정 vs 일반 사진 품질 비교 ===")
        print(f"분석 대상: {shot_path}")
        
        # Gamma 조정 사진 분석
        print("\n1. Gamma 조정 사진 분석:")
        gamma_results = self.analyze_swing_sequence(shot_path, use_gamma=True)
        
        # 일반 사진 분석
        print("\n2. 일반 사진 분석:")
        normal_results = self.analyze_swing_sequence(shot_path, use_gamma=False)
        
        # 비교 결과
        print(f"\n=== 비교 결과 ===")
        print(f"{'항목':<20} {'Gamma 조정':<15} {'일반 사진':<15} {'우수':<10}")
        print("-" * 60)
        
        gamma_valid = gamma_results['valid_frames']
        normal_valid = normal_results['valid_frames']
        gamma_quality = gamma_results['quality_analysis']['quality_score']
        normal_quality = normal_results['quality_analysis']['quality_score']
        gamma_confidence = gamma_results['quality_analysis']['avg_confidence']
        normal_confidence = normal_results['quality_analysis']['avg_confidence']
        gamma_time = gamma_results['avg_processing_time']
        normal_time = normal_results['avg_processing_time']
        
        print(f"{'유효 프레임 수':<20} {gamma_valid:<15} {normal_valid:<15} {'Gamma' if gamma_valid > normal_valid else '일반' if normal_valid > gamma_valid else '동일'}")
        print(f"{'품질 점수':<20} {gamma_quality:<15.2f} {normal_quality:<15.2f} {'Gamma' if gamma_quality > normal_quality else '일반' if normal_quality > gamma_quality else '동일'}")
        print(f"{'평균 신뢰도':<20} {gamma_confidence:<15.2f} {normal_confidence:<15.2f} {'Gamma' if gamma_confidence > normal_confidence else '일반' if normal_confidence > gamma_confidence else '동일'}")
        print(f"{'평균 처리시간':<20} {gamma_time:<15.1f} {normal_time:<15.1f} {'일반' if normal_time < gamma_time else 'Gamma' if gamma_time < normal_time else '동일'}")
        
        # 권장사항
        if gamma_quality > normal_quality and gamma_valid >= normal_valid:
            recommendation = "Gamma 조정 사진 권장"
            reason = "더 높은 품질과 신뢰도"
        elif normal_quality > gamma_quality and normal_valid >= gamma_valid:
            recommendation = "일반 사진 권장"
            reason = "더 높은 품질과 신뢰도"
        else:
            recommendation = "상황에 따라 선택"
            reason = "비슷한 성능"
        
        print(f"\n권장사항: {recommendation}")
        print(f"이유: {reason}")
        
        return {
            'gamma_results': gamma_results,
            'normal_results': normal_results,
            'recommendation': recommendation,
            'reason': reason
        }

def main():
    analyzer = ImprovedSwingSequenceAnalyzer()
    
    print("개선된 스윙 시퀀스 분석기")
    print("=" * 50)
    
    # 테스트 샷 경로
    shot_path = Path("data/video_ballData_20250930/video_ballData_20250930/5Iron_0930/1")
    
    if not shot_path.exists():
        print(f"샷 경로를 찾을 수 없습니다: {shot_path}")
        return
    
    # Gamma vs 일반 사진 비교
    comparison_results = analyzer.compare_gamma_vs_normal(shot_path)
    
    print(f"\n✅ 분석 완료!")
    print(f"📊 Gamma 조정 사진: {comparison_results['gamma_results']['valid_frames']}개 유효 프레임")
    print(f"📊 일반 사진: {comparison_results['normal_results']['valid_frames']}개 유효 프레임")
    print(f"🏆 권장: {comparison_results['recommendation']}")
    
    # 처리 시간 확인
    gamma_time = comparison_results['gamma_results']['avg_processing_time']
    normal_time = comparison_results['normal_results']['avg_processing_time']
    
    print(f"\n⏱️  처리 시간:")
    print(f"Gamma 조정: {gamma_time:.1f}ms")
    print(f"일반 사진: {normal_time:.1f}ms")
    print(f"목표 150ms 달성: {'✅' if max(gamma_time, normal_time) <= 150 else '❌'}")

if __name__ == "__main__":
    main()
