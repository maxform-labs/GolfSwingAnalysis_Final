#!/usr/bin/env python3
"""
궁극의 정확한 스윙 시퀀스 분석기
- Gamma 사진 사용 (더 나은 검출 성능)
- 정교한 골프공 필터링
- 정확도 우선 (최대 150ms 처리 시간 허용)
- 스윙 시퀀스 품질 분석
"""

import cv2
import numpy as np
import time
import os
import glob
from pathlib import Path

class UltimateAccurateAnalyzer:
    def __init__(self):
        # 캘리브레이션 파라미터
        self.baseline = 470.0  # mm
        self.focal_length = 1440  # pixels
        self.image_size = (1440, 300)
        
        # Gamma 사진에 최적화된 골프공 검출 파라미터
        self.ball_params = {
            'min_radius': 8,
            'max_radius': 60,
            'param1': 30,
            'param2': 15,
            'min_dist': 20
        }
    
    def preprocess_image(self, image):
        """Gamma 사진에 최적화된 전처리"""
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def filter_golf_ball_candidates(self, circles, image):
        """골프공 후보 필터링"""
        if circles is None:
            return []
        
        filtered_circles = []
        
        for circle in circles:
            x, y, r = circle
            
            # 기본 경계 검사
            if not (r < x < image.shape[1] - r and r < y < image.shape[0] - r):
                continue
            
            # 골프공 크기 범위 검사
            if not (self.ball_params['min_radius'] <= r <= self.ball_params['max_radius']):
                continue
            
            # 이미지 하단 70% 영역에 있는지 확인 (골프공은 보통 아래쪽에 있음)
            if y < image.shape[0] * 0.3:
                continue
            
            # 원의 품질 검사
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # 원 내부의 평균 밝기 계산
            mean_intensity = cv2.mean(image, mask)[0]
            
            # 골프공은 상대적으로 밝아야 함
            if mean_intensity < 50:  # 너무 어두우면 제외
                continue
            
            # 원의 둘레 검사 (원형인지 확인)
            perimeter = 2 * np.pi * r
            area = np.pi * r * r
            
            # 원형도 검사 (둘레/면적 비율)
            circularity = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else float('inf')
            
            if circularity > 1.5:  # 너무 불규칙하면 제외
                continue
            
            filtered_circles.append(circle)
        
        return filtered_circles
    
    def detect_golf_ball_ultimate(self, image):
        """궁극의 골프공 검출"""
        processed = self.preprocess_image(image)
        
        # Hough Circles로 원 검출
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
        
        if circles is None:
            return None, 0.0
        
        circles = np.round(circles[0, :]).astype("int")
        
        # 골프공 후보 필터링
        filtered_circles = self.filter_golf_ball_candidates(circles, image)
        
        if not filtered_circles:
            return None, 0.0
        
        # 가장 좋은 골프공 선택
        best_circle = None
        best_score = 0
        
        for circle in filtered_circles:
            x, y, r = circle
            
            # 점수 계산
            # 1. 반지름 점수 (적당한 크기일수록 높은 점수)
            radius_score = 1.0 - abs(r - 25) / 25.0  # 25픽셀을 이상적인 크기로 가정
            radius_score = max(0, radius_score)
            
            # 2. 위치 점수 (하단에 가까울수록 높은 점수)
            position_score = (image.shape[0] - y) / image.shape[0]
            
            # 3. 밝기 점수
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            mean_intensity = cv2.mean(image, mask)[0]
            brightness_score = min(1.0, mean_intensity / 100.0)
            
            # 최종 점수
            total_score = radius_score * 0.4 + position_score * 0.3 + brightness_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_circle = circle
        
        if best_circle is None:
            return None, 0.0
        
        # 신뢰도 계산
        confidence = min(1.0, best_score)
        
        return best_circle, confidence
    
    def calculate_3d_coordinates(self, point1, point2):
        """3D 좌표 계산"""
        if point1 is None or point2 is None:
            return None
        
        # Y축 시차 계산
        disparity = abs(point1[1] - point2[1])
        
        if disparity < 1:
            return None
        
        # 3D 좌표 계산
        z = (self.focal_length * self.baseline) / disparity
        
        if z < 100 or z > 10000:
            return None
        
        x = (point1[0] - self.image_size[0]/2) * z / self.focal_length
        y = (point1[1] - self.image_size[1]/2) * z / self.focal_length
        
        return np.array([x, y, z])
    
    def analyze_swing_sequence_ultimate(self, shot_path):
        """궁극의 스윙 시퀀스 분석"""
        print(f"궁극의 스윙 시퀀스 분석: {shot_path}")
        print("Gamma 사진 사용 (정교한 골프공 필터링)")
        
        # Gamma 이미지 파일들 로드
        image_files = sorted(glob.glob(str(shot_path / "*.bmp")))
        cam1_images = [f for f in image_files if "Gamma_1_" in os.path.basename(f)]
        cam2_images = [f for f in image_files if "Gamma_2_" in os.path.basename(f)]
        
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
            ball1, conf1 = self.detect_golf_ball_ultimate(img1)
            ball2, conf2 = self.detect_golf_ball_ultimate(img2)
            
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
        quality_analysis = self.analyze_sequence_quality_ultimate(frame_results)
        
        return {
            'total_frames': len(frame_results),
            'valid_frames': len(valid_frames),
            'frame_results': frame_results,
            'valid_frame_results': valid_frames,
            'quality_analysis': quality_analysis,
            'total_processing_time': total_processing_time,
            'avg_processing_time': total_processing_time / len(frame_results) if frame_results else 0
        }
    
    def analyze_sequence_quality_ultimate(self, frame_results):
        """궁극의 스윙 시퀀스 품질 분석"""
        valid_frames = [f for f in frame_results if f['valid']]
        
        if len(valid_frames) < 2:
            return {
                'quality_score': 0.0,
                'avg_confidence': 0.0,
                'avg_movement': 0.0,
                'max_movement': 0.0,
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
            sequence_type = 'stationary_ball'
            quality_score = 0.2
            recommendations = ['골프공이 멈춰있는 상태입니다. 스윙 전 상태로 보입니다.']
        elif max_movement > 100:
            sequence_type = 'ball_in_flight'
            quality_score = 0.9
            recommendations = ['골프공이 날아가는 중입니다. 활성 스윙 시퀀스입니다.']
        elif avg_movement > 20:
            sequence_type = 'active_swing'
            quality_score = 0.8
            recommendations = ['활성 스윙 시퀀스입니다. 좋은 분석 데이터입니다.']
        else:
            sequence_type = 'slow_movement'
            quality_score = 0.5
            recommendations = ['골프공이 천천히 움직이고 있습니다.']
        
        # 최종 품질 점수
        final_quality_score = (quality_score * 0.6 + avg_confidence * 0.4)
        
        # 추가 권장사항
        if len(valid_frames) < 5:
            recommendations.append('더 많은 유효한 프레임이 필요합니다.')
        if avg_confidence < 0.5:
            recommendations.append('검출 신뢰도가 낮습니다. 조명 조건을 개선해보세요.')
        
        return {
            'quality_score': final_quality_score,
            'avg_movement': avg_movement,
            'max_movement': max_movement,
            'avg_confidence': avg_confidence,
            'movement_detected': avg_movement > 5.0,
            'sequence_type': sequence_type,
            'recommendations': recommendations
        }

def main():
    analyzer = UltimateAccurateAnalyzer()
    
    print("궁극의 정확한 스윙 시퀀스 분석기")
    print("=" * 50)
    
    # 테스트 샷 경로
    shot_path = Path("data/video_ballData_20250930/video_ballData_20250930/5Iron_0930/1")
    
    if not shot_path.exists():
        print(f"샷 경로를 찾을 수 없습니다: {shot_path}")
        return
    
    # 단일 샷 분석
    print("=== 단일 샷 분석 ===")
    result = analyzer.analyze_swing_sequence_ultimate(shot_path)
    
    print(f"\n=== 분석 결과 요약 ===")
    print(f"총 프레임: {result['total_frames']}")
    print(f"유효 프레임: {result['valid_frames']}")
    print(f"검출 성공률: {result['valid_frames']/result['total_frames']*100:.1f}%")
    print(f"평균 처리 시간: {result['avg_processing_time']:.1f}ms")
    print(f"목표 150ms 달성: {'✅' if result['avg_processing_time'] <= 150 else '❌'}")
    
    quality = result['quality_analysis']
    print(f"\n=== 품질 분석 ===")
    print(f"품질 점수: {quality['quality_score']:.2f}")
    print(f"평균 움직임: {quality['avg_movement']:.1f}mm")
    print(f"최대 움직임: {quality['max_movement']:.1f}mm")
    print(f"평균 신뢰도: {quality['avg_confidence']:.2f}")
    print(f"시퀀스 타입: {quality['sequence_type']}")
    print(f"권장사항: {', '.join(quality['recommendations'])}")
    
    # 결론
    print(f"\n=== 최종 결론 ===")
    if result['valid_frames'] > 0:
        print("✅ 골프공 검출 성공!")
        if result['avg_processing_time'] <= 150:
            print("✅ 처리 시간 목표 달성!")
        else:
            print("⚠️  처리 시간 개선 필요")
    else:
        print("❌ 골프공 검출 실패 - 추가 최적화 필요")

if __name__ == "__main__":
    main()
