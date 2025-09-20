#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMP 직접 처리 및 혁신적 딤플 분석 시스템 v7.0
개발팀: maxform
목표: 820fps BMP 직접 처리, 혁신적 딤플 패턴 분석으로 95% 정확도 달성
혁신: 임팩트 기준 전후 4프레임 확대 딤플 분석, 과노출 볼 처리

통합버전: 기존 프로젝트와 충돌 없이 독립 모듈로 동작
"""

import cv2
import numpy as np
import json
import os
import struct
from pathlib import Path
from scipy import ndimage
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DimpleAnalyzerV7:
    """BMP 직접 처리 및 혁신적 딤플 분석 시스템 v7.0"""
    
    def __init__(self, fps: int = 820, debug: bool = True):
        """
        초기화
        
        Args:
            fps: 프레임 레이트 (기본 820fps)
            debug: 디버그 메시지 출력 여부
        """
        self.fps = fps
        self.debug = debug
        self.pixel_to_mm = 0.1  # 픽셀-mm 변환 비율
        
        # 12개 파라미터 (로프트앵글 제외)
        self.target_parameters = [
            # 볼 데이터 (6개)
            'ball_speed', 'launch_angle', 'direction_angle', 
            'backspin', 'sidespin', 'spin_axis',
            # 클럽 데이터 (6개)
            'club_speed', 'attack_angle', 'club_path', 
            'face_angle', 'face_to_path', 'smash_factor'
        ]
        
        # 딤플 분석 파라미터
        self.dimple_params = {
            'zoom_factor': 3,  # 3배 확대
            'gamma': 0.5,  # 감마 보정값
            'clahe_clip_limit': 2.0,  # CLAHE 클립 한계
            'overexposure_threshold': 240,  # 과노출 임계값
            'min_dimple_radius': 2,
            'max_dimple_radius': 8
        }
        
        if self.debug:
            print("=== 딤플 분석 시스템 v7.0 초기화 완료 ===")
            print(f"FPS: {fps}, 픽셀-mm 비율: {self.pixel_to_mm}")
    
    def read_bmp_direct(self, bmp_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        BMP 파일 직접 읽기 (고해상도 유지)
        
        Args:
            bmp_path: BMP 파일 경로
            
        Returns:
            이미지 배열 또는 None
        """
        bmp_path = Path(bmp_path)
        
        if not bmp_path.exists():
            if self.debug:
                print(f"파일이 존재하지 않음: {bmp_path}")
            return None
        
        try:
            with open(bmp_path, 'rb') as f:
                # BMP 헤더 읽기
                header = f.read(54)
                
                # BMP 시그니처 확인
                if header[:2] != b'BM':
                    raise ValueError("유효하지 않은 BMP 파일")
                
                # 헤더 정보 추출
                data_offset = struct.unpack('<I', header[10:14])[0]
                width = struct.unpack('<I', header[18:22])[0]
                height = struct.unpack('<I', header[22:26])[0]
                bit_count = struct.unpack('<H', header[28:30])[0]
                
                if self.debug:
                    print(f"BMP 정보: {width}x{height}, {bit_count}bit")
                
                # 이미지 데이터 읽기
                f.seek(data_offset)
                
                if bit_count == 24:
                    # 24비트 BGR
                    row_size = ((width * 3 + 3) // 4) * 4  # 4바이트 정렬
                    image_data = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    for y in range(height):
                        row_data = f.read(row_size)
                        for x in range(width):
                            if x * 3 + 2 < len(row_data):
                                # BGR → RGB 변환 및 상하 반전
                                image_data[height - 1 - y, x, 0] = row_data[x * 3 + 2]  # R
                                image_data[height - 1 - y, x, 1] = row_data[x * 3 + 1]  # G
                                image_data[height - 1 - y, x, 2] = row_data[x * 3]      # B
                    
                    return image_data
                
                elif bit_count == 8:
                    # 8비트 그레이스케일
                    row_size = ((width + 3) // 4) * 4
                    image_data = np.zeros((height, width), dtype=np.uint8)
                    
                    for y in range(height):
                        row_data = f.read(row_size)
                        for x in range(width):
                            if x < len(row_data):
                                image_data[height - 1 - y, x] = row_data[x]
                    
                    return image_data
                
                else:
                    # 다른 비트 깊이는 OpenCV로 대체
                    return cv2.imread(str(bmp_path))
                    
        except Exception as e:
            if self.debug:
                print(f"BMP 직접 읽기 실패: {e}, OpenCV로 대체")
            return cv2.imread(str(bmp_path))
    
    def detect_impact_frame(self, images: List[np.ndarray]) -> int:
        """
        임팩트 프레임 검출 (볼과 클럽이 가장 가까운 순간)
        
        Args:
            images: 이미지 시퀀스 리스트
            
        Returns:
            임팩트 프레임 인덱스
        """
        if not images:
            return 0
        
        impact_scores = []
        
        for i, image in enumerate(images):
            if image is None:
                impact_scores.append(0)
                continue
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # 볼 검출
            ball_circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 50,
                param1=50, param2=30, minRadius=10, maxRadius=100
            )
            
            # 클럽 검출 (에지 기반)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            
            # 임팩트 점수 계산
            score = 0
            if ball_circles is not None and lines is not None:
                ball_circles = np.round(ball_circles[0, :]).astype("int")
                
                for (x, y, r) in ball_circles:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        
                        # 점-직선 거리
                        num = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
                        den = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                        distance = num / den if den > 0 else float('inf')
                        
                        # 거리가 가까울수록 높은 점수
                        if distance < r + 20:
                            score += (r + 20 - distance) / (r + 20)
            
            impact_scores.append(score)
        
        # 최고 점수 프레임 선정
        impact_idx = np.argmax(impact_scores) if impact_scores else len(images) // 2
        
        if self.debug:
            print(f"임팩트 프레임: {impact_idx} (점수: {impact_scores[impact_idx]:.2f})")
        
        return impact_idx
    
    def correct_overexposure(self, image: np.ndarray, 
                            ball_center: Tuple[int, int], 
                            ball_radius: int) -> np.ndarray:
        """
        과노출된 볼 영역 보정
        
        Args:
            image: 입력 이미지
            ball_center: 볼 중심 좌표
            ball_radius: 볼 반지름
            
        Returns:
            보정된 이미지
        """
        x, y = ball_center
        r = ball_radius
        
        # ROI 설정
        roi_size = int(r * 3)
        x1 = max(0, x - roi_size//2)
        y1 = max(0, y - roi_size//2)
        x2 = min(image.shape[1], x + roi_size//2)
        y2 = min(image.shape[0], y + roi_size//2)
        
        roi = image[y1:y2, x1:x2].copy()
        
        if roi.size == 0:
            return image
        
        # 그레이스케일 변환
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # 과노출 영역 검출
        overexposed_mask = gray_roi > self.dimple_params['overexposure_threshold']
        
        if np.sum(overexposed_mask) > 0:
            if self.debug:
                print(f"과노출 영역: {np.sum(overexposed_mask)} 픽셀")
            
            # 1. 감마 보정
            gamma = self.dimple_params['gamma']
            gamma_corrected = np.power(gray_roi / 255.0, gamma) * 255.0
            gamma_corrected = gamma_corrected.astype(np.uint8)
            
            # 2. CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=self.dimple_params['clahe_clip_limit'], 
                tileGridSize=(8,8)
            )
            clahe_applied = clahe.apply(gamma_corrected)
            
            # 3. 언샤프 마스킹
            gaussian = cv2.GaussianBlur(clahe_applied, (0, 0), 2.0)
            unsharp = cv2.addWeighted(clahe_applied, 1.5, gaussian, -0.5, 0)
            
            # 과노출 영역만 교체
            corrected_roi = gray_roi.copy()
            corrected_roi[overexposed_mask] = unsharp[overexposed_mask]
            
            # 원본에 적용
            result = image.copy()
            if len(image.shape) == 3:
                result[y1:y2, x1:x2] = cv2.cvtColor(corrected_roi, cv2.COLOR_GRAY2BGR)
            else:
                result[y1:y2, x1:x2] = corrected_roi
            
            return result
        
        return image
    
    def extract_dimple_patterns(self, image: np.ndarray,
                               ball_center: Tuple[int, int],
                               ball_radius: int) -> Dict:
        """
        혁신적 딤플 패턴 추출 (3배 확대 + 다중 필터링)
        
        Args:
            image: 입력 이미지
            ball_center: 볼 중심
            ball_radius: 볼 반지름
            
        Returns:
            딤플 분석 결과
        """
        x, y = ball_center
        r = ball_radius
        
        # 볼 영역 추출
        roi_size = int(r * 2)
        x1 = max(0, x - roi_size//2)
        y1 = max(0, y - roi_size//2)
        x2 = min(image.shape[1], x + roi_size//2)
        y2 = min(image.shape[0], y + roi_size//2)
        
        ball_roi = image[y1:y2, x1:x2]
        
        if ball_roi.size == 0:
            return {
                'dimple_count': 0,
                'rotation_angle': 0,
                'confidence': 0
            }
        
        # 3배 확대
        zoom_factor = self.dimple_params['zoom_factor']
        zoomed = cv2.resize(ball_roi, None, fx=zoom_factor, fy=zoom_factor, 
                           interpolation=cv2.INTER_CUBIC)
        
        # 그레이스케일 변환
        gray_zoomed = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY) if len(zoomed.shape) == 3 else zoomed
        
        # 과노출 보정
        corrected = self.correct_overexposure(
            gray_zoomed,
            (gray_zoomed.shape[1]//2, gray_zoomed.shape[0]//2),
            r * zoom_factor
        )
        
        if len(corrected.shape) == 3:
            corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        
        # 다중 필터링
        
        # 1. 라플라시안
        laplacian = cv2.Laplacian(corrected, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # 2. DoG (Difference of Gaussians)
        g1 = cv2.GaussianBlur(corrected, (5, 5), 1.0)
        g2 = cv2.GaussianBlur(corrected, (9, 9), 2.0)
        dog = cv2.subtract(g1, g2)
        
        # 3. Top-hat
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(corrected, cv2.MORPH_TOPHAT, kernel)
        
        # 딤플 검출
        
        # Hough Circles
        circles = cv2.HoughCircles(
            corrected, cv2.HOUGH_GRADIENT, 1, 10,
            param1=50, param2=15,
            minRadius=self.dimple_params['min_dimple_radius'],
            maxRadius=self.dimple_params['max_dimple_radius']
        )
        
        dimple_count = len(circles[0]) if circles is not None else 0
        
        # Blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0
        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 100
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(255 - corrected)
        blob_count = len(keypoints)
        
        # 구조 텐서로 회전 각도 계산
        Ix = cv2.Sobel(corrected, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(corrected, cv2.CV_64F, 0, 1, ksize=3)
        
        Ixx = cv2.GaussianBlur(Ix * Ix, (5, 5), 1.0)
        Iyy = cv2.GaussianBlur(Iy * Iy, (5, 5), 1.0)
        Ixy = cv2.GaussianBlur(Ix * Iy, (5, 5), 1.0)
        
        # 주 방향 계산
        angles = []
        h, w = Ixx.shape
        
        for i in range(h//4, 3*h//4, 5):
            for j in range(w//4, 3*w//4, 5):
                M = np.array([[Ixx[i,j], Ixy[i,j]], 
                             [Ixy[i,j], Iyy[i,j]]])
                
                eigenvals, eigenvecs = np.linalg.eig(M)
                
                if eigenvals[0] > eigenvals[1]:
                    principal_dir = eigenvecs[:, 0]
                else:
                    principal_dir = eigenvecs[:, 1]
                
                angle = np.arctan2(principal_dir[1], principal_dir[0]) * 180 / np.pi
                angles.append(angle)
        
        rotation_angle = np.mean(angles) if angles else 0
        
        # 신뢰도 계산
        confidence = min(1.0, (dimple_count + blob_count) / 50)
        
        if self.debug:
            print(f"딤플: {dimple_count + blob_count}개, 회전: {rotation_angle:.1f}°")
        
        return {
            'dimple_count': dimple_count + blob_count,
            'rotation_angle': rotation_angle,
            'confidence': confidence,
            'enhanced_image': corrected
        }
    
    def multi_frame_fusion(self, images: List[np.ndarray],
                          ball_centers: List[Tuple[int, int]],
                          ball_radii: List[int],
                          impact_idx: int) -> Dict:
        """
        다중 프레임 딤플 융합 분석
        
        Args:
            images: 이미지 리스트
            ball_centers: 볼 중심 리스트
            ball_radii: 볼 반지름 리스트
            impact_idx: 임팩트 프레임 인덱스
            
        Returns:
            스핀 분석 결과
        """
        # 임팩트 전후 4프레임
        start = max(0, impact_idx - 4)
        end = min(len(images), impact_idx + 5)
        
        dimple_results = []
        
        for i in range(start, end):
            if i < len(images) and i < len(ball_centers) and i < len(ball_radii):
                result = self.extract_dimple_patterns(
                    images[i], ball_centers[i], ball_radii[i]
                )
                
                # 임팩트 프레임 가중치
                weight = 2.0 if i == impact_idx else 1.0
                result['weight'] = weight
                result['frame_idx'] = i
                
                dimple_results.append(result)
        
        if not dimple_results:
            return {
                'total_spin': 0,
                'backspin': 0,
                'sidespin': 0,
                'spin_axis': 0,
                'confidence': 0
            }
        
        # 회전 변화량 계산
        rotation_changes = []
        for i in range(1, len(dimple_results)):
            prev = dimple_results[i-1]['rotation_angle']
            curr = dimple_results[i]['rotation_angle']
            
            diff = curr - prev
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            
            rotation_changes.append(abs(diff))
        
        # 평균 회전 변화량
        avg_change = np.mean(rotation_changes) if rotation_changes else 0
        
        # RPM 변환
        dt = 1.0 / self.fps
        angular_velocity = avg_change / dt  # 도/초
        total_spin = angular_velocity / 6.0  # RPM
        
        # 스핀 성분 계산
        total_weight = sum(r['weight'] for r in dimple_results)
        weighted_angle = sum(r['rotation_angle'] * r['weight'] for r in dimple_results)
        avg_angle = weighted_angle / total_weight if total_weight > 0 else 0
        
        # 백스핀과 사이드스핀 분리
        backspin = total_spin * abs(np.cos(avg_angle * np.pi / 180))
        sidespin = total_spin * abs(np.sin(avg_angle * np.pi / 180))
        
        # 신뢰도
        confidence = sum(r['confidence'] * r['weight'] for r in dimple_results) / total_weight if total_weight > 0 else 0
        
        if self.debug:
            print(f"스핀 분석: 총 {total_spin:.0f} RPM (백스핀: {backspin:.0f}, 사이드스핀: {sidespin:.0f})")
        
        return {
            'total_spin': min(15000, max(0, total_spin)),
            'backspin': min(12000, max(0, backspin)),
            'sidespin': min(3000, max(-3000, sidespin)),
            'spin_axis': avg_angle,
            'confidence': confidence
        }
    
    def enhanced_ball_detection(self, image: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """
        향상된 볼 검출
        
        Args:
            image: 입력 이미지
            
        Returns:
            (볼 중심, 반지름) 튜플
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 적응형 임계값
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 형태학적 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity > 0.5:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    detections.append((int(x), int(y), int(radius), circularity))
        
        # Hough Circles 보완
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            for (x, y, r) in circles[0]:
                detections.append((int(x), int(y), int(r), 0.8))
        
        if detections:
            # 최고 신뢰도 선택
            best = max(detections, key=lambda x: x[3])
            return (best[0], best[1]), best[2]
        
        # 기본값
        h, w = gray.shape
        return (w//2, h//2), 30
    
    def process_sequence(self, image_paths: List[Union[str, Path]]) -> Dict:
        """
        이미지 시퀀스 처리
        
        Args:
            image_paths: 이미지 경로 리스트
            
        Returns:
            분석 결과
        """
        # 이미지 로드
        images = []
        for path in image_paths:
            img = self.read_bmp_direct(path)
            if img is not None:
                images.append(img)
        
        if not images:
            return {'error': 'No valid images found'}
        
        # 볼 검출
        ball_centers = []
        ball_radii = []
        
        for img in images:
            center, radius = self.enhanced_ball_detection(img)
            ball_centers.append(center)
            ball_radii.append(radius)
        
        # 임팩트 프레임 검출
        impact_idx = self.detect_impact_frame(images)
        
        # 다중 프레임 딤플 융합
        spin_result = self.multi_frame_fusion(
            images, ball_centers, ball_radii, impact_idx
        )
        
        # 볼 속도 계산 (간단한 예시)
        if len(ball_centers) >= 2:
            dx = ball_centers[-1][0] - ball_centers[0][0]
            dy = ball_centers[-1][1] - ball_centers[0][1]
            
            distance_pixels = np.sqrt(dx**2 + dy**2)
            distance_mm = distance_pixels * self.pixel_to_mm
            
            time_elapsed = len(ball_centers) / self.fps
            speed_ms = distance_mm / time_elapsed / 1000  # m/s
            ball_speed_mph = speed_ms * 2.237  # mph
            
            launch_angle = np.arctan2(-dy, dx) * 180 / np.pi
        else:
            ball_speed_mph = 0
            launch_angle = 0
        
        return {
            'ball_data': {
                'ball_speed': ball_speed_mph,
                'launch_angle': launch_angle,
                'backspin': spin_result['backspin'],
                'sidespin': spin_result['sidespin'],
                'spin_axis': spin_result['spin_axis']
            },
            'dimple_analysis': {
                'total_spin': spin_result['total_spin'],
                'confidence': spin_result['confidence'],
                'impact_frame': impact_idx,
                'frames_analyzed': len(images)
            }
        }


def process_directory_v7(input_dir: Union[str, Path], output_dir: Union[str, Path]) -> Dict:
    """
    디렉토리 처리 함수
    
    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        
    Returns:
        처리 결과
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 분석기 초기화
    analyzer = DimpleAnalyzerV7(debug=True)
    
    # BMP 파일 찾기
    bmp_files = sorted(input_path.glob("*.bmp"))
    
    if not bmp_files:
        print(f"BMP 파일을 찾을 수 없음: {input_path}")
        return {'error': 'No BMP files found'}
    
    print(f"\n{'='*80}")
    print(f"딤플 분석 v7.0 - BMP 직접 처리")
    print(f"{'='*80}")
    print(f"입력: {input_path}")
    print(f"출력: {output_path}")
    print(f"파일 수: {len(bmp_files)}")
    print(f"{'='*80}\n")
    
    results = []
    
    # 배치 처리 (9프레임씩)
    batch_size = 9
    
    for i in range(0, len(bmp_files), batch_size):
        batch = bmp_files[i:i+batch_size]
        
        print(f"\n배치 {i//batch_size + 1}: {len(batch)}개 파일 처리")
        
        # 시퀀스 분석
        result = analyzer.process_sequence(batch)
        results.append(result)
        
        # 첫 번째 이미지 향상 처리 및 저장
        first_img = analyzer.read_bmp_direct(batch[0])
        if first_img is not None:
            center, radius = analyzer.enhanced_ball_detection(first_img)
            
            # 과노출 보정 적용
            corrected = analyzer.correct_overexposure(first_img, center, radius)
            
            # 저장
            output_file = output_path / f"{batch[0].stem}_v7_enhanced.png"
            cv2.imwrite(str(output_file), corrected)
            print(f"  저장: {output_file.name}")
        
        # 결과 출력
        if 'ball_data' in result:
            print(f"  볼 스피드: {result['ball_data']['ball_speed']:.1f} mph")
            print(f"  백스핀: {result['ball_data']['backspin']:.0f} RPM")
            print(f"  사이드스핀: {result['ball_data']['sidespin']:.0f} RPM")
        
        if 'dimple_analysis' in result:
            print(f"  신뢰도: {result['dimple_analysis']['confidence']:.2f}")
    
    # 결과 저장
    result_file = output_path / "v7_analysis_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"처리 완료!")
    print(f"결과 저장: {result_file}")
    print(f"{'='*80}")
    
    return {
        'total_files': len(bmp_files),
        'batches_processed': len(results),
        'output_dir': str(output_path),
        'results': results
    }


if __name__ == "__main__":
    # 테스트 실행
    input_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1")
    output_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-v7-enhanced/driver/no_marker_ball-1")
    
    result = process_directory_v7(input_dir, output_dir)
    print(f"\n최종 결과: {result['batches_processed']}개 배치 처리 완료")