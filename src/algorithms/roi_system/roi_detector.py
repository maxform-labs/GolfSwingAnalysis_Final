#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive ROI Ball Detection System v2.0
적응형 ROI 기반 다단계 골프공 검출 시스템
- 전체화면 → 임팩트존 확대 → 전체화면 추적 전략
"""

import cv2
import numpy as np
import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum

class DetectionPhase(Enum):
    """검출 단계"""
    FULL_SCREEN = "full_screen"        # 전체화면 스캔
    IMPACT_ZONE = "impact_zone"        # 임팩트존 확대 검출
    TRACKING = "tracking"              # 볼 추적 모드
    FLIGHT_TRACKING = "flight_tracking" # 비행 추적 모드

class DetectionMethod(Enum):
    """검출 방법"""
    HOUGH_CIRCLES = "hough_circles"
    HOUGH_GAMMA = "hough_gamma"
    CONTOUR_BRIGHT = "contour_bright" 
    TEMPLATE_MATCH = "template_match"
    MOTION_DETECT = "motion_detect"

@dataclass
class ROIRegion:
    """ROI 영역 정의"""
    x: int
    y: int
    width: int
    height: int
    scale: float = 1.0
    method: DetectionMethod = DetectionMethod.HOUGH_CIRCLES

@dataclass
class AdaptiveBallResult:
    """적응형 볼 검출 결과"""
    center_x: float
    center_y: float
    radius: float
    confidence: float
    detection_method: DetectionMethod
    motion_state: str
    roi_region: ROIRegion
    phase: DetectionPhase

class AdaptiveROIDetector:
    """적응형 ROI 볼 검출기"""
    
    def __init__(self):
        self.previous_positions = []
        self.current_phase = DetectionPhase.FULL_SCREEN
        self.impact_detected = False
        self.ball_template = None
        self.background_model = None
        
        # 이미지 해상도 기반 ROI 설정 (1440x300)
        self.img_width = 1440
        self.img_height = 300
        
        # 단계별 ROI 영역 정의
        self.roi_regions = {
            DetectionPhase.FULL_SCREEN: ROIRegion(0, 0, 1440, 300, 1.0),
            DetectionPhase.IMPACT_ZONE: ROIRegion(600, 100, 300, 150, 2.0),  # 중앙 임팩트 존
            DetectionPhase.TRACKING: ROIRegion(0, 0, 1440, 300, 1.0),
            DetectionPhase.FLIGHT_TRACKING: ROIRegion(0, 0, 1440, 300, 1.0)
        }
        
        # 검출 파라미터 세트
        self.detection_params = {
            DetectionMethod.HOUGH_CIRCLES: {
                'dp': 1, 'minDist': 20, 'param1': 30, 'param2': 15,
                'minRadius': 3, 'maxRadius': 40
            },
            DetectionMethod.HOUGH_GAMMA: {
                'dp': 1, 'minDist': 15, 'param1': 25, 'param2': 12,
                'minRadius': 3, 'maxRadius': 35, 'gamma': 2.0
            },
            DetectionMethod.CONTOUR_BRIGHT: {
                'threshold': 120, 'min_area': 10, 'max_area': 800,
                'circularity_threshold': 0.3
            },
            DetectionMethod.MOTION_DETECT: {
                'threshold': 15, 'min_area': 20, 'max_area': 500
            }
        }
    
    def detect_ball_adaptive(self, img: np.ndarray, frame_number: int) -> Optional[AdaptiveBallResult]:
        """적응형 볼 검출 메인 함수"""
        
        if img is None or img.size == 0:
            return None
        
        # 현재 단계 결정
        self._update_detection_phase(frame_number)
        
        # ROI 영역 선택
        roi_region = self.roi_regions[self.current_phase]
        
        # ROI 추출 및 전처리
        roi_img = self._extract_roi(img, roi_region)
        if roi_img is None:
            return None
        
        # 단계별 검출 방법 적용
        detection_methods = self._get_detection_methods_for_phase()
        
        best_result = None
        best_score = 0.0
        
        for method in detection_methods:
            result = self._detect_with_method(roi_img, method, roi_region, frame_number)
            if result and result.confidence > best_score:
                best_score = result.confidence
                best_result = result
        
        # 결과 후처리
        if best_result:
            # ROI 좌표를 전체 이미지 좌표로 변환
            best_result.center_x += roi_region.x
            best_result.center_y += roi_region.y
            best_result.phase = self.current_phase
            
            # 위치 추적 업데이트
            self._update_tracking(best_result, frame_number)
        
        return best_result
    
    def _update_detection_phase(self, frame_number: int):
        """검출 단계 업데이트"""
        
        # 프레임 기반 단계 전환 로직
        if frame_number <= 5:
            # 초기: 전체화면 스캔
            self.current_phase = DetectionPhase.FULL_SCREEN
            
        elif frame_number <= 12:
            # 중기: 임팩트존 집중 (프레임 6-12)
            self.current_phase = DetectionPhase.IMPACT_ZONE
            
        elif frame_number <= 18:
            # 후기: 볼 추적 모드 (프레임 13-18)
            self.current_phase = DetectionPhase.TRACKING
            
        else:
            # 최종: 비행 추적 (프레임 19-23)
            self.current_phase = DetectionPhase.FLIGHT_TRACKING
        
        # 이전 검출 결과 기반 적응적 조정
        if len(self.previous_positions) >= 3:
            # 움직임이 감지되면 추적 모드로 전환
            recent_movement = self._calculate_recent_movement()
            if recent_movement > 20 and self.current_phase == DetectionPhase.FULL_SCREEN:
                self.current_phase = DetectionPhase.TRACKING
    
    def _get_detection_methods_for_phase(self) -> List[DetectionMethod]:
        """단계별 검출 방법 선택"""
        
        method_priority = {
            DetectionPhase.FULL_SCREEN: [
                DetectionMethod.HOUGH_GAMMA,
                DetectionMethod.CONTOUR_BRIGHT,
                DetectionMethod.HOUGH_CIRCLES
            ],
            DetectionPhase.IMPACT_ZONE: [
                DetectionMethod.HOUGH_GAMMA,
                DetectionMethod.MOTION_DETECT,
                DetectionMethod.CONTOUR_BRIGHT
            ],
            DetectionPhase.TRACKING: [
                DetectionMethod.TEMPLATE_MATCH,
                DetectionMethod.MOTION_DETECT,
                DetectionMethod.HOUGH_CIRCLES
            ],
            DetectionPhase.FLIGHT_TRACKING: [
                DetectionMethod.MOTION_DETECT,
                DetectionMethod.HOUGH_CIRCLES,
                DetectionMethod.CONTOUR_BRIGHT
            ]
        }
        
        return method_priority.get(self.current_phase, [DetectionMethod.HOUGH_CIRCLES])
    
    def _extract_roi(self, img: np.ndarray, roi_region: ROIRegion) -> Optional[np.ndarray]:
        """ROI 영역 추출 및 전처리"""
        
        # ROI 경계 확인
        x1 = max(0, roi_region.x)
        y1 = max(0, roi_region.y)
        x2 = min(img.shape[1], roi_region.x + roi_region.width)
        y2 = min(img.shape[0], roi_region.y + roi_region.height)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # ROI 추출
        roi = img[y1:y2, x1:x2].copy()
        
        # 스케일링 (확대)
        if roi_region.scale > 1.0:
            new_width = int(roi.shape[1] * roi_region.scale)
            new_height = int(roi.shape[0] * roi_region.scale)
            roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return roi
    
    def _detect_with_method(self, roi_img: np.ndarray, method: DetectionMethod, 
                           roi_region: ROIRegion, frame_number: int) -> Optional[AdaptiveBallResult]:
        """특정 방법으로 볼 검출"""
        
        if method == DetectionMethod.HOUGH_GAMMA:
            return self._detect_hough_gamma(roi_img, method, roi_region, frame_number)
            
        elif method == DetectionMethod.HOUGH_CIRCLES:
            return self._detect_hough_circles(roi_img, method, roi_region, frame_number)
            
        elif method == DetectionMethod.CONTOUR_BRIGHT:
            return self._detect_contour_bright(roi_img, method, roi_region, frame_number)
            
        elif method == DetectionMethod.MOTION_DETECT:
            return self._detect_motion(roi_img, method, roi_region, frame_number)
            
        elif method == DetectionMethod.TEMPLATE_MATCH:
            return self._detect_template_match(roi_img, method, roi_region, frame_number)
        
        return None
    
    def _detect_hough_gamma(self, roi_img: np.ndarray, method: DetectionMethod,
                           roi_region: ROIRegion, frame_number: int) -> Optional[AdaptiveBallResult]:
        """감마 보정된 Hough Circle 검출"""
        
        try:
            # 그레이스케일 변환
            if len(roi_img.shape) == 3:
                gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi_img.copy()
            
            # 감마 보정으로 어두운 이미지 개선
            params = self.detection_params[method]
            gamma = params.get('gamma', 2.0)
            
            # 룩업 테이블을 이용한 감마 보정
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(gray, table)
            
            # 가우시안 블러
            blurred = cv2.GaussianBlur(gamma_corrected, (5, 5), 1.0)
            
            # Hough Circle 검출
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=params['dp'],
                minDist=params['minDist'],
                param1=params['param1'],
                param2=params['param2'],
                minRadius=params['minRadius'],
                maxRadius=params['maxRadius']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # 가장 좋은 원 선택
                best_circle = self._select_best_circle(circles, gamma_corrected, roi_region.scale)
                
                if best_circle:
                    x, y, r, confidence = best_circle
                    return AdaptiveBallResult(
                        center_x=float(x / roi_region.scale),
                        center_y=float(y / roi_region.scale),
                        radius=float(r / roi_region.scale),
                        confidence=confidence,
                        detection_method=method,
                        motion_state=self._determine_motion_state(x, y, frame_number),
                        roi_region=roi_region,
                        phase=self.current_phase
                    )
            
            return None
            
        except Exception as e:
            print(f"Hough gamma detection error: {e}")
            return None
    
    def _detect_hough_circles(self, roi_img: np.ndarray, method: DetectionMethod,
                             roi_region: ROIRegion, frame_number: int) -> Optional[AdaptiveBallResult]:
        """일반 Hough Circle 검출"""
        
        try:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img.copy()
            
            # 히스토그램 균등화
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # 가우시안 블러
            blurred = cv2.GaussianBlur(enhanced, (7, 7), 1.5)
            
            params = self.detection_params[method]
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=params['dp'],
                minDist=params['minDist'],
                param1=params['param1'],
                param2=params['param2'],
                minRadius=params['minRadius'],
                maxRadius=params['maxRadius']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                best_circle = self._select_best_circle(circles, enhanced, roi_region.scale)
                
                if best_circle:
                    x, y, r, confidence = best_circle
                    return AdaptiveBallResult(
                        center_x=float(x / roi_region.scale),
                        center_y=float(y / roi_region.scale),
                        radius=float(r / roi_region.scale),
                        confidence=confidence * 0.8,  # 기본 Hough는 약간 낮은 가중치
                        detection_method=method,
                        motion_state=self._determine_motion_state(x, y, frame_number),
                        roi_region=roi_region,
                        phase=self.current_phase
                    )
            
            return None
            
        except Exception as e:
            print(f"Hough circles detection error: {e}")
            return None
    
    def _detect_contour_bright(self, roi_img: np.ndarray, method: DetectionMethod,
                              roi_region: ROIRegion, frame_number: int) -> Optional[AdaptiveBallResult]:
        """밝은 영역 컨투어 기반 검출"""
        
        try:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img.copy()
            
            params = self.detection_params[method]
            
            # 적응적 임계값
            threshold_val = max(params['threshold'], int(gray.mean() + gray.std()))
            _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
            
            # 모폴로지 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # 컨투어 검출
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_contour = None
            best_score = 0.0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if params['min_area'] <= area <= params['max_area']:
                    # 외접원 계산
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # 원형도 계산
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        
                        if circularity >= params['circularity_threshold']:
                            # 점수 계산 (원형도 + 크기 + 밝기)
                            roi_mask = np.zeros(gray.shape, dtype=np.uint8)
                            cv2.circle(roi_mask, (int(x), int(y)), int(radius), 255, -1)
                            mean_brightness = cv2.mean(gray, mask=roi_mask)[0]
                            
                            score = circularity * 0.4 + (area / params['max_area']) * 0.3 + (mean_brightness / 255.0) * 0.3
                            
                            if score > best_score:
                                best_score = score
                                best_contour = (x, y, radius, score)
            
            if best_contour:
                x, y, r, confidence = best_contour
                return AdaptiveBallResult(
                    center_x=float(x / roi_region.scale),
                    center_y=float(y / roi_region.scale),
                    radius=float(r / roi_region.scale),
                    confidence=confidence,
                    detection_method=method,
                    motion_state=self._determine_motion_state(x, y, frame_number),
                    roi_region=roi_region,
                    phase=self.current_phase
                )
            
            return None
            
        except Exception as e:
            print(f"Contour bright detection error: {e}")
            return None
    
    def _detect_motion(self, roi_img: np.ndarray, method: DetectionMethod,
                      roi_region: ROIRegion, frame_number: int) -> Optional[AdaptiveBallResult]:
        """모션 기반 검출"""
        
        # 배경 모델이 없으면 현재 프레임을 배경으로 설정
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img.copy()
        
        # 배경 모델 크기가 다르거나 없으면 초기화
        if (self.background_model is None or 
            self.background_model.shape != gray.shape):
            self.background_model = gray.astype(np.float32)
            return None
        
        try:
            params = self.detection_params[method]
            
            # 배경 차분
            diff = cv2.absdiff(gray, self.background_model.astype(np.uint8))
            
            # 임계값 적용
            _, thresh = cv2.threshold(diff, params['threshold'], 255, cv2.THRESH_BINARY)
            
            # 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # 컨투어 검출
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_contour = None
            best_score = 0.0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if params['min_area'] <= area <= params['max_area']:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # 원형도 계산
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        score = circularity * 0.6 + (area / params['max_area']) * 0.4
                        
                        if score > best_score:
                            best_score = score
                            best_contour = (x, y, radius, score)
            
            # 배경 모델 업데이트 (느린 학습)
            cv2.accumulateWeighted(gray, self.background_model, 0.05)
            
            if best_contour:
                x, y, r, confidence = best_contour
                return AdaptiveBallResult(
                    center_x=float(x / roi_region.scale),
                    center_y=float(y / roi_region.scale),
                    radius=float(r / roi_region.scale),
                    confidence=confidence * 0.7,  # 모션 검출은 약간 낮은 신뢰도
                    detection_method=method,
                    motion_state=self._determine_motion_state(x, y, frame_number),
                    roi_region=roi_region,
                    phase=self.current_phase
                )
            
            return None
            
        except Exception as e:
            print(f"Motion detection error: {e}")
            return None
    
    def _detect_template_match(self, roi_img: np.ndarray, method: DetectionMethod,
                              roi_region: ROIRegion, frame_number: int) -> Optional[AdaptiveBallResult]:
        """템플릿 매칭 기반 검출"""
        
        # 볼 템플릿이 없으면 생성 시도
        if self.ball_template is None and len(self.previous_positions) > 0:
            self._create_ball_template()
        
        if self.ball_template is None:
            return None
        
        try:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img.copy()
            
            # 템플릿 매칭
            result = cv2.matchTemplate(gray, self.ball_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.6:  # 충분한 매칭 점수
                template_h, template_w = self.ball_template.shape
                center_x = max_loc[0] + template_w / 2
                center_y = max_loc[1] + template_h / 2
                radius = min(template_w, template_h) / 2
                
                return AdaptiveBallResult(
                    center_x=float(center_x / roi_region.scale),
                    center_y=float(center_y / roi_region.scale),
                    radius=float(radius / roi_region.scale),
                    confidence=max_val,
                    detection_method=method,
                    motion_state=self._determine_motion_state(center_x, center_y, frame_number),
                    roi_region=roi_region,
                    phase=self.current_phase
                )
            
            return None
            
        except Exception as e:
            print(f"Template matching error: {e}")
            return None
    
    def _select_best_circle(self, circles: np.ndarray, gray_img: np.ndarray, scale: float) -> Optional[Tuple[float, float, float, float]]:
        """최적의 원 선택"""
        
        best_circle = None
        best_score = 0.0
        
        for (x, y, r) in circles:
            # 경계 체크
            if x - r < 0 or y - r < 0 or x + r >= gray_img.shape[1] or y + r >= gray_img.shape[0]:
                continue
            
            # 원형 영역 마스크 생성
            mask = np.zeros(gray_img.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # 원형 영역의 밝기 분석
            roi_pixels = gray_img[mask > 0]
            if len(roi_pixels) == 0:
                continue
            
            mean_brightness = np.mean(roi_pixels)
            std_brightness = np.std(roi_pixels)
            
            # 경계 선명도 (라플라시안)
            roi = gray_img[max(0, y-r):min(gray_img.shape[0], y+r), 
                          max(0, x-r):min(gray_img.shape[1], x+r)]
            if roi.size > 0:
                edge_variance = cv2.Laplacian(roi, cv2.CV_64F).var()
            else:
                edge_variance = 0
            
            # 종합 점수 계산
            brightness_score = min(mean_brightness / 255.0, 1.0)
            uniformity_score = max(0, 1.0 - std_brightness / 50.0)  # 균등한 밝기 선호
            edge_score = min(edge_variance / 500.0, 1.0)
            size_score = min(r / 20.0, 1.0)  # 적당한 크기 선호
            
            total_score = (brightness_score * 0.3 + 
                          uniformity_score * 0.2 + 
                          edge_score * 0.3 + 
                          size_score * 0.2)
            
            if total_score > best_score:
                best_score = total_score
                best_circle = (x, y, r, total_score)
        
        return best_circle
    
    def _determine_motion_state(self, x: float, y: float, frame_number: int) -> str:
        """움직임 상태 판단"""
        
        current_pos = (x, y)
        self.previous_positions.append((current_pos, frame_number))
        
        # 최근 5프레임만 유지
        if len(self.previous_positions) > 5:
            self.previous_positions.pop(0)
        
        if len(self.previous_positions) < 3:
            return "unknown"
        
        # 최근 움직임 계산
        recent_movement = self._calculate_recent_movement()
        
        # 상태 결정
        if recent_movement < 3.0:
            return "static"
        elif recent_movement < 8.0:
            return "ready" 
        elif recent_movement < 20.0:
            return "moving"
        else:
            return "flying"
    
    def _calculate_recent_movement(self) -> float:
        """최근 움직임 계산"""
        
        if len(self.previous_positions) < 2:
            return 0.0
        
        distances = []
        for i in range(len(self.previous_positions) - 1):
            pos1, _ = self.previous_positions[i]
            pos2, _ = self.previous_positions[i + 1]
            
            distance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _update_tracking(self, result: AdaptiveBallResult, frame_number: int):
        """추적 정보 업데이트"""
        
        # 볼 템플릿 업데이트 (추적 단계에서)
        if (self.current_phase in [DetectionPhase.TRACKING, DetectionPhase.FLIGHT_TRACKING] and 
            result.confidence > 0.7):
            # 다음 프레임을 위한 템플릿 생성 준비
            pass
    
    def _create_ball_template(self):
        """볼 템플릿 생성"""
        
        # 이전 검출 결과 기반으로 템플릿 생성
        # 실제 구현에서는 가장 좋은 검출 결과에서 템플릿 추출
        if len(self.previous_positions) > 0:
            # 기본적인 원형 템플릿 생성
            template_size = 20
            self.ball_template = np.zeros((template_size, template_size), dtype=np.uint8)
            cv2.circle(self.ball_template, 
                      (template_size//2, template_size//2), 
                      template_size//3, 
                      255, -1)
            
            # 가우시안 블러로 부드럽게
            self.ball_template = cv2.GaussianBlur(self.ball_template, (3, 3), 1.0)
    
    def reset_tracking(self):
        """추적 초기화"""
        self.previous_positions = []
        self.current_phase = DetectionPhase.FULL_SCREEN
        self.impact_detected = False
        self.ball_template = None
        self.background_model = None
    
    def get_phase_info(self) -> Dict:
        """현재 단계 정보 반환"""
        return {
            'current_phase': self.current_phase.value,
            'roi_region': self.roi_regions[self.current_phase],
            'detection_methods': [m.value for m in self._get_detection_methods_for_phase()],
            'previous_positions_count': len(self.previous_positions)
        }