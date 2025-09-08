#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Golf Analyzer v4.0
소프트웨어 기반 이미지 향상 및 검출 개선 시스템
하드웨어 추가 없이 검출률 극대화
"""

import cv2
import numpy as np
import os
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import math


@dataclass
class BallData:
    """골프공 데이터"""
    frame_num: int
    x: float
    y: float
    radius: float = 0.0
    velocity: float = 0.0
    launch_angle: float = 0.0
    direction_angle: float = 0.0
    backspin: float = 0.0
    sidespin: float = 0.0
    spin_axis: float = 0.0
    confidence: float = 0.0
    motion_state: str = "unknown"
    detection_method: str = ""


@dataclass
class ClubData:
    """클럽 데이터"""
    frame_num: int
    x: float
    y: float
    area: float = 0.0
    club_speed: float = 0.0
    attack_angle: float = 0.0
    face_angle: float = 0.0
    club_path: float = 0.0
    face_to_path: float = 0.0
    smash_factor: float = 0.0
    confidence: float = 0.0


class ImageEnhancer:
    """이미지 향상 처리기"""
    
    @staticmethod
    def enhance_dark_image(img: np.ndarray) -> Dict[str, np.ndarray]:
        """어두운 이미지 향상 - 여러 기법 적용"""
        enhanced_images = {}
        
        # 1. Gamma Correction (밝기 향상)
        gamma = 2.5  # 어두운 이미지를 위한 높은 감마값
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced_images['gamma'] = cv2.LUT(img, table)
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        enhanced_images['clahe'] = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        # 3. Histogram Stretching
        # 각 채널별로 스트레칭
        stretched = img.copy()
        for i in range(3):
            channel = img[:,:,i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                stretched[:,:,i] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        enhanced_images['stretched'] = stretched
        
        # 4. Adaptive Threshold (이진화로 객체 강조)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, 
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
        enhanced_images['adaptive'] = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)
        
        # 5. 밝기/대비 조정
        alpha = 3.0  # 대비 (1.0-3.0)
        beta = 50    # 밝기 (0-100)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        enhanced_images['brightness'] = adjusted
        
        # 6. Unsharp Masking (선명도 향상)
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        unsharp = cv2.addWeighted(img, 2.0, gaussian, -1.0, 0)
        enhanced_images['unsharp'] = unsharp
        
        return enhanced_images


class ROIBasedDetector:
    """ROI 기반 검출기 - 골프공이 나타날 가능성이 높은 영역 집중"""
    
    def __init__(self):
        self.ball_roi_history = []  # 이전 볼 위치 기록
        self.roi_size = 100  # ROI 크기
        
    def get_search_rois(self, img_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """검색할 ROI 영역들 반환"""
        height, width = img_shape[:2]
        rois = []
        
        # 1. 이전 검출 위치 주변
        if self.ball_roi_history:
            last_pos = self.ball_roi_history[-1]
            x, y = last_pos
            roi = (
                max(0, x - self.roi_size//2),
                max(0, y - self.roi_size//2),
                min(width, x + self.roi_size//2),
                min(height, y + self.roi_size//2)
            )
            rois.append(roi)
            
        # 2. 일반적인 골프공 위치 (화면 중앙 영역)
        center_x, center_y = width // 2, height // 2
        rois.append((
            center_x - width//4, center_y - height//3,
            center_x + width//4, center_y + height//3
        ))
        
        # 3. 전체 이미지 (폴백)
        rois.append((0, 0, width, height))
        
        return rois
        
    def update_history(self, x: int, y: int):
        """검출 위치 업데이트"""
        self.ball_roi_history.append((x, y))
        if len(self.ball_roi_history) > 10:
            self.ball_roi_history.pop(0)


class AdvancedBallDetector:
    """개선된 볼 검출기"""
    
    def __init__(self):
        self.roi_detector = ROIBasedDetector()
        self.min_radius = 2  # 더 작은 반경도 검출
        self.max_radius = 20
        
    def detect_ball_multi_method(self, img: np.ndarray, enhanced_imgs: Dict[str, np.ndarray], 
                                 frame_num: int) -> Optional[BallData]:
        """다중 방법으로 볼 검출"""
        
        # 모든 향상된 이미지에서 검출 시도
        all_detections = []
        
        for enhance_type, enhanced_img in enhanced_imgs.items():
            # ROI 영역들 가져오기
            rois = self.roi_detector.get_search_rois(enhanced_img.shape)
            
            for roi in rois:
                x1, y1, x2, y2 = roi
                roi_img = enhanced_img[y1:y2, x1:x2]
                
                # 방법 1: 밝은 점 검출
                detection = self._detect_bright_spot(roi_img, frame_num, enhance_type)
                if detection:
                    detection.x += x1
                    detection.y += y1
                    all_detections.append(detection)
                    
                # 방법 2: Circle Hough Transform
                detection = self._detect_circle(roi_img, frame_num, enhance_type)
                if detection:
                    detection.x += x1
                    detection.y += y1
                    all_detections.append(detection)
                    
                # 방법 3: Blob Detection
                detection = self._detect_blob(roi_img, frame_num, enhance_type)
                if detection:
                    detection.x += x1
                    detection.y += y1
                    all_detections.append(detection)
                    
        # 가장 신뢰도 높은 검출 선택
        if all_detections:
            best_detection = max(all_detections, key=lambda x: x.confidence)
            # ROI 히스토리 업데이트
            self.roi_detector.update_history(int(best_detection.x), int(best_detection.y))
            return best_detection
            
        return None
        
    def _detect_bright_spot(self, img: np.ndarray, frame_num: int, method: str) -> Optional[BallData]:
        """밝은 점 검출"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 상위 5% 밝기 픽셀 추출
        threshold = np.percentile(gray, 95)
        _, bright = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
        
        # 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
        
        # 연결된 컴포넌트 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 골프공 크기 범위
            if np.pi * self.min_radius**2 <= area <= np.pi * self.max_radius**2:
                cx, cy = centroids[i]
                radius = np.sqrt(area / np.pi)
                
                # 원형성 검사
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                           stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.7 < aspect_ratio < 1.3:  # 원형에 가까운 경우
                    return BallData(
                        frame_num=frame_num,
                        x=float(cx),
                        y=float(cy),
                        radius=radius,
                        confidence=0.7,
                        detection_method=f"bright_spot_{method}"
                    )
        
        return None
        
    def _detect_circle(self, img: np.ndarray, frame_num: int, method: str) -> Optional[BallData]:
        """Hough Circle 검출"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 가우시안 블러
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        
        # 여러 파라미터로 시도
        param_sets = [
            (30, 10),  # 낮은 임계값
            (50, 15),  # 중간 임계값
            (70, 20),  # 높은 임계값
        ]
        
        for param1, param2 in param_sets:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.5,
                minDist=15,
                param1=param1,
                param2=param2,
                minRadius=self.min_radius,
                maxRadius=self.max_radius
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    x, y, r = circle
                    return BallData(
                        frame_num=frame_num,
                        x=float(x),
                        y=float(y),
                        radius=float(r),
                        confidence=0.8,
                        detection_method=f"hough_{method}"
                    )
        
        return None
        
    def _detect_blob(self, img: np.ndarray, frame_num: int, method: str) -> Optional[BallData]:
        """Blob 검출"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Blob 검출기 설정
        params = cv2.SimpleBlobDetector_Params()
        
        # 임계값
        params.minThreshold = 10
        params.maxThreshold = 200
        
        # 면적
        params.filterByArea = True
        params.minArea = np.pi * self.min_radius**2
        params.maxArea = np.pi * self.max_radius**2
        
        # 원형성
        params.filterByCircularity = True
        params.minCircularity = 0.6
        
        # 볼록성
        params.filterByConvexity = True
        params.minConvexity = 0.8
        
        # 관성
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        
        if keypoints:
            kp = keypoints[0]
            return BallData(
                frame_num=frame_num,
                x=kp.pt[0],
                y=kp.pt[1],
                radius=kp.size / 2,
                confidence=0.75,
                detection_method=f"blob_{method}"
            )
        
        return None


class AdvancedClubDetector:
    """개선된 클럽 검출기"""
    
    def detect_club(self, img: np.ndarray, enhanced_imgs: Dict[str, np.ndarray], 
                    frame_num: int) -> Optional[ClubData]:
        """클럽 검출"""
        
        # 가장 밝은 영역 찾기 (클럽 헤드는 반사율이 높음)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 상위 10% 밝기 영역
        threshold = np.percentile(gray, 90)
        _, bright = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
        
        # 노이즈 제거 및 연결
        kernel = np.ones((7,7), np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 윤곽선 (클럽 헤드일 가능성)
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if area > 30:  # 최소 면적
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    return ClubData(
                        frame_num=frame_num,
                        x=float(cx),
                        y=float(cy),
                        area=area,
                        confidence=min(1.0, area / 200.0)
                    )
        
        return None


class PhysicsCalculator:
    """물리 계산기"""
    
    def __init__(self):
        self.fps = 820
        self.pixel_to_mm = 0.3
        self.ball_history = []
        self.club_history = []
        
    def calculate_ball_physics(self, ball: BallData, history: List[BallData]) -> BallData:
        """볼 물리 계산"""
        if len(history) >= 2:
            # 속도 계산
            p1 = history[-2]
            p2 = history[-1]
            
            dx = ball.x - p2.x
            dy = ball.y - p2.y
            dt = 1.0 / self.fps
            
            pixel_speed = np.sqrt(dx*dx + dy*dy) / dt
            ball.velocity = pixel_speed * self.pixel_to_mm * 0.002237  # mph
            
            # 발사각
            if len(history) >= 3:
                p0 = history[-3]
                trajectory_angle = np.degrees(np.arctan2(-(ball.y - p0.y), ball.x - p0.x))
                ball.launch_angle = trajectory_angle
            
            # 방향각
            ball.direction_angle = np.degrees(np.arctan2(dx, -dy))
            
            # 모션 상태
            if pixel_speed > 10:
                ball.motion_state = "launched"
            elif pixel_speed > 2:
                ball.motion_state = "moving"
            else:
                ball.motion_state = "static"
                
            # 스핀 추정 (속도 기반)
            if ball.motion_state == "launched":
                ball.backspin = 2500 + pixel_speed * 15
                ball.sidespin = abs(ball.direction_angle) * 30
                ball.spin_axis = ball.direction_angle * 0.3
        
        return ball
        
    def calculate_club_physics(self, club: ClubData, ball: Optional[BallData], 
                              history: List[ClubData]) -> ClubData:
        """클럽 물리 계산"""
        if len(history) >= 2:
            p1 = history[-2]
            p2 = history[-1]
            
            dx = club.x - p2.x
            dy = club.y - p2.y
            dt = 1.0 / self.fps
            
            pixel_speed = np.sqrt(dx*dx + dy*dy) / dt
            club.club_speed = pixel_speed * self.pixel_to_mm * 0.002237
            
            # 어택 앵글
            club.attack_angle = np.degrees(np.arctan2(-dy, dx))
            
            # 기본값
            club.face_angle = 2.5
            club.club_path = club.attack_angle * 0.8
            club.face_to_path = club.face_angle - club.club_path
            
            # 스매쉬팩터
            if ball and ball.velocity > 0 and club.club_speed > 0:
                club.smash_factor = ball.velocity / club.club_speed
        
        return club


class EnhancedGolfAnalyzer:
    """향상된 골프 분석기"""
    
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.ball_detector = AdvancedBallDetector()
        self.club_detector = AdvancedClubDetector()
        self.physics = PhysicsCalculator()
        
        self.ball_history = []
        self.club_history = []
        
    def analyze_frame(self, img_path: str, frame_num: int, save_debug: bool = False) -> Dict:
        """프레임 분석"""
        img = cv2.imread(img_path)
        if img is None:
            return {'frame_num': frame_num, 'ball_data': None, 'club_data': None}
            
        # 이미지 향상
        enhanced_imgs = self.enhancer.enhance_dark_image(img)
        
        # 볼 검출
        ball_data = self.ball_detector.detect_ball_multi_method(img, enhanced_imgs, frame_num)
        if ball_data:
            self.ball_history.append(ball_data)
            ball_data = self.physics.calculate_ball_physics(ball_data, self.ball_history)
            
        # 클럽 검출
        club_data = self.club_detector.detect_club(img, enhanced_imgs, frame_num)
        if club_data:
            self.club_history.append(club_data)
            club_data = self.physics.calculate_club_physics(club_data, ball_data, self.club_history)
            
        # 디버그 이미지 저장
        if save_debug and (ball_data or club_data):
            self._save_debug_image(img, enhanced_imgs, ball_data, club_data, frame_num)
            
        return {
            'frame_num': frame_num,
            'ball_data': ball_data,
            'club_data': club_data
        }
        
    def _save_debug_image(self, orig_img: np.ndarray, enhanced_imgs: Dict[str, np.ndarray],
                         ball_data: Optional[BallData], club_data: Optional[ClubData], 
                         frame_num: int):
        """디버그 이미지 저장"""
        # 가장 좋은 향상 이미지 선택
        debug_img = enhanced_imgs.get('gamma', orig_img).copy()
        
        # 검출 결과 표시
        if ball_data:
            cv2.circle(debug_img, (int(ball_data.x), int(ball_data.y)), 
                      int(ball_data.radius) if ball_data.radius > 0 else 5,
                      (0, 255, 0), 2)
            text = f"Ball: {ball_data.motion_state} ({ball_data.detection_method})"
            cv2.putText(debug_img, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
        if club_data:
            cv2.circle(debug_img, (int(club_data.x), int(club_data.y)), 
                      10, (255, 0, 0), 2)
            cv2.putText(debug_img, "Club", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                       
        # 저장
        debug_dir = "C:/src/GolfSwingAnalysis_Final_ver8/enhanced_debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        # 원본과 향상된 이미지 나란히 저장
        combined = np.hstack([orig_img, debug_img])
        debug_path = os.path.join(debug_dir, f"debug_{frame_num:03d}.jpg")
        cv2.imwrite(debug_path, combined)
        
    def process_sequence(self, image_dir: str) -> List[Dict]:
        """이미지 시퀀스 처리"""
        jpg_files = []
        for file in sorted(os.listdir(image_dir)):
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(image_dir, file))
                
        print(f"총 {len(jpg_files)}개 이미지 처리")
        
        results = []
        for i, img_path in enumerate(jpg_files, 1):
            print(f"\r프레임 {i}/{len(jpg_files)} 처리 중...", end="")
            
            # 주요 프레임만 디버그 이미지 저장
            save_debug = (i % 10 == 0) or (i <= 5)
            
            result = self.analyze_frame(img_path, i, save_debug)
            results.append(result)
            
        print("\n처리 완료!")
        
        # 통계
        ball_detections = sum(1 for r in results if r['ball_data'])
        club_detections = sum(1 for r in results if r['club_data'])
        launched = sum(1 for r in results if r.get('ball_data') and r['ball_data'].motion_state == 'launched')
        
        print(f"\n=== 검출 결과 ===")
        print(f"볼 검출: {ball_detections}/{len(results)} ({100*ball_detections/len(results):.1f}%)")
        print(f"클럽 검출: {club_detections}/{len(results)} ({100*club_detections/len(results):.1f}%)")
        print(f"발사 감지: {launched} 프레임")
        
        return results
        
    def export_to_excel(self, results: List[Dict], output_path: str):
        """엑셀 출력"""
        ball_rows = []
        club_rows = []
        
        for result in results:
            frame = result['frame_num']
            
            # 볼 데이터
            if result['ball_data']:
                bd = result['ball_data']
                ball_rows.append({
                    'Frame': frame,
                    'X': bd.x,
                    'Y': bd.y,
                    'Radius': bd.radius,
                    'Detection_Method': bd.detection_method,
                    'Motion_State': bd.motion_state,
                    'Ball_Speed_mph': bd.velocity,
                    'Launch_Angle': bd.launch_angle,
                    'Direction_Angle': bd.direction_angle,
                    'Backspin_rpm': bd.backspin,
                    'Sidespin_rpm': bd.sidespin,
                    'Spin_Axis': bd.spin_axis,
                    'Confidence': bd.confidence
                })
                
        # 데이터프레임 생성
        ball_df = pd.DataFrame(ball_rows) if ball_rows else pd.DataFrame()
        club_df = pd.DataFrame(club_rows) if club_rows else pd.DataFrame()
        
        # 엑셀 저장
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            if not ball_df.empty:
                ball_df.to_excel(writer, sheet_name='Ball_Data', index=False)
            if not club_df.empty:
                club_df.to_excel(writer, sheet_name='Club_Data', index=False)
                
            # 요약
            summary = {
                'Metric': ['Total_Frames', 'Ball_Detections', 'Detection_Rate'],
                'Value': [
                    len(results),
                    len(ball_rows),
                    f"{100*len(ball_rows)/len(results):.1f}%"
                ]
            }
            summary_df = pd.DataFrame(summary)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
        print(f"\n결과 저장: {output_path}")


def main():
    """메인 실행"""
    print("=== 향상된 골프 분석 시스템 v4.0 ===")
    print("소프트웨어 기반 이미지 향상 및 검출 개선")
    
    analyzer = EnhancedGolfAnalyzer()
    
    # 이미지 디렉토리
    image_dir = "C:/src/GolfSwingAnalysis_Final_ver8/shot-image-jpg/7iron_no_marker_ball_shot1"
    
    # 분석 실행
    results = analyzer.process_sequence(image_dir)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"C:/src/GolfSwingAnalysis_Final_ver8/enhanced_golf_analysis_{timestamp}.xlsx"
    analyzer.export_to_excel(results, output_path)
    
    print("\n=== 분석 완료 ===")


if __name__ == "__main__":
    main()