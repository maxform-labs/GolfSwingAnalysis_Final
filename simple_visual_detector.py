#!/usr/bin/env python3
"""
간단한 볼 검출 시각화 시스템
OpenCV 기반으로 검출 결과를 이미지에 표시
"""

import cv2
import numpy as np
import os
import glob
import logging
from datetime import datetime
from ultra_precise_ball_detector import UltraPreciseBallDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVisualDetector:
    def __init__(self):
        """간단한 시각화 검출기 초기화"""
        self.detector = UltraPreciseBallDetector()
        self.output_base_path = "ball_detection_results"
        
    def create_output_directories(self):
        """출력 디렉토리 생성"""
        clubs = ['5iron', '7iron', 'driver', 'pw']
        
        for club in clubs:
            club_output_path = os.path.join(self.output_base_path, club)
            os.makedirs(club_output_path, exist_ok=True)
            
            # 각 샷별 디렉토리 생성
            for shot_num in range(1, 11):
                shot_path = os.path.join(club_output_path, f"shot_{shot_num}")
                os.makedirs(shot_path, exist_ok=True)
        
        logger.info(f"Created output directories under {self.output_base_path}")
    
    def draw_detection_result(self, img, ball_result, camera_name):
        """검출 결과를 이미지에 그리기"""
        result_img = img.copy()
        
        if ball_result is not None:
            x, y, radius = ball_result[:3]
            
            # 볼 원 그리기 (녹색)
            cv2.circle(result_img, (int(x), int(y)), int(radius), (0, 255, 0), 3)
            
            # 중심점 표시 (빨간색)
            cv2.circle(result_img, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # 좌표 텍스트 표시
            text = f"Ball: ({int(x)}, {int(y)}) r:{int(radius)}"
            cv2.putText(result_img, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 성공 메시지
            success_text = f"{camera_name}: DETECTED"
            cv2.putText(result_img, success_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # 실패 메시지
            fail_text = f"{camera_name}: NO BALL DETECTED"
            cv2.putText(result_img, fail_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result_img
    
    def create_comparison_image(self, img1, img2, ball1, ball2, club_name, shot_num):
        """비교 이미지 생성"""
        # 각 카메라별 결과 이미지 생성
        result_img1 = self.draw_detection_result(img1, ball1, "Camera 1")
        result_img2 = self.draw_detection_result(img2, ball2, "Camera 2")
        
        # 이미지 크기 조정 (비율 유지)
        height, width = img1.shape[:2]
        new_width = 400
        new_height = int(height * new_width / width)
        
        img1_resized = cv2.resize(img1, (new_width, new_height))
        result_img1_resized = cv2.resize(result_img1, (new_width, new_height))
        img2_resized = cv2.resize(img2, (new_width, new_height))
        result_img2_resized = cv2.resize(result_img2, (new_width, new_height))
        
        # 2x2 그리드로 결합
        top_row = np.hstack([img1_resized, result_img1_resized])
        bottom_row = np.hstack([img2_resized, result_img2_resized])
        comparison = np.vstack([top_row, bottom_row])
        
        # 제목 추가
        title_text = f"{club_name.upper()} - Shot {shot_num} - Ball Detection Results"
        cv2.putText(comparison, title_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 범례 추가
        legend_y = comparison.shape[0] - 20
        cv2.putText(comparison, "Original | Detection Result", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return comparison
    
    def process_single_shot(self, shot_path, club_name, shot_num):
        """단일 샷 처리"""
        # 이미지 파일 찾기
        img1_files = sorted(glob.glob(os.path.join(shot_path, "1_*.bmp")))
        img2_files = sorted(glob.glob(os.path.join(shot_path, "2_*.bmp")))
        
        if not img1_files or not img2_files:
            logger.warning(f"No image files found in {shot_path}")
            return False
        
        # 첫 프레임 처리
        img1_path = img1_files[0]
        img2_path = img2_files[0]
        
        try:
            # 이미지 로드
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                logger.warning(f"Could not load images from {shot_path}")
                return False
            
            # 볼 검출
            ball1 = self.detector.detect_ball_ultra_precise(img1)
            ball2 = self.detector.detect_ball_ultra_precise(img2)
            
            # 출력 디렉토리
            output_dir = os.path.join(self.output_base_path, club_name, f"shot_{shot_num}")
            
            # 개별 카메라 결과 저장
            result_img1 = self.draw_detection_result(img1, ball1, "Camera 1")
            result_img2 = self.draw_detection_result(img2, ball2, "Camera 2")
            
            cv2.imwrite(os.path.join(output_dir, "camera1_result.jpg"), result_img1)
            cv2.imwrite(os.path.join(output_dir, "camera2_result.jpg"), result_img2)
            
            # 비교 이미지 생성 및 저장
            comparison = self.create_comparison_image(img1, img2, ball1, ball2, club_name, shot_num)
            cv2.imwrite(os.path.join(output_dir, "comparison.jpg"), comparison)
            
            # 검출 결과 로깅
            if ball1 is not None:
                logger.info(f"Camera 1: Ball detected at ({ball1[0]}, {ball1[1]}) with radius {ball1[2]}")
            else:
                logger.warning("Camera 1: No ball detected")
            
            if ball2 is not None:
                logger.info(f"Camera 2: Ball detected at ({ball2[0]}, {ball2[1]}) with radius {ball2[2]}")
            else:
                logger.warning("Camera 2: No ball detected")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing shot {shot_num} for {club_name}: {str(e)}")
            return False
    
    def process_all_clubs(self, data_path):
        """모든 클럽의 모든 샷 처리"""
        clubs = {
            '5Iron_0930': '5iron',
            '7Iron_0930': '7iron', 
            'driver_0930': 'driver',
            'PW_0930': 'pw'
        }
        
        total_shots = 0
        successful_shots = 0
        
        for club_folder, club_name in clubs.items():
            club_path = os.path.join(data_path, club_folder)
            
            if not os.path.exists(club_path):
                logger.warning(f"Club path not found: {club_path}")
                continue
            
            logger.info(f"Processing {club_name} shots for visualization...")
            
            # 샷 디렉토리 찾기
            shot_dirs = [d for d in os.listdir(club_path) if d.isdigit()]
            shot_dirs.sort(key=int)
            
            for shot_num in shot_dirs[:10]:  # 처음 10개 샷만 처리
                shot_path = os.path.join(club_path, shot_num)
                total_shots += 1
                
                success = self.process_single_shot(shot_path, club_name, int(shot_num))
                if success:
                    successful_shots += 1
                
                logger.info(f"Processed {club_name} shot {shot_num}")
        
        logger.info(f"Visualization completed: {successful_shots}/{total_shots} shots processed successfully")
        return successful_shots, total_shots
    
    def generate_summary_report(self, successful_shots, total_shots):
        """요약 리포트 생성"""
        report = []
        report.append("# 볼 검출 시각화 결과 요약")
        report.append("")
        report.append(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## 처리 결과")
        report.append(f"- **총 처리 샷**: {total_shots}")
        report.append(f"- **성공한 처리**: {successful_shots}")
        report.append(f"- **처리 성공률**: {(successful_shots/total_shots)*100:.1f}%")
        report.append("")
        
        report.append("## 생성된 파일 구조")
        report.append("")
        report.append("```")
        report.append("ball_detection_results/")
        report.append("├── 5iron/")
        report.append("│   ├── shot_1/")
        report.append("│   │   ├── camera1_result.jpg")
        report.append("│   │   ├── camera2_result.jpg")
        report.append("│   │   └── comparison.jpg")
        report.append("│   ├── shot_2/")
        report.append("│   └── ...")
        report.append("├── 7iron/")
        report.append("├── driver/")
        report.append("└── pw/")
        report.append("```")
        report.append("")
        
        report.append("## 파일 설명")
        report.append("- **camera1_result.jpg**: 상단 카메라 검출 결과")
        report.append("- **camera2_result.jpg**: 하단 카메라 검출 결과")
        report.append("- **comparison.jpg**: 2x2 비교 이미지 (원본 | 검출결과)")
        report.append("")
        
        report.append("## 시각화 요소")
        report.append("- 🟢 **녹색 원**: 검출된 볼의 경계")
        report.append("- 🔴 **빨간 점**: 볼의 중심점")
        report.append("- 📍 **좌표 텍스트**: (x, y) 좌표와 반지름")
        report.append("- ✅ **성공 메시지**: 검출 성공 시 녹색 텍스트")
        report.append("- ❌ **실패 메시지**: 검출 실패 시 빨간색 텍스트")
        
        return "\n".join(report)

def main():
    """메인 실행 함수"""
    logger.info("Starting simple visual ball detection...")
    
    detector = SimpleVisualDetector()
    
    if not detector.detector.calibration_data:
        logger.error("Failed to load calibration data. Exiting.")
        return
    
    # 출력 디렉토리 생성
    detector.create_output_directories()
    
    # 모든 클럽 처리
    data_path = "data/video_ballData_20250930/video_ballData_20250930"
    if os.path.exists(data_path):
        successful_shots, total_shots = detector.process_all_clubs(data_path)
        
        # 요약 리포트 생성
        report = detector.generate_summary_report(successful_shots, total_shots)
        
        # 리포트 저장
        with open("ball_detection_visual_summary.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info("Visual detection completed!")
        logger.info(f"Results saved to: {detector.output_base_path}")
        logger.info("Summary report saved to: ball_detection_visual_summary.md")
        
        # 결과 요약 출력
        print(f"\n=== 시각화 결과 요약 ===")
        print(f"총 처리 샷: {total_shots}")
        print(f"성공한 처리: {successful_shots}")
        print(f"처리 성공률: {(successful_shots/total_shots)*100:.1f}%")
        print(f"결과 저장 위치: {detector.output_base_path}")
        
    else:
        logger.error(f"Data path not found: {data_path}")

if __name__ == "__main__":
    main()





