#!/usr/bin/env python3
"""
볼 검출 결과 시각화 검증 시스템
검출된 볼을 이미지에 표시하여 정확도 확인
"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import logging
from ultra_precise_ball_detector import UltraPreciseBallDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisualBallDetectionValidator:
    def __init__(self):
        """시각적 볼 검출 검증기 초기화"""
        self.detector = UltraPreciseBallDetector()
        self.output_base_path = "ball_detection_visual_results"
        
    def create_output_directories(self):
        """출력 디렉토리 생성"""
        clubs = ['5Iron_0930', '7Iron_0930', 'driver_0930', 'PW_0930']
        
        for club in clubs:
            club_output_path = os.path.join(self.output_base_path, club)
            os.makedirs(club_output_path, exist_ok=True)
            
            # 각 샷별 디렉토리 생성
            for shot_num in range(1, 11):
                shot_path = os.path.join(club_output_path, f"shot_{shot_num}")
                os.makedirs(shot_path, exist_ok=True)
        
        logger.info(f"Created output directories under {self.output_base_path}")
    
    def draw_ball_detection(self, img, ball_result, method="opencv"):
        """이미지에 볼 검출 결과 그리기"""
        if ball_result is None:
            return img.copy()
        
        x, y, radius = ball_result[:3]
        
        if method == "opencv":
            # OpenCV로 그리기
            result_img = img.copy()
            
            # 볼 원 그리기 (녹색)
            cv2.circle(result_img, (int(x), int(y)), int(radius), (0, 255, 0), 3)
            
            # 중심점 표시 (빨간색)
            cv2.circle(result_img, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # 좌표 텍스트 표시
            text = f"({int(x)}, {int(y)}) r:{int(radius)}"
            cv2.putText(result_img, text, (int(x) - 50, int(y) - radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return result_img
        
        elif method == "matplotlib":
            # Matplotlib으로 그리기
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 원본 이미지
            ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # 검출 결과 이미지
            ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # 볼 원 그리기
            circle = patches.Circle((x, y), radius, linewidth=3, edgecolor='lime', facecolor='none')
            ax2.add_patch(circle)
            
            # 중심점 표시
            ax2.plot(x, y, 'ro', markersize=8)
            
            # 좌표 텍스트
            ax2.text(x - 50, y - radius - 10, f"({int(x)}, {int(y)}) r:{int(radius)}", 
                    fontsize=10, color='white', bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            ax2.set_title('Ball Detection Result')
            ax2.axis('off')
            
            plt.tight_layout()
            return fig
        
        return img.copy()
    
    def process_single_shot(self, shot_path, club_name, shot_num):
        """단일 샷 처리 및 시각화"""
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
            
            # OpenCV 방식으로 결과 이미지 생성
            if ball1 is not None:
                result_img1 = self.draw_ball_detection(img1, ball1, method="opencv")
                cv2.imwrite(os.path.join(output_dir, "camera1_detection.jpg"), result_img1)
                logger.info(f"Camera 1: Ball detected at ({ball1[0]}, {ball1[1]}) with radius {ball1[2]}")
            else:
                cv2.imwrite(os.path.join(output_dir, "camera1_no_detection.jpg"), img1)
                logger.warning("Camera 1: No ball detected")
            
            if ball2 is not None:
                result_img2 = self.draw_ball_detection(img2, ball2, method="opencv")
                cv2.imwrite(os.path.join(output_dir, "camera2_detection.jpg"), result_img2)
                logger.info(f"Camera 2: Ball detected at ({ball2[0]}, {ball2[1]}) with radius {ball2[2]}")
            else:
                cv2.imwrite(os.path.join(output_dir, "camera2_no_detection.jpg"), img2)
                logger.warning("Camera 2: No ball detected")
            
            # Matplotlib 방식으로 비교 이미지 생성
            if ball1 is not None or ball2 is not None:
                fig = plt.figure(figsize=(20, 10))
                
                # 카메라 1
                ax1 = plt.subplot(2, 2, 1)
                ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                ax1.set_title(f'Camera 1 - Original', fontsize=12)
                ax1.axis('off')
                
                ax2 = plt.subplot(2, 2, 2)
                ax2.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                if ball1 is not None:
                    circle1 = patches.Circle((ball1[0], ball1[1]), ball1[2], 
                                           linewidth=3, edgecolor='lime', facecolor='none')
                    ax2.add_patch(circle1)
                    ax2.plot(ball1[0], ball1[1], 'ro', markersize=8)
                    ax2.text(ball1[0] - 50, ball1[1] - ball1[2] - 10, 
                            f"({int(ball1[0])}, {int(ball1[1])}) r:{int(ball1[2])}", 
                            fontsize=10, color='white', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                else:
                    ax2.text(img1.shape[1]//2, img1.shape[0]//2, "NO BALL DETECTED", 
                            fontsize=16, color='red', ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
                ax2.set_title(f'Camera 1 - Detection Result', fontsize=12)
                ax2.axis('off')
                
                # 카메라 2
                ax3 = plt.subplot(2, 2, 3)
                ax3.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
                ax3.set_title(f'Camera 2 - Original', fontsize=12)
                ax3.axis('off')
                
                ax4 = plt.subplot(2, 2, 4)
                ax4.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
                if ball2 is not None:
                    circle2 = patches.Circle((ball2[0], ball2[1]), ball2[2], 
                                           linewidth=3, edgecolor='lime', facecolor='none')
                    ax4.add_patch(circle2)
                    ax4.plot(ball2[0], ball2[1], 'ro', markersize=8)
                    ax4.text(ball2[0] - 50, ball2[1] - ball2[2] - 10, 
                            f"({int(ball2[0])}, {int(ball2[1])}) r:{int(ball2[2])}", 
                            fontsize=10, color='white', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                else:
                    ax4.text(img2.shape[1]//2, img2.shape[0]//2, "NO BALL DETECTED", 
                            fontsize=16, color='red', ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
                ax4.set_title(f'Camera 2 - Detection Result', fontsize=12)
                ax4.axis('off')
                
                plt.suptitle(f'{club_name} - Shot {shot_num} - Ball Detection Visualization', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # 결과 저장
                comparison_path = os.path.join(output_dir, "comparison_visualization.png")
                plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Comparison visualization saved to {comparison_path}")
            
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
        report.append("ball_detection_visual_results/")
        report.append("├── 5iron/")
        report.append("│   ├── shot_1/")
        report.append("│   │   ├── camera1_detection.jpg")
        report.append("│   │   ├── camera2_detection.jpg")
        report.append("│   │   └── comparison_visualization.png")
        report.append("│   ├── shot_2/")
        report.append("│   └── ...")
        report.append("├── 7iron/")
        report.append("├── driver/")
        report.append("└── pw/")
        report.append("```")
        report.append("")
        
        report.append("## 파일 설명")
        report.append("- **camera1_detection.jpg**: 상단 카메라 검출 결과")
        report.append("- **camera2_detection.jpg**: 하단 카메라 검출 결과")
        report.append("- **comparison_visualization.png**: 4분할 비교 이미지")
        report.append("")
        
        report.append("## 시각화 요소")
        report.append("- 🟢 **녹색 원**: 검출된 볼의 경계")
        report.append("- 🔴 **빨간 점**: 볼의 중심점")
        report.append("- 📍 **좌표 텍스트**: (x, y) 좌표와 반지름")
        report.append("- ⚠️ **노란 배경 텍스트**: 검출 실패 시 표시")
        
        return "\n".join(report)

def main():
    """메인 실행 함수"""
    logger.info("Starting visual ball detection validation...")
    
    validator = VisualBallDetectionValidator()
    
    if not validator.detector.calibration_data:
        logger.error("Failed to load calibration data. Exiting.")
        return
    
    # 출력 디렉토리 생성
    validator.create_output_directories()
    
    # 모든 클럽 처리
    data_path = "data/video_ballData_20250930/video_ballData_20250930"
    if os.path.exists(data_path):
        successful_shots, total_shots = validator.process_all_clubs(data_path)
        
        # 요약 리포트 생성
        report = validator.generate_summary_report(successful_shots, total_shots)
        
        # 리포트 저장
        with open("ball_detection_visual_summary.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info("Visual validation completed!")
        logger.info(f"Results saved to: {validator.output_base_path}")
        logger.info("Summary report saved to: ball_detection_visual_summary.md")
        
        # 결과 요약 출력
        print(f"\n=== 시각화 결과 요약 ===")
        print(f"총 처리 샷: {total_shots}")
        print(f"성공한 처리: {successful_shots}")
        print(f"처리 성공률: {(successful_shots/total_shots)*100:.1f}%")
        print(f"결과 저장 위치: {validator.output_base_path}")
        
    else:
        logger.error(f"Data path not found: {data_path}")

if __name__ == "__main__":
    main()
