#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 골프공 검출 알고리즘
두 카메라를 일관되게 처리하는 통합 검출 방법
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_unified_characteristics():
    """통합 특성 분석"""
    
    print("=== 통합 골프공 검출 특성 분석 ===")
    
    # 이미지 디렉토리
    image_dir = 'data/swingData/5Iron_0930/1'
    
    # 샘플 이미지들 로드
    cam1_samples = []
    cam2_samples = []
    
    for i in range(1, 6):  # 처음 5개 이미지로 분석
        cam1_img = cv2.imread(os.path.join(image_dir, f'Gamma_1_{i}.bmp'))
        cam2_img = cv2.imread(os.path.join(image_dir, f'Gamma_2_{i}.bmp'))
        
        if cam1_img is not None:
            cam1_samples.append(cam1_img)
        if cam2_img is not None:
            cam2_samples.append(cam2_img)
    
    if not cam1_samples or not cam2_samples:
        print("이미지 로드 실패")
        return None
    
    # 통합 특성 분석
    all_brightness = []
    all_std = []
    all_histograms = []
    
    for img in cam1_samples + cam2_samples:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        all_brightness.append(np.mean(gray))
        all_std.append(np.std(gray))
        all_histograms.append(cv2.calcHist([gray], [0], None, [256], [0, 256]))
    
    # 통합 통계
    avg_brightness = np.mean(all_brightness)
    avg_std = np.mean(all_std)
    
    print(f"통합 평균 밝기: {avg_brightness:.1f}")
    print(f"통합 평균 표준편차: {avg_std:.1f}")
    print(f"분석 이미지 수: {len(cam1_samples + cam2_samples)}개")
    
    # 최적 임계값 계산
    optimal_threshold = int(avg_brightness + 0.5 * avg_std)
    print(f"최적 밝기 임계값: {optimal_threshold}")
    
    return {
        'avg_brightness': avg_brightness,
        'avg_std': avg_std,
        'optimal_threshold': optimal_threshold,
        'sample_count': len(cam1_samples + cam2_samples)
    }

def unified_golf_ball_detection(image_path):
    """통합 골프공 검출 알고리즘"""
    
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 통합 전처리
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. 적응형 밝기 임계값
    # 이미지별 평균 밝기 기반 임계값 설정
    mean_brightness = np.mean(gray)
    threshold = max(150, min(200, int(mean_brightness + 50)))
    
    # 3. 다중 임계값 처리
    # 여러 임계값으로 시도
    thresholds = [threshold, threshold + 20, threshold - 20, 180, 160]
    
    best_circle = None
    best_score = 0
    
    for thresh_val in thresholds:
        # 밝기 임계값 처리
        white_mask = cv2.inRange(gray, thresh_val, 255)
        
        # 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20 or area > 5000:  # 넓은 범위
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.4:  # 낮은 원형도 임계값
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                if 3 <= radius <= 50:  # 넓은 반지름 범위
                    # 위치 기반 점수 (이미지 중심에 가까울수록 높은 점수)
                    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
                    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    position_score = max(0, 1 - distance_from_center / (img.shape[1] * 0.5))
                    
                    # 종합 점수
                    score = circularity * area * position_score
                    
                    if score > best_score:
                        best_score = score
                        best_circle = (int(x), int(y), int(radius))
    
    if best_circle is not None:
        return (best_circle[0], best_circle[1]), best_circle[2], img
    else:
        # 허프 원 변환으로 재시도
        return unified_hough_circles(img, gray)

def unified_hough_circles(img, gray):
    """통합 허프 원 변환"""
    
    # 가우시안 블러
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 통합 허프 원 변환 파라미터
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=25,
        minRadius=5,
        maxRadius=40
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # 가장 큰 원 선택
        best_circle = max(circles, key=lambda x: x[2])
        return (best_circle[0], best_circle[1]), best_circle[2], img
    
    return None, None, img

def detect_all_unified_golf_balls():
    """통합 골프공 검출"""
    
    print("=== 통합 골프공 검출 시작 ===")
    
    # 통합 특성 분석
    characteristics = analyze_unified_characteristics()
    
    # 이미지 디렉토리
    image_dir = 'data/swingData/5Iron_0930/1'
    
    # 결과 저장
    results = {
        'cam1': [],
        'cam2': [],
        'detected_images': []
    }
    
    print("\n통합 골프공 검출 진행:")
    print("-" * 50)
    
    # Gamma_1 (Cam1) 이미지들
    print("Cam1 (Gamma_1) 검출:")
    for i in range(1, 11):
        image_path = os.path.join(image_dir, f'Gamma_1_{i}.bmp')
        ball_pos, radius, img = unified_golf_ball_detection(image_path)
        
        if ball_pos is not None:
            results['cam1'].append(ball_pos)
            results['detected_images'].append(f'Gamma_1_{i}')
            print(f"  Gamma_1_{i}: 검출 성공 - 위치: {ball_pos}, 반지름: {radius}")
        else:
            print(f"  Gamma_1_{i}: 검출 실패")
    
    print()
    
    # Gamma_2 (Cam2) 이미지들
    print("Cam2 (Gamma_2) 검출:")
    for i in range(1, 11):
        image_path = os.path.join(image_dir, f'Gamma_2_{i}.bmp')
        ball_pos, radius, img = unified_golf_ball_detection(image_path)
        
        if ball_pos is not None:
            results['cam2'].append(ball_pos)
            results['detected_images'].append(f'Gamma_2_{i}')
            print(f"  Gamma_2_{i}: 검출 성공 - 위치: {ball_pos}, 반지름: {radius}")
        else:
            print(f"  Gamma_2_{i}: 검출 실패")
    
    print()
    
    # 검출 결과 요약
    total_images = 20
    detected_count = len(results['detected_images'])
    success_rate = (detected_count / total_images) * 100
    
    print(f"검출 결과 요약:")
    print(f"  총 이미지: {total_images}개")
    print(f"  검출 성공: {detected_count}개")
    print(f"  성공률: {success_rate:.1f}%")
    print(f"  Cam1: {len(results['cam1'])}/10개 ({len(results['cam1'])/10*100:.1f}%)")
    print(f"  Cam2: {len(results['cam2'])}/10개 ({len(results['cam2'])/10*100:.1f}%)")
    print()
    
    if success_rate == 100:
        print("SUCCESS: 모든 이미지에서 통합 골프공 검출 성공!")
    elif success_rate >= 80:
        print("GOOD: 대부분의 이미지에서 통합 골프공 검출 성공!")
    else:
        print(f"WARNING: {total_images - detected_count}개 이미지에서 검출 실패")
    
    return results

def create_unified_detection_visualization(results):
    """통합 검출 결과 시각화"""
    
    print("=== 통합 검출 결과 시각화 생성 ===")
    
    # 이미지 디렉토리
    image_dir = 'data/swingData/5Iron_0930/1'
    
    # 시각화 생성
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < 10:
            # Gamma_1 이미지들
            image_path = os.path.join(image_dir, f'Gamma_1_{i+1}.bmp')
            title = f'Gamma_1_{i+1}'
        else:
            # Gamma_2 이미지들
            image_path = os.path.join(image_dir, f'Gamma_2_{i-9}.bmp')
            title = f'Gamma_2_{i-9}'
        
        # 이미지 로드
        img = cv2.imread(image_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            
            # 통합 골프공 검출
            ball_pos, radius, _ = unified_golf_ball_detection(image_path)
            
            if ball_pos is not None:
                # 검출된 골프공 표시 (통일된 스타일)
                circle = plt.Circle(ball_pos, radius, color='lime', fill=False, linewidth=3)
                ax.add_patch(circle)
                ax.plot(ball_pos[0], ball_pos[1], 'ko', markersize=8, markeredgecolor='lime', markeredgewidth=2)
                ax.set_title(f'{title} - 검출됨', color='green', fontsize=10)
            else:
                ax.set_title(f'{title} - 검출 실패', color='red', fontsize=10)
        
        ax.axis('off')
    
    plt.tight_layout()
    
    # 저장
    output_path = '../ball_detect/unified_golf_ball_detection_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"통합 검출 결과 시각화 저장: {output_path}")
    
    return output_path

def create_unified_detection_report(results):
    """통합 검출 보고서 생성"""
    
    total_images = 20
    detected_count = len(results['detected_images'])
    success_rate = (detected_count / total_images) * 100
    
    report_content = f"""# 통합 골프공 검출 결과 보고서

## 📊 검출 결과

### 전체 검출률
- **총 이미지**: {total_images}개
- **검출 성공**: {detected_count}개
- **성공률**: {success_rate:.1f}%

### 카메라별 검출률
- **Cam1 (Gamma_1)**: {len(results['cam1'])}/10개 ({len(results['cam1'])/10*100:.1f}%)
- **Cam2 (Gamma_2)**: {len(results['cam2'])}/10개 ({len(results['cam2'])/10*100:.1f}%)

### 검출된 이미지
"""
    
    for img in results['detected_images']:
        report_content += f"- {img}\n"
    
    report_content += f"""
## 🔧 통합 검출 기법

### 핵심 원리
- **일관된 처리**: 두 카메라에 동일한 알고리즘 적용
- **적응형 임계값**: 이미지별 평균 밝기 기반 자동 조정
- **다중 임계값**: 여러 임계값으로 시도하여 최적 결과 선택
- **위치 기반 점수**: 이미지 중심에 가까운 골프공 우선 선택

### 검출 파라미터
- **밝기 임계값**: 이미지별 적응형 (150-200)
- **원형도 임계값**: 0.4 (낮은 임계값으로 유연성 확보)
- **면적 범위**: 20-5000 픽셀 (넓은 범위)
- **반지름 범위**: 3-50 픽셀 (넓은 범위)
- **허프 원 변환**: dp=1, minDist=30, param1=50, param2=25

### 장점
- **일관성**: 두 카메라에 동일한 기준 적용
- **적응성**: 이미지별 특성에 자동 조정
- **안정성**: 다중 검출 방법으로 실패율 최소화
- **효율성**: 단일 알고리즘으로 모든 카메라 처리

## 📁 생성된 파일
- **통합 검출 결과**: ball_detect/unified_golf_ball_detection_results.png
- **통합 검출 보고서**: 이 문서

## ✅ 결론
"""
    
    if success_rate == 100:
        report_content += "모든 이미지에서 통합 골프공 검출에 성공했습니다!\n"
    elif success_rate >= 80:
        report_content += "대부분의 이미지에서 통합 골프공 검출에 성공했습니다!\n"
    else:
        report_content += f"{total_images - detected_count}개 이미지에서 검출에 실패했습니다.\n"
    
    report_content += """
통합 검출 알고리즘으로 두 카메라를 일관되게 처리할 수 있게 되었습니다.

---
*검출 완료: 2025-10-20*
"""
    
    # 보고서 저장
    with open('../ball_detect/unified_golf_ball_detection_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("통합 검출 보고서 저장: ../ball_detect/unified_golf_ball_detection_report.md")
    return '../ball_detect/unified_golf_ball_detection_report.md'

if __name__ == "__main__":
    print("통합 골프공 검출 시작...")
    
    # 모든 이미지에서 통합 골프공 검출
    results = detect_all_unified_golf_balls()
    
    # 통합 검출 결과 시각화
    viz_path = create_unified_detection_visualization(results)
    
    # 통합 검출 보고서 생성
    report_path = create_unified_detection_report(results)
    
    print(f"\n=== 통합 골프공 검출 완료 ===")
    print(f"검출 결과: {viz_path}")
    print(f"보고서: {report_path}")
    print(f"\n검출 결과:")
    print(f"  - 총 이미지: 20개")
    print(f"  - 검출 성공: {len(results['detected_images'])}개")
    print(f"  - 성공률: {(len(results['detected_images']) / 20) * 100:.1f}%")
    print(f"  - Cam1: {len(results['cam1'])}/10개")
    print(f"  - Cam2: {len(results['cam2'])}/10개")
    print(f"  - 상태: {'완벽' if len(results['detected_images']) == 20 else '개선됨'}")
