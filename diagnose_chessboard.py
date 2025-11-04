"""
체스보드 이미지 진단 - 왜 코너 검출이 실패하는가?
"""

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

def diagnose_chessboard_image(image_path, pattern_sizes=[(9, 6), (8, 5), (7, 6), (6, 9)]):
    """
    체스보드 이미지 진단
    """
    print(f"\n{'='*60}")
    print(f"이미지 분석: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지 로드 실패")
        return None
    
    print(f"이미지 크기: {img.shape[1]} × {img.shape[0]}")
    print(f"채널: {img.shape[2] if len(img.shape) > 2 else 1}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 밝기 분석
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    min_brightness = np.min(gray)
    max_brightness = np.max(gray)
    
    print(f"\n밝기 분석:")
    print(f"  평균: {mean_brightness:.1f}")
    print(f"  표준편차: {std_brightness:.1f}")
    print(f"  범위: {min_brightness} - {max_brightness}")
    
    if mean_brightness < 50:
        print(f"  ⚠️ 이미지가 너무 어두움")
    elif mean_brightness > 200:
        print(f"  ⚠️ 이미지가 너무 밝음")
    
    if std_brightness < 30:
        print(f"  ⚠️ 대비가 부족함")
    
    # 여러 패턴 크기 시도
    print(f"\n체스보드 패턴 검출 시도:")
    found_pattern = None
    
    for pattern_size in pattern_sizes:
        print(f"  패턴 {pattern_size}...", end=" ")
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            print(f"✓ 성공! ({len(corners)} 코너 발견)")
            found_pattern = (pattern_size, corners)
            break
        else:
            print(f"✗ 실패")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 원본 이미지
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Original Image\n{os.path.basename(image_path)}')
    axes[0, 0].axis('off')
    
    # 그레이스케일
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title(f'Grayscale\nMean: {mean_brightness:.1f}, Std: {std_brightness:.1f}')
    axes[0, 1].axis('off')
    
    # 히스토그램
    axes[1, 0].hist(gray.ravel(), bins=256, range=(0, 256), color='gray')
    axes[1, 0].set_title('Brightness Histogram')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 검출된 코너 (있으면)
    if found_pattern:
        pattern_size, corners = found_pattern
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, True)
        axes[1, 1].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Detected Corners\nPattern: {pattern_size}')
    else:
        # Edge 검출로 체스보드 구조 확인
        edges = cv2.Canny(gray, 50, 150)
        axes[1, 1].imshow(edges, cmap='gray')
        axes[1, 1].set_title('Edge Detection\n(No corners found)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    output_dir = "chessboard_diagnosis"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"diagnosis_{os.path.basename(image_path).replace('.bmp', '.png')}")
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"\n진단 결과 저장: {output_file}")
    plt.close()
    
    return found_pattern

def diagnose_all_images():
    """
    모든 체스보드 이미지 진단
    """
    print("="*60)
    print("체스보드 이미지 진단 시작")
    print("="*60)
    
    cam1_images = sorted(glob.glob("chessboard_images/Cam1_*.bmp"))[:3]  # 처음 3장만
    cam2_images = sorted(glob.glob("chessboard_images/Cam2_*.bmp"))[:3]  # 처음 3장만
    
    print(f"\n[Camera 1 진단] - 샘플 {len(cam1_images)}장")
    cam1_results = []
    for img_path in cam1_images:
        result = diagnose_chessboard_image(img_path)
        cam1_results.append((img_path, result))
    
    print(f"\n[Camera 2 진단] - 샘플 {len(cam2_images)}장")
    cam2_results = []
    for img_path in cam2_images:
        result = diagnose_chessboard_image(img_path)
        cam2_results.append((img_path, result))
    
    # 요약
    print(f"\n{'='*60}")
    print("진단 요약")
    print(f"{'='*60}")
    
    cam1_success = sum(1 for _, r in cam1_results if r is not None)
    cam2_success = sum(1 for _, r in cam2_results if r is not None)
    
    print(f"Camera 1: {cam1_success}/{len(cam1_results)} 성공")
    print(f"Camera 2: {cam2_success}/{len(cam2_results)} 성공")
    
    if cam1_success == 0 and cam2_success == 0:
        print(f"\n❌ 문제 발견:")
        print(f"   - 체스보드 패턴 검출 완전 실패")
        print(f"   - 가능한 원인:")
        print(f"     1. 체스보드 패턴 크기가 다름 (9x6이 아님)")
        print(f"     2. 이미지 품질 문제 (blur, 저대비, 과다노출/과소노출)")
        print(f"     3. 체스보드가 이미지에 완전히 포함되지 않음")
        print(f"     4. 체스보드가 아닌 다른 물체가 촬영됨")
        print(f"\n   진단 이미지를 확인하세요: chessboard_diagnosis/ 폴더")
    
    # 성공한 패턴 크기 출력
    patterns_found = set()
    for _, result in cam1_results + cam2_results:
        if result:
            patterns_found.add(result[0])
    
    if patterns_found:
        print(f"\n✓ 발견된 패턴 크기: {patterns_found}")
        print(f"  → recalibrate_stereo.py의 pattern_size를 이 값으로 수정하세요")

if __name__ == "__main__":
    diagnose_all_images()
