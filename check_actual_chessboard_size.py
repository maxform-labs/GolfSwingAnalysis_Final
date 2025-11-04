"""
체스보드 실제 크기 확인 및 전체 보드 검출 시도
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 가장 밝고 검출이 잘 된 이미지들 확인
test_images = [
    "chessboard_images/Cam1_10.bmp",
    "chessboard_images/Cam1_11.bmp", 
    "chessboard_images/Cam2_10.bmp",
    "chessboard_images/Cam2_11.bmp"
]

# 실제 체스보드 크기를 추정해보기 위한 큰 패턴들
large_patterns = [
    (11, 8), (10, 8), (9, 8), (8, 8),
    (11, 7), (10, 7), (9, 7), (8, 7),
    (11, 6), (10, 6), (9, 6),
]

print("="*70)
print("실제 체스보드 크기 확인")
print("="*70)
print("\n큰 패턴들을 시도해서 전체 보드를 찾아봅니다...")

for img_path in test_images:
    print(f"\n{img_path}:")
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    
    found_patterns = []
    
    for pattern in large_patterns:
        # 원본
        ret1, corners1 = cv2.findChessboardCorners(gray, pattern, 
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret1:
            found_patterns.append(('original', pattern, len(corners1)))
        
        # CLAHE
        ret2, corners2 = cv2.findChessboardCorners(gray_clahe, pattern,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret2:
            found_patterns.append(('clahe', pattern, len(corners2)))
    
    if found_patterns:
        print(f"  발견된 큰 패턴들:")
        for method, pattern, corners in found_patterns:
            print(f"    {pattern} ({method}): {corners}개 코너")
        
        # 가장 큰 패턴
        largest = max(found_patterns, key=lambda x: x[2])
        print(f"  ✓ 가장 큰 패턴: {largest[1]} ({largest[0]})")
    else:
        print(f"  ✗ 큰 패턴 검출 실패")

print("\n" + "="*70)
print("이미지를 직접 확인합니다")
print("="*70)
print("\n다음 이미지들을 열어서 육안으로 체스보드 크기를 세어주세요:")
print("1. chessboard_results/viz_Cam1_10.png")
print("2. chessboard_results/viz_Cam1_11.png")
print("3. chessboard_results/viz_Cam2_10.png")
print("4. chessboard_results/viz_Cam2_11.png")
print("\n체스보드의 '내부 코너'(검은 사각형이 만나는 점) 개수를 세어주세요")
print("예: 9x6 = 가로 9개, 세로 6개의 내부 코너")

# 원본 이미지 하나를 크게 표시
img = cv2.imread("chessboard_images/Cam1_11.bmp")
if img is not None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 원본
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image - Cam1_11.bmp\n육안으로 내부 코너 개수를 세어주세요', 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[0].grid(True, alpha=0.3)
    
    # CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    axes[1].imshow(gray_clahe, cmap='gray')
    axes[1].set_title('Enhanced (CLAHE)\n체스보드 패턴이 더 명확하게 보입니다', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chessboard_size_check.png', dpi=150, bbox_inches='tight')
    print("\n✓ 저장: chessboard_size_check.png")
    print("  이 이미지를 열어서 체스보드 크기를 확인하세요")
    plt.close()

# 추가: 여러 이미지에서 발견된 모든 패턴 정리
print("\n" + "="*70)
print("지금까지 발견된 모든 패턴")
print("="*70)

from collections import Counter

all_patterns = [
    (8, 4), (8, 5), (8, 6), (6, 5), (4, 4), (7, 6), (6, 4),  # Cam1
    (6, 4), (8, 7), (6, 6), (9, 4), (6, 5), (8, 5), (8, 6), (9, 5), (9, 6), (10, 6)  # Cam2
]

counter = Counter(all_patterns)
print("\n패턴 빈도:")
for pattern, count in counter.most_common():
    area = pattern[0] * pattern[1]
    print(f"  {pattern}: {count}회, 코너수 {area}")

print("\n✓ 패턴이 다양하다는 것은:")
print("  → 이미지마다 체스보드의 다른 부분/각도가 보인다")
print("  → 또는 체스보드의 일부만 프레임에 들어온다")
print("  → 실제 전체 체스보드는 더 클 가능성이 높다")

print("\n다음 단계:")
print("1. chessboard_size_check.png를 열어서 실제 체스보드 크기 확인")
print("2. 크기를 알려주시면, 그 크기로 강제 검출 시도")
print("3. 또는 '부분 보드 허용' 모드로 캘리브레이션 진행")
