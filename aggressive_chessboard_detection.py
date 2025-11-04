"""
체스보드 검출 - 공격적인 방법들 시도
육안으로 보이는 체스보드를 반드시 검출하기
"""

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def enhance_image_for_chessboard(img):
    """
    체스보드 검출을 위한 이미지 전처리
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    enhanced_methods = {}
    
    # 1. 원본
    enhanced_methods['original'] = gray
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_methods['clahe'] = clahe.apply(gray)
    
    # 3. 히스토그램 균등화
    enhanced_methods['hist_eq'] = cv2.equalizeHist(gray)
    
    # 4. 밝기 증가 + 대비 증가
    alpha = 2.0  # 대비
    beta = 50    # 밝기
    enhanced_methods['brightness'] = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # 5. 감마 보정 (어두운 이미지를 밝게)
    gamma = 2.0
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    enhanced_methods['gamma'] = cv2.LUT(gray, table)
    
    # 6. Bilateral 필터 + CLAHE
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    enhanced_methods['bilateral_clahe'] = clahe.apply(bilateral)
    
    # 7. 극단적 밝기 증가
    enhanced_methods['ultra_bright'] = cv2.convertScaleAbs(gray, alpha=3.0, beta=100)
    
    return enhanced_methods

def try_all_chessboard_patterns(gray_img, image_name):
    """
    다양한 체스보드 패턴 크기 시도
    """
    # 시도할 패턴들 (내부 코너 개수)
    patterns = [
        (10, 7), (9, 7), (8, 7), (7, 7),
        (10, 6), (9, 6), (8, 6), (7, 6), (6, 6),
        (10, 5), (9, 5), (8, 5), (7, 5), (6, 5), (5, 5),
        (9, 4), (8, 4), (7, 4), (6, 4),
        (7, 9), (6, 9), (6, 8), (5, 8), (5, 7)
    ]
    
    # 여러 플래그 조합 시도
    flag_combinations = [
        None,  # 기본
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
    ]
    
    print(f"\n{'='*70}")
    print(f"이미지: {image_name}")
    print(f"{'='*70}")
    
    for pattern in patterns:
        for flags in flag_combinations:
            if flags is None:
                ret, corners = cv2.findChessboardCorners(gray_img, pattern, None)
            else:
                ret, corners = cv2.findChessboardCorners(gray_img, pattern, flags=flags)
            
            if ret:
                flag_name = "DEFAULT" if flags is None else f"FLAGS_{flags}"
                print(f"✓✓✓ 성공! 패턴: {pattern}, 플래그: {flag_name}")
                return pattern, corners, True
    
    return None, None, False

def aggressive_chessboard_detection(image_path, output_dir="chessboard_results"):
    """
    공격적인 체스보드 검출 시도
    """
    os.makedirs(output_dir, exist_ok=True)
    
    basename = os.path.basename(image_path)
    print(f"\n{'='*70}")
    print(f"처리 중: {basename}")
    print(f"{'='*70}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지 로드 실패")
        return None
    
    # 다양한 전처리 방법 적용
    enhanced_images = enhance_image_for_chessboard(img)
    
    print(f"\n총 {len(enhanced_images)}가지 전처리 방법 시도 중...")
    
    best_result = None
    
    for method_name, enhanced_gray in enhanced_images.items():
        print(f"\n[{method_name}] 시도 중...", end=" ")
        
        pattern, corners, success = try_all_chessboard_patterns(enhanced_gray, method_name)
        
        if success:
            print(f"✓✓✓ 성공!")
            
            # 서브픽셀 정확도로 코너 개선
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(enhanced_gray, corners, (11, 11), (-1, -1), criteria)
            
            # 시각화
            img_with_corners = img.copy()
            cv2.drawChessboardCorners(img_with_corners, pattern, corners_refined, True)
            
            # 저장
            output_file = os.path.join(output_dir, f"detected_{basename.replace('.bmp', '.png')}")
            cv2.imwrite(output_file, img_with_corners)
            
            # 결과 시각화
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(enhanced_gray, cmap='gray')
            axes[1].set_title(f'Enhanced: {method_name}')
            axes[1].axis('off')
            
            axes[2].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
            axes[2].set_title(f'Detected Pattern: {pattern}')
            axes[2].axis('off')
            
            plt.tight_layout()
            viz_file = os.path.join(output_dir, f"viz_{basename.replace('.bmp', '.png')}")
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   저장: {output_file}")
            print(f"   시각화: {viz_file}")
            
            best_result = {
                'image_path': image_path,
                'pattern': pattern,
                'corners': corners_refined,
                'method': method_name,
                'enhanced_gray': enhanced_gray
            }
            
            break  # 성공하면 더 이상 시도하지 않음
        else:
            print(f"✗ 실패")
    
    if best_result is None:
        print(f"\n❌ 모든 방법 실패: {basename}")
        
        # 실패한 경우 진단 이미지 저장
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Failed Detection: {basename}', fontsize=16)
        
        methods = list(enhanced_images.items())
        for idx, (name, img_enhanced) in enumerate(methods[:8]):
            row = idx // 4
            col = idx % 4
            axes[row, col].imshow(img_enhanced, cmap='gray')
            axes[row, col].set_title(name)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        fail_file = os.path.join(output_dir, f"failed_{basename.replace('.bmp', '.png')}")
        plt.savefig(fail_file, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"   실패 진단 저장: {fail_file}")
    
    return best_result

def process_all_images(pattern="chessboard_images/Cam*.bmp", max_images=5):
    """
    모든 이미지 처리
    """
    print("="*70)
    print("공격적인 체스보드 검출 시작")
    print("="*70)
    
    images = sorted(glob.glob(pattern))[:max_images]
    
    if len(images) == 0:
        print(f"❌ 이미지를 찾을 수 없습니다: {pattern}")
        return
    
    print(f"\n처리할 이미지: {len(images)}개")
    
    results = []
    for img_path in images:
        result = aggressive_chessboard_detection(img_path)
        if result:
            results.append(result)
    
    # 요약
    print(f"\n{'='*70}")
    print("검출 요약")
    print(f"{'='*70}")
    print(f"전체: {len(images)}개")
    print(f"성공: {len(results)}개 ({len(results)/len(images)*100:.1f}%)")
    print(f"실패: {len(images)-len(results)}개")
    
    if len(results) > 0:
        print(f"\n✓ 성공한 이미지:")
        patterns_found = {}
        for r in results:
            print(f"   {os.path.basename(r['image_path'])}: 패턴 {r['pattern']}, 방법: {r['method']}")
            pattern_key = r['pattern']
            patterns_found[pattern_key] = patterns_found.get(pattern_key, 0) + 1
        
        print(f"\n패턴 통계:")
        for pattern, count in sorted(patterns_found.items(), key=lambda x: x[1], reverse=True):
            print(f"   {pattern}: {count}개 이미지")
        
        # 가장 많이 발견된 패턴
        most_common_pattern = max(patterns_found, key=patterns_found.get)
        print(f"\n✓✓✓ 가장 많이 발견된 패턴: {most_common_pattern}")
        print(f"   → recalibrate_stereo.py의 pattern_size를 {most_common_pattern}로 수정하세요")
        
        return results, most_common_pattern
    else:
        print(f"\n❌ 모든 이미지에서 검출 실패")
        print(f"   가능한 원인:")
        print(f"   1. 체스보드가 아닌 다른 패턴")
        print(f"   2. 체스보드가 너무 작거나 일부만 보임")
        print(f"   3. 심각한 블러 또는 왜곡")
        print(f"   4. 체스보드 격자가 불균일")
        print(f"\n   chessboard_results/failed_*.png 파일들을 확인하세요")
        
        return None, None

def test_specific_image(image_path):
    """
    특정 이미지 하나만 집중 테스트
    """
    print("="*70)
    print("단일 이미지 집중 테스트")
    print("="*70)
    
    result = aggressive_chessboard_detection(image_path, output_dir="chessboard_test")
    
    if result:
        print(f"\n✓✓✓ 성공!")
        print(f"   패턴: {result['pattern']}")
        print(f"   방법: {result['method']}")
        print(f"   코너 개수: {len(result['corners'])}")
    else:
        print(f"\n❌ 실패")
    
    return result

if __name__ == "__main__":
    # 먼저 Camera1, Camera2 각각 몇 장씩 테스트
    print("\n[Camera 1 테스트]")
    cam1_results, cam1_pattern = process_all_images("chessboard_images/Cam1_*.bmp", max_images=5)
    
    print("\n\n[Camera 2 테스트]")
    cam2_results, cam2_pattern = process_all_images("chessboard_images/Cam2_*.bmp", max_images=5)
    
    # 결과 요약
    print(f"\n{'='*70}")
    print("최종 요약")
    print(f"{'='*70}")
    
    if cam1_results and cam2_results:
        print(f"✓✓✓ 양쪽 카메라 모두 검출 성공!")
        print(f"\nCamera1 패턴: {cam1_pattern}")
        print(f"Camera2 패턴: {cam2_pattern}")
        
        if cam1_pattern == cam2_pattern:
            print(f"\n✓ 두 카메라가 같은 패턴 사용: {cam1_pattern}")
            print(f"  → recalibrate_stereo.py 실행 가능")
        else:
            print(f"\n⚠️ 두 카메라가 다른 패턴:")
            print(f"  Camera1: {cam1_pattern}")
            print(f"  Camera2: {cam2_pattern}")
            print(f"  → 가장 많이 발견된 패턴으로 재시도 필요")
    elif cam1_results or cam2_results:
        print(f"⚠️ 한쪽만 성공")
        if cam1_results:
            print(f"   Camera1: 성공 (패턴 {cam1_pattern})")
            print(f"   Camera2: 실패")
        else:
            print(f"   Camera1: 실패")
            print(f"   Camera2: 성공 (패턴 {cam2_pattern})")
    else:
        print(f"❌ 양쪽 모두 실패")
        print(f"\n육안으로 보인다고 하셨는데 검출이 안 됩니다.")
        print(f"다음을 확인해주세요:")
        print(f"1. chessboard_results/failed_*.png 파일 확인")
        print(f"2. 실제 체스보드 이미지 확인 (혹시 다른 패턴?)")
        print(f"3. 한 장을 선택해서 test_specific_image() 실행")
