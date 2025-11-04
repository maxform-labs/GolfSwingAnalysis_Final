"""
모든 체스보드 이미지의 패턴 분석
어떤 패턴이 가장 일관성 있게 나타나는지 확인
"""

import cv2
import numpy as np
import glob
import os
from collections import Counter

def enhance_image_for_chessboard(img):
    """체스보드 검출을 위한 이미지 전처리"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    methods = {}
    methods['original'] = gray
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    methods['clahe'] = clahe.apply(gray)
    methods['hist_eq'] = cv2.equalizeHist(gray)
    methods['brightness'] = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
    
    gamma = 2.0
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    methods['gamma'] = cv2.LUT(gray, table)
    
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    methods['bilateral_clahe'] = clahe.apply(bilateral)
    
    return methods

def try_all_patterns(gray_img):
    """다양한 체스보드 패턴 시도"""
    patterns = [
        (10, 7), (9, 7), (8, 7), (7, 7),
        (10, 6), (9, 6), (8, 6), (7, 6), (6, 6),
        (10, 5), (9, 5), (8, 5), (7, 5), (6, 5), (5, 5),
        (9, 4), (8, 4), (7, 4), (6, 4),
        (7, 9), (6, 9), (6, 8), (5, 8), (5, 7), (5, 6), (4, 4)
    ]
    
    flags_list = [
        None,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
    ]
    
    for pattern in patterns:
        for flags in flags_list:
            if flags is None:
                ret, corners = cv2.findChessboardCorners(gray_img, pattern, None)
            else:
                ret, corners = cv2.findChessboardCorners(gray_img, pattern, flags=flags)
            
            if ret:
                return pattern, corners, True
    
    return None, None, False

def analyze_single_image(image_path):
    """단일 이미지 분석"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    enhanced_images = enhance_image_for_chessboard(img)
    
    for method_name, enhanced_gray in enhanced_images.items():
        pattern, corners, success = try_all_patterns(enhanced_gray)
        if success:
            return {
                'path': image_path,
                'basename': os.path.basename(image_path),
                'pattern': pattern,
                'corners': corners,
                'method': method_name,
                'corner_count': len(corners)
            }
    
    return None

def analyze_all_images():
    """모든 이미지 분석"""
    cam1_images = sorted(glob.glob("chessboard_images/Cam1_*.bmp"))
    cam2_images = sorted(glob.glob("chessboard_images/Cam2_*.bmp"))
    
    print("="*70)
    print("전체 체스보드 이미지 패턴 분석")
    print("="*70)
    print(f"Camera1 이미지: {len(cam1_images)}개")
    print(f"Camera2 이미지: {len(cam2_images)}개")
    print()
    
    # Camera 1 분석
    print("Camera 1 분석 중...")
    cam1_results = []
    cam1_patterns = []
    
    for img_path in cam1_images:
        result = analyze_single_image(img_path)
        if result:
            cam1_results.append(result)
            cam1_patterns.append(result['pattern'])
            print(f"  ✓ {result['basename']}: {result['pattern']} ({result['method']})")
        else:
            print(f"  ✗ {os.path.basename(img_path)}: 실패")
    
    # Camera 2 분석
    print("\nCamera 2 분석 중...")
    cam2_results = []
    cam2_patterns = []
    
    for img_path in cam2_images:
        result = analyze_single_image(img_path)
        if result:
            cam2_results.append(result)
            cam2_patterns.append(result['pattern'])
            print(f"  ✓ {result['basename']}: {result['pattern']} ({result['method']})")
        else:
            print(f"  ✗ {os.path.basename(img_path)}: 실패")
    
    # 통계 분석
    print("\n" + "="*70)
    print("통계 분석")
    print("="*70)
    
    print(f"\nCamera 1:")
    print(f"  성공: {len(cam1_results)}/{len(cam1_images)} ({len(cam1_results)/len(cam1_images)*100:.1f}%)")
    if cam1_patterns:
        cam1_counter = Counter(cam1_patterns)
        print(f"  패턴 분포:")
        for pattern, count in cam1_counter.most_common():
            print(f"    {pattern}: {count}개 ({count/len(cam1_patterns)*100:.1f}%)")
        cam1_most_common = cam1_counter.most_common(1)[0]
        print(f"  ✓ 가장 많은 패턴: {cam1_most_common[0]} ({cam1_most_common[1]}개)")
    
    print(f"\nCamera 2:")
    print(f"  성공: {len(cam2_results)}/{len(cam2_images)} ({len(cam2_results)/len(cam2_images)*100:.1f}%)")
    if cam2_patterns:
        cam2_counter = Counter(cam2_patterns)
        print(f"  패턴 분포:")
        for pattern, count in cam2_counter.most_common():
            print(f"    {pattern}: {count}개 ({count/len(cam2_patterns)*100:.1f}%)")
        cam2_most_common = cam2_counter.most_common(1)[0]
        print(f"  ✓ 가장 많은 패턴: {cam2_most_common[0]} ({cam2_most_common[1]}개)")
    
    # 공통 패턴 찾기
    print("\n" + "="*70)
    print("공통 패턴 분석")
    print("="*70)
    
    if cam1_patterns and cam2_patterns:
        common_patterns = set(cam1_patterns) & set(cam2_patterns)
        
        if common_patterns:
            print(f"✓ 양쪽 카메라에서 발견된 공통 패턴: {len(common_patterns)}개")
            
            # 공통 패턴 중 가장 많이 나타나는 것
            common_counts = {}
            for pattern in common_patterns:
                count1 = cam1_counter.get(pattern, 0)
                count2 = cam2_counter.get(pattern, 0)
                common_counts[pattern] = min(count1, count2)  # 최소값 사용
                print(f"  {pattern}: Cam1 {count1}개, Cam2 {count2}개")
            
            # 가장 좋은 패턴 추천
            best_pattern = max(common_counts, key=common_counts.get)
            print(f"\n✓✓✓ 추천 패턴: {best_pattern}")
            print(f"    이유: 양쪽 카메라에서 모두 검출되며, 가장 많은 이미지에서 발견됨")
            
            # 이 패턴을 사용한 이미지 페어 찾기
            cam1_with_pattern = [r for r in cam1_results if r['pattern'] == best_pattern]
            cam2_with_pattern = [r for r in cam2_results if r['pattern'] == best_pattern]
            
            print(f"\n이 패턴을 사용한 이미지:")
            print(f"  Camera1: {len(cam1_with_pattern)}개")
            for r in cam1_with_pattern:
                print(f"    - {r['basename']}")
            print(f"  Camera2: {len(cam2_with_pattern)}개")
            for r in cam2_with_pattern:
                print(f"    - {r['basename']}")
            
            # 스테레오 페어 찾기
            cam1_numbers = {r['basename'].split('_')[1].split('.')[0]: r for r in cam1_with_pattern}
            cam2_numbers = {r['basename'].split('_')[1].split('.')[0]: r for r in cam2_with_pattern}
            
            common_numbers = set(cam1_numbers.keys()) & set(cam2_numbers.keys())
            
            print(f"\n✓ 스테레오 페어 (양쪽 카메라 모두 {best_pattern} 패턴): {len(common_numbers)}개")
            for num in sorted(common_numbers):
                print(f"    Cam1_{num}.bmp + Cam2_{num}.bmp")
            
            if len(common_numbers) >= 10:
                print(f"\n✓✓✓ 충분한 스테레오 페어가 있습니다!")
                print(f"    recalibrate_stereo.py를 pattern_size={best_pattern}로 실행하세요")
                return best_pattern, len(common_numbers)
            else:
                print(f"\n⚠️ 스테레오 페어가 부족합니다 ({len(common_numbers)}/10)")
                print(f"   다른 패턴도 고려해야 합니다")
        else:
            print("✗ 공통 패턴이 없습니다")
            print(f"  Camera1 패턴: {set(cam1_patterns)}")
            print(f"  Camera2 패턴: {set(cam2_patterns)}")
    
    # 대안: 가장 많이 검출된 패턴 사용
    print("\n" + "="*70)
    print("대안: 가장 많이 검출된 패턴")
    print("="*70)
    
    if cam1_patterns and cam2_patterns:
        all_patterns = cam1_patterns + cam2_patterns
        all_counter = Counter(all_patterns)
        most_common_overall = all_counter.most_common(1)[0]
        
        print(f"전체에서 가장 많은 패턴: {most_common_overall[0]} ({most_common_overall[1]}개)")
        
        cam1_with = len([p for p in cam1_patterns if p == most_common_overall[0]])
        cam2_with = len([p for p in cam2_patterns if p == most_common_overall[0]])
        
        print(f"  Camera1: {cam1_with}개")
        print(f"  Camera2: {cam2_with}개")
        
        if min(cam1_with, cam2_with) >= 10:
            print(f"\n✓ 이 패턴으로 충분한 페어를 만들 수 있습니다")
            return most_common_overall[0], min(cam1_with, cam2_with)
        else:
            print(f"\n⚠️ 이 패턴으로는 충분한 페어가 안 됩니다")
    
    return None, 0

if __name__ == "__main__":
    best_pattern, pair_count = analyze_all_images()
    
    if best_pattern:
        print(f"\n{'='*70}")
        print("최종 결론")
        print(f"{'='*70}")
        print(f"✓✓✓ 추천 패턴: {best_pattern}")
        print(f"✓✓✓ 사용 가능한 페어: {pair_count}개")
        print(f"\n다음 단계:")
        print(f"  1. recalibrate_stereo.py 수정:")
        print(f"     pattern_size = {best_pattern}")
        print(f"  2. python recalibrate_stereo.py 실행")
    else:
        print(f"\n{'='*70}")
        print("⚠️ 캘리브레이션 불가")
        print(f"{'='*70}")
        print(f"충분한 스테레오 페어를 찾을 수 없습니다")
        print(f"chessboard_results/ 폴더의 이미지들을 확인하세요")
