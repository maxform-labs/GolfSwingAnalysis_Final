"""
실제 체스보드 크기로 재검출
11x8 또는 더 큰 패턴 우선 시도
"""

import cv2
import numpy as np
import glob
import os
from collections import Counter

def enhance_for_detection(img):
    """체스보드 검출을 위한 전처리"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    methods = {}
    methods['original'] = gray
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    methods['clahe'] = clahe.apply(gray)
    methods['hist_eq'] = cv2.equalizeHist(gray)
    
    # 밝기/대비 증가
    methods['enhanced'] = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
    
    # 감마 보정
    gamma = 2.0
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    methods['gamma'] = cv2.LUT(gray, table)
    
    return methods

def try_chessboard_detection(gray_img, preferred_patterns=None):
    """
    체스보드 검출 - 큰 패턴부터 시도
    """
    # 기본 큰 패턴들 (실제 보드 크기 추정)
    if preferred_patterns is None:
        preferred_patterns = [
            (11, 8), (10, 8), (11, 7), (10, 7),  # 매우 큰 패턴
            (9, 8), (9, 7), (8, 8), (8, 7),      # 큰 패턴
            (9, 6), (8, 6), (7, 6), (6, 6),      # 중간 패턴
            (8, 5), (7, 5), (6, 5), (5, 5),      # 작은 패턴
            (8, 4), (7, 4), (6, 4), (5, 4), (4, 4)  # 매우 작은 패턴
        ]
    
    flags_list = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        None,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
    ]
    
    for pattern in preferred_patterns:
        for flags in flags_list:
            try:
                if flags is None:
                    ret, corners = cv2.findChessboardCorners(gray_img, pattern, None)
                else:
                    ret, corners = cv2.findChessboardCorners(gray_img, pattern, flags=flags)
                
                if ret:
                    # 서브픽셀 정확도
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
                    return pattern, corners_refined, True
            except:
                continue
    
    return None, None, False

def process_all_images_with_large_pattern():
    """모든 이미지를 큰 패턴으로 재검출"""
    
    cam1_images = sorted(glob.glob("chessboard_images/Cam1_*.bmp"))
    cam2_images = sorted(glob.glob("chessboard_images/Cam2_*.bmp"))
    
    print("="*70)
    print("큰 패턴 우선 검출 (11×8, 10×8 등)")
    print("="*70)
    
    cam1_results = {}
    cam2_results = {}
    
    # Camera 1
    print("\nCamera 1:")
    for img_path in cam1_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        basename = os.path.basename(img_path)
        img_number = basename.split('_')[1].split('.')[0]
        
        enhanced = enhance_for_detection(img)
        
        found = False
        for method_name, gray in enhanced.items():
            pattern, corners, success = try_chessboard_detection(gray)
            if success:
                cam1_results[img_number] = {
                    'path': img_path,
                    'basename': basename,
                    'pattern': pattern,
                    'corners': corners,
                    'method': method_name,
                    'gray': gray
                }
                print(f"  ✓ {basename}: {pattern} ({method_name})")
                found = True
                break
        
        if not found:
            print(f"  ✗ {basename}: 실패")
    
    # Camera 2
    print("\nCamera 2:")
    for img_path in cam2_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        basename = os.path.basename(img_path)
        img_number = basename.split('_')[1].split('.')[0]
        
        enhanced = enhance_for_detection(img)
        
        found = False
        for method_name, gray in enhanced.items():
            pattern, corners, success = try_chessboard_detection(gray)
            if success:
                cam2_results[img_number] = {
                    'path': img_path,
                    'basename': basename,
                    'pattern': pattern,
                    'corners': corners,
                    'method': method_name,
                    'gray': gray
                }
                print(f"  ✓ {basename}: {pattern} ({method_name})")
                found = True
                break
        
        if not found:
            print(f"  ✗ {basename}: 실패")
    
    # 통계
    print("\n" + "="*70)
    print("검출 통계")
    print("="*70)
    
    print(f"\nCamera 1: {len(cam1_results)}/{len(cam1_images)} 성공 ({len(cam1_results)/len(cam1_images)*100:.1f}%)")
    cam1_patterns = [r['pattern'] for r in cam1_results.values()]
    if cam1_patterns:
        cam1_counter = Counter(cam1_patterns)
        for pattern, count in cam1_counter.most_common():
            print(f"  {pattern}: {count}개")
    
    print(f"\nCamera 2: {len(cam2_results)}/{len(cam2_images)} 성공 ({len(cam2_results)/len(cam2_images)*100:.1f}%)")
    cam2_patterns = [r['pattern'] for r in cam2_results.values()]
    if cam2_patterns:
        cam2_counter = Counter(cam2_patterns)
        for pattern, count in cam2_counter.most_common():
            print(f"  {pattern}: {count}개")
    
    # 스테레오 페어 찾기
    print("\n" + "="*70)
    print("스테레오 페어 분석")
    print("="*70)
    
    common_numbers = set(cam1_results.keys()) & set(cam2_results.keys())
    print(f"\n양쪽 카메라에서 모두 검출된 이미지 번호: {len(common_numbers)}개")
    
    if len(common_numbers) > 0:
        print("\n스테레오 페어별 패턴:")
        
        # 패턴별로 그룹화
        pattern_pairs = {}
        
        for num in sorted(common_numbers):
            cam1_pattern = cam1_results[num]['pattern']
            cam2_pattern = cam2_results[num]['pattern']
            
            print(f"  [{num}] Cam1: {cam1_pattern}, Cam2: {cam2_pattern}", end="")
            
            if cam1_pattern == cam2_pattern:
                print(" ✓ 같은 패턴!")
                pattern_key = cam1_pattern
                if pattern_key not in pattern_pairs:
                    pattern_pairs[pattern_key] = []
                pattern_pairs[pattern_key].append(num)
            else:
                print(" ✗ 다른 패턴")
        
        # 같은 패턴을 가진 페어들
        print("\n" + "="*70)
        print("같은 패턴의 스테레오 페어")
        print("="*70)
        
        if pattern_pairs:
            for pattern, numbers in sorted(pattern_pairs.items(), key=lambda x: len(x[1]), reverse=True):
                print(f"\n패턴 {pattern}: {len(numbers)}개 페어")
                for num in sorted(numbers):
                    print(f"  Cam1_{num}.bmp + Cam2_{num}.bmp")
                
                if len(numbers) >= 10:
                    print(f"\n✓✓✓ 충분합니다! 이 패턴으로 캘리브레이션 가능")
                    return pattern, numbers, cam1_results, cam2_results
                else:
                    print(f"  ⚠️ 부족 ({len(numbers)}/10)")
        else:
            print("\n✗ 같은 패턴의 페어가 하나도 없습니다")
        
        # 대안: 가장 많은 패턴 찾기
        print("\n" + "="*70)
        print("대안: 가장 많이 검출된 패턴으로 재시도")
        print("="*70)
        
        all_patterns = cam1_patterns + cam2_patterns
        overall_counter = Counter(all_patterns)
        most_common = overall_counter.most_common(3)
        
        print("\n전체 패턴 통계:")
        for pattern, count in most_common:
            cam1_count = sum(1 for p in cam1_patterns if p == pattern)
            cam2_count = sum(1 for p in cam2_patterns if p == pattern)
            min_count = min(cam1_count, cam2_count)
            print(f"  {pattern}: Cam1 {cam1_count}개, Cam2 {cam2_count}개 → 최소 {min_count}개 페어 가능")
            
            if min_count >= 10:
                print(f"    ✓ 이 패턴 사용 가능!")
                # 이 패턴을 가진 이미지들 찾기
                cam1_with_pattern = [num for num, r in cam1_results.items() if r['pattern'] == pattern]
                cam2_with_pattern = [num for num, r in cam2_results.items() if r['pattern'] == pattern]
                matching_pairs = list(set(cam1_with_pattern) & set(cam2_with_pattern))
                
                if len(matching_pairs) >= 10:
                    return pattern, matching_pairs, cam1_results, cam2_results
        
        print("\n⚠️ 충분한 페어를 찾을 수 없습니다")
        return None, [], cam1_results, cam2_results
    
    else:
        print("\n✗ 공통으로 검출된 이미지가 없습니다")
        return None, [], cam1_results, cam2_results

if __name__ == "__main__":
    best_pattern, pair_numbers, cam1_results, cam2_results = process_all_images_with_large_pattern()
    
    if best_pattern and len(pair_numbers) >= 10:
        print("\n" + "="*70)
        print("✓✓✓ 캘리브레이션 준비 완료!")
        print("="*70)
        print(f"\n추천 패턴: {best_pattern}")
        print(f"사용 가능한 스테레오 페어: {len(pair_numbers)}개")
        print(f"\n페어 목록:")
        for num in sorted(pair_numbers)[:15]:  # 처음 15개만 표시
            print(f"  Cam1_{num}.bmp + Cam2_{num}.bmp")
        if len(pair_numbers) > 15:
            print(f"  ... 외 {len(pair_numbers)-15}개")
        
        print(f"\n다음 명령 실행:")
        print(f"  1. recalibrate_stereo.py를 열어서")
        print(f"     pattern_size = {best_pattern}  # 이 값으로 수정")
        print(f"  2. python recalibrate_stereo.py")
    else:
        print("\n" + "="*70)
        print("⚠️ 캘리브레이션 불가")
        print("="*70)
        
        if best_pattern:
            print(f"\n패턴 {best_pattern}를 찾았으나 페어가 부족합니다 ({len(pair_numbers)}/10)")
        else:
            print("\n공통 패턴을 찾을 수 없습니다")
        
        print("\n가능한 해결책:")
        print("1. 더 많은 체스보드 이미지 촬영")
        print("2. 조명을 개선하여 재촬영")
        print("3. 체스보드를 더 정면으로 촬영")
