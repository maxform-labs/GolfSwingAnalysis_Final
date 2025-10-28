import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def enhance_image_for_chessboard(image):
    """체스보드 검출을 위한 이미지 향상"""
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 적응형 히스토그램 균등화 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. 감마 보정
    gamma = 2.5
    lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                            for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, lookup_table)
    
    # 4. 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)
    
    # 5. 적응형 임계값
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return enhanced, gamma_corrected, adaptive_thresh

def detect_chessboard_enhanced(image_path, pattern_size=(9, 6)):
    """향상된 체스보드 검출"""
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None
    
    # 이미지 향상
    enhanced, gamma_corrected, adaptive_thresh = enhance_image_for_chessboard(image)
    
    # 다양한 방법으로 체스보드 검출 시도
    methods = [
        # 방법 1: 향상된 이미지로 검출
        (enhanced, "Enhanced"),
        # 방법 2: 감마 보정된 이미지로 검출
        (gamma_corrected, "Gamma Corrected"),
        # 방법 3: 적응형 임계값 이미지로 검출
        (adaptive_thresh, "Adaptive Threshold"),
        # 방법 4: 원본 그레이스케일
        (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), "Original Gray")
    ]
    
    for processed_img, method_name in methods:
        print(f"\n{method_name} 방법으로 검출 시도:")
        
        # 체스보드 검출
        ret, corners = cv2.findChessboardCorners(
            processed_img, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK
        )
        
        if ret:
            print(f"  ✅ {method_name}에서 성공!")
            
            # 서브픽셀 정확도
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(processed_img, corners, (11, 11), (-1, -1), criteria)
            
            # 결과 시각화
            result_img = image.copy()
            cv2.drawChessboardCorners(result_img, pattern_size, corners, ret)
            
            return ret, corners, result_img, processed_img, method_name
        else:
            print(f"  ❌ {method_name}에서 실패")
    
    return None, None, None, None, None

def debug_enhanced_chessboard():
    """향상된 체스보드 검출 디버깅"""
    calibration_path = "data2/calibrations"
    
    # 첫 번째 이미지 쌍 로드
    cam1_path = os.path.join(calibration_path, "Cam1_1.bmp")
    cam2_path = os.path.join(calibration_path, "Cam2_1.bmp")
    
    if not os.path.exists(cam1_path) or not os.path.exists(cam2_path):
        print("이미지 파일을 찾을 수 없습니다.")
        return
    
    print("=== 향상된 체스보드 검출 디버깅 ===")
    
    # Cam1 검출
    print("\n--- Cam1 검출 ---")
    ret1, corners1, result1, processed1, method1 = detect_chessboard_enhanced(cam1_path)
    
    # Cam2 검출
    print("\n--- Cam2 검출 ---")
    ret2, corners2, result2, processed2, method2 = detect_chessboard_enhanced(cam2_path)
    
    if ret1 and ret2:
        print(f"\n✅ 성공! Cam1: {method1}, Cam2: {method2}")
        
        # 결과 저장
        cv2.imwrite('enhanced_chessboard_cam1.jpg', result1)
        cv2.imwrite('enhanced_chessboard_cam2.jpg', result2)
        cv2.imwrite('processed_cam1.jpg', processed1)
        cv2.imwrite('processed_cam2.jpg', processed2)
        
        print("결과 이미지 저장 완료:")
        print("- enhanced_chessboard_cam1.jpg")
        print("- enhanced_chessboard_cam2.jpg")
        print("- processed_cam1.jpg")
        print("- processed_cam2.jpg")
        
        # 전처리 과정 시각화
        visualize_preprocessing(cam1_path, "Cam1")
        visualize_preprocessing(cam2_path, "Cam2")
        
    else:
        print("\n❌ 모든 방법에서 실패했습니다.")
        
        # 실패 원인 분석
        analyze_failure_reasons(cam1_path, cam2_path)

def visualize_preprocessing(image_path, camera_name):
    """전처리 과정 시각화"""
    image = cv2.imread(image_path)
    enhanced, gamma_corrected, adaptive_thresh = enhance_image_for_chessboard(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 원본 이미지
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'{camera_name} - Original')
    axes[0, 0].axis('off')
    
    # 그레이스케일
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title(f'{camera_name} - Grayscale')
    axes[0, 1].axis('off')
    
    # 향상된 이미지
    axes[0, 2].imshow(enhanced, cmap='gray')
    axes[0, 2].set_title(f'{camera_name} - Enhanced (CLAHE)')
    axes[0, 2].axis('off')
    
    # 감마 보정
    axes[1, 0].imshow(gamma_corrected, cmap='gray')
    axes[1, 0].set_title(f'{camera_name} - Gamma Corrected')
    axes[1, 0].axis('off')
    
    # 적응형 임계값
    axes[1, 1].imshow(adaptive_thresh, cmap='gray')
    axes[1, 1].set_title(f'{camera_name} - Adaptive Threshold')
    axes[1, 1].axis('off')
    
    # 히스토그램
    axes[1, 2].hist(gray.ravel(), 256, [0, 256], color='blue', alpha=0.7, label='Original')
    axes[1, 2].hist(enhanced.ravel(), 256, [0, 256], color='red', alpha=0.7, label='Enhanced')
    axes[1, 2].set_title(f'{camera_name} - Histogram')
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(f'preprocessing_{camera_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_failure_reasons(cam1_path, cam2_path):
    """실패 원인 분석"""
    print("\n=== 실패 원인 분석 ===")
    
    for path, name in [(cam1_path, "Cam1"), (cam2_path, "Cam2")]:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print(f"\n{name} 분석:")
        print(f"  이미지 크기: {image.shape}")
        print(f"  평균 밝기: {np.mean(gray):.2f}")
        print(f"  표준편차: {np.std(gray):.2f}")
        print(f"  최소값: {np.min(gray)}")
        print(f"  최대값: {np.max(gray)}")
        
        # 대비 분석
        contrast = np.std(gray)
        print(f"  대비 (표준편차): {contrast:.2f}")
        
        if contrast < 20:
            print("  ⚠️ 대비가 너무 낮습니다.")
        if np.mean(gray) < 50:
            print("  ⚠️ 이미지가 너무 어둡습니다.")
        if np.mean(gray) > 200:
            print("  ⚠️ 이미지가 너무 밝습니다.")

if __name__ == "__main__":
    debug_enhanced_chessboard()

