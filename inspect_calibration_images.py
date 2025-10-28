import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def inspect_calibration_images():
    """캘리브레이션 이미지 직접 검사"""
    calibration_path = "data2/calibrations"
    
    # 첫 번째 이미지 로드
    cam1_path = os.path.join(calibration_path, "Cam1_1.bmp")
    cam2_path = os.path.join(calibration_path, "Cam2_1.bmp")
    
    if not os.path.exists(cam1_path) or not os.path.exists(cam2_path):
        print("이미지 파일을 찾을 수 없습니다.")
        return
    
    # 이미지 로드
    img1 = cv2.imread(cam1_path)
    img2 = cv2.imread(cam2_path)
    
    print(f"이미지 크기: {img1.shape}")
    
    # 이미지 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 원본 이미지
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Cam1 Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Cam2 Original')
    axes[0, 1].axis('off')
    
    # 그레이스케일
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    axes[0, 2].imshow(gray1, cmap='gray')
    axes[0, 2].set_title('Cam1 Grayscale')
    axes[0, 2].axis('off')
    
    # 히스토그램
    axes[1, 0].hist(gray1.ravel(), 256, [0, 256], color='blue', alpha=0.7, label='Cam1')
    axes[1, 0].hist(gray2.ravel(), 256, [0, 256], color='red', alpha=0.7, label='Cam2')
    axes[1, 0].set_title('Histogram')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # 대비 분석
    contrast1 = np.std(gray1)
    contrast2 = np.std(gray2)
    axes[1, 1].bar(['Cam1', 'Cam2'], [contrast1, contrast2], color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_title('Contrast Analysis')
    axes[1, 1].set_ylabel('Standard Deviation')
    
    # 밝기 분석
    brightness1 = np.mean(gray1)
    brightness2 = np.mean(gray2)
    axes[1, 2].bar(['Cam1', 'Cam2'], [brightness1, brightness2], color=['blue', 'red'], alpha=0.7)
    axes[1, 2].set_title('Brightness Analysis')
    axes[1, 2].set_ylabel('Mean Pixel Value')
    
    plt.tight_layout()
    plt.savefig('calibration_image_inspection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 이미지 품질 분석
    print(f"\n이미지 품질 분석:")
    print(f"Cam1 - 평균 밝기: {brightness1:.2f}, 대비: {contrast1:.2f}")
    print(f"Cam2 - 평균 밝기: {brightness2:.2f}, 대비: {contrast2:.2f}")
    
    # 체스보드가 실제로 보이는지 확인
    print(f"\n체스보드 패턴 확인:")
    print("이미지에서 체스보드 패턴이 명확하게 보이나요?")
    print("체스보드의 크기와 위치를 확인해보세요.")
    
    # ROI 추출 (이미지 중앙 부분)
    h, w = gray1.shape
    roi1 = gray1[h//4:3*h//4, w//4:3*w//4]
    roi2 = gray2[h//4:3*h//4, w//4:3*w//4]
    
    print(f"\nROI 분석 (이미지 중앙 50%):")
    print(f"Cam1 ROI - 평균: {np.mean(roi1):.2f}, 대비: {np.std(roi1):.2f}")
    print(f"Cam2 ROI - 평균: {np.mean(roi2):.2f}, 대비: {np.std(roi2):.2f}")
    
    # ROI 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(roi1, cmap='gray')
    axes[0].set_title('Cam1 ROI')
    axes[0].axis('off')
    
    axes[1].imshow(roi2, cmap='gray')
    axes[1].set_title('Cam2 ROI')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('calibration_roi_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_manual_corner_detection():
    """수동 코너 검출 테스트"""
    calibration_path = "data2/calibrations"
    
    cam1_path = os.path.join(calibration_path, "Cam1_1.bmp")
    cam2_path = os.path.join(calibration_path, "Cam2_1.bmp")
    
    img1 = cv2.imread(cam1_path)
    img2 = cv2.imread(cam2_path)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 극강 향상
    enhanced1 = ultra_enhance(gray1)
    enhanced2 = ultra_enhance(gray2)
    
    # 코너 검출
    corners1 = cv2.goodFeaturesToTrack(enhanced1, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners2 = cv2.goodFeaturesToTrack(enhanced2, maxCorners=100, qualityLevel=0.01, minDistance=10)
    
    print(f"\n수동 코너 검출:")
    print(f"Cam1에서 {len(corners1) if corners1 is not None else 0}개 코너 검출")
    print(f"Cam2에서 {len(corners2) if corners2 is not None else 0}개 코너 검출")
    
    if corners1 is not None and corners2 is not None:
        # 결과 시각화
        result1 = img1.copy()
        result2 = img2.copy()
        
        for corner in corners1:
            x, y = corner.ravel()
            cv2.circle(result1, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        for corner in corners2:
            x, y = corner.ravel()
            cv2.circle(result2, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        cv2.imwrite('manual_corners_cam1.jpg', result1)
        cv2.imwrite('manual_corners_cam2.jpg', result2)
        
        print("수동 코너 검출 결과 저장 완료")

def ultra_enhance(image):
    """초강력 향상"""
    # 감마 보정
    gamma = 0.3
    lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                            for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(image, lookup_table)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
    clahe_enhanced = clahe.apply(gamma_corrected)
    
    # 히스토그램 균등화
    hist_eq = cv2.equalizeHist(clahe_enhanced)
    
    # 언샤프 마스킹
    gaussian = cv2.GaussianBlur(hist_eq, (0, 0), 2.0)
    unsharp_mask = cv2.addWeighted(hist_eq, 1.5, gaussian, -0.5, 0)
    
    # 노이즈 제거
    denoised = cv2.bilateralFilter(unsharp_mask, 9, 75, 75)
    
    return denoised

if __name__ == "__main__":
    inspect_calibration_images()
    test_manual_corner_detection()