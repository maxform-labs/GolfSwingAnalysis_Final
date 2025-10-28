#!/usr/bin/env python3
"""
볼 검출 테스트 스크립트
실제 이미지에서 볼 검출이 가능한지 확인
"""

import cv2
import numpy as np
import os
import glob

def test_ball_detection(image_path):
    """단일 이미지에서 볼 검출 테스트"""
    print(f"Testing: {image_path}")
    
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print("  Failed to load image")
        return False
    
    print(f"  Image size: {img.shape}")
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 다양한 파라미터로 테스트
    test_params = [
        {"name": "Very Conservative", "param1": 100, "param2": 80, "minRadius": 15, "maxRadius": 40},
        {"name": "Conservative", "param1": 80, "param2": 50, "minRadius": 10, "maxRadius": 30},
        {"name": "Moderate", "param1": 50, "param2": 30, "minRadius": 8, "maxRadius": 40},
        {"name": "Sensitive", "param1": 30, "param2": 20, "minRadius": 5, "maxRadius": 50},
        {"name": "Very Sensitive", "param1": 20, "param2": 15, "minRadius": 3, "maxRadius": 60}
    ]
    
    for params in test_params:
        try:
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=params["param1"],
                param2=params["param2"],
                minRadius=params["minRadius"],
                maxRadius=params["maxRadius"]
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                print(f"  SUCCESS {params['name']}: Found {len(circles)} circles")
                
                # 가장 큰 원 표시
                if len(circles) > 0:
                    circles = sorted(circles, key=lambda x: x[2], reverse=True)
                    x, y, r = circles[0]
                    print(f"    Best circle: center=({x}, {y}), radius={r}")
                    
                    # 결과 이미지 저장
                    result_img = img.copy()
                    cv2.circle(result_img, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(result_img, (x, y), 2, (0, 0, 255), 3)
                    
                    output_path = f"test_detection_{params['name'].replace(' ', '_').lower()}.jpg"
                    cv2.imwrite(output_path, result_img)
                    print(f"    Saved: {output_path}")
            else:
                print(f"  FAILED {params['name']}: No circles found")
                
        except Exception as e:
            print(f"  ERROR {params['name']}: Error - {str(e)}")
    
    print()
    return True

def main():
    """메인 테스트 함수"""
    print("Ball Detection Test")
    print("=" * 50)
    
    # 테스트할 이미지 찾기
    test_images = []
    
    # 5iron 샷 1의 첫 번째 이미지들
    data_path = "data/video_ballData_20250930/video_ballData_20250930/5Iron_0930/1"
    if os.path.exists(data_path):
        img1_files = sorted(glob.glob(os.path.join(data_path, "1_*.bmp")))
        img2_files = sorted(glob.glob(os.path.join(data_path, "2_*.bmp")))
        
        if img1_files:
            test_images.append(("Camera 1", img1_files[0]))
        if img2_files:
            test_images.append(("Camera 2", img2_files[0]))
    
    if not test_images:
        print("No test images found!")
        return
    
    for camera_name, image_path in test_images:
        print(f"Camera: {camera_name}")
        print("-" * 30)
        test_ball_detection(image_path)

if __name__ == "__main__":
    main()
