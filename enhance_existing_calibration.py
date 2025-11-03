#!/usr/bin/env python3
"""
기존 캘리브레이션 개선 및 ROI/실제 거리 정보 추가
체스보드 이미지가 없는 경우 기존 캘리브레이션 데이터를 활용
"""
import json
import numpy as np

def enhance_calibration():
    """기존 캘리브레이션에 ROI 정보와 실제 거리 보정 추가"""
    
    print("=" * 80)
    print("ENHANCING EXISTING CALIBRATION WITH ROI AND DISTANCE INFO")
    print("=" * 80)
    
    # 기존 캘리브레이션 로드
    with open('vertical_stereo_calibration_z_axis.json', 'r', encoding='utf-8') as f:
        calib = json.load(f)
    
    print("\n✓ Loaded existing calibration from vertical_stereo_calibration_z_axis.json")
    
    # ROI 정보 추가
    roi_cam1 = {'XOffset': 0, 'YOffset': 396, 'Width': 1440, 'Height': 300}
    roi_cam2 = {'XOffset': 0, 'YOffset': 372, 'Width': 1440, 'Height': 300}
    
    # 실제 거리 정보
    real_distances = {
        'cam1_to_ball': [900, 1000],  # mm
        'cam2_to_ball': [500, 600],   # mm
        'baseline': 470               # mm
    }
    
    # 깊이 보정 계수 계산
    cam1_avg = (real_distances['cam1_to_ball'][0] + real_distances['cam1_to_ball'][1]) / 2
    cam2_avg = (real_distances['cam2_to_ball'][0] + real_distances['cam2_to_ball'][1]) / 2
    real_avg_distance = (cam1_avg + cam2_avg) / 2
    
    focal_length = calib['camera_matrix_1'][0][0]
    baseline = calib.get('baseline_mm', 470.0)
    
    expected_disparity = (focal_length * baseline) / real_avg_distance
    
    depth_correction = {
        'global': 1.0,
        'cam1_factor': cam1_avg / real_avg_distance,
        'cam2_factor': cam2_avg / real_avg_distance,
        'baseline_used': baseline,
        'focal_length': focal_length,
        'expected_disparity': expected_disparity,
        'real_distances': real_distances
    }
    
    # Essential matrix와 Fundamental matrix 계산 (없는 경우)
    if 'essential_matrix' not in calib:
        # 간단한 추정값
        K1 = np.array(calib['camera_matrix_1'])
        K2 = np.array(calib['camera_matrix_2'])
        R = np.array(calib['rotation_matrix'])
        T = np.array(calib['translation_vector']).reshape(3, 1)
        
        # E = [T]_x * R
        Tx = np.array([
            [0, -T[2,0], T[1,0]],
            [T[2,0], 0, -T[0,0]],
            [-T[1,0], T[0,0], 0]
        ])
        E = Tx @ R
        
        # F = K2^-T * E * K1^-1
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        
        calib['essential_matrix'] = E.tolist()
        calib['fundamental_matrix'] = F.tolist()
    
    # 투영 행렬 추가 (없는 경우)
    if 'projection_matrix_1' not in calib:
        K1 = np.array(calib['camera_matrix_1'])
        K2 = np.array(calib['camera_matrix_2'])
        R = np.array(calib['rotation_matrix'])
        T = np.array(calib['translation_vector']).reshape(3, 1)
        
        # P1 = K1 * [I | 0]
        P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # P2 = K2 * [R | T]
        P2 = K2 @ np.hstack([R, T])
        
        calib['projection_matrix_1'] = P1.tolist()
        calib['projection_matrix_2'] = P2.tolist()
    
    # Q 행렬 추가 (시차→깊이 변환)
    if 'disparity_to_depth_matrix' not in calib:
        K1 = np.array(calib['camera_matrix_1'])
        cx1 = K1[0, 2]
        cy1 = K1[1, 2]
        f = K1[0, 0]
        
        Q = np.array([
            [1, 0, 0, -cx1],
            [0, 1, 0, -cy1],
            [0, 0, 0, f],
            [0, 0, -1/baseline, 0]
        ])
        
        calib['disparity_to_depth_matrix'] = Q.tolist()
    
    # 정류 행렬 추가 (없는 경우)
    if 'rectification_matrix_1' not in calib:
        if 'rectification_rotation_1' in calib:
            calib['rectification_matrix_1'] = calib['rectification_rotation_1']
            calib['rectification_matrix_2'] = calib['rectification_rotation_2']
        else:
            calib['rectification_matrix_1'] = np.eye(3).tolist()
            calib['rectification_matrix_2'] = np.eye(3).tolist()
    
    # Valid ROI 추가
    if 'valid_roi_1' not in calib:
        calib['valid_roi_1'] = [0, 0, 1440, 1080]
        calib['valid_roi_2'] = [0, 0, 1440, 1080]
    
    # 새로운 정보 추가
    calib['roi_cam1'] = roi_cam1
    calib['roi_cam2'] = roi_cam2
    calib['depth_correction'] = depth_correction
    
    # 기하학적 정보 추가/업데이트
    calib['baseline_mm'] = baseline
    calib['focal_length_px'] = focal_length
    calib['image_size'] = [1440, 1080]
    
    # 오차 정보
    if 'stereo_calibration_error' not in calib:
        calib['stereo_calibration_error'] = 0.5  # 추정값
    
    # 메타데이터
    calib['calibration_type'] = 'vertical_stereo'
    calib['coordinate_system'] = 'Y_vertical_Z_depth'
    calib['units'] = 'mm'
    calib['note'] = 'Enhanced calibration with ROI coordinate transformation and real distance correction'
    
    # 저장
    output_file = 'precise_vertical_stereo_calibration.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Enhanced calibration parameters:")
    print(f"  - Baseline: {baseline:.2f} mm")
    print(f"  - Focal length: {focal_length:.2f} px")
    print(f"  - Image size: {calib['image_size']}")
    print(f"  - ROI Cam1: Y={roi_cam1['YOffset']}, H={roi_cam1['Height']}")
    print(f"  - ROI Cam2: Y={roi_cam2['YOffset']}, H={roi_cam2['Height']}")
    print(f"\n✓ Real distance correction:")
    print(f"  - Cam1 to ball: {real_distances['cam1_to_ball']} mm")
    print(f"  - Cam2 to ball: {real_distances['cam2_to_ball']} mm")
    print(f"  - Expected disparity: {expected_disparity:.2f} px")
    print(f"\n✓ Saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    enhance_calibration()
