#!/usr/bin/env python3
"""
효율적인 스테레오 캘리브레이션
470mm 베이스라인을 고려한 수직 스테레오 비전 캘리브레이션
"""

import cv2
import numpy as np
import glob
import json
import os
from datetime import datetime

class EfficientStereoCalibration:
    def __init__(self, calibration_images_path="data2/Calibration_image_1025"):
        self.calibration_images_path = calibration_images_path
        self.image_size = (1440, 1080)  # 실제 이미지 크기
        self.baseline_mm = 470.0  # 실제 베이스라인 (mm)
        
        # 체스보드 패턴 크기 (가장 일반적인 패턴들만 시도)
        self.chessboard_patterns = [
            (8, 6), (9, 6), (10, 7), (8, 7), (9, 7)
        ]
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        print(f"Efficient Stereo Calibration Initialized")
        print(f"Image path: {calibration_images_path}")
        print(f"Image size: {self.image_size}")
        print(f"Baseline: {self.baseline_mm}mm")
    
    def enhance_image(self, img):
        """Image enhancement"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def find_best_chessboard_pattern(self, images, camera_name):
        """Find optimal chessboard pattern"""
        print(f"\n{camera_name} searching for optimal pattern...")
        
        best_pattern = None
        best_success_count = 0
        best_objpoints = None
        best_imgpoints = None
        
        for pattern in self.chessboard_patterns:
            print(f"  Testing pattern {pattern}...")
            
            # Prepare 3D points
            objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
            
            objpoints = []
            imgpoints = []
            success_count = 0
            
            for i, img in enumerate(images):
                enhanced = self.enhance_image(img)
                
                # Detect chessboard corners
                ret, corners = cv2.findChessboardCorners(enhanced, pattern, None)
                
                if ret:
                    # Refine corners
                    corners2 = cv2.cornerSubPix(enhanced, corners, (11, 11), (-1, -1), self.criteria)
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    success_count += 1
            
            print(f"    Success: {success_count}/{len(images)} images")
            
            # Select pattern with best success rate
            if success_count > best_success_count and success_count >= 5:
                best_pattern = pattern
                best_success_count = success_count
                best_objpoints = objpoints
                best_imgpoints = imgpoints
        
        if best_pattern is None:
            print(f"  X {camera_name}: No suitable pattern found")
            return None, None, None
        
        print(f"  OK {camera_name} optimal pattern: {best_pattern} ({best_success_count} images)")
        return best_objpoints, best_imgpoints, best_pattern
    
    def calibrate_individual_camera(self, images, camera_name):
        """Individual camera calibration"""
        print(f"\n{camera_name} individual calibration...")
        
        objpoints, imgpoints, pattern = self.find_best_chessboard_pattern(images, camera_name)
        
        if objpoints is None:
            return None, None, None
        
        # Camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_size, None, None
        )
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        
        print(f"OK {camera_name} calibration completed")
        print(f"  Reprojection error: {mean_error:.3f} pixels")
        print(f"  Used pattern: {pattern}")
        print(f"  Successful images: {len(objpoints)}")
        
        return mtx, dist, (objpoints, imgpoints)
    
    def calibrate_stereo(self, cam1_images, cam2_images):
        """Stereo calibration"""
        print("\n=== Stereo Calibration Started ===")
        
        # Individual camera calibration
        mtx1, dist1, cam1_data = self.calibrate_individual_camera(cam1_images, "Cam1")
        mtx2, dist2, cam2_data = self.calibrate_individual_camera(cam2_images, "Cam2")
        
        if mtx1 is None or mtx2 is None:
            print("X Individual calibration failed")
            return None
        
        # Stereo calibration
        print("\nPerforming stereo calibration...")
        
        objpoints1, imgpoints1 = cam1_data
        objpoints2, imgpoints2 = cam2_data
        
        # Stereo calibration
        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints1, imgpoints1, imgpoints2,
            mtx1, dist1, mtx2, dist2,
            self.image_size,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        # Stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx1, dist1, mtx2, dist2, self.image_size, R, T
        )
        
        # Calculate baseline
        baseline_pixels = np.linalg.norm(T)
        
        # Calculate pixel to mm ratio (based on 470mm baseline)
        pixel_to_mm = self.baseline_mm / baseline_pixels
        
        print(f"OK Stereo calibration completed")
        print(f"  Reprojection error: {ret:.3f}")
        print(f"  Baseline (pixels): {baseline_pixels:.2f}")
        print(f"  Baseline (actual): {self.baseline_mm}mm")
        print(f"  Pixel to mm ratio: {pixel_to_mm:.3f}mm")
        
        return {
            'camera_matrix_1': mtx1.tolist(),
            'distortion_coeffs_1': dist1.tolist(),
            'camera_matrix_2': mtx2.tolist(),
            'distortion_coeffs_2': dist2.tolist(),
            'rotation_matrix': R.tolist(),
            'translation_vector': T.tolist(),
            'rectification_rotation_1': R1.tolist(),
            'rectification_rotation_2': R2.tolist(),
            'projection_matrix_1': P1.tolist(),
            'projection_matrix_2': P2.tolist(),
            'disparity_to_depth_matrix': Q.tolist(),
            'roi_1': roi1.tolist(),
            'roi_2': roi2.tolist(),
            'baseline_pixels': baseline_pixels,
            'baseline_mm': self.baseline_mm,
            'pixel_to_mm_ratio': pixel_to_mm,
            'reprojection_error': ret,
            'image_size': self.image_size,
            'calibration_date': datetime.now().isoformat(),
            'calibration_images_path': self.calibration_images_path
        }
    
    def load_calibration_images(self):
        """Load calibration images"""
        print(f"\nLoading calibration images...")
        
        cam1_pattern = os.path.join(self.calibration_images_path, "Cam1_*.bmp")
        cam2_pattern = os.path.join(self.calibration_images_path, "Cam2_*.bmp")
        
        cam1_files = sorted(glob.glob(cam1_pattern))
        cam2_files = sorted(glob.glob(cam2_pattern))
        
        print(f"Cam1 images: {len(cam1_files)}")
        print(f"Cam2 images: {len(cam2_files)}")
        
        if len(cam1_files) == 0 or len(cam2_files) == 0:
            print("X Calibration images not found")
            return None, None
        
        # Load images
        cam1_images = []
        cam2_images = []
        
        for file in cam1_files:
            img = cv2.imread(file)
            if img is not None:
                cam1_images.append(img)
        
        for file in cam2_files:
            img = cv2.imread(file)
            if img is not None:
                cam2_images.append(img)
        
        print(f"Loaded Cam1 images: {len(cam1_images)}")
        print(f"Loaded Cam2 images: {len(cam2_images)}")
        
        return cam1_images, cam2_images
    
    def save_calibration(self, calibration_data, filename="efficient_stereo_calibration_470mm.json"):
        """Save calibration data"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nOK Calibration data saved: {filename}")
    
    def run_calibration(self):
        """Run complete calibration"""
        print("=== Efficient Stereo Calibration Started ===")
        
        # 1. Load images
        cam1_images, cam2_images = self.load_calibration_images()
        
        if cam1_images is None:
            return None
        
        # 2. Stereo calibration
        calibration_data = self.calibrate_stereo(cam1_images, cam2_images)
        
        if calibration_data is None:
            print("X Calibration failed")
            return None
        
        # 3. Save results
        self.save_calibration(calibration_data)
        
        # 4. Print summary
        self.print_calibration_summary(calibration_data)
        
        return calibration_data
    
    def print_calibration_summary(self, calibration_data):
        """Print calibration results summary"""
        print("\n=== Calibration Results Summary ===")
        print(f"Reprojection error: {calibration_data['reprojection_error']:.3f} pixels")
        print(f"Baseline (pixels): {calibration_data['baseline_pixels']:.2f}")
        print(f"Baseline (actual): {calibration_data['baseline_mm']}mm")
        print(f"Pixel to mm ratio: {calibration_data['pixel_to_mm_ratio']:.3f}mm")
        print(f"Image size: {calibration_data['image_size']}")
        print(f"Calibration date: {calibration_data['calibration_date']}")
        
        # Camera matrix information
        K1 = np.array(calibration_data['camera_matrix_1'])
        K2 = np.array(calibration_data['camera_matrix_2'])
        
        print(f"\nCam1 intrinsic parameters:")
        print(f"  Focal length: fx={K1[0,0]:.2f}, fy={K1[1,1]:.2f}")
        print(f"  Principal point: cx={K1[0,2]:.2f}, cy={K1[1,2]:.2f}")
        
        print(f"\nCam2 intrinsic parameters:")
        print(f"  Focal length: fx={K2[0,0]:.2f}, fy={K2[1,1]:.2f}")
        print(f"  Principal point: cx={K2[0,2]:.2f}, cy={K2[1,2]:.2f}")
        
        # Stereo transformation information
        T = np.array(calibration_data['translation_vector'])
        print(f"\nStereo transformation:")
        print(f"  Translation vector T: [{T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}]")

def main():
    """Main function"""
    calibrator = EfficientStereoCalibration()
    calibration_data = calibrator.run_calibration()
    
    if calibration_data:
        print("\nOK Calibration completed successfully!")
        print("Result file: efficient_stereo_calibration_470mm.json")
        print("\nNext steps:")
        print("1. Verify calibration results")
        print("2. Generate stereo rectification and depth maps")
        print("3. Test 3D position estimation")
    else:
        print("\nX Calibration failed.")

if __name__ == "__main__":
    main()
