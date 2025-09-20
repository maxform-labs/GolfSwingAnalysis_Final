#!/usr/bin/env python3
"""
Dimple Visibility Enhancer
Selective ball processing to enhance dimple visibility while preserving club details
"""

import numpy as np
import cv2
from pathlib import Path

class DimpleVisibilityEnhancer:
    """Enhanced BMP processor with selective ball dimple enhancement"""
    
    def __init__(self):
        self.debug_mode = True
        
    def detect_ball_region(self, img):
        """Detect golf ball region using circular Hough transform"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Detect circles using HoughCircles
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=15,
                maxRadius=80
            )
            
            ball_mask = np.zeros(gray.shape, dtype=np.uint8)
            ball_center = None
            ball_radius = 0
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # Find the most prominent circle (largest or brightest)
                best_circle = None
                best_score = 0
                
                for (x, y, r) in circles:
                    # Check if circle is within image bounds
                    if (x - r >= 0 and y - r >= 0 and 
                        x + r < gray.shape[1] and y + r < gray.shape[0]):
                        
                        # Calculate score based on brightness and size
                        roi = gray[y-r:y+r, x-r:x+r]
                        brightness_score = np.mean(roi)
                        size_score = r
                        score = brightness_score * 0.7 + size_score * 0.3
                        
                        if score > best_score:
                            best_score = score
                            best_circle = (x, y, r)
                
                if best_circle:
                    x, y, r = best_circle
                    cv2.circle(ball_mask, (x, y), r, 255, -1)
                    ball_center = (x, y)
                    ball_radius = r
                    
                    if self.debug_mode:
                        print(f"Ball detected at ({x}, {y}) with radius {r}")
            
            return ball_mask, ball_center, ball_radius
            
        except Exception as e:
            print(f"Error in ball detection: {e}")
            return np.zeros(img.shape[:2], dtype=np.uint8), None, 0
    
    def enhance_dimple_visibility(self, img, ball_mask, ball_center, ball_radius):
        """Enhance dimple visibility within ball region"""
        try:
            if ball_center is None or ball_radius == 0:
                return img
            
            result = img.copy()
            
            # Extract ball region
            x, y = ball_center
            r = ball_radius
            
            # Create padded ROI to avoid boundary issues
            pad = 10
            x1, y1 = max(0, x-r-pad), max(0, y-r-pad)
            x2, y2 = min(img.shape[1], x+r+pad), min(img.shape[0], y+r+pad)
            
            ball_roi = img[y1:y2, x1:x2].copy()
            mask_roi = ball_mask[y1:y2, x1:x2]
            
            if ball_roi.size == 0:
                return img
            
            # Convert to LAB for better processing
            if len(ball_roi.shape) == 3:
                ball_lab = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(ball_lab)
            else:
                l = ball_roi.copy()
            
            # Apply mask to get ball pixels only
            ball_pixels = l[mask_roi > 0]
            
            if len(ball_pixels) == 0:
                return img
            
            # Calculate ball statistics
            ball_mean = np.mean(ball_pixels)
            ball_std = np.std(ball_pixels)
            
            print(f"Ball region - Mean: {ball_mean:.2f}, Std: {ball_std:.2f}")
            
            # Enhance dimple visibility
            l_enhanced = l.copy().astype(np.float32)
            
            # Method 1: Reduce overall ball brightness while enhancing contrast
            for i in range(l_enhanced.shape[0]):
                for j in range(l_enhanced.shape[1]):
                    if mask_roi[i, j] > 0:  # Inside ball
                        pixel_val = l_enhanced[i, j]
                        
                        # Reduce brightness but enhance local contrast
                        # Dimples (darker areas) become more visible
                        if pixel_val > ball_mean:
                            # Bright areas (non-dimple) - reduce brightness more
                            l_enhanced[i, j] = pixel_val * 0.6
                        else:
                            # Dark areas (dimples) - reduce brightness less
                            l_enhanced[i, j] = pixel_val * 0.8
            
            # Method 2: Apply unsharp masking for dimple edge enhancement
            gaussian = cv2.GaussianBlur(l_enhanced, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(l_enhanced, 1.5, gaussian, -0.5, 0)
            
            # Apply only to ball region
            l_final = l.astype(np.float32)
            l_final[mask_roi > 0] = unsharp_mask[mask_roi > 0]
            l_final = np.clip(l_final, 0, 255).astype(np.uint8)
            
            # Merge back
            if len(ball_roi.shape) == 3:
                ball_enhanced = cv2.merge([l_final, a, b])
                ball_enhanced = cv2.cvtColor(ball_enhanced, cv2.COLOR_LAB2BGR)
            else:
                ball_enhanced = l_final
            
            # Place enhanced ball back into original image
            result[y1:y2, x1:x2] = ball_enhanced
            
            return result
            
        except Exception as e:
            print(f"Error in dimple enhancement: {e}")
            return img
    
    def process_image(self, bmp_path, output_path):
        """Process single image with dimple enhancement"""
        try:
            # Load original BMP
            img = cv2.imread(str(bmp_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"ERROR: Cannot load {bmp_path}")
                return False
            
            print(f"\n=== Processing: {bmp_path.name} ===")
            
            # Step 1: Basic histogram equalization (as before)
            gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if np.mean(gray_check) < 10:
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            else:
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(img_lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                img_enhanced = cv2.merge([l, a, b])
                img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)
            
            # Step 2: Detect ball region
            ball_mask, ball_center, ball_radius = self.detect_ball_region(img_enhanced)
            
            # Step 3: Enhance dimple visibility
            if ball_center is not None:
                result = self.enhance_dimple_visibility(img_enhanced, ball_mask, ball_center, ball_radius)
                print(f"Dimple enhancement applied to ball at {ball_center}")
            else:
                result = img_enhanced
                print("No ball detected - using standard enhancement")
            
            # Save result
            success = cv2.imwrite(str(output_path), result)
            
            if success:
                print(f"[SUCCESS] Saved: {output_path}")
                return True
            else:
                print(f"[FAILED] Failed to save: {output_path}")
                return False
                
        except Exception as e:
            print(f"ERROR processing {bmp_path}: {e}")
            return False

def test_dimple_enhancement():
    """Test dimple enhancement on specific problematic image"""
    
    enhancer = DimpleVisibilityEnhancer()
    
    # Test on the problematic image
    input_file = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1/Gamma_1_6.bmp")
    output_file = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-bmp-treated-3/driver/no_marker_ball-1/Gamma_1_6_dimple_enhanced.png")
    
    print("="*80)
    print("TESTING DIMPLE VISIBILITY ENHANCEMENT")
    print("="*80)
    
    if input_file.exists():
        success = enhancer.process_image(input_file, output_file)
        
        if success:
            print(f"\n[SUCCESS] Enhanced image saved to:")
            print(f"{output_file}")
            print(f"\nCompare with original processed version:")
            print(f"C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-bmp-treated-3/driver/no_marker_ball-1/Gamma_1_6.png")
        else:
            print("[FAILED] Enhancement failed")
    else:
        print(f"[ERROR] Input file not found: {input_file}")
    
    print("="*80)

if __name__ == "__main__":
    test_dimple_enhancement()