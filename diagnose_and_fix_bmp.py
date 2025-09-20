#!/usr/bin/env python3
"""
Diagnose and Fix BMP Processing Issues
Create improved BMP processor with visual verification
"""

import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

class ImprovedBMPProcessor:
    """Improved BMP processor with better gamma handling and visualization"""
    
    def __init__(self):
        self.debug_mode = True
        
    def analyze_original_image(self, bmp_path):
        """Analyze original BMP image properties"""
        try:
            # Load BMP with different methods
            img_cv2 = cv2.imread(str(bmp_path), cv2.IMREAD_COLOR)
            img_gray = cv2.imread(str(bmp_path), cv2.IMREAD_GRAYSCALE)
            
            if img_cv2 is None:
                print(f"ERROR: Cannot load {bmp_path}")
                return None
            
            print(f"\n=== ANALYZING: {bmp_path.name} ===")
            print(f"Image shape: {img_cv2.shape}")
            print(f"Data type: {img_cv2.dtype}")
            print(f"Color range: [{img_cv2.min()}, {img_cv2.max()}]")
            print(f"Gray range: [{img_gray.min()}, {img_gray.max()}]")
            print(f"Mean brightness: {np.mean(img_gray):.2f}")
            print(f"Std deviation: {np.std(img_gray):.2f}")
            
            # Check if image is extremely dark
            if np.mean(img_gray) < 5:
                print("WARNING: Extremely dark image detected!")
                
            return {
                'img_color': img_cv2,
                'img_gray': img_gray,
                'stats': {
                    'mean': np.mean(img_gray),
                    'std': np.std(img_gray),
                    'min': img_gray.min(),
                    'max': img_gray.max()
                }
            }
            
        except Exception as e:
            print(f"Error analyzing {bmp_path}: {e}")
            return None
    
    def process_dark_image(self, img, method='adaptive'):
        """Process extremely dark images with multiple enhancement methods"""
        
        if img is None:
            return None
        
        # Convert to float
        img_float = img.astype(np.float32) / 255.0
        
        print(f"\nProcessing with method: {method}")
        
        if method == 'adaptive':
            # Adaptive method for very dark images
            
            # Step 1: Strong brightening for very dark images
            if np.mean(img_float) < 0.02:  # Very dark threshold
                print("Applying strong brightening for very dark image")
                # Use logarithmic brightening
                img_bright = np.log1p(img_float * 50) / np.log1p(50)
            else:
                # Moderate gamma correction
                img_bright = np.power(img_float, 1.0/2.2)
            
            # Step 2: Adaptive histogram equalization
            if len(img_bright.shape) == 3:
                # Convert to LAB for better processing
                img_lab = cv2.cvtColor(img_bright.astype(np.float32), cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(img_lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l_enhanced = clahe.apply((l * 255).astype(np.uint8)).astype(np.float32) / 255.0
                
                # Merge back
                img_enhanced = cv2.merge([l_enhanced, a, b])
                img_final = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)
            else:
                # Grayscale processing
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_final = clahe.apply((img_bright * 255).astype(np.uint8)).astype(np.float32) / 255.0
            
        elif method == 'linear_stretch':
            # Linear contrast stretching
            img_min, img_max = np.percentile(img_float, [1, 99])
            if img_max > img_min:
                img_final = (img_float - img_min) / (img_max - img_min)
                img_final = np.clip(img_final, 0, 1)
            else:
                img_final = img_float
                
        elif method == 'histogram_eq':
            # Global histogram equalization
            if len(img.shape) == 3:
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                img_final = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR).astype(np.float32) / 255.0
            else:
                img_final = cv2.equalizeHist(img).astype(np.float32) / 255.0
        
        # Ensure valid range
        img_final = np.clip(img_final, 0, 1)
        
        # Convert back to 8-bit
        result = (img_final * 255).astype(np.uint8)
        
        print(f"Result range: [{result.min()}, {result.max()}]")
        print(f"Result mean: {np.mean(result):.2f}")
        
        return result
    
    def process_and_save(self, bmp_path, output_path, methods=['adaptive', 'linear_stretch', 'histogram_eq']):
        """Process image with multiple methods and save best result"""
        
        # Analyze original
        analysis = self.analyze_original_image(bmp_path)
        if not analysis:
            return False
        
        img_color = analysis['img_color']
        img_gray = analysis['img_gray']
        
        best_result = None
        best_method = None
        best_score = 0
        
        # Try different methods
        for method in methods:
            print(f"\nTrying method: {method}")
            
            # Process color image
            result = self.process_dark_image(img_color, method)
            
            if result is not None:
                # Calculate quality score
                result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
                
                # Score based on mean brightness and contrast
                brightness_score = min(np.mean(result_gray) / 128.0, 1.0)  # Target ~128 brightness
                contrast_score = min(np.std(result_gray) / 64.0, 1.0)      # Target ~64 contrast
                
                score = (brightness_score + contrast_score) / 2
                
                print(f"Method {method} score: {score:.3f} (brightness: {brightness_score:.3f}, contrast: {contrast_score:.3f})")
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_method = method
        
        # Save best result
        if best_result is not None:
            success = cv2.imwrite(str(output_path), best_result)
            if success:
                print(f"\n[SUCCESS] Saved with method '{best_method}' (score: {best_score:.3f})")
                print(f"Output: {output_path}")
                return True
            else:
                print(f"[ERROR] Failed to save to {output_path}")
                return False
        else:
            print("[ERROR] All processing methods failed")
            return False

def test_sample_images():
    """Test improved processing on sample images"""
    
    processor = ImprovedBMPProcessor()
    
    # Test paths
    input_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1")
    output_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-bmp-treated-3/driver/no_marker_ball-1")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test on specific problematic files
    test_files = ['Gamma_1_4.bmp', '1_1.bmp', '2_1.bmp']
    
    print("="*80)
    print("TESTING IMPROVED BMP PROCESSOR")
    print("="*80)
    
    for filename in test_files:
        bmp_path = input_dir / filename
        output_path = output_dir / f"{bmp_path.stem}_fixed.png"
        
        if bmp_path.exists():
            print(f"\nProcessing: {filename}")
            success = processor.process_and_save(bmp_path, output_path)
            
            if success:
                print(f"✓ Successfully processed {filename}")
            else:
                print(f"✗ Failed to process {filename}")
        else:
            print(f"✗ File not found: {filename}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE - Check output files for visual verification")
    print("="*80)

if __name__ == "__main__":
    test_sample_images()