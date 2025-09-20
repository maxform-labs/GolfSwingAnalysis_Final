#!/usr/bin/env python3
"""
Simple BMP Optimization Quality Check
ASCII-only output for Windows cmd compatibility
"""

import os
import numpy as np
import cv2
from pathlib import Path

def analyze_image(img_path):
    """Analyze single image quality"""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return {
            'file': img_path.name,
            'brightness': np.mean(gray),
            'contrast': np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0,
            'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'size_kb': img_path.stat().st_size / 1024
        }
    except Exception as e:
        print(f"Error analyzing {img_path}: {e}")
        return None

def main():
    """Compare original vs optimized images"""
    
    original_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1")
    optimized_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-bmp-treated-3/driver/no_marker_ball-1")
    
    print("="*80)
    print("BMP Optimization Quality Analysis Report")
    print("="*80)
    
    # Get sample files (first 5 for quick analysis)
    original_files = list(original_dir.glob("*.bmp"))[:5]
    
    if not original_files:
        print("ERROR: No BMP files found in original directory")
        return
    
    print(f"\nAnalyzing {len(original_files)} sample images...")
    print(f"Original: {original_dir}")
    print(f"Optimized: {optimized_dir}")
    
    results = []
    
    for original_file in original_files:
        optimized_file = optimized_dir / f"{original_file.stem}.png"
        
        if optimized_file.exists():
            orig_metrics = analyze_image(original_file)
            opt_metrics = analyze_image(optimized_file)
            
            if orig_metrics and opt_metrics:
                results.append({
                    'filename': original_file.stem,
                    'original': orig_metrics,
                    'optimized': opt_metrics
                })
                
                print(f"\n[ANALYZED] {original_file.stem}")
                print(f"  Original BMP - Brightness: {orig_metrics['brightness']:.1f}, Contrast: {orig_metrics['contrast']:.3f}, Size: {orig_metrics['size_kb']:.1f}KB")
                print(f"  Optimized PNG - Brightness: {opt_metrics['brightness']:.1f}, Contrast: {opt_metrics['contrast']:.3f}, Size: {opt_metrics['size_kb']:.1f}KB")
    
    if not results:
        print("ERROR: No valid image pairs found for comparison")
        return
    
    # Calculate improvements
    brightness_improvements = []
    contrast_improvements = []
    sharpness_improvements = []
    size_changes = []
    
    for result in results:
        orig = result['original']
        opt = result['optimized']
        
        brightness_improvements.append((opt['brightness'] / orig['brightness'] - 1) * 100)
        contrast_improvements.append((opt['contrast'] / orig['contrast'] - 1) * 100)
        sharpness_improvements.append((opt['sharpness'] / orig['sharpness'] - 1) * 100)
        size_changes.append((opt['size_kb'] / orig['size_kb'] - 1) * 100)
    
    avg_brightness = np.mean(brightness_improvements)
    avg_contrast = np.mean(contrast_improvements)
    avg_sharpness = np.mean(sharpness_improvements)
    avg_size = np.mean(size_changes)
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nAverage Improvements:")
    print(f"Brightness:  {avg_brightness:+6.1f}%")
    print(f"Contrast:    {avg_contrast:+6.1f}%")
    print(f"Sharpness:   {avg_sharpness:+6.1f}%")
    print(f"File Size:   {avg_size:+6.1f}%")
    
    print(f"\nQuality Assessment:")
    
    if avg_brightness > 5:
        print(f"[SUCCESS] Brightness improved significantly (+{avg_brightness:.1f}%)")
    elif avg_brightness > 0:
        print(f"[GOOD] Brightness improved (+{avg_brightness:.1f}%)")
    else:
        print(f"[WARNING] Brightness decreased ({avg_brightness:.1f}%)")
    
    if avg_contrast > 10:
        print(f"[SUCCESS] Contrast improved significantly (+{avg_contrast:.1f}%)")
    elif avg_contrast > 0:
        print(f"[GOOD] Contrast improved (+{avg_contrast:.1f}%)")
    else:
        print(f"[WARNING] Contrast decreased ({avg_contrast:.1f}%)")
    
    if avg_sharpness > 5:
        print(f"[SUCCESS] Sharpness improved significantly (+{avg_sharpness:.1f}%)")
    elif avg_sharpness > 0:
        print(f"[GOOD] Sharpness improved (+{avg_sharpness:.1f}%)")
    else:
        print(f"[WARNING] Sharpness decreased ({avg_sharpness:.1f}%)")
    
    # Overall assessment
    positive_improvements = sum([1 for x in [avg_brightness, avg_contrast, avg_sharpness] if x > 0])
    
    print(f"\nOverall Assessment:")
    print(f"Improved metrics: {positive_improvements}/3")
    
    if positive_improvements == 3:
        print("[EXCELLENT] All quality metrics improved!")
    elif positive_improvements == 2:
        print("[GOOD] Most quality metrics improved!")
    elif positive_improvements == 1:
        print("[PARTIAL] Some quality improvements achieved!")
    else:
        print("[WARNING] Quality optimization needs adjustment!")
    
    # Technical details
    print(f"\nTechnical Details:")
    print(f"Processing applied: Gamma correction (2.2), LAB colorspace, CLAHE enhancement")
    print(f"Output format: PNG (lossless compression)")
    print(f"Target improvements: Dimple visibility, club surface angles")
    
    # Final file count verification
    total_original = len(list(original_dir.glob("*.bmp")))
    total_optimized = len(list(optimized_dir.glob("*.png")))
    
    print(f"\nProcessing Summary:")
    print(f"Original BMP files: {total_original}")
    print(f"Processed PNG files: {total_optimized}")
    print(f"Success rate: {(total_optimized/total_original)*100:.1f}%")
    
    print("\n" + "="*80)
    print("BMP Optimization Verification Complete!")
    print("="*80)

if __name__ == "__main__":
    main()