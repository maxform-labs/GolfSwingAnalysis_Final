#!/usr/bin/env python3
"""
Verify BMP Optimization Results
Compare original vs optimized images for quality assessment
"""

import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageStat
import matplotlib.pyplot as plt

class ImageQualityAnalyzer:
    """Analyze image quality metrics"""
    
    def __init__(self):
        self.metrics = []
        
    def calculate_metrics(self, img_path, label=""):
        """Calculate image quality metrics"""
        try:
            # Load image
            if img_path.suffix.lower() == '.bmp':
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                img_pil = Image.open(img_path)
            else:  # PNG
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                img_pil = Image.open(img_path)
            
            if img is None:
                return None
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            metrics = {
                'label': label,
                'file': img_path.name,
                'mean_brightness': np.mean(gray),
                'std_brightness': np.std(gray),
                'contrast': np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0,
                'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
                'file_size_kb': img_path.stat().st_size / 1024,
                'dimensions': img.shape[:2]
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing {img_path}: {e}")
            return None
    
    def compare_sample_images(self, original_dir, optimized_dir, sample_count=5):
        """Compare sample images between original and optimized"""
        
        original_path = Path(original_dir)
        optimized_path = Path(optimized_dir)
        
        # Get sample files
        original_files = list(original_path.glob("*.bmp"))[:sample_count]
        
        print(f"=== BMP Optimization Quality Analysis ===")
        print(f"Original directory: {original_dir}")
        print(f"Optimized directory: {optimized_dir}")
        print(f"Sample size: {len(original_files)} images\n")
        
        results = []
        
        for original_file in original_files:
            # Find corresponding optimized file
            optimized_file = optimized_path / f"{original_file.stem}.png"
            
            if optimized_file.exists():
                # Analyze both images
                original_metrics = self.calculate_metrics(original_file, "Original BMP")
                optimized_metrics = self.calculate_metrics(optimized_file, "Optimized PNG")
                
                if original_metrics and optimized_metrics:
                    results.append({
                        'original': original_metrics,
                        'optimized': optimized_metrics,
                        'filename': original_file.stem
                    })
        
        return self.generate_report(results)
    
    def generate_report(self, results):
        """Generate detailed comparison report"""
        
        if not results:
            print("No valid image pairs found for comparison")
            return
        
        print(f"{'='*80}")
        print(f"{'Image Quality Comparison Report':^80}")
        print(f"{'='*80}")
        
        # Calculate averages
        orig_brightness = np.mean([r['original']['mean_brightness'] for r in results])
        opt_brightness = np.mean([r['optimized']['mean_brightness'] for r in results])
        
        orig_contrast = np.mean([r['original']['contrast'] for r in results])
        opt_contrast = np.mean([r['optimized']['contrast'] for r in results])
        
        orig_sharpness = np.mean([r['original']['sharpness'] for r in results])
        opt_sharpness = np.mean([r['optimized']['sharpness'] for r in results])
        
        orig_size = np.mean([r['original']['file_size_kb'] for r in results])
        opt_size = np.mean([r['optimized']['file_size_kb'] for r in results])
        
        print(f"\n📊 AVERAGE METRICS COMPARISON")
        print(f"{'Metric':<20} {'Original BMP':<15} {'Optimized PNG':<15} {'Improvement':<15}")
        print(f"{'-'*65}")
        print(f"{'Brightness':<20} {orig_brightness:<15.2f} {opt_brightness:<15.2f} {((opt_brightness/orig_brightness-1)*100):+.1f}%")
        print(f"{'Contrast':<20} {orig_contrast:<15.4f} {opt_contrast:<15.4f} {((opt_contrast/orig_contrast-1)*100):+.1f}%")
        print(f"{'Sharpness':<20} {orig_sharpness:<15.2f} {opt_sharpness:<15.2f} {((opt_sharpness/orig_sharpness-1)*100):+.1f}%")
        print(f"{'File Size (KB)':<20} {orig_size:<15.2f} {opt_size:<15.2f} {((opt_size/orig_size-1)*100):+.1f}%")
        
        print(f"\n🎯 QUALITY ASSESSMENT")
        brightness_improvement = (opt_brightness / orig_brightness - 1) * 100
        contrast_improvement = (opt_contrast / orig_contrast - 1) * 100
        sharpness_improvement = (opt_sharpness / orig_sharpness - 1) * 100
        
        if brightness_improvement > 10:
            print(f"✅ Brightness: EXCELLENT improvement (+{brightness_improvement:.1f}%)")
        elif brightness_improvement > 0:
            print(f"✅ Brightness: Good improvement (+{brightness_improvement:.1f}%)")
        else:
            print(f"⚠️  Brightness: Decreased ({brightness_improvement:.1f}%)")
        
        if contrast_improvement > 20:
            print(f"✅ Contrast: EXCELLENT improvement (+{contrast_improvement:.1f}%)")
        elif contrast_improvement > 0:
            print(f"✅ Contrast: Good improvement (+{contrast_improvement:.1f}%)")
        else:
            print(f"⚠️  Contrast: Decreased ({contrast_improvement:.1f}%)")
        
        if sharpness_improvement > 10:
            print(f"✅ Sharpness: EXCELLENT improvement (+{sharpness_improvement:.1f}%)")
        elif sharpness_improvement > 0:
            print(f"✅ Sharpness: Good improvement (+{sharpness_improvement:.1f}%)")
        else:
            print(f"⚠️  Sharpness: Decreased ({sharpness_improvement:.1f}%)")
        
        print(f"\n📁 PROCESSING SUMMARY")
        print(f"Total image pairs analyzed: {len(results)}")
        print(f"Original format: BMP")
        print(f"Optimized format: PNG (lossless)")
        print(f"Processing techniques applied:")
        print(f"  • Gamma lens correction (γ=2.2)")
        print(f"  • LAB colorspace optimization")
        print(f"  • CLAHE contrast enhancement")
        print(f"  • Dimple visibility enhancement")
        print(f"  • Club surface angle detection")
        
        # Calculate overall quality score
        overall_score = (max(0, brightness_improvement) + max(0, contrast_improvement) + max(0, sharpness_improvement)) / 3
        
        print(f"\n🏆 OVERALL QUALITY SCORE: {overall_score:.1f}%")
        
        if overall_score > 20:
            print(f"🎉 OUTSTANDING optimization results!")
        elif overall_score > 10:
            print(f"✅ EXCELLENT optimization results!")
        elif overall_score > 0:
            print(f"✅ Good optimization results!")
        else:
            print(f"⚠️  Optimization may need adjustment")
        
        return results

def main():
    """Run image quality verification"""
    
    analyzer = ImageQualityAnalyzer()
    
    # Define paths
    original_dir = "C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1"
    optimized_dir = "C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-bmp-treated-3/driver/no_marker_ball-1"
    
    # Run comparison with more samples
    results = analyzer.compare_sample_images(original_dir, optimized_dir, sample_count=10)
    
    print(f"\n{'='*80}")
    print(f"BMP Optimization Verification Complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()