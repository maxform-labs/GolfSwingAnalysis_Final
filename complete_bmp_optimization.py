#!/usr/bin/env python3
"""
Complete BMP Optimization - Resume Processing
Optimized version to handle remaining images efficiently
"""

import os
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import time

class OptimizedBMPProcessor:
    """Lightweight BMP processor optimized for speed"""
    
    def __init__(self):
        self.gamma = 2.2
        self.output_format = 'PNG'
        
    def process_image(self, bmp_path, output_path):
        """Process single BMP image with core optimizations"""
        try:
            # Load BMP directly
            img = cv2.imread(str(bmp_path), cv2.IMREAD_COLOR)
            if img is None:
                return False
                
            # Convert to float for processing
            img_float = img.astype(np.float32) / 255.0
            
            # Gamma correction for lens characteristics
            img_corrected = np.power(img_float, 1.0 / self.gamma)
            
            # Convert to LAB for better dimple visibility
            img_lab = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(img_lab)
            
            # Enhance L channel (lightness) for dimple visibility
            l_enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply((l * 255).astype(np.uint8))
            l_enhanced = l_enhanced.astype(np.float32) / 255.0
            
            # Merge back to LAB
            img_enhanced_lab = cv2.merge([l_enhanced, a, b])
            
            # Convert back to BGR
            img_final = cv2.cvtColor(img_enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Convert to 8-bit
            img_final = np.clip(img_final * 255, 0, 255).astype(np.uint8)
            
            # Save as PNG (lossless)
            cv2.imwrite(str(output_path), img_final)
            return True
            
        except Exception as e:
            print(f"Error processing {bmp_path}: {e}")
            return False

def main():
    """Complete the remaining BMP optimization"""
    
    # Paths
    input_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1")
    output_dir = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-bmp-treated-3/driver/no_marker_ball-1")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = OptimizedBMPProcessor()
    
    # Get all BMP files
    bmp_files = list(input_dir.glob("*.bmp"))
    print(f"Found {len(bmp_files)} BMP files")
    
    # Check which files are already processed
    processed_files = set()
    for png_file in output_dir.glob("*.png"):
        bmp_name = png_file.stem + ".bmp"
        processed_files.add(bmp_name)
    
    print(f"Already processed: {len(processed_files)} files")
    
    # Process remaining files
    remaining_files = [f for f in bmp_files if f.name not in processed_files]
    print(f"Remaining to process: {len(remaining_files)} files")
    
    if not remaining_files:
        print("All files already processed!")
        return
    
    # Process in batches
    batch_size = 10
    total_processed = len(processed_files)
    
    for i in range(0, len(remaining_files), batch_size):
        batch = remaining_files[i:i + batch_size]
        batch_start = time.time()
        
        for bmp_file in batch:
            output_file = output_dir / f"{bmp_file.stem}.png"
            
            print(f"Processing: {bmp_file.name}")
            success = processor.process_image(bmp_file, output_file)
            
            if success:
                total_processed += 1
                print(f"  [SUCCESS] Saved: {output_file.name}")
            else:
                print(f"  [ERROR] Failed: {bmp_file.name}")
        
        batch_time = time.time() - batch_start
        print(f"Batch completed in {batch_time:.2f}s - Total processed: {total_processed}/{len(bmp_files)}")
        print("-" * 50)
    
    print(f"\nFinal Results:")
    print(f"Total BMP files: {len(bmp_files)}")
    print(f"Successfully processed: {total_processed}")
    print(f"Success rate: {total_processed/len(bmp_files)*100:.1f}%")
    
    # Verify output directory
    final_count = len(list(output_dir.glob("*.png")))
    print(f"Final PNG files: {final_count}")

if __name__ == "__main__":
    main()