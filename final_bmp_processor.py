#!/usr/bin/env python3
"""
Final BMP Processor - Production Version
Process all BMP images with optimized histogram equalization
"""

import os
import numpy as np
import cv2
from pathlib import Path
import time

class FinalBMPProcessor:
    """Production BMP processor using optimal histogram equalization"""
    
    def __init__(self):
        self.method = 'histogram_eq'
        self.processed_count = 0
        self.failed_count = 0
        
    def process_image(self, bmp_path, output_path):
        """Process single BMP image with histogram equalization"""
        try:
            # Load BMP
            img = cv2.imread(str(bmp_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"ERROR: Cannot load {bmp_path}")
                return False
            
            # Check if extremely dark (mean < 10)
            gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if np.mean(gray_check) < 10:
                # Apply histogram equalization for dark images
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            else:
                # Apply CLAHE for moderate images
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(img_lab)
                
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                result = cv2.merge([l, a, b])
                result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
            
            # Save as PNG
            success = cv2.imwrite(str(output_path), result)
            
            if success:
                self.processed_count += 1
                return True
            else:
                print(f"ERROR: Failed to save {output_path}")
                self.failed_count += 1
                return False
                
        except Exception as e:
            print(f"ERROR processing {bmp_path}: {e}")
            self.failed_count += 1
            return False
    
    def process_directory(self, input_dir, output_dir):
        """Process all BMP files in directory"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all BMP files
        bmp_files = list(input_path.glob("*.bmp"))
        
        if not bmp_files:
            print(f"No BMP files found in {input_dir}")
            return
        
        print(f"="*80)
        print(f"FINAL BMP PROCESSING - PRODUCTION VERSION")
        print(f"="*80)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Total BMP files: {len(bmp_files)}")
        print(f"Processing method: Histogram Equalization + CLAHE")
        print(f"="*80)
        
        start_time = time.time()
        
        # Process in batches for progress reporting
        batch_size = 10
        
        for i in range(0, len(bmp_files), batch_size):
            batch = bmp_files[i:i + batch_size]
            batch_start = time.time()
            
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(bmp_files)-1)//batch_size + 1}")
            
            for bmp_file in batch:
                output_file = output_path / f"{bmp_file.stem}.png"
                
                print(f"  Processing: {bmp_file.name}", end=" ... ")
                success = self.process_image(bmp_file, output_file)
                
                if success:
                    print("OK")
                else:
                    print("FAILED")
            
            batch_time = time.time() - batch_start
            progress = (i + len(batch)) / len(bmp_files) * 100
            print(f"  Batch completed in {batch_time:.2f}s - Progress: {progress:.1f}%")
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"PROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"Total files processed: {self.processed_count}")
        print(f"Failed files: {self.failed_count}")
        print(f"Success rate: {(self.processed_count/(self.processed_count+self.failed_count))*100:.1f}%")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average per file: {total_time/len(bmp_files):.2f} seconds")
        
        # Verify output
        output_files = list(output_path.glob("*.png"))
        print(f"Output PNG files: {len(output_files)}")
        
        if len(output_files) == len(bmp_files):
            print("[SUCCESS] All files processed successfully!")
        else:
            print(f"[WARNING] Expected {len(bmp_files)}, got {len(output_files)} files")

def main():
    """Process all BMP images with final optimized method"""
    
    processor = FinalBMPProcessor()
    
    # Process the driver/no_marker_ball-1 directory
    input_dir = "C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-original/driver/no_marker_ball-1"
    output_dir = "C:/src/GolfSwingAnalysis_Final_ver8/data/images/shot-image-bmp-treated-3/driver/no_marker_ball-1"
    
    processor.process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()