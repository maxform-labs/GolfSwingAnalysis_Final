"""
Complete remaining image enhancement processing
Simple single-threaded approach to finish the job
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time

def enhance_image_simple(image):
    """Simple but effective image enhancement"""
    try:
        # Fast denoising
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Adaptive gamma
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 50:
            gamma = 2.2
        elif mean_brightness < 100:
            gamma = 1.8
        else:
            gamma = 1.4
        
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(denoised, table)
        
        # Simple CLAHE
        lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Light sharpening
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
        
    except Exception as e:
        print(f"Enhancement error: {e}")
        return image

def main():
    """Complete the image enhancement processing"""
    print("Completing image enhancement processing...")
    
    source_base = 'shot-image-jpg'
    target_base = 'shot-image-treated'
    
    # Create target base if not exists
    Path(target_base).mkdir(exist_ok=True)
    
    processed = 0
    enhanced = 0
    errors = 0
    start_time = time.time()
    
    # Walk through all source files
    for root, dirs, files in os.walk(source_base):
        for file in files:
            if file.lower().endswith('.jpg'):
                source_path = os.path.join(root, file)
                
                # Calculate target path
                rel_path = os.path.relpath(source_path, source_base)
                rel_dir = os.path.dirname(rel_path)
                filename = os.path.basename(rel_path)
                
                # Create target directory
                if rel_dir != '.':
                    target_dir = os.path.join(target_base, rel_dir)
                else:
                    target_dir = target_base
                Path(target_dir).mkdir(parents=True, exist_ok=True)
                
                # Target file with treated_ prefix
                target_filename = 'treated_' + filename
                target_path = os.path.join(target_dir, target_filename)
                
                processed += 1
                
                # Skip if already exists
                if os.path.exists(target_path):
                    continue
                
                try:
                    # Read and enhance image
                    image = cv2.imread(source_path)
                    if image is None:
                        print(f"Could not read: {source_path}")
                        errors += 1
                        continue
                    
                    # Enhance
                    enhanced_image = enhance_image_simple(image)
                    
                    # Save
                    success = cv2.imwrite(target_path, enhanced_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    
                    if success:
                        enhanced += 1
                        if enhanced % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = enhanced / elapsed
                            print(f"Enhanced: {enhanced}/{processed} files, Rate: {rate:.1f}/sec")
                    else:
                        errors += 1
                        print(f"Failed to save: {target_path}")
                        
                except Exception as e:
                    errors += 1
                    print(f"Error processing {source_path}: {e}")
    
    # Final summary
    elapsed_time = time.time() - start_time
    total_treated = len([f for f in Path(target_base).rglob("treated_*.jpg")])
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Total source files: {processed}")
    print(f"Newly enhanced: {enhanced}")
    print(f"Total treated files: {total_treated}")
    print(f"Errors: {errors}")
    print(f"Processing time: {elapsed_time/60:.2f} minutes")
    
    # Show structure
    print("\n" + "-"*50)
    print("Final folder structure:")
    print("-"*50)
    
    for root, dirs, files in os.walk(target_base):
        level = root.replace(target_base, '').count(os.sep)
        indent = '  ' * level
        folder_name = os.path.basename(root) if level > 0 else 'shot-image-treated'
        print(f"{indent}{folder_name}/")
        
        treated_files = [f for f in files if f.startswith('treated_') and f.endswith('.jpg')]
        if treated_files:
            sub_indent = '  ' * (level + 1)
            print(f"{sub_indent}({len(treated_files)} treated JPG files)")
    
    return 0 if errors == 0 else 1

if __name__ == "__main__":
    exit(main())