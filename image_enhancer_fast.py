"""
Fast Image Enhancement System for Golf Swing Analysis
Optimized version with parallel processing and efficient memory management
"""

import cv2
import numpy as np
import os
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import time

class FastImageEnhancer:
    """Fast image enhancement with parallel processing"""
    
    def __init__(self, num_workers=None):
        """Initialize with optimal worker count"""
        self.num_workers = num_workers or min(8, mp.cpu_count())
        self.total_processed = 0
        self.total_enhanced = 0
        self.errors = []
        
    @staticmethod
    def enhance_image_optimized(image):
        """Optimized image enhancement - faster version"""
        try:
            # Step 1: Fast denoising (reduced parameters)
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
            
            # Step 2: Simple gamma correction
            gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # Simplified gamma determination
            if mean_brightness < 50:
                gamma = 2.5
            elif mean_brightness < 100:
                gamma = 2.0
            else:
                gamma = 1.5
            
            # Apply gamma
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            gamma_corrected = cv2.LUT(denoised, table)
            
            # Step 3: Fast CLAHE
            lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Step 4: Simple sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            final = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)
            
            return final
            
        except Exception as e:
            print(f"Enhancement error: {e}")
            return image
    
    @staticmethod
    def process_single_file(args):
        """Process a single file - static method for multiprocessing"""
        input_path, output_path = args
        
        try:
            # Check if output already exists
            if os.path.exists(output_path):
                return True, f"Skipped (exists): {os.path.basename(input_path)}"
            
            # Read image
            image = cv2.imread(input_path)
            if image is None:
                return False, f"Could not load: {input_path}"
            
            # Enhance image
            enhanced = FastImageEnhancer.enhance_image_optimized(image)
            
            # Save enhanced image
            success = cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Clean up memory
            del image, enhanced
            gc.collect()
            
            if success:
                return True, f"Enhanced: {os.path.basename(input_path)}"
            else:
                return False, f"Save failed: {output_path}"
                
        except Exception as e:
            return False, f"Error processing {input_path}: {str(e)}"
    
    def collect_all_files(self, source_base='shot-image-jpg'):
        """Collect all files to process"""
        file_pairs = []
        
        for root, dirs, files in os.walk(source_base):
            for file in files:
                if file.lower().endswith('.jpg'):
                    input_path = os.path.join(root, file)
                    
                    # Calculate output path
                    rel_path = os.path.relpath(input_path, source_base)
                    rel_dir = os.path.dirname(rel_path)
                    filename = os.path.basename(rel_path)
                    
                    # Create target directory path
                    target_dir = os.path.join('shot-image-treated', rel_dir) if rel_dir != '.' else 'shot-image-treated'
                    Path(target_dir).mkdir(parents=True, exist_ok=True)
                    
                    # Create output filename with prefix
                    output_filename = 'treated_' + filename
                    output_path = os.path.join(target_dir, output_filename)
                    
                    file_pairs.append((input_path, output_path))
        
        return file_pairs
    
    def process_all_parallel(self, source_base='shot-image-jpg'):
        """Process all images in parallel"""
        print("=" * 60)
        print("FAST IMAGE ENHANCEMENT SYSTEM v6.0")
        print(f"Parallel Processing with {self.num_workers} workers")
        print("=" * 60)
        
        start_time = time.time()
        
        # Collect all files
        print("Collecting files to process...")
        file_pairs = self.collect_all_files(source_base)
        print(f"Found {len(file_pairs)} files to process")
        
        if not file_pairs:
            print("No files to process!")
            return 0, 0
        
        # Process in parallel
        processed_count = 0
        enhanced_count = 0
        
        print(f"\nStarting parallel processing...")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all jobs
            future_to_file = {executor.submit(self.process_single_file, args): args for args in file_pairs}
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                processed_count += 1
                args = future_to_file[future]
                
                try:
                    success, message = future.result()
                    if success:
                        enhanced_count += 1
                    else:
                        self.errors.append(message)
                        
                    # Progress update every 100 files
                    if processed_count % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed
                        eta = (len(file_pairs) - processed_count) / rate if rate > 0 else 0
                        print(f"  Progress: {processed_count}/{len(file_pairs)} ({processed_count/len(file_pairs)*100:.1f}%) "
                              f"- Rate: {rate:.1f} files/sec - ETA: {eta/60:.1f}min")
                        
                except Exception as e:
                    self.errors.append(f"Future error: {str(e)}")
        
        # Final summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("ENHANCEMENT COMPLETE")
        print("=" * 60)
        print(f"Total files: {len(file_pairs)}")
        print(f"Successfully enhanced: {enhanced_count}")
        print(f"Errors: {len(self.errors)}")
        print(f"Processing time: {elapsed_time/60:.2f} minutes")
        print(f"Average rate: {len(file_pairs)/elapsed_time:.2f} files/second")
        
        if self.errors:
            print(f"\nFirst 10 errors:")
            for error in self.errors[:10]:
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        
        # Show final structure
        self._show_final_structure()
        
        return enhanced_count, len(self.errors)
    
    def _show_final_structure(self):
        """Show the final folder structure"""
        print("\n" + "-" * 60)
        print("Created folder structure:")
        print("-" * 60)
        
        if os.path.exists('shot-image-treated'):
            for root, dirs, files in os.walk('shot-image-treated'):
                level = root.replace('shot-image-treated', '').count(os.sep)
                indent = '  ' * level
                folder_name = os.path.basename(root) if level > 0 else 'shot-image-treated'
                print(f"{indent}{folder_name}/")
                
                sub_indent = '  ' * (level + 1)
                treated_files = [f for f in files if f.startswith('treated_') and f.endswith('.jpg')]
                if treated_files:
                    print(f"{sub_indent}({len(treated_files)} treated JPG files)")


def main():
    """Main execution function"""
    # Use optimal number of workers (but not too many to avoid memory issues)
    num_workers = min(6, mp.cpu_count())
    enhancer = FastImageEnhancer(num_workers=num_workers)
    
    try:
        # Process all images
        enhanced_count, error_count = enhancer.process_all_parallel()
        
        if error_count > 0:
            print(f"\n⚠️  Processing completed with {error_count} errors")
            return 1
        else:
            print(f"\n✅ Successfully enhanced {enhanced_count} images")
            return 0
            
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()