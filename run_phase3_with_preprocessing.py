"""
Phase 3: 기존 Ultra Enhancement 전처리 적용 후 Driver 분석
Apply existing Ultra Enhancement preprocessing then analyze driver images
"""

import sys
import os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import cv2
import numpy as np
from pathlib import Path
import shutil

# Import existing Ultra Enhancer
sys.path.append(str(Path(__file__).parent))
from src.processing.image_enhancement.ultra_enhancer import UltraImageEnhancer
from run_phase3_driver_simple import SimpleDriverAnalyzer

class Phase3WithPreprocessing:
    """Phase 3 with Ultra Enhancement preprocessing"""

    def __init__(self,
                 source_dir: str,
                 excel_path: str,
                 calibration_path: str):
        self.source_dir = Path(source_dir)
        self.excel_path = Path(excel_path)
        self.calibration_path = Path(calibration_path)

        # Create temporary directory for enhanced images
        self.enhanced_dir = self.source_dir.parent / "driver_enhanced"
        self.enhanced_dir.mkdir(exist_ok=True)

        # Initialize enhancer
        self.enhancer = UltraImageEnhancer()

    def preprocess_all_images(self):
        """Apply Ultra Enhancement to all BMP images"""
        print("="*70)
        print("STEP 1: Ultra Enhancement Preprocessing")
        print("="*70)

        # Find all BMP files
        bmp_files = list(self.source_dir.glob("*.bmp"))
        print(f"Found {len(bmp_files)} BMP files")

        enhanced_count = 0

        for bmp_file in bmp_files:
            try:
                # Load BMP
                img = cv2.imread(str(bmp_file))

                if img is None:
                    print(f"[WARN] Failed to load: {bmp_file.name}")
                    continue

                # Get original stats
                gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_orig = gray_orig.mean()

                # Apply Ultra Enhancement
                enhanced = self.enhancer.enhance_image_ultra(img)

                # Get enhanced stats
                gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                mean_enh = gray_enh.mean()

                # Save enhanced image
                output_path = self.enhanced_dir / bmp_file.name
                cv2.imwrite(str(output_path), enhanced)

                enhanced_count += 1

                improvement = ((mean_enh - mean_orig) / mean_orig) * 100
                print(f"  {bmp_file.name}: {mean_orig:.1f} -> {mean_enh:.1f} (+{improvement:.1f}%)")

            except Exception as e:
                print(f"[ERROR] {bmp_file.name}: {e}")

        print(f"\n[OK] Enhanced {enhanced_count}/{len(bmp_files)} images")
        print(f"[OK] Output: {self.enhanced_dir}")

        return enhanced_count

    def analyze_enhanced_images(self):
        """Run Phase 3 analysis on enhanced images"""
        print("\n" + "="*70)
        print("STEP 2: Analyze Enhanced Images")
        print("="*70)

        # Create analyzer with enhanced images directory
        analyzer = SimpleDriverAnalyzer(
            image_dir=str(self.enhanced_dir),
            excel_path=str(self.excel_path),
            calibration_path=str(self.calibration_path)
        )

        # Run analysis
        analyzer.run()

    def cleanup(self):
        """Clean up temporary files (optional)"""
        if self.enhanced_dir.exists():
            print(f"\nCleaning up: {self.enhanced_dir}")
            # Uncomment to delete:
            # shutil.rmtree(self.enhanced_dir)
            print("[INFO] Enhanced images kept for inspection")

    def run(self):
        """Full pipeline: preprocess -> analyze"""
        print("="*70)
        print("Phase 3 with Ultra Enhancement Preprocessing")
        print("="*70)

        # Step 1: Preprocess
        enhanced_count = self.preprocess_all_images()

        if enhanced_count == 0:
            print("[FAIL] No images enhanced")
            return

        # Step 2: Analyze
        self.analyze_enhanced_images()

        # Step 3: Cleanup (optional)
        # self.cleanup()

        print("\n" + "="*70)
        print("Phase 3 with Preprocessing COMPLETED")
        print("="*70)

if __name__ == "__main__":
    processor = Phase3WithPreprocessing(
        source_dir='C:/src/GolfSwingAnalysis_Final/data/1440_300_data/driver/1',
        excel_path='C:/src/GolfSwingAnalysis_Final/data/data-standard.xlsx',
        calibration_path='C:/src/GolfSwingAnalysis_Final/config/calibration_default.json'
    )

    processor.run()
