"""
BMP to JPG Converter for Golf Swing Analysis Project
Converts all BMP files from /shot-image to JPG format in /shot-image-jpg
"""

import os
from PIL import Image
from pathlib import Path
import shutil

def convert_bmp_to_jpg(source_dir='shot-image', target_dir='shot-image-jpg'):
    """
    Convert all BMP files from source directory to JPG in target directory
    Maintains the same folder structure
    """
    
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(exist_ok=True)
    
    # Statistics
    total_files = 0
    converted_files = 0
    skipped_files = 0
    errors = []
    
    print(f"Starting BMP to JPG conversion...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print("-" * 50)
    
    # Walk through all directories and files
    for root, dirs, files in os.walk(source_dir):
        # Calculate relative path
        rel_path = os.path.relpath(root, source_dir)
        
        # Create corresponding directory in target
        if rel_path != '.':
            target_subdir = os.path.join(target_dir, rel_path)
        else:
            target_subdir = target_dir
            
        Path(target_subdir).mkdir(parents=True, exist_ok=True)
        
        # Process BMP files in current directory
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        
        if bmp_files:
            print(f"\nProcessing {len(bmp_files)} files in: {rel_path}")
            
        for filename in bmp_files:
            total_files += 1
            source_file = os.path.join(root, filename)
            
            # Create target filename with .jpg extension
            target_filename = os.path.splitext(filename)[0] + '.jpg'
            target_file = os.path.join(target_subdir, target_filename)
            
            try:
                # Check if target already exists
                if os.path.exists(target_file):
                    print(f"  Skipping (already exists): {filename}")
                    skipped_files += 1
                    continue
                
                # Open and convert BMP to JPG
                with Image.open(source_file) as img:
                    # Convert RGBA to RGB if necessary
                    if img.mode in ('RGBA', 'LA', 'P'):
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as JPG with high quality
                    img.save(target_file, 'JPEG', quality=95, optimize=True)
                    converted_files += 1
                    
                    if converted_files % 50 == 0:
                        print(f"  Progress: {converted_files}/{total_files} files converted")
                        
            except Exception as e:
                error_msg = f"Error converting {source_file}: {str(e)}"
                print(f"  ERROR: {error_msg}")
                errors.append(error_msg)
    
    # Print summary
    print("\n" + "=" * 50)
    print("CONVERSION COMPLETE")
    print("=" * 50)
    print(f"Total BMP files found: {total_files}")
    print(f"Successfully converted: {converted_files}")
    print(f"Skipped (already exist): {skipped_files}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nError details:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    # Verify folder structure
    print("\n" + "-" * 50)
    print("Created folder structure:")
    for root, dirs, files in os.walk(target_dir):
        level = root.replace(target_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 2 * (level + 1)
        jpg_count = len([f for f in files if f.lower().endswith('.jpg')])
        if jpg_count > 0:
            print(f"{sub_indent}({jpg_count} JPG files)")
    
    return converted_files, skipped_files, errors

if __name__ == "__main__":
    # Run the conversion
    converted, skipped, errors = convert_bmp_to_jpg()
    
    # Exit with appropriate code
    if errors:
        exit(1)
    else:
        exit(0)