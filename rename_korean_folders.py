"""
Korean to English Folder Name Converter
Converts Korean folder names to English for OpenCV compatibility
"""

import os
import shutil
from pathlib import Path

def rename_korean_folders():
    """
    Rename Korean folder names to English in both shot-image/driver and shot-image-jpg/driver
    """
    
    # Korean to English mapping based on CLAUDE.md
    folder_mapping = {
        "녹색 로고볼": "green_logo_ball",
        "로고, 마커없는 볼-1": "no_marker_ball-1", 
        "로고, 마커없는 볼-2": "no_marker_ball-2",
        "로고볼-1": "logo_ball-1",
        "로고볼-2": "logo_ball-2", 
        "마커볼": "marker_ball",
        "주황색 로고볼-1": "orange_logo_ball-1",
        "주황색 로고볼-2": "orange_logo_ball-2"
    }
    
    # Directories to process
    directories_to_process = [
        "shot-image/driver",
        "shot-image-jpg/driver"
    ]
    
    total_renamed = 0
    errors = []
    
    print("Starting Korean to English folder renaming...")
    print("=" * 50)
    
    for base_dir in directories_to_process:
        print(f"\nProcessing: {base_dir}")
        print("-" * 30)
        
        if not os.path.exists(base_dir):
            print(f"Directory {base_dir} does not exist, skipping...")
            continue
            
        # Get list of folders in directory
        try:
            folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
            
            for korean_name in folders:
                if korean_name in folder_mapping:
                    english_name = folder_mapping[korean_name]
                    old_path = os.path.join(base_dir, korean_name)
                    new_path = os.path.join(base_dir, english_name)
                    
                    try:
                        # Check if target already exists
                        if os.path.exists(new_path):
                            print(f"  Target already exists: {korean_name} -> {english_name}")
                            continue
                            
                        # Rename folder
                        os.rename(old_path, new_path)
                        print(f"  [OK] Renamed: {korean_name} -> {english_name}")
                        total_renamed += 1
                        
                    except Exception as e:
                        error_msg = f"Failed to rename {korean_name}: {str(e)}"
                        print(f"  [ERROR] Error: {error_msg}")
                        errors.append(error_msg)
                        
                elif korean_name not in folder_mapping.values():
                    print(f"  [INFO] Unknown folder: {korean_name} (skipping)")
        
        except Exception as e:
            error_msg = f"Failed to process directory {base_dir}: {str(e)}"
            print(f"[ERROR] Error: {error_msg}")
            errors.append(error_msg)
    
    # Summary
    print("\n" + "=" * 50)
    print("RENAMING COMPLETE")
    print("=" * 50)
    print(f"Total folders renamed: {total_renamed}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nError details:")
        for error in errors:
            print(f"  - {error}")
    
    # Verify results
    print("\n" + "-" * 50)
    print("Current folder structure:")
    for base_dir in directories_to_process:
        if os.path.exists(base_dir):
            print(f"\n{base_dir}:")
            folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
            for folder in sorted(folders):
                file_count = len([f for f in os.listdir(os.path.join(base_dir, folder)) 
                                if os.path.isfile(os.path.join(base_dir, folder, f))])
                print(f"  - {folder} ({file_count} files)")
    
    return total_renamed, errors

if __name__ == "__main__":
    renamed_count, errors = rename_korean_folders()
    
    if errors:
        exit(1)
    else:
        print(f"\n[SUCCESS] Successfully renamed {renamed_count} folders to English names")
        exit(0)