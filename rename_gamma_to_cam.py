#!/usr/bin/env python3
"""
Gamma_ 파일명을 Cam으로 변경하는 스크립트
"""

import os
import glob

def rename_gamma_to_cam(directory):
    """디렉토리 내의 Gamma_ 파일명을 Cam으로 변경"""
    print(f"Processing directory: {directory}")
    
    # Gamma_로 시작하는 bmp 파일 찾기
    pattern = os.path.join(directory, "Gamma_*.bmp")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} files to rename")
    
    renamed_count = 0
    for file_path in files:
        # 파일명에서 Gamma_를 Cam으로 변경
        new_name = file_path.replace("Gamma_", "Cam")
        
        try:
            os.rename(file_path, new_name)
            print(f"Renamed: {os.path.basename(file_path)} -> {os.path.basename(new_name)}")
            renamed_count += 1
        except Exception as e:
            print(f"Error renaming {file_path}: {e}")
    
    print(f"Successfully renamed {renamed_count} files in {directory}")
    return renamed_count

def main():
    """메인 함수"""
    directories = [
        "data2/driver/3",
        "data2/driver/4", 
        "data2/driver/5"
    ]
    
    total_renamed = 0
    
    for directory in directories:
        if os.path.exists(directory):
            renamed = rename_gamma_to_cam(directory)
            total_renamed += renamed
            print("-" * 50)
        else:
            print(f"Directory not found: {directory}")
    
    print(f"\nTotal files renamed: {total_renamed}")

if __name__ == "__main__":
    main()
