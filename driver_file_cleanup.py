#!/usr/bin/env python3
"""
드라이버 디렉토리 파일 정리 스크립트
- Gamma_를 Cam으로 변경
- Cam이 포함되지 않은 파일 삭제
- 이미 Cam이 포함된 파일은 보존
"""

import os
import glob
import shutil
from pathlib import Path

def cleanup_driver_directory(driver_dir="data2/driver"):
    """드라이버 디렉토리 정리"""
    print(f"=== CLEANING UP DRIVER DIRECTORY: {driver_dir} ===")
    
    # 모든 하위 디렉토리 찾기
    subdirs = [d for d in os.listdir(driver_dir) if os.path.isdir(os.path.join(driver_dir, d))]
    subdirs.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
    
    print(f"Found {len(subdirs)} subdirectories: {subdirs}")
    
    total_renamed = 0
    total_deleted = 0
    total_preserved = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(driver_dir, subdir)
        print(f"\n--- Processing directory: {subdir} ---")
        
        # 모든 파일 목록
        files = os.listdir(subdir_path)
        print(f"  Found {len(files)} files")
        
        for filename in files:
            file_path = os.path.join(subdir_path, filename)
            
            # 이미 Cam이 포함된 파일은 보존
            if "Cam" in filename:
                print(f"    PRESERVED: {filename} (already contains 'Cam')")
                total_preserved += 1
                continue
            
            # Gamma_로 시작하는 파일을 Cam으로 변경
            if filename.startswith("Gamma_"):
                new_filename = filename.replace("Gamma_", "Cam", 1)
                new_file_path = os.path.join(subdir_path, new_filename)
                
                try:
                    os.rename(file_path, new_file_path)
                    print(f"    RENAMED: {filename} -> {new_filename}")
                    total_renamed += 1
                except Exception as e:
                    print(f"    ERROR renaming {filename}: {e}")
            
            # Cam이 포함되지 않은 파일 삭제
            else:
                try:
                    os.remove(file_path)
                    print(f"    DELETED: {filename} (no 'Cam' in filename)")
                    total_deleted += 1
                except Exception as e:
                    print(f"    ERROR deleting {filename}: {e}")
    
    print(f"\n=== CLEANUP SUMMARY ===")
    print(f"Files renamed (Gamma_ -> Cam): {total_renamed}")
    print(f"Files deleted (no Cam): {total_deleted}")
    print(f"Files preserved (already Cam): {total_preserved}")
    print(f"Total processed: {total_renamed + total_deleted + total_preserved}")

def verify_cleanup(driver_dir="data2/driver"):
    """정리 결과 검증"""
    print(f"\n=== VERIFYING CLEANUP RESULTS ===")
    
    subdirs = [d for d in os.listdir(driver_dir) if os.path.isdir(os.path.join(driver_dir, d))]
    subdirs.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
    
    for subdir in subdirs:
        subdir_path = os.path.join(driver_dir, subdir)
        files = os.listdir(subdir_path)
        
        # Cam 파일과 기타 파일 분류
        cam_files = [f for f in files if "Cam" in f]
        other_files = [f for f in files if "Cam" not in f]
        
        print(f"Directory {subdir}: {len(cam_files)} Cam files, {len(other_files)} other files")
        
        if other_files:
            print(f"  WARNING: Still has non-Cam files: {other_files}")
        
        # Cam1, Cam2 파일 개수 확인
        cam1_files = [f for f in cam_files if "Cam1" in f]
        cam2_files = [f for f in cam_files if "Cam2" in f]
        
        print(f"  Cam1 files: {len(cam1_files)}")
        print(f"  Cam2 files: {len(cam2_files)}")

def main():
    """메인 함수"""
    driver_dir = "data2/driver"
    
    if not os.path.exists(driver_dir):
        print(f"Driver directory {driver_dir} not found!")
        return
    
    # 파일 정리
    cleanup_driver_directory(driver_dir)
    
    # 결과 검증
    verify_cleanup(driver_dir)
    
    print("\n=== FILE CLEANUP COMPLETE ===")

if __name__ == "__main__":
    main()
