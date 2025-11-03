#!/usr/bin/env python3
"""
Gamma_ 파일만 Cam으로 변경
ROI JSON 파일은 보존
"""

import os
import glob
from pathlib import Path

def rename_gamma_files_only(driver_dir="data2/driver"):
    """Gamma_ 파일만 Cam으로 변경 (ROI JSON 파일 보존)"""
    print(f"=== RENAMING GAMMA_ FILES ONLY (PRESERVING ROI JSON) ===")
    
    # 모든 하위 디렉토리 찾기
    subdirs = [d for d in os.listdir(driver_dir) if os.path.isdir(os.path.join(driver_dir, d))]
    subdirs.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
    
    print(f"Found {len(subdirs)} subdirectories: {subdirs}")
    
    total_renamed = 0
    total_preserved = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(driver_dir, subdir)
        print(f"\n--- Processing directory: {subdir} ---")
        
        # 모든 파일 목록
        files = os.listdir(subdir_path)
        print(f"  Found {len(files)} files")
        
        for filename in files:
            file_path = os.path.join(subdir_path, filename)
            
            # JSON 파일은 보존
            if filename.endswith('.json'):
                print(f"    PRESERVED: {filename} (JSON file)")
                total_preserved += 1
                continue
            
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
            
            # Gamma_가 아닌 다른 파일들도 보존
            else:
                print(f"    PRESERVED: {filename} (not Gamma_ file)")
                total_preserved += 1
    
    print(f"\n=== RENAME SUMMARY ===")
    print(f"Files renamed (Gamma_ -> Cam): {total_renamed}")
    print(f"Files preserved: {total_preserved}")
    print(f"Total processed: {total_renamed + total_preserved}")

def verify_rename_results(driver_dir="data2/driver"):
    """변경 결과 검증"""
    print(f"\n=== VERIFYING RENAME RESULTS ===")
    
    subdirs = [d for d in os.listdir(driver_dir) if os.path.isdir(os.path.join(driver_dir, d))]
    subdirs.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
    
    for subdir in subdirs:
        subdir_path = os.path.join(driver_dir, subdir)
        files = os.listdir(subdir_path)
        
        # 파일 유형별 분류
        cam_files = [f for f in files if "Cam" in f and not f.endswith('.json')]
        json_files = [f for f in files if f.endswith('.json')]
        other_files = [f for f in files if "Cam" not in f and not f.endswith('.json')]
        gamma_files = [f for f in files if f.startswith("Gamma_")]
        
        print(f"Directory {subdir}:")
        print(f"  Cam files: {len(cam_files)}")
        print(f"  JSON files: {len(json_files)}")
        print(f"  Other files: {len(other_files)}")
        print(f"  Gamma files: {len(gamma_files)}")
        
        if gamma_files:
            print(f"  WARNING: Still has Gamma files: {gamma_files}")
        
        if other_files:
            print(f"  Other non-Cam files: {other_files}")

def main():
    """메인 함수"""
    driver_dir = "data2/driver"
    
    if not os.path.exists(driver_dir):
        print(f"Driver directory {driver_dir} not found!")
        return
    
    # Gamma_ 파일만 변경
    rename_gamma_files_only(driver_dir)
    
    # 결과 검증
    verify_rename_results(driver_dir)
    
    print("\n=== GAMMA RENAME COMPLETE ===")

if __name__ == "__main__":
    main()

