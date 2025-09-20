#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간편한 이미지 뷰어 스크립트
IrfanView와 기본 뷰어로 이미지를 열 수 있습니다.
"""

import os
import sys
import subprocess
from pathlib import Path

def open_with_irfanview(image_path):
    """IrfanView로 이미지 열기"""
    irfanview_path = r"C:\Program Files\IrfanView\i_view64.exe"
    if os.path.exists(irfanview_path):
        try:
            subprocess.Popen([irfanview_path, str(image_path)])
            return True
        except Exception as e:
            print(f"IrfanView 실행 오류: {e}")
    return False

def open_with_default(image_path):
    """기본 프로그램으로 이미지 열기"""
    try:
        os.startfile(str(image_path))
        return True
    except Exception as e:
        print(f"기본 프로그램 실행 오류: {e}")
    return False

def main():
    if len(sys.argv) < 2:
        print("사용법: python view_images.py <이미지_경로_또는_폴더>")
        print("\n처리된 이미지 폴더들:")
        
        # 처리된 이미지 폴더 표시
        base_path = Path("C:/src/GolfSwingAnalysis_Final_ver8/data/images")
        folders = [
            "shot-image-improved-v7-final/driver/no_marker_ball-1",
            "shot-image-improved-v7/driver/no_marker_ball-1", 
            "shot-image-bmp-treated-3/driver/no_marker_ball-1"
        ]
        
        for folder in folders:
            full_path = base_path / folder
            if full_path.exists():
                count = len(list(full_path.glob("*.png")))
                print(f"  {folder} ({count}개 파일)")
        
        return

    path = Path(sys.argv[1])
    
    if not path.exists():
        print(f"경로가 존재하지 않습니다: {path}")
        return
    
    if path.is_file():
        # 단일 파일 열기
        print(f"이미지 열기: {path}")
        if not open_with_irfanview(path):
            open_with_default(path)
    
    elif path.is_dir():
        # 폴더의 모든 이미지 파일 표시
        image_files = list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.bmp"))
        
        if not image_files:
            print(f"폴더에 이미지 파일이 없습니다: {path}")
            return
        
        print(f"\n{path}에서 {len(image_files)}개 이미지 발견:")
        for i, img_file in enumerate(sorted(image_files)[:10], 1):  # 처음 10개만 표시
            print(f"  {i}. {img_file.name}")
        
        if len(image_files) > 10:
            print(f"  ... 및 {len(image_files)-10}개 더")
        
        # 첫 번째 이미지 열기
        first_image = sorted(image_files)[0]
        print(f"\n첫 번째 이미지 열기: {first_image.name}")
        if not open_with_irfanview(first_image):
            open_with_default(first_image)

if __name__ == "__main__":
    main()