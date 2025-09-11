#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple BMP Direct Processing Script
BMP 파일의 선명도를 유지하면서 이미지 향상만 수행하는 간단한 스크립트
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import argparse

class SimpleBMPProcessor:
    """간단한 BMP 직접 처리기"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.stats = {'processed': 0, 'success': 0, 'failed': 0}
    
    def load_bmp_safely(self, file_path):
        """BMP 파일을 안전하게 로드 (선명도 유지)"""
        try:
            with Image.open(file_path) as pil_img:
                if pil_img.mode == 'L':  # 그레이스케일
                    return np.array(pil_img, dtype=np.uint8)
                elif pil_img.mode == 'RGB':
                    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    pil_img = pil_img.convert('RGB')
                    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            if self.debug:
                print(f"로드 실패 {file_path}: {e}")
            return None
    
    def enhance_image(self, image):
        """이미지 향상 (딤플 가시성 향상)"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. 히스토그램 균등화 (대비 향상)
        enhanced = cv2.equalizeHist(gray)
        
        # 2. CLAHE (적응형 히스토그램 균등화)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
        
        # 3. 언샤프 마스킹 (선명도 향상)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # 4. 노이즈 감소
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        return enhanced
    
    def process_file(self, input_path, output_path):
        """단일 파일 처리"""
        self.stats['processed'] += 1
        
        # BMP 로드
        image = self.load_bmp_safely(input_path)
        if image is None:
            self.stats['failed'] += 1
            return False
        
        # 이미지 향상
        enhanced = self.enhance_image(image)
        
        # 결과 저장
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, enhanced)
            self.stats['success'] += 1
            return True
        except Exception as e:
            if self.debug:
                print(f"저장 실패 {output_path}: {e}")
            self.stats['failed'] += 1
            return False
    
    def process_directory(self, input_dir, output_dir, max_files=None):
        """디렉토리 처리"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # BMP 파일 목록
        bmp_files = list(input_path.rglob("*.bmp"))
        if max_files:
            bmp_files = bmp_files[:max_files]
        
        print(f"처리할 BMP 파일: {len(bmp_files)}개")
        
        start_time = datetime.now()
        
        for i, bmp_file in enumerate(bmp_files):
            if self.debug or i % 50 == 0:  # 진행상황 표시
                print(f"처리 중... [{i+1}/{len(bmp_files)}] {bmp_file.name}")
            
            # 출력 경로 생성 (원본 구조 유지)
            relative_path = bmp_file.relative_to(input_path)
            output_file = output_path / relative_path
            
            # 처리
            success = self.process_file(str(bmp_file), str(output_file))
            
            if self.debug and not success:
                print(f"처리 실패: {bmp_file.name}")
        
        # 결과 출력
        total_time = (datetime.now() - start_time).total_seconds()
        success_rate = (self.stats['success'] / self.stats['processed'] * 100) if self.stats['processed'] > 0 else 0
        
        print(f"\n=== 처리 완료 ===")
        print(f"전체: {self.stats['processed']}")
        print(f"성공: {self.stats['success']}")
        print(f"실패: {self.stats['failed']}")
        print(f"성공률: {success_rate:.1f}%")
        print(f"처리 시간: {total_time:.1f}초")
        print(f"평균 시간: {total_time/len(bmp_files):.3f}초/파일")

def main():
    parser = argparse.ArgumentParser(description='간단한 BMP 직접 처리 스크립트')
    parser.add_argument('--input', '-i', required=True, help='입력 디렉토리')
    parser.add_argument('--output', '-o', required=True, help='출력 디렉토리')
    parser.add_argument('--max-files', type=int, help='최대 파일 수')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"입력 디렉토리가 없습니다: {args.input}")
        return
    
    processor = SimpleBMPProcessor(debug=args.debug)
    processor.process_directory(args.input, args.output, args.max_files)

if __name__ == "__main__":
    main()