#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMP Direct Loader for Golf Swing Analysis
BMP 파일의 선명도를 유지하면서 직접 로드하는 유틸리티
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, List
import logging
from pathlib import Path

class BMPLoader:
    """BMP 파일 전용 로더 (선명도 유지)"""
    
    def __init__(self, enable_cache: bool = True):
        """
        BMP 로더 초기화
        
        Args:
            enable_cache: 메모리 캐시 사용 여부
        """
        self.cache = {} if enable_cache else None
        self.logger = logging.getLogger(__name__)
    
    def load_bmp_safely(self, file_path: str) -> Optional[np.ndarray]:
        """
        BMP 파일을 안전하게 로드 (선명도 유지)
        
        Args:
            file_path: BMP 파일 경로
            
        Returns:
            로드된 이미지 배열 또는 None
        """
        try:
            # 캐시 확인
            if self.cache is not None and file_path in self.cache:
                return self.cache[file_path]
            
            # PIL로 원본 품질 유지하며 로드
            with Image.open(file_path) as pil_img:
                # BMP의 원본 색상 모드 유지
                if pil_img.mode == 'L':  # 그레이스케일
                    result = np.array(pil_img, dtype=np.uint8)
                elif pil_img.mode == 'RGB':
                    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                elif pil_img.mode == 'RGBA':
                    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
                else:
                    # 다른 모드는 RGB로 변환
                    pil_img = pil_img.convert('RGB')
                    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # 캐시에 저장
            if self.cache is not None:
                self.cache[file_path] = result
            
            return result
                
        except Exception as e:
            self.logger.error(f"BMP 로드 실패 {file_path}: {e}")
            return None
    
    def load_bmp_sequence(self, file_paths: List[str]) -> List[np.ndarray]:
        """
        BMP 파일 시퀀스 로드
        
        Args:
            file_paths: BMP 파일 경로 리스트
            
        Returns:
            로드된 이미지 배열 리스트
        """
        images = []
        for file_path in file_paths:
            img = self.load_bmp_safely(file_path)
            if img is not None:
                images.append(img)
            else:
                self.logger.warning(f"이미지 로드 실패: {file_path}")
        
        return images
    
    def clear_cache(self):
        """캐시 정리"""
        if self.cache is not None:
            self.cache.clear()
    
    def get_cache_info(self) -> Dict:
        """캐시 정보 반환"""
        if self.cache is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "cached_files": len(self.cache),
            "memory_usage": sum(img.nbytes for img in self.cache.values()) / (1024 * 1024)  # MB
        }

def create_bmp_loader(enable_cache: bool = True) -> BMPLoader:
    """
    BMP 로더 생성 팩토리 함수
    
    Args:
        enable_cache: 메모리 캐시 사용 여부
        
    Returns:
        BMPLoader 인스턴스
    """
    return BMPLoader(enable_cache)
