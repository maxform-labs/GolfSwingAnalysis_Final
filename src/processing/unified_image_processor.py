#!/usr/bin/env python3
"""
통합 이미지 처리 시스템 v4.2
Unified Image Processing System

이 모듈은 모든 이미지 처리 기능을 통합합니다:
- BMP → JPG 변환
- Ultra 이미지 향상
- Fast 병렬 처리
- 한국어 폴더명 → 영어 변환
- 경로 정규화
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import shutil
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from datetime import datetime
import json

# 폴더명 매핑 테이블
KOREAN_TO_ENGLISH_MAPPING = {
    "7번 아이언": "7iron",
    "7번아이언": "7iron", 
    "7iron": "7iron",
    "로고, 마커없는 볼-1": "no_marker_ball-1",
    "로고, 마커없는 볼-2": "no_marker_ball-2",
    "로고,마커없는볼-1": "no_marker_ball-1",
    "로고,마커없는볼-2": "no_marker_ball-2",
    "로고볼-1": "logo_ball-1",
    "로고볼-2": "logo_ball-2",
    "마커볼": "marker_ball",
    "녹색 로고볼": "green_logo_ball",
    "녹색로고볼": "green_logo_ball",
    "주황색 로고볼-1": "orange_logo_ball-1",
    "주황색 로고볼-2": "orange_logo_ball-2",
    "주황색로고볼-1": "orange_logo_ball-1",
    "주황색로고볼-2": "orange_logo_ball-2",
    "드라이버": "driver",
    "driver": "driver"
}

class UnifiedImageProcessor:
    """통합 이미지 처리기"""
    
    def __init__(self, 
                 target_accuracy: float = 0.95,
                 max_workers: int = 4,
                 log_level: str = 'INFO'):
        self.target_accuracy = target_accuracy
        self.max_workers = max_workers
        self.logger = self._setup_logging(log_level)
        
        # 처리 통계
        self.stats = {
            'files_processed': 0,
            'files_converted': 0,
            'files_enhanced': 0,
            'folders_renamed': 0,
            'total_size_before': 0,
            'total_size_after': 0,
            'processing_time': 0.0,
            'compression_ratio': 0.0
        }
        
    def _setup_logging(self, level: str) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('UnifiedImageProcessor')
        logger.setLevel(getattr(logging, level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def process_directory(self, 
                         input_dir: str,
                         output_dir: str = None,
                         operations: List[str] = None) -> Dict[str, Any]:
        """
        디렉토리 전체 처리
        
        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리 (None이면 in-place 처리)
            operations: 수행할 작업 리스트 
                       ['convert', 'enhance', 'rename_folders', 'normalize_paths']
        
        Returns:
            Dict: 처리 결과 통계
        """
        start_time = datetime.now()
        
        if operations is None:
            operations = ['convert', 'enhance', 'rename_folders', 'normalize_paths']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else input_path
        
        self.logger.info(f"Starting directory processing: {input_path}")
        self.logger.info(f"Operations: {operations}")
        
        try:
            # 1. 폴더명 정규화 (한국어 → 영어)
            if 'rename_folders' in operations:
                self._rename_korean_folders(input_path)
            
            # 2. 파일 목록 수집
            image_files = self._collect_image_files(input_path)
            self.logger.info(f"Found {len(image_files)} image files")
            
            # 3. BMP → JPG 변환
            if 'convert' in operations:
                converted_files = self._convert_bmp_to_jpg_batch(image_files, output_path)
                self.stats['files_converted'] = len(converted_files)
            
            # 4. 이미지 향상 처리
            if 'enhance' in operations:
                enhance_files = self._collect_image_files(output_path, extensions=['.jpg', '.jpeg'])
                enhanced_files = self._enhance_images_batch(enhance_files, output_path)
                self.stats['files_enhanced'] = len(enhanced_files)
            
            # 5. 경로 정규화
            if 'normalize_paths' in operations:
                self._normalize_all_paths(output_path)
            
            # 처리 시간 계산
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processing_time'] = processing_time
            
            # 압축률 계산
            if self.stats['total_size_before'] > 0:
                self.stats['compression_ratio'] = (
                    (self.stats['total_size_before'] - self.stats['total_size_after']) / 
                    self.stats['total_size_before']
                ) * 100
            
            self.logger.info(f"Directory processing completed in {processing_time:.2f}s")
            return self.get_statistics()
            
        except Exception as e:
            self.logger.error(f"Directory processing failed: {str(e)}")
            raise
    
    def _collect_image_files(self, 
                           directory: Path, 
                           extensions: List[str] = None) -> List[Path]:
        """이미지 파일 수집"""
        if extensions is None:
            extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tiff']
        
        image_files = []
        for ext in extensions:
            image_files.extend(directory.rglob(f"*{ext}"))
            image_files.extend(directory.rglob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _rename_korean_folders(self, base_path: Path) -> int:
        """한국어 폴더명을 영어로 변환"""
        renamed_count = 0
        
        # 모든 디렉토리를 찾아서 이름 변경
        for root, dirs, files in os.walk(str(base_path), topdown=False):
            for dir_name in dirs:
                old_path = Path(root) / dir_name
                
                if dir_name in KOREAN_TO_ENGLISH_MAPPING:
                    new_name = KOREAN_TO_ENGLISH_MAPPING[dir_name]
                    new_path = Path(root) / new_name
                    
                    if not new_path.exists():
                        old_path.rename(new_path)
                        self.logger.info(f"Renamed folder: {dir_name} → {new_name}")
                        renamed_count += 1
                    else:
                        self.logger.warning(f"Target folder already exists: {new_name}")
        
        self.stats['folders_renamed'] = renamed_count
        return renamed_count
    
    def _convert_bmp_to_jpg_batch(self, 
                                 bmp_files: List[Path], 
                                 output_dir: Path) -> List[Path]:
        """BMP 파일들을 JPG로 일괄 변환"""
        bmp_files = [f for f in bmp_files if f.suffix.lower() == '.bmp']
        
        if not bmp_files:
            self.logger.info("No BMP files found for conversion")
            return []
        
        self.logger.info(f"Converting {len(bmp_files)} BMP files to JPG")
        
        def convert_single_file(bmp_file: Path) -> Optional[Path]:
            try:
                # 출력 경로 계산
                relative_path = bmp_file.relative_to(bmp_files[0].parent.parent)
                output_file = output_dir / relative_path.with_suffix('.jpg')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # PIL로 변환 (한글 경로 지원)
                with Image.open(str(bmp_file)) as img:
                    # RGB로 변환 (BMP는 때로 다른 모드일 수 있음)
                    rgb_img = img.convert('RGB')
                    rgb_img.save(str(output_file), 'JPEG', quality=85, optimize=True)
                
                # 파일 크기 통계
                original_size = bmp_file.stat().st_size
                converted_size = output_file.stat().st_size
                self.stats['total_size_before'] += original_size
                self.stats['total_size_after'] += converted_size
                
                return output_file
                
            except Exception as e:
                self.logger.error(f"Failed to convert {bmp_file}: {str(e)}")
                return None
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            converted_files = list(filter(None, executor.map(convert_single_file, bmp_files)))
        
        self.logger.info(f"Successfully converted {len(converted_files)}/{len(bmp_files)} files")
        return converted_files
    
    def _enhance_images_batch(self, 
                            image_files: List[Path],
                            output_dir: Path) -> List[Path]:
        """이미지들을 일괄 향상 처리"""
        self.logger.info(f"Enhancing {len(image_files)} images")
        
        def enhance_single_image(image_file: Path) -> Optional[Path]:
            try:
                # Ultra Enhancement 적용
                enhanced_image = self._ultra_enhance_image(str(image_file))
                
                if enhanced_image is not None:
                    # 출력 파일 경로
                    relative_path = image_file.relative_to(image_files[0].parent.parent) if image_files else image_file.name
                    output_file = output_dir / "enhanced" / relative_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # OpenCV로 저장
                    cv2.imwrite(str(output_file), enhanced_image)
                    return output_file
                    
            except Exception as e:
                self.logger.error(f"Failed to enhance {image_file}: {str(e)}")
            
            return None
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            enhanced_files = list(filter(None, executor.map(enhance_single_image, image_files)))
        
        self.logger.info(f"Successfully enhanced {len(enhanced_files)}/{len(image_files)} images")
        return enhanced_files
    
    def _ultra_enhance_image(self, image_path: str) -> Optional[np.ndarray]:
        """Ultra 이미지 향상 처리"""
        try:
            # OpenCV로 이미지 로드
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # 4단계 향상 처리
            # 1단계: 노이즈 제거
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            
            # 2단계: 선명도 향상 (언샵 마스킹)
            gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
            sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
            
            # 3단계: 명암 대비 개선 (CLAHE)
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 4단계: 감마 보정
            gamma = 1.2
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(enhanced, table)
            
            return gamma_corrected
            
        except Exception as e:
            self.logger.error(f"Ultra enhancement failed for {image_path}: {str(e)}")
            return None
    
    def _normalize_all_paths(self, base_path: Path):
        """모든 경로를 정규화"""
        # 파일명에서 특수문자 제거, 공백을 언더스코어로 변환
        for root, dirs, files in os.walk(str(base_path)):
            for file_name in files:
                old_path = Path(root) / file_name
                normalized_name = self._normalize_filename(file_name)
                
                if normalized_name != file_name:
                    new_path = Path(root) / normalized_name
                    if not new_path.exists():
                        old_path.rename(new_path)
                        self.logger.debug(f"Normalized filename: {file_name} → {normalized_name}")
    
    def _normalize_filename(self, filename: str) -> str:
        """파일명 정규화"""
        # 한글 제거, 특수문자를 언더스코어로 변환
        import re
        
        # 파일명과 확장자 분리
        name, ext = os.path.splitext(filename)
        
        # 정규화 규칙 적용
        normalized = re.sub(r'[^\w\-_\.]', '_', name)  # 영숫자, -, _, . 만 허용
        normalized = re.sub(r'_+', '_', normalized)     # 연속된 언더스코어 제거
        normalized = normalized.strip('_')               # 앞뒤 언더스코어 제거
        
        return normalized + ext
    
    def fast_process_directory(self, 
                             input_dir: str,
                             output_dir: str = None,
                             max_processes: int = None) -> Dict[str, Any]:
        """고속 병렬 디렉토리 처리"""
        if max_processes is None:
            max_processes = min(8, os.cpu_count() or 4)
        
        self.logger.info(f"Fast processing with {max_processes} processes")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else input_path / "processed"
        
        # 파일 목록 수집
        image_files = self._collect_image_files(input_path)
        
        # 파일을 청크로 나누어 병렬 처리
        chunk_size = max(1, len(image_files) // max_processes)
        file_chunks = [image_files[i:i + chunk_size] 
                      for i in range(0, len(image_files), chunk_size)]
        
        def process_chunk(chunk: List[Path]) -> int:
            """파일 청크 처리"""
            processed_count = 0
            for file_path in chunk:
                try:
                    if file_path.suffix.lower() == '.bmp':
                        # BMP → JPG 변환
                        output_file = output_path / file_path.relative_to(input_path).with_suffix('.jpg')
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        with Image.open(str(file_path)) as img:
                            rgb_img = img.convert('RGB')
                            rgb_img.save(str(output_file), 'JPEG', quality=85)
                            processed_count += 1
                    else:
                        # 이미지 향상
                        enhanced = self._ultra_enhance_image(str(file_path))
                        if enhanced is not None:
                            output_file = output_path / file_path.relative_to(input_path)
                            output_file.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(output_file), enhanced)
                            processed_count += 1
                            
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {str(e)}")
            
            return processed_count
        
        # 프로세스 풀로 병렬 실행
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            results = list(executor.map(process_chunk, file_chunks))
        
        total_processed = sum(results)
        self.stats['files_processed'] = total_processed
        
        self.logger.info(f"Fast processing completed: {total_processed} files")
        return self.get_statistics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return self.stats.copy()
    
    def save_processing_report(self, output_path: str):
        """처리 보고서 저장"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'settings': {
                'target_accuracy': self.target_accuracy,
                'max_workers': self.max_workers
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Processing report saved to {output_path}")

# 편의 함수들
def convert_bmp_directory(input_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """디렉토리의 모든 BMP 파일을 JPG로 변환"""
    processor = UnifiedImageProcessor()
    return processor.process_directory(input_dir, output_dir, operations=['convert'])

def enhance_image_directory(input_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """디렉토리의 모든 이미지를 향상 처리"""
    processor = UnifiedImageProcessor()
    return processor.process_directory(input_dir, output_dir, operations=['enhance'])

def normalize_korean_folders(input_dir: str) -> Dict[str, Any]:
    """한국어 폴더명을 영어로 정규화"""
    processor = UnifiedImageProcessor()
    return processor.process_directory(input_dir, operations=['rename_folders'])

def complete_image_processing(input_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """완전한 이미지 처리 (변환 + 향상 + 정규화)"""
    processor = UnifiedImageProcessor()
    return processor.process_directory(input_dir, output_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Image Processing System v4.2')
    parser.add_argument('input_dir', help='Input directory path')
    parser.add_argument('-o', '--output', help='Output directory path')
    parser.add_argument('-op', '--operations', nargs='+', 
                       choices=['convert', 'enhance', 'rename_folders', 'normalize_paths'],
                       default=['convert', 'enhance', 'rename_folders', 'normalize_paths'],
                       help='Operations to perform')
    parser.add_argument('--fast', action='store_true', help='Use fast processing mode')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--report', help='Save processing report to file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("통합 이미지 처리 시스템 v4.2")
    print("Unified Image Processing System v4.2") 
    print("=" * 60)
    
    processor = UnifiedImageProcessor(max_workers=args.workers)
    
    try:
        if args.fast:
            stats = processor.fast_process_directory(args.input_dir, args.output)
        else:
            stats = processor.process_directory(args.input_dir, args.output, args.operations)
        
        print(f"\n처리 완료 - Processing Completed:")
        print(f"  파일 처리: {stats['files_processed']}")
        print(f"  파일 변환: {stats['files_converted']}")
        print(f"  파일 향상: {stats['files_enhanced']}")
        print(f"  폴더 이름 변경: {stats['folders_renamed']}")
        print(f"  처리 시간: {stats['processing_time']:.2f}초")
        if stats['compression_ratio'] > 0:
            print(f"  압축률: {stats['compression_ratio']:.1f}%")
        
        if args.report:
            processor.save_processing_report(args.report)
            
    except Exception as e:
        print(f"처리 실패 - Processing failed: {str(e)}")
        exit(1)